"""Check concurrent matrix MAC and vector epilogue writeback in an RTL trace."""

import argparse
import json
from pathlib import Path
import re


def inspect(path: Path) -> dict:
    prefix = re.compile(r"T=\s*(\d+), H=(\d+), (\w+)\((.*)\)")
    fields = re.compile(r"(\w+)=\s*(0x[0-9a-f]+|\d+)")
    owners = {}
    matrix_harts = set()
    matrix_cycles = set()
    vector_cycles = set()
    matrix_cycles_by_hart = {}
    vector_cycles_by_hart = {}
    for line in path.open():
        match = prefix.fullmatch(line.strip())
        if not match:
            continue
        cycle, hart = int(match[1]), int(match[2])
        event = match[3]
        values = {key: int(value, 16 if value.startswith("0x") else 10)
                  for key, value in fields.findall(match[4])}
        tag = values.get("t1_tag")
        key = (hart, tag)
        if event == "T1Dispatch":
            owners[key] = values["inst"]
        instruction = owners.get(key, 0)
        if event == "MatrixWrite" and instruction == 0xF2880477:
            matrix_harts.add(hart)
            matrix_cycles.add(cycle)
            matrix_cycles_by_hart.setdefault(hart, set()).add(cycle)
        if event == "VrfPhysicalWrite" and 24 <= values["vd"] <= 27:
            if instruction & 0x7F == 0x57:
                vector_cycles.add(cycle)
                vector_cycles_by_hart.setdefault(hart, set()).add(cycle)
        if event == "T1Release":
            owners.pop(key, None)
    overlap = matrix_cycles & vector_cycles
    cross_hart_cycles = set()
    cross_hart_pairs = {}
    for matrix_hart, cycles in sorted(matrix_cycles_by_hart.items()):
        for vector_hart, writes in sorted(vector_cycles_by_hart.items()):
            if matrix_hart == vector_hart:
                continue
            shared_cycles = cycles & writes
            cross_hart_cycles.update(shared_cycles)
            if shared_cycles:
                cross_hart_pairs[f"{matrix_hart}->{vector_hart}"] = len(shared_cycles)
    return {
        "trace": str(path.resolve()), "matrix_harts": sorted(matrix_harts),
        "matrix_write_cycles": len(matrix_cycles),
        "vector_arithmetic_write_cycles": len(vector_cycles),
        "simultaneous_writeback_cycles": len(overlap),
        "cross_hart_simultaneous_writeback_cycles": len(cross_hart_cycles),
        "matrix_to_vector_hart_pair_cycles": cross_hart_pairs,
        "per_hart_matrix_write_cycles": {
            str(hart): len(cycles)
            for hart, cycles in sorted(matrix_cycles_by_hart.items())
        },
        "per_hart_simultaneous_writeback_cycles": {
            str(hart): len(cycles & vector_cycles_by_hart.get(hart, set()))
            for hart, cycles in sorted(matrix_cycles_by_hart.items())
        },
        "first_overlap_cycle": min(overlap) if overlap else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    arguments = parser.parse_args()
    report = inspect(arguments.trace)
    print(json.dumps(report, indent=2))
    if report["matrix_harts"] != [0, 1, 2, 3]:
        raise SystemExit("missing matrix computation on one or more harts")
    if not report["cross_hart_simultaneous_writeback_cycles"]:
        raise SystemExit("no cross-hart matrix/vector writeback overlap observed")


if __name__ == "__main__":
    main()
