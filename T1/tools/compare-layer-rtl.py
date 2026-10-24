"""Compare completed layer runs after checking their ISA readbacks."""

import argparse
import hashlib
import json
from pathlib import Path


def inspect(work: Path, reference: Path) -> dict:
    result = json.loads((work / "sim_result.json").read_text())
    if result.get("success") is not True:
        raise ValueError(f"RTL run did not succeed: {work}")
    actual = (work / "memdump.bin").read_bytes()
    expected = reference.read_bytes()
    if not actual or actual != expected:
        raise ValueError(f"RTL readback differs from ISA reference: {work}")
    events = [json.loads(line) for line in
              (work / "mmio-event.jsonl").read_text().splitlines()]
    begins = [event["cycle"] for event in events
              if event.get("value") == 0x71B00001]
    ends = [event["cycle"] for event in events
            if event.get("value") == 0x71B00002]
    if len(begins) != 1 or len(ends) != 1 or ends[0] <= begins[0]:
        raise ValueError(f"expected exactly one completed kernel interval: {work}")
    if result["total_cycles"] < ends[0]:
        raise ValueError(f"kernel interval exceeds simulation duration: {work}")
    return {
        "work": str(work.resolve()),
        "kernel_cycles": ends[0] - begins[0],
        "total_cycles": result["total_cycles"],
        "readback_bytes": len(actual),
        "readback_sha256": hashlib.sha256(actual).hexdigest(),
        "isa_readback_identical": True,
        "configuration": {key: result[key] for key in (
            "meta_vlen", "meta_dlen", "meta_lane_width", "meta_thread_count",
            "meta_te", "meta_isa", "meta_memory_model",
        )},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rvv", "--baseline", required=True, type=Path)
    parser.add_argument("--vme", "--candidate", required=True, type=Path)
    parser.add_argument("--rvv-reference", "--baseline-reference", required=True, type=Path)
    parser.add_argument("--vme-reference", "--candidate-reference", required=True, type=Path)
    parser.add_argument("--require-identical-readback", action="store_true",
                        help="also require byte-identical baseline/candidate memory dumps")
    parser.add_argument("--prefill", action="store_true",
                        help="label the paths serial/GEMM rather than RVV/VME")
    arguments = parser.parse_args()
    rvv = inspect(arguments.rvv, arguments.rvv_reference)
    vme = inspect(arguments.vme, arguments.vme_reference)
    if rvv["configuration"] != vme["configuration"]:
        raise ValueError("RTL configurations differ")
    if arguments.require_identical_readback:
        if ((arguments.rvv / "memdump.bin").read_bytes()
                != (arguments.vme / "memdump.bin").read_bytes()):
            raise ValueError("baseline and candidate readbacks differ")
    baseline_label = "serial" if arguments.prefill else "rvv"
    candidate_label = "gemm" if arguments.prefill else "vme"
    print(json.dumps({
        baseline_label: rvv,
        candidate_label: vme,
        f"{candidate_label}_over_{baseline_label}_kernel_cycles": vme["kernel_cycles"] / rvv["kernel_cycles"],
        f"speedup_{baseline_label}_over_{candidate_label}": rvv["kernel_cycles"] / vme["kernel_cycles"],
        "cross_readback_checked": arguments.require_identical_readback,
        "scope": "Kernel interval only; caller must ensure identical model, "
                 "layer, position, token count, input and simulator build. Each path is "
                 "checked against its own ISA reference; cross-path equality is "
                 "checked only when cross_readback_checked is true.",
    }, indent=2))


if __name__ == "__main__":
    main()
