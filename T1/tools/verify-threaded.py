"""Validate thread exchange and compare Llama inference with a C reference."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def run(command: list[str], name: str, output: Path, environment: dict, timeout: int) -> str:
    print(f"Running {name}", flush=True)
    with (output / f"{name}.stdout").open("w") as stdout:
        with (output / f"{name}.stderr").open("w") as stderr:
            subprocess.run(
                command, env=environment, stdout=stdout, stderr=stderr,
                check=True, timeout=timeout,
            )
    return (output / f"{name}.stdout").read_text()


def transcript(text: str) -> str:
    return text.split("=== T1 output ===\n", 1)[1].rsplit("\n=== ", 1)[0].rstrip("\n")


def tokens(text: str) -> list[tuple[int, int]]:
    return [(int(position), int(token)) for position, token in re.findall(
        r"\[pos (\d+) -> (\d+) \|", text
    )]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--emulator", required=True, type=Path)
    parser.add_argument("--kind", choices=("rtl", "pokedex"), required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-kind", choices=("fp32", "quantized"), default="fp32")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--keep-rtl-events", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.steps < 2:
        parser.error("at least two positions are required to exercise the KV cache")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = arguments.package.resolve()
    environment = os.environ.copy()
    environment.update({
        "MODULAR_DRIVER_PLUGINS": f"T1@{package}/lib/libT1Driver.so",
        "T1RT_EMULATOR": str(arguments.emulator.resolve()),
        "T1RT_TRAMPOLINE": str(package / "share/t1/trampoline.bin"),
        "T1RT_THREAD_COUNT": "4",
        "T1RT_POKEDEX_VLEN": "1024",
        "T1RT_MEMDUMP": "1",
        "T1RT_DUMP_MAX_BYTES": "4194304",
        "T1RT_VERBOSE": "1",
    })
    environment.pop("T1RT_PC_HISTOGRAM", None)
    environment.pop("T1RT_EMULATOR_KIND", None)
    if arguments.kind == "pokedex":
        environment["T1RT_EMULATOR_KIND"] = "pokedex"
    elif not arguments.keep_rtl_events:
        environment["T1RT_DISABLE_RTL_EVENTS"] = "1"
    for threads in range(1, 5):
        result = run(
            [str(package / "bin/t1-threaded"), str(threads)],
            f"exchange-{threads}", output, environment, arguments.timeout,
        )
        if f"Thread exchange OK: {threads} harts, 64 barriers" not in result:
            raise AssertionError(f"missing completion marker for {threads} harts")
        result = run(
            [str(package / "bin/t1-rmsnorm"), str(threads)],
            f"rmsnorm-{threads}", output, environment, arguments.timeout,
        )
        if f"RMSNorm OK: {threads} harts, 2048 elements" not in result:
            raise AssertionError(f"missing RMSNorm completion marker for {threads} harts")
    if arguments.reference_kind == "quantized":
        for binary, marker in (
            ("t1-vme", "VME matrix OK:"),
            ("t1-vme-quantized", "Quantized VME OK:"),
        ):
            result = run([str(package / "bin" / binary)], binary, output,
                         environment, arguments.timeout)
            if marker not in result:
                raise AssertionError(f"missing completion marker for {binary}")
    model_arguments = [
        str(arguments.model.resolve()), str(arguments.tokenizer.resolve()),
        arguments.prompt, str(arguments.steps),
    ]
    single = run([str(package / "bin/t1llama-single"), *model_arguments],
                 "single", output, environment, arguments.timeout)
    threaded = run([str(package / "bin/t1llama"), *model_arguments],
                   "threaded", output, environment, arguments.timeout)
    reference = run([
        str(arguments.reference.resolve()), str(arguments.model.resolve()),
        "-t", "0", "-n", str(arguments.steps), "-z",
        str(arguments.tokenizer.resolve()), "-i", arguments.prompt,
    ], "reference", output, environment, arguments.timeout)
    single_tokens = tokens(single)
    threaded_tokens = tokens(threaded)
    prompt_count = re.search(r"prompt tokens: (\d+)", threaded)
    if not prompt_count or arguments.steps < int(prompt_count[1]):
        raise AssertionError("run did not reach a generated token beyond the prompt")
    batch_match = re.search(r"prefill batch: (\d+)", threaded)
    batch_size = int(batch_match[1]) if batch_match else 1
    if batch_size < 1:
        raise AssertionError("invalid prefill batch size")
    expected_positions = []
    position = 0
    while position < arguments.steps:
        count = min(batch_size, arguments.steps - position,
                    max(1, int(prompt_count[1]) - position))
        position += count
        expected_positions.append(position - 1)
    if ([position for position, _ in threaded_tokens] != expected_positions
            or single_tokens != threaded_tokens):
        raise AssertionError("incomplete inference or different single/four-hart tokens")
    if transcript(single) != transcript(threaded) or transcript(threaded) != reference.rstrip("\n"):
        raise AssertionError("inference transcript differs from the C reference")
    report = {
        "kind": arguments.kind, "package": str(package),
        "reference_kind": arguments.reference_kind,
        "reference": str(arguments.reference.resolve()),
        "emulator": str(arguments.emulator.resolve()),
        "model": str(arguments.model.resolve()), "threads": 4,
        "positions": arguments.steps, "tokens": threaded_tokens,
        "transcript": transcript(threaded), "status": "passed",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
