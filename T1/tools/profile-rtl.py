"""Stream a complete RTL trace into bounded instruction and latency summaries."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess


EVENT = re.compile(r"T=\s*(\d+), H=(\d+), (\w+)\((.*)\)")
FIELDS = re.compile(r"(\w+)=\s*(0x[0-9a-f]+|\d+)")


class Profile:
    def __init__(self):
        self.events = Counter()
        self.instructions = defaultdict(Counter)
        self.previous = {}
        self.commits = {}
        self.issues = {}
        self.latencies = defaultdict(Counter)

    def consume(self, line):
        match = EVENT.match(line)
        if not match:
            return
        cycle, hart = int(match[1]), int(match[2])
        event = match[3]
        self.events[event] += 1
        if event not in ("RocketCommit", "T1Issue", "T1Result"):
            return
        fields = {name: int(value, 16 if value.startswith("0x") else 10)
                  for name, value in FIELDS.findall(match[4])}
        if event == "RocketCommit":
            key = hart, fields["pc"], fields["inst"]
            self.instructions[key]["count"] += 1
            if hart in self.previous:
                previous_key, previous_cycle = self.previous[hart]
                gap = cycle - previous_cycle
                self.instructions[previous_key]["following_gap_cycles"] += gap
                self.instructions[previous_key]["maximum_gap"] = max(
                    self.instructions[previous_key]["maximum_gap"], gap)
            self.previous[hart] = key, cycle
            if fields["is_t1"]:
                self.commits[hart, fields["t1_seq"]] = cycle
        else:
            key = hart, fields["t1_seq"]
            if event == "T1Issue":
                instruction = fields["inst"]
                self.issues[key] = cycle, instruction
                committed = self.commits.pop(key, None)
                if committed is not None:
                    self.latencies[instruction]["queue_cycles"] += cycle - committed
                    self.latencies[instruction]["queue_count"] += 1
            elif key in self.issues:
                issued, instruction = self.issues.pop(key)
                latency = cycle - issued
                summary = self.latencies[instruction]
                summary["count"] += 1
                summary["issue_to_result_cycles"] += latency
                summary["maximum_latency"] = max(summary["maximum_latency"], latency)

    def save(self, output):
        with (output / "instructions.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(("hart", "pc", "instruction", "count",
                             "following_gap_cycles", "maximum_gap"))
            for (hart, pc, instruction), counts in sorted(self.instructions.items()):
                writer.writerow((hart, hex(pc), hex(instruction), counts["count"],
                                 counts["following_gap_cycles"], counts["maximum_gap"]))
        return {
            "events": dict(self.events),
            "unmatched_issues": len(self.issues),
            "unmatched_commits": len(self.commits),
            "instruction_latencies": {hex(instruction): dict(counts)
                                      for instruction, counts in self.latencies.items()},
            "interpretation": "Retirement gaps partition each hart's observed timeline, "
            "not instruction execution costs. Queue and issue-to-result intervals "
            "can overlap across instructions/harts and must not be summed as wall time.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump-range", required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fifo = output / "events.fifo"
    os.mkfifo(fifo)
    descriptor = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    profile = Profile()
    pending = b""
    with (output / "simulator.log").open("w") as log:
        process = subprocess.Popen([
            str(arguments.emulator.resolve()),
            f"+t1_elf_file={arguments.image.resolve()}",
            f"+t1_dev_rtl_event_path={fifo}",
            f"+t1_memory_dump_path={output / 'memdump.bin'}",
            f"+t1_memory_dump_range={arguments.dump_range}",
        ], cwd=output, stdout=log, stderr=subprocess.STDOUT)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_READ)
                while True:
                    finished = process.poll() is not None
                    if selector.select(timeout=0.1) or finished:
                        try:
                            chunk = os.read(descriptor, 65536)
                        except BlockingIOError:
                            chunk = None
                        if chunk:
                            lines = (pending + chunk).split(b"\n")
                            pending = lines.pop()
                            for line in lines:
                                profile.consume(line.decode())
                        elif finished and chunk == b"":
                            break
            status = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            os.close(descriptor)
            fifo.unlink()
    report = profile.save(output)
    report.update(returncode=status, partial_line_bytes=len(pending))
    with arguments.image.open("rb") as stream:
        report["image_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    report["image"] = str(arguments.image.resolve())
    report["emulator"] = str(arguments.emulator.resolve())
    result_path = output / "sim_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    report["simulation_success"] = result.get("success") is True
    (output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"returncode": status, "events": report["events"]}))
    raise SystemExit(status != 0 or bool(pending) or not report["simulation_success"]
                     or not profile.events["RocketCommit"])


if __name__ == "__main__":
    main()
