"""Run one RTL image while retaining only a bounded tail of its event trace."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lines", type=int, default=100000)
    parser.add_argument("--stall-timeout", type=int, default=1000000)
    parser.add_argument("--wall-timeout", type=float)
    arguments = parser.parse_args()
    if arguments.lines <= 0:
        parser.error("--lines must be positive")
    if arguments.wall_timeout is not None and arguments.wall_timeout <= 0:
        parser.error("--wall-timeout must be positive")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fifo = output / "events.fifo"
    os.mkfifo(fifo)
    tail = deque(maxlen=arguments.lines)
    count = 0
    timed_out = False
    pending = b""
    descriptor = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    with (output / "simulator.log").open("w") as log:
        process = subprocess.Popen([
            str(arguments.emulator.resolve()),
            f"+t1_elf_file={arguments.image.resolve()}",
            f"+t1_dev_rtl_event_path={fifo}",
            f"+t1_timeout={arguments.stall_timeout}",
        ], cwd=output, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_READ)
                while True:
                    if (arguments.wall_timeout is not None
                            and time.monotonic() - started >= arguments.wall_timeout
                            and process.poll() is None):
                        timed_out = True
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    finished = process.poll() is not None
                    if selector.select(timeout=0.1) or finished:
                        try:
                            chunk = os.read(descriptor, 65536)
                        except BlockingIOError:
                            chunk = b""
                        if chunk:
                            lines = (pending + chunk).split(b"\n")
                            pending = lines.pop()
                            for line in lines:
                                tail.append(line.decode(errors="replace") + "\n")
                                count += 1
                        elif finished:
                            break
            status = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            os.close(descriptor)
            fifo.unlink()
            (output / "rtl-tail.log").write_text("".join(tail))
    report = {"returncode": status, "events": count, "retained": len(tail),
              "wall_timeout": timed_out, "partial_line_bytes": len(pending)}
    (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(124 if timed_out else 0 if status == 0 else 1)


if __name__ == "__main__":
    main()
