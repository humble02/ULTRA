#!/usr/bin/env python3
"""Run the released configuration and retain logs, GPU usage and exit status."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["student", "finetune"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--label", help="separate log/status prefix for a new run")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / "output/reproduction"
    directory.mkdir(parents=True, exist_ok=True)
    command = ["bash", str(root / ("scripts/reproduce_" + args.stage + ".sh"))]
    if args.stage == "finetune":
        if not args.checkpoint or not args.checkpoint.is_file():
            parser.error("finetune requires an existing --checkpoint")
        command.append(str(args.checkpoint.resolve()))
    if args.resume_from:
        if not args.resume_from.is_file():
            parser.error("--resume-from must be an existing checkpoint")
        command.extend(["--resume_from", str(args.resume_from.resolve())])
    if args.output_path:
        command.extend(["--output_path", str(args.output_path.resolve())])
    label = args.label or args.stage
    if Path(label).name != label or label in (".", ".."):
        parser.error("--label must be a filename prefix")
    state_path = directory / (label + "_status.json")
    log_path = directory / (label + ".log")
    with log_path.open("a") as log, (directory / (label + "_gpu.csv")).open("a") as metrics:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        state = {"stage": args.stage, "pid": child.pid, "command": command,
                 "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "state": "running", "log": str(log_path), "hyperparameter_overrides": {}}
        state["resume_from"] = str(args.resume_from.resolve()) if args.resume_from else None
        state["output_path"] = str(args.output_path.resolve()) if args.output_path else None
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        while child.poll() is None:
            try:
                sample = subprocess.run(
                    ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                metrics.write(sample.stdout)
                metrics.flush()
            except (OSError, subprocess.TimeoutExpired) as error:
                log.write("GPU monitor: {}\n".format(error))
                log.flush()
            time.sleep(5)
        state.update(state="finished" if child.returncode == 0 else "failed",
                     exit_code=child.returncode,
                     finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
