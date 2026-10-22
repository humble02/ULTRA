#!/usr/bin/env python3
"""Evaluate every Nth exact checkpoint from one distillation run, in order."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_student_benchmark import source_hashes, training_epoch, write_json

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def update_history(run, report, directory):
    history_path = run / "success_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    epoch = int(report["checkpoint_metadata"]["epoch"])
    for previous in history.values():
        assert previous["cohort"] == report["manifest"]["motions"], "Evaluation cohort changed"
    history[str(epoch)] = {
        "epoch": epoch, "checkpoint_sha256": report["checkpoint_sha256"],
        "initial_student_epoch": report["manifest"].get("initial_student_epoch", 0),
        "policy_architecture": report["manifest"].get("policy_architecture", "v2_distill"),
        "cohort": report["manifest"]["motions"], "summary": report["summary"],
        "report": str(directory.relative_to(ROOT)), "completed_at": now(),
    }
    write_json(history_path, history)
    rows = []
    for value in sorted(history.values(), key=lambda item: item["epoch"]):
        for engine, modes in value["summary"].items():
            for mode, metrics in modes.items():
                count = metrics["episodes"]
                rows.append(dict(epoch=value["epoch"], training_epochs=value["epoch"] - value.get("initial_student_epoch", 0),
                                 engine=engine, mode=mode, episodes=count,
                                 task_success_rate=metrics["task_success_rate"],
                                 strict_placement_success_rate=metrics["strict_placement_successes"] / count,
                                 carry_and_place_success_rate=metrics["carry_and_place_successes"] / count,
                                 falls=metrics["falls"]))
    temporary = run / "success_history.csv.tmp"
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(run / "success_history.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for engine in ("gym", "mujoco"):
        for mode in ("full_track", "object_obs"):
            selected = [r for r in rows if r["engine"] == engine and r["mode"] == mode]
            for axis, metric in zip(axes, ("task_success_rate", "carry_and_place_success_rate")):
                axis.plot([r["training_epochs"] for r in selected], [100 * r[metric] for r in selected],
                          marker="o", label=f"{engine} / {mode}")
    is_rl = report["manifest"].get("policy_architecture") == "v3_rl"
    for axis, title in zip(axes, ("Goal reaching without falls", "Lift, release and stable placement")):
        axis.set(title=title, xlabel="RL epoch" if is_rl else "Distillation epoch", ylabel="Success (%)", ylim=(-2, 102))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    label = "RL finetuning" if is_rl else "Fresh corrected student"
    fig.suptitle(f"{label} | fixed 20 motions per setting | prior-only evaluation")
    fig.savefig(run / "success_history.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=2000)
    parser.add_argument("--initial-student-epoch", type=int, default=0)
    parser.add_argument("--policy-architecture", choices=["v2_distill", "v3_rl"], default="v2_distill")
    parser.add_argument("--checkpoint-prefix", default="g1_student_vae")
    parser.add_argument("--training-status-name", default="training_status.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    run = args.run_dir.resolve()
    checkpoints = args.checkpoint_dir.resolve()
    assert ROOT in run.parents and ROOT in checkpoints.parents
    assert args.interval > 0 and args.interval % 500 == 0
    lock = (run / "evaluation_watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = json.loads((run / "source_sha256.json").read_text())
    state = dict(pid=os.getpid(), interval=args.interval, started_at=now(), completed_epochs=[],
                 initial_student_epoch=args.initial_student_epoch, policy_architecture=args.policy_architecture)
    state_path = run / "evaluation_status.json"
    epoch = args.initial_student_epoch + args.interval

    def status(phase, **extra):
        state.update(state=phase, target_epoch=epoch, target_training_epochs=epoch - args.initial_student_epoch, updated_at=now(),
                     current_training_epoch=training_epoch(run / "training.log"), **extra)
        write_json(state_path, state)

    try:
        while True:
            directory = run / "benchmarks" / f"epoch_{epoch}"
            complete = []
            if directory.exists():
                complete = [p.parent for p in directory.rglob("status.json")
                            if json.loads(p.read_text()).get("state") == "completed"]
            if complete:
                assert len(complete) == 1, "Multiple completed evaluations for the same epoch"
                report_dir = complete[0]
            else:
                checkpoint = checkpoints / f"{args.checkpoint_prefix}_{epoch:08d}.pth"
                while not checkpoint.exists():
                    status("waiting_checkpoint")
                    training_status = json.loads((run / args.training_status_name).read_text())
                    if training_status["state"] in ("failed", "completed", "stopped_after_checkpoint"):
                        status("training_stopped", training_state=training_status["state"])
                        return
                    time.sleep(20)
                current = source_hashes()
                drift = [p for p in sorted(set(expected) | set(current)) if expected.get(p) != current.get(p)]
                assert not drift, f"Source changed since training launch: {drift}"
                report_dir = None
                for attempt in range(1, 4):
                    candidate = directory / f"attempt_{attempt}"
                    if candidate.exists():
                        continue
                    candidate.mkdir(parents=True)
                    command = [sys.executable, str(ROOT / "scripts/run_student_benchmark.py"),
                               "--epoch", str(epoch), "--checkpoint", str(checkpoint), "--base", str(candidate),
                               "--training-log", str(run / "training.log"),
                               "--policy-architecture", args.policy_architecture,
                               "--initial-student-epoch", str(args.initial_student_epoch)]
                    status("evaluating", attempt=attempt, report_dir=str(candidate), command=command)
                    with (candidate / "runner.log").open("w") as log:
                        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            status("evaluating", evaluation_pid=process.pid)
                            try:
                                process.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                pass
                    if process.returncode == 0:
                        report_dir = candidate
                        break
                    status("evaluation_attempt_failed", exit_code=process.returncode)
                assert report_dir is not None, f"All evaluation attempts failed at epoch {epoch}"
            report = json.loads((report_dir / "summary.json").read_text())
            assert report["checkpoint_metadata"]["epoch"] == epoch
            assert report["verified_deploy_prior_only"]
            assert report["rollouts_use_training_pca_frames"]
            assert not report["source_changed_during_evaluation"]
            update_history(run, report, report_dir)
            state["completed_epochs"].append(epoch)
            status("completed_checkpoint", last_completed_epoch=epoch, latest_summary=report["summary"])
            epoch += args.interval
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
