#!/usr/bin/env python3
"""Preserve the next student checkpoint, release its GPU, then launch RL."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reproduction/finetune_from_10000"
LABEL = "finetune_from_10000"
STUDENT_PID = 3964027
STUDENT_RECORDER_PID = 3964018
STUDENT_SAVED = ROOT / "output/student_prior_fixed/g1_student_vae/nn/g1_student_vae_00013000.pth"
INITIAL = ROOT / "output/student_prior_fixed/g1_student_vae/nn/g1_student_vae_00010000.pth"


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def latest_epoch(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        matches = re.findall(rb"epoch_num:(\d+)", stream.read())
    return int(matches[-1]) if matches else None


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    validation = json.loads((OUT / "initialization_verification.json").read_text())
    assert validation["passed"] and validation["checkpoint_epoch"] == 10000
    assert validation["checkpoint_sha256"] == sha(INITIAL)
    assert not (ROOT / f"output/reproduction/{LABEL}_status.json").exists(), "RL run already exists"
    student = psutil.Process(STUDENT_PID)
    assert "ultra/run_distill.py" in student.cmdline()
    assert str(ROOT / "output/student_prior_fixed") in student.cmdline()
    created = student.create_time()
    for name, source in {
        "environment.yaml": ROOT / "ultra/data/cfg/g1_student_finetune.yaml",
        "training.yaml": ROOT / "ultra/data/cfg/train/rlg/g1_student_finetune.yaml",
    }.items():
        shutil.copy2(source, OUT / name)
    state = {"initial_checkpoint": str(INITIAL), "initial_epoch": 10000,
             "preserved_student_checkpoint": str(STUDENT_SAVED),
             "student_pid": STUDENT_PID, "student_process_start_time": created,
             "started_at": datetime.now(timezone.utc).isoformat(),
             "hyperparameter_overrides": {}, "original_num_envs": 4096,
             "horizon_length": 16, "minibatch_size": 8192, "mini_epochs": 4,
             "source_sha256": {p: sha(ROOT / p) for p in (
                 "ultra/data/cfg/g1_student_finetune.yaml", "ultra/data/cfg/train/rlg/g1_student_finetune.yaml",
                 "ultra/learning/ultra_agent_distill_vae_rl.py", "ultra/learning/ultra_network_builder_obj_v3.py",
                 "ultra/env/tasks/ultra_g1_distill_obj_v3rl.py", "scripts/train_finetune.sh")}}

    def status(phase, **values):
        state.update(state=phase, updated_at=datetime.now(timezone.utc).isoformat(), **values)
        write(OUT / "switch_status.json", state)

    try:
        while True:
            assert student.is_running() and student.create_time() == created
            status("waiting_student_checkpoint", current_student_epoch=latest_epoch(ROOT / "output/reproduction/student_prior_fixed.log"))
            if STUDENT_SAVED.exists():
                before = STUDENT_SAVED.stat()
                time.sleep(3)
                after = STUDENT_SAVED.stat()
                if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns and after.st_size > 0:
                    break
            time.sleep(10)
        import torch
        saved = torch.load(STUDENT_SAVED, map_location="cpu")
        assert saved["epoch"] == 13000 and "model" in saved and "optimizer" in saved
        state["preserved_student_checkpoint_sha256"] = sha(STUDENT_SAVED)
        del saved
        # This running UltraAgent overrides the base signal handler. Interrupt
        # only after its normal complete model+optimizer checkpoint is durable.
        assert student.create_time() == created
        status("stopping_student_after_checkpoint")
        student.send_signal(signal.SIGINT)
        student.wait(timeout=30)
        deadline = time.monotonic() + 20
        while psutil.pid_exists(STUDENT_RECORDER_PID) and time.monotonic() < deadline:
            time.sleep(1)
        path = ROOT / "output/reproduction/student_prior_fixed_status.json"
        old = json.loads(path.read_text())
        old.update(state="stopped_for_finetune", reason="User requested RL finetuning from epoch 10000; GPU released after epoch 13000 checkpoint was saved.",
                   last_preserved_checkpoint=str(STUDENT_SAVED), last_preserved_epoch=13000,
                   stopped_at=datetime.now(timezone.utc).isoformat())
        write(path, old)
        status("starting_rl")
        command = [sys.executable, str(ROOT / "scripts/record_training.py"), "finetune", "--checkpoint", str(INITIAL),
                   "--label", LABEL, "--output-path", str(ROOT / "output/finetune_from_10000")]
        child = subprocess.Popen(command, cwd=ROOT)
        while child.poll() is None:
            epoch = latest_epoch(ROOT / f"output/reproduction/{LABEL}.log")
            status("rl_running", recorder_pid=child.pid, current_rl_log_epoch=epoch,
                   relative_rl_epochs=None if epoch is None else epoch - 10000)
            time.sleep(10)
        status("rl_finished" if child.returncode == 0 else "rl_failed", exit_code=child.returncode)
        raise SystemExit(child.returncode)
    except Exception:
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
