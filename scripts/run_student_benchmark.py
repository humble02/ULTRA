#!/usr/bin/env python3
"""Wait for an exact saved epoch and run the same 80-rollout student benchmark.

Run under scripts/ultra_env.sh, preferably as a systemd user service. The
training process is never signalled or modified. All state lives in --base.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "output/reproduction/student_10000_benchmark"
KINDS = ("largebox", "plasticbox", "smallbox", "suitcase")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def source_hashes():
    paths = set((ROOT / "ultra").rglob("*.py"))
    paths.update((ROOT / "ultra/data/cfg").rglob("*.yaml"))
    paths.update((ROOT / "ultra/data/assets").rglob("*.xml"))
    paths.update((ROOT / "scripts").glob("*benchmark*.py"))
    paths.update([ROOT / "ultra/data/assets/objects/point_pca_frames.json", ROOT / "scripts/ultra_env.sh"])
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def training_epoch(log=None):
    log = ROOT / (log or "output/reproduction/student_prior_fixed.log")
    if not log.exists():
        return None
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 65536))
        matches = re.findall(rb"epoch_num:(\d+)", stream.read())
    return int(matches[-1]) if matches else None


def prepare(base, epoch, checkpoint=None, policy_architecture="v2_distill", training_log=None, initial_student_epoch=10000,
            finetune_restart_epoch=None):
    base.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OLD / "manifest.json").read_text())
    manifest.update(epoch=epoch,
                    checkpoint=f"output/student_prior_fixed/g1_student_vae/nn/g1_student_vae_{epoch:08d}.pth",
                    evaluation_revision="training4096_reference_pca_v1",
                    cohort_source=str((OLD / "manifest.json").relative_to(ROOT)))
    if checkpoint is not None:
        manifest["checkpoint"] = str(checkpoint.resolve().relative_to(ROOT))
    if policy_architecture == "v3_rl":
        manifest.update(policy_architecture="v3_rl",
                        policy_training_config="ultra/data/cfg/train/rlg/g1_student_finetune.yaml",
                        initial_student_epoch=initial_student_epoch, rl_finetune_epochs=epoch - initial_student_epoch,
                        training_log="output/reproduction/finetune_from_10000_checkpointed.log")
    if training_log is not None:
        manifest["training_log"] = str(training_log.resolve().relative_to(ROOT))
    if finetune_restart_epoch is not None:
        assert initial_student_epoch <= finetune_restart_epoch <= epoch
        manifest.update(finetune_restart_epoch=finetune_restart_epoch,
                        updates_since_restart=epoch - finetune_restart_epoch)
    path = base / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Existing benchmark manifest differs"
    else:
        write_json(path, manifest)
    for row in manifest["motions"]:
        source = ROOT / row["motion"]
        assert source.is_file(), source
        link = base / "motions" / row["object"] / source.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(source)
        assert link.resolve() == source.resolve(), link
    # Previously verified actor exports bootstrap the CPU benchmark; the new
    # native evaluations overwrite them, and the summary rechecks equality.
    for kind in KINDS:
        target = base / "gym" / kind / "object_inertia.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(OLD / "gym" / kind / "object_inertia.json", target)
        target = base / f"inertia_alignment_{kind}.json"
        if not target.exists():
            shutil.copy2(OLD / target.name, target)
    target = base / "point_encoding_verification.json"
    if not target.exists():
        shutil.copy2(OLD / target.name, target)
    write_json(base / "point_sampling_correction.json", {
        "point_fixed_grid_sampling": True,
        "point_pca_frame_policy": "training4096_reference_v1",
        "all_rollouts_are_new_after_pca_fix": True,
        "previous_reports_preserved_at": str(OLD.relative_to(ROOT)),
    })
    target = base / "scheduled_source_sha256.json"
    if not target.exists():
        write_json(target, source_hashes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-architecture", choices=["v2_distill", "v3_rl"], default="v2_distill")
    parser.add_argument("--training-log", type=Path, help="Log for the training run being evaluated")
    parser.add_argument("--initial-student-epoch", type=int, default=10000)
    parser.add_argument("--finetune-restart-epoch", type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    base = args.base.resolve()
    assert base.is_relative_to(ROOT) if hasattr(base, "is_relative_to") else str(base).startswith(str(ROOT) + "/")
    manifest = prepare(base, args.epoch, args.checkpoint, args.policy_architecture, args.training_log,
                       args.initial_student_epoch, args.finetune_restart_epoch)
    if args.prepare_only:
        print(base, flush=True)
        return
    # One owner per output directory, in addition to the shared GPU lock.
    owner = (base / "runner.lock").open("w")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (base / "summary.json").exists():
        print("Already completed:", base, flush=True)
        return
    if any((base / "gym" / k / "metrics.json").exists() for k in KINDS) or (base / "mujoco/metrics.json").exists():
        raise RuntimeError("Partial evaluation exists; preserve it and use a new --base")
    state = {"epoch": args.epoch, "pid": os.getpid(), "base": str(base),
             "checkpoint": manifest["checkpoint"], "started_at": datetime.now(timezone.utc).isoformat()}

    def status(phase, **values):
        state.update(state=phase, updated_at=datetime.now(timezone.utc).isoformat(),
                     current_training_epoch=training_epoch(manifest.get("training_log")), **values)
        write_json(base / "status.json", state)

    cpu = None
    active = None
    try:
        checkpoint = ROOT / manifest["checkpoint"]
        while True:
            status("waiting_checkpoint")
            if checkpoint.is_file() and checkpoint.stat().st_size > 0:
                before = checkpoint.stat()
                time.sleep(10)
                after = checkpoint.stat()
                if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                    break
            time.sleep(30)
        # Validate the saved internal epoch before any rollout, not the name alone.
        import torch
        torch.set_num_threads(2)
        saved = torch.load(checkpoint, map_location="cpu")
        assert saved["epoch"] == args.epoch, (saved["epoch"], args.epoch)
        assert saved.get("model"), "Missing model state"
        metadata = {"checkpoint": manifest["checkpoint"], "epoch": saved["epoch"],
                    "frame": saved.get("frame"), "has_model": True,
                    "file_size_bytes": checkpoint.stat().st_size, "sha256": sha(checkpoint)}
        del saved
        write_json(base / "checkpoint_metadata.json", metadata)
        lock = (base.parent / "gym_eval.lock").open("w")
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status("waiting_evaluation_lock")
                time.sleep(30)
        sources = source_hashes()
        expected = json.loads((base / "scheduled_source_sha256.json").read_text())
        changed = sorted(p for p in set(expected) | set(sources) if expected.get(p) != sources.get(p))
        write_json(base / "evaluation_source_sha256.json", sources)
        status("running", source_changed_since_scheduled=changed)

        def launch(script, log, *extra):
            cmd = [sys.executable, str(ROOT / "scripts" / script), "--base", str(base), *extra]
            with (base / log).open("w") as stream:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            print("Started", proc.pid, " ".join(cmd), flush=True)
            return proc

        cpu = launch("benchmark_student_mujoco.py", "mujoco.log")
        for kind in KINDS:
            while True:
                if cpu.poll() not in (None, 0):
                    raise RuntimeError(f"MuJoCo evaluation exited {cpu.returncode}; see mujoco.log")
                free = int(subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "--id=0"
                ], text=True).strip())
                if free >= 1900:
                    break
                status("waiting_gpu_memory", object=kind, gpu_free_mib=free)
                time.sleep(30)
            active = launch("benchmark_student_gym.py", f"gym_{kind}.log", "--object", kind)
            while active.poll() is None:
                status("running_gym", object=kind, gym_pid=active.pid, mujoco_pid=cpu.pid)
                try:
                    active.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if active.returncode:
                raise RuntimeError(f"Gym {kind} exited {active.returncode}; see gym_{kind}.log")
            active = None
        while cpu.poll() is None:
            status("waiting_mujoco")
            try:
                cpu.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if cpu.returncode:
            raise RuntimeError(f"MuJoCo evaluation exited {cpu.returncode}; see mujoco.log")
        active = launch("summarize_student_benchmark.py", "summary.log")
        if active.wait():
            raise RuntimeError("Summary validation failed; see summary.log")
        active = None
        assert sha(checkpoint) == metadata["sha256"], "Checkpoint changed during evaluation"
        end_sources = source_hashes()
        drift = sorted(p for p in set(sources) | set(end_sources) if sources.get(p) != end_sources.get(p))
        report = json.loads((base / "summary.json").read_text())
        report.update(source_changed_since_scheduled=changed, source_changed_during_evaluation=drift,
                      evaluation_source_sha256=sources)
        write_json(base / "summary.json", report)
        status("completed", summary=report["summary"], source_changed_during_evaluation=drift,
               completed_at=datetime.now(timezone.utc).isoformat())
        reports = {}
        for epoch in (10000, 12000):
            path = base.parent / f"epoch_{epoch}" / "summary.json"
            if path.exists():
                reports[str(epoch)] = json.loads(path.read_text())
        if len(reports) == 2:
            comparable = (reports["10000"]["manifest"]["motions"] == reports["12000"]["manifest"]["motions"]
                          and reports["10000"]["evaluation_source_sha256"] == reports["12000"]["evaluation_source_sha256"]
                          and not any(r["source_changed_during_evaluation"] for r in reports.values()))
            write_json(base.parent / "comparison_epochs.json", {
                "same_cohort_and_source": comparable,
                "epochs": {epoch: {"checkpoint_sha256": r["checkpoint_sha256"], "summary": r["summary"]}
                           for epoch, r in reports.items()}})
        print(json.dumps(state, indent=2), flush=True)
    except BaseException:
        for proc in (active, cpu):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
