#!/usr/bin/env python3
"""Prepare reproducible model+optimizer continuation and its evaluation schedule."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401
import torch
import yaml
from run_student_benchmark import source_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "experiments/local_5090/reset_only")
    parser.add_argument("--initial-student-epoch", type=int, default=6000)
    parser.add_argument("--source-num-envs", type=int, default=4096)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-envs", type=int, default=8)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    assert args.label and all(c.isalnum() or c in "_-" for c in args.label)
    checkpoint = args.checkpoint.resolve()
    run = ROOT / "output/reproduction" / args.label
    output = ROOT / "output" / args.label
    assert not run.exists() and not output.exists(), "Use a new experiment label"
    saved = torch.load(checkpoint, map_location="cpu")
    epoch = int(saved["epoch"])
    assert epoch > args.initial_student_epoch and saved["optimizer"]["state"]
    assert any("rl_finetune_film" in key for key in saved["model"]), "Expected a complete V3 RL checkpoint"
    environment = yaml.safe_load((args.config_dir / "environment.yaml").read_text())
    training = yaml.safe_load((args.config_dir / "training.yaml").read_text())
    config = training["params"]["config"]
    assert config["actor_schedule_start_epoch"] == args.initial_student_epoch
    if args.seed is not None:
        training["params"]["seed"] = args.seed
    gate_path = args.config_dir / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() and not args.smoke else None
    if gate:
        assert gate["interval"] == 200 and gate["max_new_updates"] == 1000
        config["max_epochs"] = epoch + gate["max_new_updates"] - 1
        # Evaluated checkpoints must never be overwritten by the base loop.
        config["save_frequency"] = 0
    run.mkdir(parents=True)
    if gate:
        for name in ("gate.json", "development_cohort.json", "acceptance_cohort.json"):
            shutil.copy2(args.config_dir / name, run / name)
    if args.smoke:
        manifest = json.loads((ROOT / "output/reproduction/student_10000_benchmark/manifest.json").read_text())
        motions = run / "motions"
        motions.mkdir()
        for kind in ("largebox", "plasticbox", "smallbox", "suitcase"):
            selected = [row for row in manifest["motions"] if row["object"] == kind][:2]
            assert len(selected) == 2
            for row in selected:
                source = ROOT / row["motion"]
                (motions / source.name).symlink_to(source)
        assert args.smoke_envs >= 8 and args.smoke_envs % 4 == 0
        environment["env"].update(numEnvs=args.smoke_envs, motion_file=str(motions))
        config.update(horizon_length=2, minibatch_size=args.smoke_envs * 2, mini_epochs=2,
                      max_epochs=epoch + 2, save_frequency=1)
        output = run / "training_output"
    for name, value in (("training.yaml", training), ("environment.yaml", environment)):
        (run / name).write_text(yaml.safe_dump(value, sort_keys=False))
    num_envs = environment["env"]["numEnvs"]
    frame = int(saved["frame"])
    request = {
        "requested_at": datetime.now(timezone.utc).isoformat(), "initial_checkpoint": str(checkpoint),
        "experiment_profile": str(args.config_dir.resolve()),
        "initial_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "initial_checkpoint_epoch": epoch, "initial_student_epoch": args.initial_student_epoch,
        "initial_rl_epochs": epoch - args.initial_student_epoch, "resume_optimizer": True,
        "actor_schedule_start_epoch": args.initial_student_epoch,
        "initial_frame": frame, "initial_environment_steps": frame // args.source_num_envs,
        "source_num_envs": args.source_num_envs, "output_path": str(output),
        "num_envs": num_envs, "distillation_envs": num_envs // 4, "ppo_envs": num_envs - num_envs // 4,
        "horizon_length": config["horizon_length"], "minibatch_size": config["minibatch_size"],
        "mini_epochs": config["mini_epochs"], "mixed_precision": config["mixed_precision"],
        "save_frequency": config["save_frequency"], "activation_checkpointing": True,
        "prior_action_loss_coefficient": config["prior_action_loss_coef"],
        "hyperparameter_changes_from_released_config": {k: config[k] for k in (
            "prior_action_loss_coef", "ppo_target_kl", "learning_rate", "lr_schedule", "schedule_type",
            "kl_threshold", "lr_min", "lr_max", "e_clip", "actor_warmup_epochs", "actor_ramp_epochs")},
        "environment_training_changes": {k: environment["env"]["rl_finetune"][k] for k in (
            "points_goal_fraction", "feasible_object_goals", "root_xy_reward")},
        "goal_deadline_penalty": environment["env"]["rl_finetune"].get("goal_deadline_penalty", False),
        "new_correctness_fixes": ["write_root_height_to_original_tensor", "ground_feasible_object_initial_state"],
        "full_simulator_state_available": saved.get("env_state") is not None,
        "smoke_only": args.smoke,
        "seed": training["params"]["seed"],
        "evaluation_gate": gate,
        "distillation_mode_coverage": {k: environment["env"]["rl_finetune"].get(k, 0.0) for k in (
            "distill_full_track_fraction", "distill_point_goal_fraction")},
        "evaluation": {"early_checkpoint_epoch": epoch + 200, "periodic_first_checkpoint_epoch": epoch + 500,
                       "interval": 500, "rollouts_per_checkpoint": 80},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    if gate:
        request["development_cohort_manifest"] = str(run / "development_cohort.json")
        request["acceptance_cohort_manifest"] = str(run / "acceptance_cohort.json")
        request["evaluation"] = {"baseline_epoch": epoch, "interval": gate["interval"],
                                 "max_new_updates": gate["max_new_updates"], "rollouts_per_checkpoint": 80,
                                 "synchronous": True, "long_training_allowed": False}
    write_json(run / "training_request.json", request)
    hashes = source_hashes()
    write_json(run / "source_sha256.json", hashes)
    config_files = ["training.yaml", "environment.yaml"]
    if gate:
        config_files += ["gate.json", "development_cohort.json", "acceptance_cohort.json"]
    write_json(run / "config_sha256.json", {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                            for name in config_files})
    snapshot = run / "source_snapshot"
    paths = [p for p in hashes if p.endswith(".py") or p.startswith("ultra/data/cfg/")]
    paths += ["scripts/run_finetune_checkpointed.py", "scripts/prepare_rl_continuation.py",
              "scripts/verify_rl_initialization.py", "scripts/verify_rl_resume.py"]
    for name in sorted(set(paths)):
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    command = [sys.executable, str(ROOT / "scripts/run_finetune_checkpointed.py"),
               "--task", "UltraDistillObjV3RL", "--cfg_env", str(run / "environment.yaml"),
               "--cfg_train", str(run / "training.yaml"), "--resume_from", str(checkpoint),
               "--headless", "--output_path", str(output)]
    write_json(run / "launch.json", {"environment": {"ULTRA_RL_RUN_DIR": str(run),
                                                     "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                                     "command": command})
    print(json.dumps({"run_dir": str(run), "output_path": str(output), "checkpoint_epoch": epoch,
                      "initial_RL_updates": epoch - args.initial_student_epoch}, indent=2))


if __name__ == "__main__":
    main()
