#!/usr/bin/env python3
"""Configured RL training with prior activation recomputation and memory telemetry.

Only training-time autograd storage changes. Rollouts, precision, batch sizes,
losses, and optimizer steps follow the selected training configuration.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import runpy
import sys
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401; must precede torch
import torch
from torch.utils.checkpoint import checkpoint
from learning import ultra_network_builder_obj_v3


def enable_checkpointing():
    cls = ultra_network_builder_obj_v3.UltraBuilder.Network
    original = cls._prior
    if getattr(original, "_activation_checkpointed", False):
        return

    def prior(self, inputs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(original, self, inputs, use_reentrant=False,
                              preserve_rng_state=True)
        return original(self, inputs)

    prior._activation_checkpointed = True
    cls._prior = prior


def verify_cpu(out=None, checkpoint_path=None):
    import contextlib
    import yaml
    from learning import ultra_models
    torch.set_num_threads(2)
    torch.manual_seed(42)
    out = out or ROOT / "output/reproduction/finetune_from_10000"
    out.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load((ROOT / "ultra/data/cfg/train/rlg/g1_student_finetune.yaml").read_text())["params"]
    with (out / "checkpointing_build.log").open("w") as log, contextlib.redirect_stdout(log):
        builder = ultra_network_builder_obj_v3.UltraBuilder()
        builder.load(params["network"])
        model = ultra_models.ModelUltraContinuous(builder).build({
            "actions_num": 29, "input_shape": (1496,), "num_seqs": 1, "value_size": 1})
    checkpoint_path = checkpoint_path or ROOT / "output/student_prior_fixed/g1_student_vae/nn/g1_student_vae_00010000.pth"
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    saved = checkpoint_data["model"]
    current = model.state_dict()
    model.load_state_dict({k: v for k, v in saved.items() if k in current and current[k].shape == v.shape
                           and not (".critic" in k or ".value" in k)}, strict=False)
    model.train()
    obs = torch.randn(5, 1496) * .1
    obs[:, -242:] = 1
    base = dict(is_train=True, prev_actions=torch.randn(5, 29) * .1,
                obs=obs, teacher_obs=torch.randn(5, 4052) * .1,
                vae_noise=torch.randn(5, 64), with_vae=True,
                rnn_states=None, skip_critic=False)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        posterior = model({**base, "with_encoder": True})
        prior = model({**base, "with_encoder": False})
        loss = (posterior["mus"].square().mean() + prior["mus"].square().mean()
                + 5 * prior["values"].square().mean()
                + prior["prev_neglogp"].mean() * .001)
        loss.backward()
        return prior["mus"].detach().clone(), {
            k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}

    baseline_action, baseline_grads = forward_backward()
    enable_checkpointing()
    action, grads = forward_backward()
    assert torch.equal(baseline_action, action)
    assert baseline_grads.keys() == grads.keys()
    for key, value in baseline_grads.items():
        assert torch.allclose(value, grads[key], atol=1e-6, rtol=1e-5), key
        assert torch.isfinite(grads[key]).all(), key
    report = {"passed": True, "checkpoint": str(checkpoint_path), "initial_student_epoch": checkpoint_data["epoch"],
              "action_max_abs_error": float((baseline_action - action).abs().max()),
              "gradient_max_abs_error": max(float((v - grads[k]).abs().max()) for k, v in baseline_grads.items()),
              "compared_gradient_tensors": len(grads),
              "hyperparameter_changes": {}, "mixed_precision": False,
              "method": "torch.utils.checkpoint on V3 _prior; non-reentrant; RNG preserved; no checkpointing during rollouts"}
    (out / "checkpointing_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    os.chdir(ROOT)
    if "--verify-cpu" in sys.argv[1:]:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--verify-cpu", action="store_true")
        parser.add_argument("--checkpoint", type=Path)
        parser.add_argument("--output-dir", type=Path)
        args = parser.parse_args()
        verify_cpu(args.output_dir, args.checkpoint)
        return
    enable_checkpointing()
    run = Path(os.environ.get("ULTRA_RL_RUN_DIR", ROOT / "output/reproduction/finetune_from_10000")).resolve()
    assert ROOT in run.parents
    run.mkdir(parents=True, exist_ok=True)
    status_path = run / "memory_status.json"
    request_path = run / "training_request.json"
    run_request = json.loads(request_path.read_text()) if request_path.exists() else {}
    stopping = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopping.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: stopping.append(signum))

    def write_status(value):
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(status_path)

    write_status({"state": "starting", "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat()})
    from learning import ultra_agent_distill_vae_rl
    cls = ultra_agent_distill_vae_rl.UltraAgentDistill
    original_log = cls._log_train_info
    original_init = cls._init_train
    watched = {}

    def initialize(self):
        original_init(self)
        request_path = run / "training_request.json"
        if not request_path.exists():
            return
        request = json.loads(request_path.read_text())
        assert self.epoch_num_start == request["initial_student_epoch"]
        source = torch.load(request["initial_checkpoint"], map_location="cpu")["model"]
        current = self.model.state_dict()
        checked = 0
        for name, value in source.items():
            if ".critic" not in name and ".value" not in name:
                assert torch.equal(current[name].detach().cpu(), value), name
                checked += 1
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or "prior_mu_head" in name or "critic_value" in name:
                watched[name] = parameter.detach().cpu().clone()
        observation_checks = {}
        if request.get("environment_training_changes", {}).get("points_goal_fraction", 0) > 0:
            task = self.vec_env.env.task
            ids = torch.arange(task.num_envs, device=task.device)
            count = int((task.num_envs - task.num_envs // 4) * task.rl_points_goal_fraction)
            selected = ids[(ids >= task.num_envs // 4) & (ids < task.num_envs // 4 + count)]
            for name in ("keep_obj_point_mask", "keep_obj_trans_mask", "keep_goal_mask"):
                assert bool(getattr(task, name)[selected].all()), name
            for name in ("keep_obj_pos_mask", "keep_obj_rot_mask", "keep_global_goal_mask", "keep_local_goal_mask"):
                assert not bool(getattr(task, name)[selected].any()), name
            commands = task.goal_phase[:, 1:3]
            assert bool(((commands == 0) | (commands == 1)).all())
            observation_checks = {"reserved_points_goal_envs": count, "point_mode_masks_correct": True,
                                  "commands_binary": True}
        path = run / "runtime_initialization_verification.json"
        path.write_text(json.dumps({"passed": True, "initial_student_epoch": self.epoch_num_start,
                                   "actor_state_entries_verified_exact": checked,
                                   "fresh_rl_optimizer": not bool(self.optimizer.state),
                                   "distillation_envs": int((~self.prior_mask).sum()),
                                   "ppo_envs": int(self.prior_mask.sum()),
                                   "observation_checks": observation_checks}, indent=2) + "\n")

    def log(self, train_info, frame):
        original_log(self, train_info, frame)
        metrics = {}
        for key, values in train_info.items():
            if isinstance(values, list) and values:
                try:
                    metrics[key] = sum(float(v.detach().float().mean()) if torch.is_tensor(v) else float(v)
                                       for v in values) / len(values)
                except (TypeError, ValueError):
                    pass
        assert all(math.isfinite(v) for v in metrics.values()), metrics
        if watched:
            parameters = dict(self.model.named_parameters())
            frozen = [name for name in watched if not parameters[name].requires_grad]
            assert all(torch.equal(parameters[name].detach().cpu(), watched[name]) for name in frozen)
            changed = [name for name in watched if parameters[name].requires_grad
                       and not torch.equal(parameters[name].detach().cpu(), watched[name])]
            assert any("prior_mu_head" in name for name in changed), changed
            assert any("critic_value" in name for name in changed), changed
            assert metrics["distill_mask_fraction"] == 0.25
            (run / "first_update_verification.json").write_text(json.dumps({
                "passed": True, "epoch": self.epoch_num, "relative_rl_epochs": self.epoch_num - self.epoch_num_start,
                "frozen_parameter_tensors_unchanged": len(frozen), "updated_prior_and_critic_parameters": changed,
                "distillation_env_fraction": metrics["distill_mask_fraction"], "all_metrics_finite": True,
            }, indent=2) + "\n")
            watched.clear()
        data = {"state": "training", "log_epoch": self.epoch_num,
                "epoch": self.epoch_num, "pid": os.getpid(),
                "relative_rl_epochs": self.epoch_num - self.epoch_num_start,
                "initial_student_epoch": self.epoch_num_start,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "activation_checkpointing": True,
                "hyperparameter_overrides": run_request.get("hyperparameter_changes_from_released_config", {}),
                "environment_training_changes": run_request.get("environment_training_changes", {}),
                "num_envs": self.num_actors, "horizon_length": self.horizon_length,
                "minibatch_size": self.minibatch_size, "mini_epochs": self.mini_epochs_num,
                "mixed_precision": self.mixed_precision, "metrics": metrics,
                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
        task = self.vec_env.env.task
        data["environment_metrics"] = {
            key: float(value.detach().float().mean()) if torch.is_tensor(value) else float(value)
            for key, value in task.extras.items() if key.startswith("rl_")
        }
        write_status(data)
        if stopping:
            path = Path(self.nn_dir) / f"{self.config['name']}_{self.epoch_num:08d}"
            self.save(str(path))
            self.writer.flush()
            data.update(state="stopped_after_checkpoint", checkpoint=str(path) + ".pth")
            write_status(data)
            raise SystemExit(0)
    ultra_agent_distill_vae_rl.UltraAgentDistill._log_train_info = log
    cls._init_train = initialize
    print("RL prior activation checkpointing enabled; FP32 batches and configured objectives retained.", flush=True)
    sys.argv[0] = str(ROOT / "ultra/run_distill.py")
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except Exception as error:
        data = json.loads(status_path.read_text())
        data.update(state="failed", error=repr(error), updated_at=datetime.now(timezone.utc).isoformat())
        write_status(data)
        raise


if __name__ == "__main__":
    main()
