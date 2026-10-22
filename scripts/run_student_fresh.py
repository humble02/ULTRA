#!/usr/bin/env python3
"""Start corrected distillation from scratch with auditable branch monitoring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="Recover this same fresh run after interruption")
    args = parser.parse_args()
    os.chdir(ROOT)
    run = args.run_dir.resolve()
    output = args.output_path.resolve()
    assert ROOT in run.parents and ROOT in output.parents
    request = json.loads((run / "training_request.json").read_text())
    assert request["resume_from"] is None and request["from_scratch"] is True
    resume = args.resume_from.resolve() if args.resume_from else None
    if resume is None:
        assert not any(output.rglob("*.pth")), "Fresh training refuses existing checkpoints"
    else:
        assert resume.is_file() and resume.parent == output / "g1_student_vae/nn"
    state_path = run / "training_status.json"
    if state_path.exists():
        previous_state = json.loads(state_path.read_text())
        allowed = ("failed", "stopped_after_checkpoint") if resume else ("prepared",)
        assert previous_state["state"] in allowed, "Run already started or is still running"
        if resume:
            archive = run / "recoveries" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive.mkdir(parents=True)
            write_json(archive / "previous_training_status.json", previous_state)
    log_fd = os.open(run / "training.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    sys.path.insert(0, str(ROOT / "ultra"))
    import isaacgym  # noqa: F401; must precede torch
    import torch
    import yaml
    from learning import ultra_agent_distill_vae

    configuration = yaml.safe_load((run / "training.yaml").read_text())
    assert not configuration["params"].get("load_checkpoint", False)
    assert configuration["params"]["config"].get("resume_from") is None
    assert not configuration["params"]["network"]["modal_config"].get("freeze_encoder_decoder", False)
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] > 0 and checkpoint.get("model") and checkpoint.get("optimizer")
        assert checkpoint["epoch"] == int(resume.stem.rsplit("_", 1)[1])
        assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())
    state = dict(state="starting", pid=os.getpid(), started_at=now(), epoch=checkpoint["epoch"] if resume else 0,
                 from_scratch=True, resume_from=str(resume) if resume else None, output_path=str(output))
    write_json(state_path, state)
    stopping = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopping.append(signum))
    signal.signal(signal.SIGINT, lambda signum, frame: stopping.append(signum))

    cls = ultra_agent_distill_vae.UltraAgentDistill
    original_train = cls.train
    original_log = cls._log_train_info
    original_restore = cls.restore
    original_init_train = cls._init_train
    originals = {}
    handles = []
    branch_counts = {"posterior": 0, "prior_only": 0, "privileged_encoder": 0}
    gradient_checks = {}

    def attach_monitor(self):
        net = self.model.a2c_network
        groups = ("encoder_net.", "encoder_mu_head.", "encoder_logvar_head.",
                  "prior_mu_head.", "prior_logvar_head.")
        for key, value in net.named_parameters():
            if key.startswith(groups):
                assert value.requires_grad, key
                originals[key] = value.detach().cpu().clone()
        if resume is None:
            save_initial_model(self)

        def network_call(module, inputs):
            if not torch.is_grad_enabled():
                return
            data = inputs[0]
            if data.get("with_encoder", False):
                assert data.get("with_vae", False) and "teacher_obs" in data
                branch_counts["posterior"] += 1
            else:
                assert not data.get("with_vae", False), "V2 prior-only must use act()"
                assert "teacher_obs" not in data, "Prior-only must not consume teacher inputs"
                branch_counts["prior_only"] += 1

        def encoder_call(module, inputs):
            if torch.is_grad_enabled():
                branch_counts["privileged_encoder"] += 1

        handles.extend([net.register_forward_pre_hook(network_call),
                        net.encoder_net.register_forward_pre_hook(encoder_call)])
        for name in ("encoder_mu_head", "prior_mu_head"):
            parameter = next(getattr(net, name).parameters())

            def check(gradient, name=name):
                finite = bool(torch.isfinite(gradient).all())
                value = float(gradient.detach().abs().max())
                previous = gradient_checks.setdefault(name, {"calls": 0, "all_finite": True, "max_abs": 0.0})
                previous["calls"] += 1
                previous["all_finite"] &= finite
                previous["max_abs"] = max(previous["max_abs"], value)

            handles.append(parameter.register_hook(check))

    def save_initial_model(self):
        initial = {"model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                   "epoch": 0, "frame": 0, "initialization_only": True,
                   "resume_from": None, "seed": configuration["params"]["seed"]}
        initial_path = run / "initial_model.pth"
        torch.save(initial, initial_path)
        initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        del initial
        write_json(run / "initialization.json", {
            "from_scratch": True, "epoch": self.epoch_num, "resume_from": self.resume_from,
            "initial_model_sha256": initial_sha,
            "all_prior_and_privileged_heads_trainable": True,
            "teacher_checkpoint": request["teacher_checkpoint"],
            "student_checkpoint_loaded": False,
        })


    def verify_equal(actual, expected):
        if torch.is_tensor(expected):
            assert torch.equal(actual.detach().cpu(), expected)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                verify_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                verify_equal(a, b)
        else:
            assert actual == expected

    def monitored_restore(self, path):
        original_restore(self, path)
        if resume:
            assert Path(path).resolve() == resume
            verify_equal(self.model.state_dict(), checkpoint["model"])
            verify_equal(self.optimizer.state_dict(), checkpoint["optimizer"])
            assert self.epoch_num == checkpoint["epoch"]
            attach_monitor(self)
            write_json(archive / "restore_verification.json", {
                "passed": True, "checkpoint": str(resume), "epoch": self.epoch_num,
                "model_exact": True, "optimizer_exact": True,
                "environment_and_rng_reinitialized": True,
                "original_initialization_artifacts_preserved": True,
                "training_hyperparameters_changed": False,
            })

    def monitored_train(self):
        if resume:
            assert Path(self.resume_from).resolve() == resume
        else:
            assert self.resume_from in (None, "None", "") and self.epoch_num == 0
            attach_monitor(self)
        print(f"Student training starting at epoch {checkpoint['epoch'] if resume else 0}; privileged encoder and prior are trainable", flush=True)
        return original_train(self)

    def monitored_init_train(self):
        original_init_train(self)
        if resume:
            # UltraAgent.train resets frame after restore; preserve TensorBoard chronology.
            self.frame = checkpoint["frame"]

    def monitored_log(self, train_info, frame):
        original_log(self, train_info, frame)
        metrics = {}
        for key, values in train_info.items():
            if not isinstance(values, list) or not values:
                continue
            numeric = [float(v.detach().float().mean()) if torch.is_tensor(v) else float(v) for v in values]
            assert all(math.isfinite(v) for v in numeric), f"Nonfinite {key}"
            metrics[key] = sum(numeric) / len(numeric)
        if originals:
            assert branch_counts["posterior"] > 0 and branch_counts["prior_only"] > 0
            assert branch_counts["privileged_encoder"] == branch_counts["posterior"], branch_counts
            assert set(gradient_checks) == {"encoder_mu_head", "prior_mu_head"}
            assert all(v["all_finite"] and v["max_abs"] > 0 for v in gradient_checks.values())
            parameters = dict(self.model.a2c_network.named_parameters())
            changed = {k: not torch.equal(v, parameters[k].detach().cpu()) for k, v in originals.items()}
            assert all(changed.values()), changed
            verification_path = archive / "branch_training_verification.json" if resume else run / "branch_training_verification.json"
            write_json(verification_path, {
                "passed": True, "verified_at_epoch": self.epoch_num,
                "gradient_enabled_forward_calls": branch_counts,
                "head_gradient_checks": gradient_checks,
                "updated_parameters": changed,
                "prior_only_has_no_privileged_inputs": True,
                "main_distillation_still_uses_privileged_encoder": True,
                "optimization_and_hyperparameters_changed_by_monitor": False,
            })
            originals.clear()
            for handle in handles:
                handle.remove()
        free, total = torch.cuda.mem_get_info()
        state.update(state="training", epoch=self.epoch_num, frame=frame, updated_at=now(),
                     metrics=metrics, gpu_free_mib=free / 2**20, gpu_total_mib=total / 2**20,
                     torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
        write_json(state_path, state)
        if stopping:
            path = Path(self.nn_dir) / f"{self.config['name']}_{self.epoch_num:08d}"
            self.save(str(path))
            self.writer.flush()
            state.update(state="stopped_after_checkpoint", checkpoint=str(path) + ".pth", stopped_at=now())
            write_json(state_path, state)
            raise SystemExit(0)

    cls.train = monitored_train
    cls._log_train_info = monitored_log
    if resume:
        cls.restore = monitored_restore
        cls._init_train = monitored_init_train
    sys.argv = [str(ROOT / "ultra/run_distill.py"), "--task", "UltraDistillObjV2Point",
                "--cfg_env", str(run / "environment.yaml"), "--cfg_train", str(run / "training.yaml"),
                "--headless", "--output_path", str(output)]
    if resume:
        sys.argv.extend(["--resume_from", str(resume)])
    try:
        runpy.run_path(str(ROOT / "ultra/run_distill.py"), run_name="__main__")
    except SystemExit as error:
        if error.code not in (None, 0):
            state.update(state="failed", error=str(error), updated_at=now())
            write_json(state_path, state)
        raise
    except BaseException as error:
        state.update(state="failed", error=repr(error), updated_at=now())
        write_json(state_path, state)
        raise
    else:
        state.update(state="completed", updated_at=now())
        write_json(state_path, state)


if __name__ == "__main__":
    main()
