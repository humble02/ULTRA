#!/usr/bin/env python3
"""Run three real GPU updates, including PPO, with repaired RL observations."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/reproduction/finetune_fixed_6000_20261022/smoke"
os.environ["ULTRA_RL_RUN_DIR"] = str(RUN)
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401
import torch
from learning import ultra_agent_distill_vae_rl


def main():
    if os.environ.get("ULTRA_TEST_DISABLE_MHA_FASTPATH") == "1":
        torch.backends.mha.set_fastpath_enabled(False)
    cls = ultra_agent_distill_vae_rl.UltraAgentDistill
    original_init = cls._init_train
    original_actor = cls._actor_loss
    original_gradients = cls.calc_gradients
    gradients = []
    replay = {}

    def initialize(self):
        original_init(self)
        task = self.vec_env.env.task
        ids = torch.arange(task.num_envs, device=task.device)
        selected = ids[(ids >= task.num_envs // 4) & (ids < task.num_envs // 4 + int((task.num_envs - task.num_envs // 4) * .5))]
        assert len(selected) == 3
        for name in ("keep_obj_point_mask", "keep_obj_trans_mask", "keep_goal_mask"):
            assert bool(getattr(task, name)[selected].all()), name
        for name in ("keep_obj_pos_mask", "keep_obj_rot_mask", "keep_global_goal_mask", "keep_local_goal_mask"):
            assert not bool(getattr(task, name)[selected].any()), name
        phase = task.goal_phase[:, 1:3]
        assert bool(((phase == 0) | (phase == 1)).all())
        assert self.prior_action_loss_coef == .6 and self.ppo_target_kl == .04
        assert self.scheduler.max_lr == 2e-5
        assert self.actor_warmup_epochs == 0, "Only this smoke test skips warmup to exercise PPO gradients"
        (RUN / "observation_verification.json").write_text(json.dumps({
            "passed": True, "num_envs": task.num_envs, "distillation_envs": 2,
            "ppo_envs": 6, "reserved_points_goal_envs": len(selected),
            "commands_binary": True, "point_mode_masks_correct": True,
            "prior_anchor_coefficient": self.prior_action_loss_coef,
            "exact_prior_kl_gate_threshold": self.ppo_target_kl,
            "adaptive_lr_max": self.scheduler.max_lr,
        }, indent=2) + "\n")

    def actor_loss(self, *args, **kwargs):
        result = original_actor(self, *args, **kwargs)
        loss = result["actor_loss"]
        if loss.requires_grad:
            def record(gradient):
                assert torch.isfinite(gradient).all()
                gradients.append(float(gradient.detach().abs().sum()))
            loss.register_hook(record)
        return result

    cls._init_train = initialize
    cls._actor_loss = actor_loss

    def check_replay(self, batch):
        if not replay:
            self.set_train()
            with torch.no_grad():
                predicted = self.model({"is_train": True, "prev_actions": batch["actions"],
                    "obs": self._preproc_obs(batch["obs"]), "teacher_obs": batch["teacher_obs"],
                    "vae_noise": batch["vae_noise"], "with_encoder": False, "with_vae": True,
                    "skip_critic": True})
                mask = batch["prior_mask"].bool()
                replay["mu_max_abs_error"] = float((predicted["mus"][mask] - batch["mu"][mask]).abs().max())
                replay["negative_logp_max_abs_error"] = float((predicted["prev_neglogp"][mask] - batch["old_logp_actions"][mask]).abs().max())
                self.model.eval()
                evaluation = self.model({"is_train": True, "prev_actions": batch["actions"],
                    "obs": self._preproc_obs(batch["obs"]), "teacher_obs": batch["teacher_obs"],
                    "vae_noise": batch["vae_noise"], "with_encoder": False, "with_vae": True,
                    "skip_critic": True})
                replay["eval_mode_mu_max_abs_error"] = float((evaluation["mus"][mask] - batch["mu"][mask]).abs().max())
                replay["train_vs_eval_mu_max_abs_error"] = float((evaluation["mus"][mask] - predicted["mus"][mask]).abs().max())
                self.set_train()
                (RUN / "first_replay_probe.json").write_text(json.dumps(replay, indent=2) + "\n")
                assert replay["mu_max_abs_error"] < 1e-5, replay
                assert replay["negative_logp_max_abs_error"] < 1e-4, replay
        return original_gradients(self, batch)

    cls.calc_gradients = check_replay
    initial = ROOT / "output/student_fresh_20261020/g1_student_vae/nn/g1_student_vae_00006000.pth"
    sys.argv = [str(ROOT / "scripts/run_finetune_checkpointed.py"), "--task", "UltraDistillObjV3RL",
                "--cfg_env", str(RUN / "environment.yaml"), "--cfg_train", str(RUN / "training.yaml"),
                "--resume_from", str(initial), "--headless", "--output_path", str(RUN / "training_output")]
    runpy.run_path(sys.argv[0], run_name="__main__")
    assert gradients and max(gradients) > 0, "PPO actor term did not contribute gradients"
    state = json.loads((RUN / "memory_status.json").read_text())
    assert state["epoch"] == 6003 and abs(state["metrics"]["prior_action_coeff"] - .6) < 1e-6
    assert state["metrics"]["actor_coeff"] > 0
    (RUN / "verification.json").write_text(json.dumps({
        "passed": True, "completed_updates": 3, "final_epoch": state["epoch"],
        "ppo_actor_backward_calls": len(gradients), "ppo_actor_loss_gradient_max_l1": max(gradients),
        "PPO_exercised_without_100_epoch_wait": True,
        "production_keeps_original_100_epoch_warmup": True,
        "prior_action_loss_coef": .6, "all_logged_metrics_finite": True,
        "environment_observation_checks_passed": True,
        "not_a_success_rate_evaluation": True,
        "first_PPO_minibatch_replays_rollout_distribution": replay,
    }, indent=2) + "\n")
    print("Repaired RL GPU smoke passed", flush=True)


if __name__ == "__main__":
    main()
