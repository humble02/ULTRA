#!/usr/bin/env python3
"""Exercise a complete RL checkpoint continuation for three small GPU updates."""
import argparse
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401
import torch
from utils import torch_utils
from env.tasks.ultra_g1_distill_obj_v3rl import UltraDistillObjV3RL
from learning.ultra_agent_distill_vae_rl import UltraAgentDistill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    request = json.loads((run / "training_request.json").read_text())
    assert request["resume_optimizer"]
    os.environ["ULTRA_RL_RUN_DIR"] = str(run)
    checks = {"root_resets": 0, "object_resets": 0, "max_root_height_error_m": 0.0,
              "minimum_box_clearance_m": float("inf")}
    original_root = UltraDistillObjV3RL._set_env_state
    original_target = UltraDistillObjV3RL._reset_target

    def root_reset(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        original_root(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel)
        ids = env_ids
        mask = self.prior_mask[ids]
        if mask.any():
            error = float((self._humanoid_root_states[ids[mask], 2] - root_pos[mask, 2] - .01).abs().max())
            assert error < 1e-6
            checks["root_resets"] += int(mask.sum())
            checks["max_root_height_error_m"] = max(error, checks["max_root_height_error_m"])

    def target_reset(self, env_ids):
        original_target(self, env_ids)
        ids = env_ids
        selected = ids[self.prior_mask[ids]]
        if selected.numel():
            states = self._target_states[selected]
            corners = self.object_corners[self.object_id[self.data_id[selected]]]
            rotations = states[:, None, 3:7].expand(-1, corners.shape[1], -1)
            rotated = torch_utils.quat_rotate(rotations.reshape(-1, 4), corners.reshape(-1, 3)).reshape_as(corners)
            clearance = states[:, 2] + rotated[:, :, 2].amin(1)
            assert float(clearance.min()) >= .01 - 1e-6
            checks["object_resets"] += len(selected)
            checks["minimum_box_clearance_m"] = min(float(clearance.min()), checks["minimum_box_clearance_m"])

    UltraDistillObjV3RL._set_env_state = root_reset
    UltraDistillObjV3RL._reset_target = target_reset
    original_actor = UltraAgentDistill._actor_loss
    original_gradients = UltraAgentDistill.calc_gradients
    gradients, replay, gradient_audit = [], {}, {}
    routing = {"distillation_reference_goal_checks": 0, "PPO_random_goal_checks": 0}
    original_humanoid = UltraDistillObjV3RL._compute_humanoid_obs
    original_task_obs = UltraDistillObjV3RL._compute_task_obs
    reference_goal = {}

    def humanoid_obs(self, env_ids, ref_obs, *args, **kwargs):
        reference_goal["positions"] = ref_obs[:, 71:74].clone()
        return original_humanoid(self, env_ids, ref_obs, *args, **kwargs)

    def task_obs(self, env_ids, ref_obs, student_obs=False):
        if student_obs:
            prior = self.prior_mask[env_ids]
            assert torch.equal(ref_obs[~prior, 71:74], reference_goal["positions"][~prior])
            assert torch.equal(ref_obs[prior, 71:74], self.rl_target_obj_pos[env_ids[prior]])
            routing["distillation_reference_goal_checks"] += int((~prior).sum())
            routing["PPO_random_goal_checks"] += int(prior.sum())
        return original_task_obs(self, env_ids, ref_obs, student_obs)

    UltraDistillObjV3RL._compute_humanoid_obs = humanoid_obs
    UltraDistillObjV3RL._compute_task_obs = task_obs
    original_backward = torch.Tensor.backward

    def audited_backward(tensor, *args, **kwargs):
        caller = inspect.currentframe().f_back
        if not gradient_audit and caller.f_code.co_name == "calc_gradients" and "distill_loss" in caller.f_locals:
            local = caller.f_locals
            agent = local["self"]
            selected = [(name, p) for name, p in agent.model.named_parameters()
                        if p.requires_grad and "prior_mu_head" in name]
            vectors = {}
            for label, loss in (("distillation", local["distill_loss"]),
                                ("PPO_and_critic", local["rl_loss"]),
                                ("prior_action_anchor", local["prior_action_loss"] * local["prior_action_coeff"])):
                grads = torch.autograd.grad(loss, [p for _, p in selected], retain_graph=True, allow_unused=True)
                vector = torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                                    for (_, p), g in zip(selected, grads)])
                assert torch.isfinite(vector).all()
                vectors[label] = vector
                gradient_audit[label + "_gradient_l2"] = float(vector.norm())
            a, b = vectors["distillation"], vectors["PPO_and_critic"]
            gradient_audit.update(parameters=[name for name, _ in selected],
                                 distill_PPO_cosine=float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)),
                                 scope="First small-batch update only; not representative training prevalence")
        return original_backward(tensor, *args, **kwargs)

    torch.Tensor.backward = audited_backward

    def actor_loss(self, *arguments, **keywords):
        result = original_actor(self, *arguments, **keywords)
        def record(gradient):
            assert torch.isfinite(gradient).all()
            gradients.append(float(gradient.detach().abs().sum()))
        result["actor_loss"].register_hook(record)
        return result

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
                assert replay["mu_max_abs_error"] < 1e-5, replay
                assert replay["negative_logp_max_abs_error"] < 1e-4, replay
                assert self.actor_schedule_start_epoch == request["initial_student_epoch"]
                assert self.epoch_num - self.actor_schedule_start_epoch > self.actor_warmup_epochs + self.actor_ramp_epochs
        return original_gradients(self, batch)

    UltraAgentDistill._actor_loss = actor_loss
    UltraAgentDistill.calc_gradients = check_replay
    sys.argv = [str(ROOT / "scripts/run_finetune_checkpointed.py"), "--task", "UltraDistillObjV3RL",
                "--cfg_env", str(run / "environment.yaml"), "--cfg_train", str(run / "training.yaml"),
                "--resume_from", request["initial_checkpoint"], "--headless",
                "--output_path", str(run / "training_output")]
    runpy.run_path(sys.argv[0], run_name="__main__")
    status = json.loads((run / "memory_status.json").read_text())
    initial = json.loads((run / "runtime_initialization_verification.json").read_text())
    assert status["state"] == "completed" and status["updates_since_restart"] == 3
    assert status["relative_rl_epochs"] == request["initial_checkpoint_epoch"] - request["initial_student_epoch"] + 3
    assert initial["resume_checks"]["optimizer_verified_exact"] and not initial["fresh_rl_optimizer"]
    assert gradients and max(gradients) > 0
    assert checks["root_resets"] >= 6 and checks["object_resets"] >= 6
    assert routing["distillation_reference_goal_checks"] > 0 and routing["PPO_random_goal_checks"] > 0
    assert gradient_audit["distillation_gradient_l2"] > 0
    assert gradient_audit["PPO_and_critic_gradient_l2"] > 0
    result = {"passed": True, "completed_updates": 3, "reset_checks": checks,
              "resume_checks": initial["resume_checks"], "rollout_replay": replay,
              "PPO_actor_backward_calls": len(gradients), "PPO_actor_gradient_max_l1": max(gradients),
              "actor_warmup_was_not_restarted": True, "not_a_success_rate_evaluation": True}
    result.update(goal_routing=routing, gradient_audit=gradient_audit,
                  observation_checks=initial["observation_checks"])
    (run / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
