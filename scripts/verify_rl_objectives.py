"""Controlled tests of the actual deadline penalty, bootstrap and masked advantages."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401
import torch
import env.tasks.ultra_g1_distill_obj_v3rl as task_module
from env.tasks.smooth_rewards import SmoothRewards
from learning.common_agent import CommonAgent
from learning.ultra_agent_distill_vae_rl import UltraAgentDistill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    n = 6
    def task(enabled):
        t = SimpleNamespace(
            cfg={"smooth_rewards": {"scales": {"termination": -50.0}}},
            prior_mask=torch.ones(n, dtype=torch.bool), reset_buf=torch.zeros(n, dtype=torch.long),
            _terminate_buf=torch.zeros(n, dtype=torch.long), progress_buf=torch.full((n,), 40),
            start_times=torch.zeros(n, dtype=torch.long), obs_buf=torch.zeros(n, 10),
            _contact_forces=torch.zeros(n, 39, 3), _rigid_body_pos=torch.zeros(n, 39, 3),
            max_episode_length=torch.full((n,), 300), data_id=torch.arange(n),
            _enable_early_termination=True, _termination_heights=.25, _termination_heights_init=.25,
            _curr_ref_obs=torch.zeros(n, 10), _curr_obs=torch.zeros(n, 10), rollout_length=200,
            is_stand_still=torch.zeros(n, dtype=torch.bool), rl_goal_term_warmup_steps=30,
            long_term_t=torch.tensor([1, 1, 2, 1, 1, 1]), keep_obj_trans_mask=torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.bool),
            keep_obj_pos_mask=torch.zeros(n, dtype=torch.bool), keep_obj_point_mask=torch.ones(n, dtype=torch.bool),
            keep_global_goal_mask=torch.zeros(n, dtype=torch.bool), _target_states=torch.zeros(n, 13),
            _humanoid_root_states=torch.zeros(n, 13), rl_target_obj_pos=torch.zeros(n, 3),
            rl_target_root_pos=torch.zeros(n, 3), rl_obj_goal_term_threshold=.15,
            rl_root_goal_term_threshold=.15, rl_root_xy_reward=True, rl_goal_deadline_penalty=enabled,
            common_step_counter=8000, rew_buf=torch.zeros(n), extras={})
        t._rigid_body_pos[:, 0, 2] = .9; t._rigid_body_pos[5, 0, 2] = .1
        t.rl_target_obj_pos[[1, 2, 4], 0] = 1.
        t.rl_target_root_pos[3, 0] = 100.
        t.smooth_rewards = SmoothRewards(t)
        return t
    # Replace only the unrelated inherited reset calculation. The actual RL
    # override computes visibility, deadline, fall, penalty and bootstrap flags.
    original_reset = task_module.compute_humanoid_reset_retarget
    task_module.compute_humanoid_reset_retarget = lambda reset, *args: (torch.zeros_like(reset), torch.zeros_like(reset))
    try:
        disabled, enabled = task(False), task(True)
        task_module.UltraDistillObjV3RL._compute_reset(disabled)
        task_module.UltraDistillObjV3RL._compute_reset(enabled)
    finally:
        task_module.compute_humanoid_reset_retarget = original_reset
    expected = torch.tensor([0., -12.5, 0., 0., 0., 0.])
    assert torch.equal(enabled.rew_buf - disabled.rew_buf, expected)
    assert torch.equal(enabled._terminate_buf, torch.tensor([0, 0, 0, 0, 0, 1]))
    assert torch.equal(enabled._terminate_buf, disabled._terminate_buf)
    assert enabled.reset_buf[1] == 0 and enabled.reset_buf[5] == 1
    assert enabled.rew_buf[5] == disabled.rew_buf[5] == -12.5
    # The previous implementation read only _terminate_buf: a deadline miss
    # generated no penalty despite a nonzero _reward_terminate_buf.
    old_reward_read = enabled._terminate_buf.float() * -50 * .25
    assert old_reward_read[1] == 0 and enabled.rew_buf[1] == -12.5

    actor = SimpleNamespace(horizon_length=2, gamma=.99, tau=.95)
    values = torch.zeros(2, 3, 1)
    rewards = torch.tensor([[[1.], [1.], [1.]], [[2.], [2.], [2.]]])
    next_values = torch.tensor([[[0.], [4.], [4.]], [[0.], [0.], [0.]]])
    dones = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
    advantages = CommonAgent.discount_values(actor, dones, values, rewards, next_values)
    assert torch.allclose(advantages[0, :, 0], torch.tensor([1., 4.96, 6.841]), atol=1e-6)
    batch = dict(returns=torch.tensor([[1e9], [2.], [4.], [6.]]), values=torch.zeros(4, 1),
                 rand_action_mask=torch.ones(4), prior_mask=torch.tensor([0., 1., 1., 1.]))
    normalized = UltraAgentDistill._calc_advs(SimpleNamespace(normalize_advantage=True), batch)
    assert abs(float(normalized[1:].mean())) < 1e-6
    assert abs(float(normalized[1:].square().mean()) - 1) < 1e-6

    # Confirm the goal component itself still cannot distinguish a push from
    # a lift with the same final visible-goal distance. This is a separate
    # objective-design question; the current candidate does not add shaping.
    r = SimpleNamespace(prior_mask=torch.ones(2, dtype=torch.bool), rew_buf=torch.zeros(2),
        _target_states=torch.zeros(2, 3), _humanoid_root_states=torch.zeros(2, 3),
        rl_target_obj_pos=torch.zeros(2, 3), rl_target_root_pos=torch.zeros(2, 3),
        rl_prev_obj_goal_dist=torch.zeros(2), rl_prev_root_goal_dist=torch.zeros(2),
        keep_obj_trans_mask=torch.ones(2, dtype=torch.bool), keep_obj_point_mask=torch.ones(2, dtype=torch.bool),
        keep_obj_pos_mask=torch.zeros(2, dtype=torch.bool), keep_global_goal_mask=torch.zeros(2, dtype=torch.bool),
        rl_obj_goal_weight=1., rl_root_goal_weight=1., rl_progress_reward_weight=1.,
        rl_goal_success_bonus=10., rl_obj_goal_success_threshold=.15, rl_root_goal_success_threshold=.15,
        rl_goal_success_awarded=torch.zeros(2, dtype=torch.bool), long_term_t=torch.ones(2, dtype=torch.long),
        rl_root_xy_reward=True, extras={}, past_lift_height=torch.tensor([0., .5]))
    task_module.UltraDistillObjV3RL._compute_reward(r, None)
    assert torch.equal(r.rew_buf, torch.tensor([11., 11.]))
    result = {"passed": True, "case_order": ["reached", "missed_deadline", "before_deadline", "hidden_root", "hidden_object", "fall"],
              "old_penalty_contributions": old_reward_read.tolist(), "new_penalty_contributions": enabled.rew_buf.tolist(),
              "flag_disabled_matches_previous_reward_behavior": True,
              "deadline_failure_penalty_effective": -12.5, "deadline_failure_does_not_end_episode": True,
              "physical_termination_and_GAE_bootstrap_preserved": True,
              "GAE_terminal_timeout_continuing_expected": advantages[0, :, 0].tolist(),
              "advantage_normalization_excludes_distillation": True,
              "equal_goal_component_with_and_without_lift_history": r.rew_buf.tolist(),
              "scope": "Controlled CPU inputs, actual reward/reset/GAE methods; no measured policy improvement."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
