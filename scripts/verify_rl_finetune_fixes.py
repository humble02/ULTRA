#!/usr/bin/env python3
"""Reproduce concrete old RL defects and verify the repaired functions on CPU."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401
import torch
from utils import torch_utils
from rl_games.algos_torch import torch_ext

OUT = ROOT / "output/reproduction/rl_finetune_diagnosis_20261022"
RELATIVE = Path("ultra/env/tasks/ultra_g1_distill_obj_v3rl.py")


def method(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "torch_utils": torch_utils}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def phase_state():
    n = 8
    return SimpleNamespace(num_envs=n, device="cpu", ig_hand_body_ids=[28, 38],
        prev_hand_ig_valid=torch.zeros(n, dtype=torch.bool), prev_hand_ig=torch.zeros(n),
        ig_interaction_threshold=.15, ig_transition_delta=.0001,
        progress_buf=torch.full((n,), 190), start_times=torch.zeros(n, dtype=torch.long),
        max_episode_length=torch.full((n,), 400), data_id=torch.arange(n),
        prior_mask=torch.arange(n) >= 2, long_term_t=torch.full((n,), 10),
        rollout_length=200, goal_phase=torch.zeros(n, 4))


def goal_state():
    n = 1024
    refs = torch.zeros(1, 1, 2, 747)
    refs[..., 2] = .9
    refs[..., 73] = .1
    refs[..., 77] = 1
    corners = torch.cartesian_prod(*[torch.tensor([-.1, .1])] * 3).unsqueeze(0)
    return SimpleNamespace(num_envs=n, device="cpu", progress_buf=torch.zeros(n, dtype=torch.long),
        start_times=torch.zeros(n, dtype=torch.long), long_term_t=torch.zeros(n, dtype=torch.long),
        data_id=torch.zeros(n, dtype=torch.long), max_episode_length=torch.tensor([2]), hoi_refs=refs,
        long_term_speed_threshold=.2, prior_mask=torch.arange(n) >= n // 4,
        rl_goal_success_awarded=torch.zeros(n, dtype=torch.bool), goal_offset_range_xy=.4, goal_offset_range_z=.2,
        goal_offset_obj=torch.zeros(n, 3), goal_offset_root=torch.zeros(n, 3),
        rl_target_obj_pos=torch.zeros(n, 3), rl_target_root_pos=torch.zeros(n, 3),
        rl_prev_obj_goal_dist=torch.zeros(n), rl_prev_root_goal_dist=torch.zeros(n),
        _target_states=torch.tensor([0., 0., .1]).expand(n, 3).clone(),
        _humanoid_root_states=torch.tensor([0., 0., .9]).expand(n, 3).clone(),
        object_id=torch.tensor([0]), object_corners=corners,
        rl_feasible_object_goals=True, rl_root_xy_reward=True,
        obs_buf_student=torch.zeros(n, 1496), _compute_observations_iter=lambda *a, **kw: torch.zeros(n, 1496))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    old = OUT / "before" / RELATIVE
    new = ROOT / RELATIVE
    ig = torch.full((8, 39, 1), .3)
    old_phase = method(old, "_update_goal_phase")
    fixed_phase = method(new, "_update_goal_phase")
    a, b = phase_state(), phase_state()
    old_phase(a, torch.arange(8), ig)
    first = a.goal_phase.clone()
    a.prev_hand_ig_valid.zero_()
    old_phase(a, torch.arange(8), ig)
    assert not torch.equal(first[2:, 1:3], a.goal_phase[2:, 1:3])
    assert not first[2:, 0].any()
    fixed_phase(b, torch.arange(8), ig)
    repaired = b.goal_phase.clone()
    b.prev_hand_ig_valid.zero_()
    fixed_phase(b, torch.arange(8), ig)
    assert torch.equal(repaired, b.goal_phase)
    assert bool(((repaired[:, 1:3] == 0) | (repaired[:, 1:3] == 1)).all())
    assert bool(repaired[2:, 0].all())

    old_goals, fixed_goals = goal_state(), goal_state()
    for path, task in ((old, old_goals), (new, fixed_goals)):
        torch.manual_seed(42)
        method(path, "_compute_observations_student")(task, torch.arange(1024))
    old_below = int((old_goals.rl_target_obj_pos[256:, 2] < .1).sum())
    new_below = int((fixed_goals.rl_target_obj_pos[256:, 2] < .1).sum())
    assert old_below > 0 and new_below == 0

    phase_updates = []
    for path in (old, new):
        calls = []
        task = SimpleNamespace(progress_buf=torch.zeros(8, dtype=torch.long), data_id=torch.zeros(8, dtype=torch.long),
            max_episode_length=torch.tensor([2]), hoi_data=torch.zeros(1, 2, 747), _curr_ref_obs=torch.zeros(8, 747),
            is_stand_still=torch.zeros(8, dtype=torch.bool), stand_still_frame=torch.zeros(8, dtype=torch.long),
            prior_mask=torch.arange(8) >= 2, rl_target_obj_pos=torch.zeros(8, 3),
            _compute_humanoid_obs=lambda *a, **kw: torch.zeros(8, 1046),
            _compute_task_obs=lambda *a, **kw: (torch.zeros(8, 204), None),
            _compute_ig_features=lambda *a, **kw: (torch.ones(8, 39, 3), ig),
            _update_goal_phase=lambda ids, norm: calls.append(norm.clone()),
            _finalize_student_obs=lambda ids, obs, *a: obs)
        method(path, "_compute_observations_iter")(task, torch.arange(8), 1, student_obs=True)
        phase_updates.append(len(calls))
        if calls:
            assert torch.equal(calls[0], ig)
    assert phase_updates == [0, 1]

    root_rewards = []
    for path in (old, new):
        task = SimpleNamespace(prior_mask=torch.ones(2, dtype=torch.bool), rew_buf=torch.zeros(2),
            _target_states=torch.zeros(2, 3), _humanoid_root_states=torch.tensor([[0., 0., .9], [0., 0., 1.1]]),
            rl_target_obj_pos=torch.zeros(2, 3), rl_target_root_pos=torch.tensor([[0., 0., .9], [0., 0., .9]]),
            rl_prev_obj_goal_dist=torch.zeros(2), rl_prev_root_goal_dist=torch.zeros(2),
            keep_obj_trans_mask=torch.zeros(2, dtype=torch.bool), keep_obj_pos_mask=torch.zeros(2, dtype=torch.bool),
            keep_obj_point_mask=torch.zeros(2, dtype=torch.bool), keep_global_goal_mask=torch.ones(2, dtype=torch.bool),
            rl_obj_goal_weight=1., rl_root_goal_weight=1., rl_progress_reward_weight=0., rl_goal_success_bonus=0.,
            rl_root_xy_reward=True, extras={})
        method(path, "_compute_reward")(task, None)
        root_rewards.append(task.rew_buf.tolist())
    assert root_rewards[0][0] != root_rewards[0][1] and root_rewards[1][0] == root_rewards[1][1]

    mean = torch.zeros(8, 29)
    std = torch.full_like(mean, .055)
    legacy_same_kl = float(torch_ext.policy_kl(mean, std, mean, std))
    exact = torch.distributions.kl_divergence(torch.distributions.Normal(mean, std),
                                            torch.distributions.Normal(mean, std)).sum(-1)
    assert bool((exact == 0).all())
    posterior_vs_prior = float(torch_ext.policy_kl(mean + .1, std, mean, std))
    assert posterior_vs_prior > 1

    # Verify the actual agent source uses the exact prior-only KL and a gated
    # actor coefficient, rather than only checking a standalone formula.
    agent_source = (ROOT / "ultra/learning/ultra_agent_distill_vae_rl.py").read_text()
    assert "kl_dist = prior_kl" in agent_source and "actor_coeff_tensor = a_loss.new_tensor(actor_coeff) * kl_gate" in agent_source
    report = {"passed": True,
        "old_random_commands_change_with_identical_state": True,
        "fixed_commands_repeatable_and_binary": True,
        "old_rl_end_flag_never_set_near_episode_end": True,
        "fixed_rl_end_flag_set_near_episode_end": True,
        "student_observation_phase_updates_before_after": phase_updates,
        "fixed_phase_uses_current_sensed_interaction": True,
        "old_infeasible_below_ground_goals_of_768": old_below,
        "fixed_infeasible_below_ground_goals_of_768": new_below,
        "legacy_same_policy_kl": legacy_same_kl,
        "correct_same_policy_kl": 0.0,
        "wrong_posterior_to_prior_kl_for_unchanged_prior": posterior_vs_prior,
        "root_reward_same_xy_different_hidden_z_before_after": root_rewards,
        "no_success_rate_or_causal_ablation_claim": True}
    (OUT / "regression_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
