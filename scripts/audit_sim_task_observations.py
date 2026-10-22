#!/usr/bin/env python3
"""Compare task/command/mask math with native Gym code on identical CPU inputs.

This isolates encoding errors; it does not claim equivalence of engine state
extraction, PCA CPU/GPU eigenvector conventions, randomization, or dynamics.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ultra"), str(ROOT / "scripts")]
import isaacgym  # noqa: F401
import numpy as np
import torch
from env.tasks.ultra_g1_distill_obj_v2vae import (
    UltraDistillObjV2Point as Native,
    compute_obj_observations_pca_corners as native_points,
)
from diagnose_sim_gap import create_env, StudentObs


def cpu_eigh(value):
    values, axes = np.linalg.eigh(value.detach().numpy())
    return torch.from_numpy(values).to(value), torch.from_numpy(axes).to(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-equal", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT); torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".log").open("w") as log, contextlib.redirect_stdout(log):
        env = create_env()
    capture = torch.load("output/reproduction/gap/gym_snapshots/full_track_snapshots.pt", map_location="cpu")
    reports = []
    modes = [("full_track", "points"), ("sparse_track", "points"),
             ("object_obs", "points"), ("object_obs", "pos"), ("object_obs", "none")]
    for index in (1, 30, 60):
        snap = capture["snapshots"][index]
        state = snap["_curr_obs"].unsqueeze(0)
        ref = env.hoi_data[[min(int(snap["progress_buf"] + snap["long_term_t"]), env.max_episode_length - 1)]]
        camera = capture["body_names"].index("d435_link")
        cam_pos = state[:, 84:201].reshape(1, 39, 3)[:, camera]
        cam_rot = state[:, 201:357].reshape(1, 39, 4)[:, camera]
        obs = StudentObs(env.model, env.object_name, env.max_episode_length, env.hoi_data,
                         env.object_points, env.object_corners, goal_phase_dim=4, obj_goal_decouple="off")
        inputs = (state[:, :13], state[:, 71:84], env.object_corners, ref, cam_pos, cam_rot)
        # Both implementations use exactly the same NumPy PCA backend here.
        with patch("torch.linalg.eigh", cpu_eigh):
            expected_task, expected_world = native_points(*inputs, fixed_surface_sampling=True, disable_geometry_noise=True)
            actual_task, actual_world, _ = obs.compute_obj_observations_pca_corners(
                *inputs, fixed_surface_sampling=True, disable_geometry_noise=True)
        for mode, point_mode in modes:
            obs.configure_task_mode(mode, point_mode)
            fake = SimpleNamespace(**vars(obs))
            fake.num_envs = 1; fake.device = "cpu"
            for name in vars(fake).copy():
                if name.startswith("keep_") and name.endswith("_mask"):
                    setattr(fake, name, getattr(fake, name).reshape(1).clone())
            fake._apply_task_mode_masks = lambda ids: Native._apply_task_mode_masks(fake, ids)
            fake._maybe_flip_keep_masks = lambda ids: None
            fake.debug_obs_layout = False
            with contextlib.redirect_stdout(None):
                Native.configure_task_mode(fake, mode, point_mode)
            ids = torch.tensor([0])
            humanoid = torch.zeros(1, 1046)
            fake.goal_phase = torch.tensor([[0., 1., 0., .5]])
            obs.command_goal = fake.goal_phase.clone()
            expected = Native._finalize_student_obs(fake, ids, torch.cat((humanoid, expected_task), -1), 1046, 204)
            actual = obs._finalize_student_obs(torch.cat((humanoid, actual_task), -1), 1046, 204)
            reports.append({"control": index, "mode": mode, "object_input": point_mode,
                            "total_dimensions": actual.shape[-1],
                            "task_goal_max_error": (actual_task[:, :3] - expected_task[:, :3]).abs().max().item(),
                            "camera_points_max_error": (actual_task[:, 12:] - expected_task[:, 12:]).abs().max().item(),
                            "visible_world_points_max_error": (actual_world - expected_world).abs().max().item(),
                            "mask_mismatched_dimensions": int((actual[:, -242:] != expected[:, -242:]).sum()),
                            "assembled_max_error": (actual - expected).abs().max().item()})
    # Exercise command transitions, the end flag, and time-to-go over a full episode.
    obs.prev_hand_ig_valid[:] = False
    fake.prev_hand_ig_valid = torch.tensor([False]); fake.prev_hand_ig = torch.zeros(1)
    fake.max_episode_length = torch.tensor([env.max_episode_length]); fake.data_id = torch.tensor([0])
    phase_error = 0.
    for step in range(env.max_episode_length):
        ig = torch.full((1, 39, 1), .1 + .2 * (1 + np.sin(step / 13)))
        obs.long_term_t[:] = (137 - step) % 120
        fake.long_term_t = obs.long_term_t.clone(); fake.progress_buf = torch.tensor([step])
        Native._update_goal_phase(fake, ids, ig)
        actual_phase = obs._update_goal_phase(ig, step)
        phase_error = max(phase_error, (actual_phase - fake.goal_phase).abs().max().item())
    result = {"checks": reports, "command_phase_max_error": phase_error,
              "scope": "Identical inputs, deterministic fixed-grid surface sampling, shared NumPy PCA, zero noise. Humanoid math separately checked in observation_equivalence.json. Not an end-to-end engine or policy equivalence test."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.require_equal:
        assert phase_error < 2e-6
        assert all(row["assembled_max_error"] < 2e-6 for row in reports)


if __name__ == "__main__":
    main()
