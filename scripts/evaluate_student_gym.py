#!/usr/bin/env python3
"""Evaluate a saved student's deploy prior on one motion in native Isaac Gym."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401; must precede torch
import numpy as np
import torch
import yaml
from learning import ultra_players_distill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["full_track", "object_obs"],
                        default=["object_obs", "full_track"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-noise", choices=["zero", "sampled"], default="zero")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--capture-observations", action="store_true")
    parser.add_argument("--max-controls", type=int, help="limit a diagnostic rollout")
    args = parser.parse_args()
    if args.repeat and not args.viewer:
        parser.error("--repeat requires --viewer")
    args.checkpoint = args.checkpoint.resolve()
    args.motion = args.motion.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file() or not args.motion.is_file():
        parser.error("checkpoint and motion must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion_dir = args.output_dir / "motion"
    motion_dir.mkdir(exist_ok=True)
    link = motion_dir / args.motion.name
    if not link.exists():
        link.symlink_to(args.motion)
    if len(list(motion_dir.glob("*.pt"))) != 1:
        parser.error("evaluation motion directory must contain exactly one motion")
    os.chdir(ROOT)
    config = yaml.safe_load((ROOT / "ultra/data/cfg/g1_student_vae.yaml").read_text())
    teacher = yaml.safe_load((ROOT / "ultra/data/cfg/g1_teacher_no_dr.yaml").read_text())
    config["domain_rand"] = teacher["domain_rand"]
    config["noise"] = teacher["noise"]
    config["env"].update(numEnvs=1, stateInit="Start", enableEarlyTermination=False,
                          enableDebugVis=False, obs_mask_noise_std=0.0)
    for key in ("pointNoiseStd", "pointDropoutProb", "pointOutlierProb", "pointDepthNoiseScale",
                "pointClusterNoiseStd", "pointTranslationNoise", "pointOcclusionProb",
                "cameraRotNoise", "cameraPosNoise"):
        config["env"][key] = 0.0
    for key in ("pointDensityMin", "pointDensityMax", "pointScaleMin", "pointScaleMax"):
        config["env"][key] = 1.0
    config["sim"]["physx"].update(max_gpu_contact_pairs=65536, default_buffer_size_multiplier=2.0)
    cfg_path = args.output_dir / "evaluation.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    source = torch.load(args.motion, map_location="cpu")

    def run_rollouts(player):
        task = player.env.task
        if args.capture_observations:
            def vector(value):
                return [float(value.x), float(value.y), float(value.z)]

            def bodies(handle):
                props = task.gym.get_actor_rigid_body_properties(task.envs[0], handle)
                result = []
                for name, prop in zip(task.gym.get_actor_rigid_body_names(task.envs[0], handle), props):
                    com = prop.com.p if hasattr(prop.com, "p") else prop.com
                    result.append({"name": name, "mass": float(prop.mass), "com": vector(com),
                                   "inertia": [vector(prop.inertia.x), vector(prop.inertia.y), vector(prop.inertia.z)]})
                return result

            def shapes(handle):
                props = task.gym.get_actor_rigid_shape_properties(task.envs[0], handle)
                return [{key: float(getattr(p, key)) for key in
                         ("friction", "rolling_friction", "torsion_friction", "restitution", "contact_offset", "rest_offset")}
                        for p in props]

            humanoid, target = task.humanoid_handles[0], task._target_handles[0]
            properties = {"humanoid_bodies": bodies(humanoid), "object_bodies": bodies(target),
                          "humanoid_shapes": shapes(humanoid), "object_shapes": shapes(target)}
            (args.output_dir / "native_actor_properties.json").write_text(json.dumps(properties, indent=2) + "\n")
        task._enable_early_termination = False
        task.rollout_length = int(task.max_episode_length[0]) + 1
        frames = min(len(source) - 1, args.max_controls) if args.max_controls else len(source) - 1
        encoder_calls = [0]

        def reject_encoder(module, inputs):
            encoder_calls[0] += 1
            raise RuntimeError("Deploy evaluation must not call the privileged encoder")

        hook = player.model.a2c_network.encoder_net.register_forward_pre_hook(reject_encoder)
        reports = []
        try:
            while True:
                for mode in args.modes:
                    # Manual mode changes do not follow the task's normal done
                    # path, which clears these histories after termination.
                    for name in ("actions", "last_actions", "last_dof_pos", "last_dof_vel",
                                 "torques", "action_history_buf", "obs_history_buf",
                                 "episode_length_buf", "last_contact_forces"):
                        value = getattr(task, name, None)
                        if isinstance(value, torch.Tensor):
                            value.zero_()
                    task.configure_task_mode(mode, "points")
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    obs = player.env_reset()
                    player.get_batch_size(obs["obs"], 1)
                    if args.latent_noise == "zero":
                        player._vae_noise = torch.zeros(1, player.model.a2c_network.vae_dim, device=player.device)
                    observed, reference, actions = [], [], []
                    initial_box = task._target_states[0, :3].detach().cpu().clone()
                    initial_student_obs = obs["obs"].detach().cpu().clone()
                    snapshots = []
                    print(f"Evaluating student deploy prior: {mode}, {frames} controls", flush=True)
                    for index in range(frames):
                        started = time.perf_counter()
                        # The environment computes labels for training; inference
                        # deliberately excludes them from the policy input.
                        player._last_expert = None
                        if args.capture_observations:
                            task._compute_hoi_observations(torch.arange(task.num_envs, device=task.device))
                            snapshot = {name: getattr(task, name)[0].detach().cpu().clone()
                                        for name in ("_curr_obs", "obs_buf", "torques", "last_dof_pos",
                                                     "last_dof_vel", "obs_history_buf", "goal_phase",
                                                     "long_term_t", "progress_buf", "episode_length_buf")}
                            snapshot["student_obs"] = obs["obs"][0].detach().cpu().clone()
                            snapshot["previous_action"] = task.action_history_buf[0, -1].detach().cpu().clone()
                            snapshot["root_actor_states"] = task._humanoid_root_states[0].detach().cpu().clone()
                            snapshot["object_actor_states"] = task._target_states[0].detach().cpu().clone()
                            snapshot["contact_forces"] = task._contact_forces[0].detach().cpu().clone()
                            snapshots.append(snapshot)
                        action = player.get_action(obs, True)
                        if not torch.isfinite(action).all():
                            raise RuntimeError("Nonfinite student action")
                        next_obs, _, done, _ = player.env_step(player.env, action)
                        obs = next_obs if isinstance(next_obs, dict) else {"obs": next_obs}
                        observed.append(task._curr_obs[0].detach().cpu().clone())
                        reference.append(task._curr_ref_obs[0].detach().cpu().clone())
                        actions.append(action[0].detach().cpu().clone())
                        if (index + 1) % 100 == 0:
                            print(f"{mode}: {index + 1}/{frames}", flush=True)
                        if task.viewer:
                            time.sleep(max(0.0, task.dt - (time.perf_counter() - started)))
                        if bool(done[0]):
                            break
                    actual = torch.stack(observed)
                    refs = torch.stack(reference)
                    low = (actual[:, 2] < 0.25).nonzero().flatten()
                    report = {
                        "engine": "Isaac Gym GPU PhysX", "checkpoint": str(args.checkpoint),
                        "mode": mode, "seed": args.seed, "latent_noise": args.latent_noise,
                        "controls": len(actual), "privileged_encoder_calls": encoder_calls[0],
                        "minimum_pelvis_height_m": actual[:, 2].min().item(),
                        "first_control_pelvis_below_0_25m": int(low[0]) if len(low) else None,
                        "maximum_box_lift_from_reference_start_m": (actual[:, 73].max() - source[0, 73]).item(),
                        "maximum_box_lift_from_actual_start_m": (actual[:, 73].max() - initial_box[2]).item(),
                        "reference_maximum_box_lift_m": (source[:, 73].max() - source[0, 73]).item(),
                        "box_position_rmse_m": (actual[:, 71:74] - refs[:, 71:74]).square().sum(-1).mean().sqrt().item(),
                        "final_box_xy_error_m": torch.linalg.vector_norm(actual[-1, 71:73] - refs[-1, 71:73]).item(),
                        "box_xy_travel_m": torch.linalg.vector_norm(actual[-1, 71:73] - source[0, 71:73]).item(),
                        "all_finite": bool(torch.isfinite(actual).all()),
                        "evaluation_config": str(cfg_path),
                        "note": "One motion without domain/observation randomization; not a dataset-wide success rate. Early termination disabled to record failures."
                    }
                    torch.save({"observed_hoi": actual, "reference_hoi": refs,
                                "actions": torch.stack(actions), "motion_file": str(args.motion),
                                "initial_box_position": initial_box,
                                "initial_student_obs": initial_student_obs}, args.output_dir / f"{mode}_rollout.pt")
                    if snapshots:
                        torch.save({"snapshots": snapshots,
                                    "body_names": task.gym.get_actor_rigid_body_names(task.envs[0], task.humanoid_handles[0]),
                                    "object_points": task.object_points.detach().cpu(),
                                    "object_corners": task.object_corners.detach().cpu(),
                                    "motion_file": str(args.motion)}, args.output_dir / f"{mode}_snapshots.pt")
                    (args.output_dir / f"{mode}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
                    reports.append(report)
                    (args.output_dir / "metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
                    print(json.dumps(report), flush=True)
                if not args.repeat:
                    break
        finally:
            hook.remove()

    ultra_players_distill.UltraPlayerContinuousDistill.run = run_rollouts
    sys.argv = [str(ROOT / "ultra/run_distill.py"), "--task", "UltraDistillObjV2Point",
                "--cfg_env", str(cfg_path), "--cfg_train", "ultra/data/cfg/train/rlg/g1_student_vae.yaml",
                "--test", "--num_envs", "1", "--seed", str(args.seed),
                "--checkpoint", str(args.checkpoint), "--motion_file", str(motion_dir),
                "--task_mode", args.modes[0], "--obj_obs", "points",
                "--output_path", str(args.output_dir / "rlgames")]
    if not args.viewer:
        sys.argv.append("--headless")
    runpy.run_path(str(ROOT / "ultra/run_distill.py"), run_name="__main__")


if __name__ == "__main__":
    main()
