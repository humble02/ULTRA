#!/usr/bin/env python3
"""Run the fixed student benchmark using the deployed MuJoCo prior policy."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ultra"), str(ROOT / "scripts")]
import mujoco
import mujoco.viewer
import numpy as np
import torch
from sim2sim_vae import HumanoidEnv
from utils.obs_vae import MujocoObs as StudentObs
from student_benchmark_metrics import measure_trajectory
from student_benchmark_policy import verify_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "output/reproduction/student_10000_benchmark")
    args = parser.parse_args()
    os.chdir(ROOT); torch.set_num_threads(2)
    base = args.base.resolve()
    manifest = json.loads((base / "manifest.json").read_text())
    pca_sha = hashlib.sha256((ROOT / 'ultra/data/assets/objects/point_pca_frames.json').read_bytes()).hexdigest()
    out = base / "mujoco"; out.mkdir(parents=True, exist_ok=True)
    dummy = SimpleNamespace(cam=SimpleNamespace(), opt=SimpleNamespace(sitegroup=np.zeros(6)),
                            user_scn=SimpleNamespace(ngeom=0))
    reports, inertia_checks = [], []
    hook_calls = [0]
    for kind in sorted({r["object"] for r in manifest["motions"]}):
        print("Waiting for verified inertial alignment:", kind, flush=True)
        deadline = time.monotonic() + 1800
        while not (base / f"inertia_alignment_{kind}.json").exists():
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out waiting for {kind} Gym export/alignment")
            time.sleep(2)
        rows = [r for r in manifest["motions"] if r["object"] == kind]
        launch = mujoco.viewer.launch_passive
        mujoco.viewer.launch_passive = lambda *a, **kw: dummy
        try:
            with (out / f"{kind}_build.log").open("w") as log, contextlib.redirect_stdout(log):
                env = HumanoidEnv(str(ROOT / manifest["checkpoint"]), device="cpu", motion_path=rows[0]["motion"],
                                  point_viz_sites=0, goal_phase_dim=4)
                env.policy.eval()
                policy_identity = verify_policy(env.policy, manifest)
        finally:
            mujoco.viewer.launch_passive = launch
        prop = json.loads((base / "gym" / kind / "object_inertia.json").read_text())
        bid = env.model.body(kind).id
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, env.model.body_iquat[bid]); rot = rot.reshape(3, 3)
        inertia = rot @ np.diag(env.model.body_inertia[bid]) @ rot.T
        assert np.allclose(env.model.body_mass[bid], prop["mass"], atol=1e-10, rtol=0)
        assert np.allclose(env.model.body_ipos[bid], prop["com"], atol=1e-10, rtol=0)
        assert np.allclose(inertia, prop["inertia"], atol=1e-8, rtol=0)
        inertia_checks.append({"object": kind, "mass": float(env.model.body_mass[bid]), "com": env.model.body_ipos[bid].tolist(),
                               "inertia": inertia.tolist(), "verified_equal_to_native_actor_export": True})
        def reject(module, inputs):
            hook_calls[0] += 1
            raise RuntimeError("Privileged encoder must not run during deploy evaluation")
        hook = env.policy.a2c_network.encoder_net.register_forward_pre_hook(reject)
        for row in rows:
            if env.motion_path != row["motion"]:
                env.motion_path = row["motion"]
                with (out / f"{kind}_build.log").open("a") as log, contextlib.redirect_stdout(log):
                    env._load_motion(row["motion"])
            source = torch.load(row["motion"], map_location="cpu")
            frames = len(source) - 1
            for mode in manifest["modes"]:
                torch.manual_seed(manifest["rollout_seed"]); np.random.seed(manifest["rollout_seed"])
                env.reset_human_and_object_from_ref()
                obs = StudentObs(env.model, kind, env.max_episode_length, env.hoi_data, env.object_points,
                                 env.object_corners, history_step=10, goal_phase_dim=4, obj_goal_decouple="off",
                                 point_fixed_grid_sampling=True)
                obs.configure_task_mode(mode, "points")
                action, torque = torch.zeros(1, 29), torch.zeros(1, 29)
                states, contacts, supports = [], [], []
                hands = {env.model.body(n).id for n in ("left_rubber_hand", "right_rubber_hand")}
                for step in range(frames):
                    with torch.inference_mode():
                        observation = obs._compute_observations(env.data, step, action, torque, student_obs=True, episode_length=step).clamp(-5, 5)
                        assert observation.shape == (1, 1496), observation.shape
                        action = env.policy({"is_train": False, "obs": observation, "prev_actions": None, "rnn_states": None,
                                             "with_vae": False, "with_encoder": False, "vae_noise": torch.zeros(1, 64),
                                             "skip_critic": True})["mus"].clamp(-1, 1)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite policy action")
                    target = action[0].numpy() * 3
                    for _ in range(env.sim_decimation):
                        tau = np.clip((target - env.data.qpos[obs.dof_qpos_ids].astype(np.float32)) * env.stiffness
                                      - env.data.qvel[obs.dof_qvel_ids].astype(np.float32) * env.damping,
                                      -env.torque_limits, env.torque_limits)
                        env.data.ctrl[:] = tau
                        torque = torch.from_numpy(tau).float().unsqueeze(0)
                        mujoco.mj_step(env.model, env.data)
                    # Read joint poses/velocities, object position and body-origin
                    # velocities through the same observation helpers as deployment.
                    from utils.obs_vae import _single_body_world_velocities, wxyz_to_xyzw
                    root_bid = env.model.body("pelvis").id
                    rv, rw = _single_body_world_velocities(env.model, env.data, root_bid)
                    bv, bw = _single_body_world_velocities(env.model, env.data, bid)
                    state = np.concatenate((env.data.xpos[root_bid], wxyz_to_xyzw(env.data.xquat[root_bid]), rv[0], rw[0],
                                            env.data.qpos[obs.dof_qpos_ids], env.data.qvel[obs.dof_qvel_ids],
                                            env.data.xpos[bid], wxyz_to_xyzw(env.data.xquat[bid]), bv[0], bw[0]))
                    states.append(state)
                    hand_touch, ground_touch = False, False
                    for ci in range(env.data.ncon):
                        c = env.data.contact[ci]
                        b1, b2 = (int(env.model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
                        other = b2 if b1 == bid else b1 if b2 == bid else None
                        if other is None:
                            continue
                        force = np.zeros(6); mujoco.mj_contactForce(env.model, env.data, ci, force)
                        if abs(force[0]) > .1:
                            hand_touch |= other in hands
                            ground_touch |= other == 0
                    contacts.append(hand_touch); supports.append(ground_touch)
                reference = source[1:, :84].numpy()
                result = measure_trajectory(states, reference, contacts, supports, frames)
                result.update(engine="mujoco", mode=mode, motion=row["motion"], object=kind,
                              **policy_identity,
                              checkpoint=manifest["checkpoint"], seed=manifest["rollout_seed"],
                              point_fixed_grid_sampling=obs.point_fixed_grid_sampling,
                              point_pca_frame_policy='training4096_reference_v1', point_pca_frame_registry_sha256=pca_sha,
                              privileged_encoder_calls=hook_calls[0], contact_measurement="resolved hand-box and ground-box contact normal force >.1 N")
                reports.append(result)
                np.savez_compressed(out / (Path(row["motion"]).stem + f"_{mode}.npz"), states=states,
                                    reference=reference, hand_contact=contacts, support_contact=supports)
                (out / "metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        hook.remove()
        del env; gc.collect()
    (out / "inertia_verification.json").write_text(json.dumps(inertia_checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
