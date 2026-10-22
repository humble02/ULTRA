#!/usr/bin/env python3
"""Interactive CPU policy comparison using the repository's MuJoCo models and observations."""
import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from learning import ultra_models, ultra_network_builder
from sim2sim_vae import HumanoidEnv, _gather_dof_slices
from utils.obs import MujocoObs as TeacherObs
from utils.obs_vae import MujocoObs as StudentObs

MODES = {
    1: "Student: reference tracking / deploy prior",
    2: "Student: object points + goal / deploy prior",
    3: "Student: object points + goal / privileged encoder (training diagnostic)",
    4: "Original tracking teacher",
}
DEFAULT_MOTION = "InterAct/OMOMO_retarget_aug/sub3_largebox_028_082_084_079_100_100_100.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/student/g1_student_vae/nn/g1_student_vae_00008000.pth"))
    parser.add_argument("--motion", type=Path, default=Path(DEFAULT_MOTION))
    parser.add_argument("--mode", type=int, choices=MODES, default=1)
    parser.add_argument("--physics", choices=["released", "training-joints"], default="training-joints",
                        help="Match measured URDF joint dynamics and released training armature/effort limits")
    parser.add_argument("--initial-state", choices=["gym", "released"], default="gym")
    parser.add_argument("--teacher-observations", choices=["full", "masked"], default="full")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    state = {"mode": args.mode, "reset": True, "paused": False, "quit": False, "ghost": True}

    def on_key(key):
        if key in (49, 50, 51, 52):
            state.update(mode=key - 48, reset=True)
        elif key == 82:
            state["reset"] = True
        elif key == 32:
            state["paused"] = not state["paused"]
        elif key == 71:
            state["ghost"] = not state["ghost"]
        elif key == 256:
            state["quit"] = True

    launch = mujoco.viewer.launch_passive
    mujoco.viewer.launch_passive = lambda *a, **kw: launch(*a, key_callback=on_key, **kw)
    try:
        env = HumanoidEnv(str(args.checkpoint), device="cpu", motion_path=str(args.motion),
                          record_video=False, object_alpha=1.0, point_viz_sites=0, goal_phase_dim=4,
                          align_training_dynamics=args.physics == "training-joints",
                          match_training_reset=args.initial_state == "gym")
    finally:
        mujoco.viewer.launch_passive = launch
    env.policy.eval()
    initial_reference = env.hoi_ref.clone()
    source = torch.load(args.motion, map_location="cpu")
    checkpoint_match = re.search(r"_(\d+)$", args.checkpoint.stem)
    checkpoint_label = str(int(checkpoint_match.group(1))) if checkpoint_match else args.checkpoint.stem
    hand_bodies = {int(env.model.body(name).id) for name in ["left_rubber_hand", "right_rubber_hand"]}
    box_body = int(env.model.body(env.object_name).id)
    teacher_checkpoint = torch.load(ROOT / "ultra/weights/teacher_ultra_inference.pth", map_location="cpu", weights_only=False)
    with (ROOT / "ultra/data/cfg/train/rlg/g1_teacher.yaml").open() as stream:
        teacher_config = yaml.safe_load(stream)["params"]["network"]
    with (ROOT / "output/reproduction/viewer_model_build.log").open("w") as log, contextlib.redirect_stdout(log):
        builder = ultra_network_builder.UltraBuilder()
        builder.load(teacher_config)
        teacher = ultra_models.ModelUltraContinuous(builder).build(
            {"actions_num": 29, "input_shape": (4052,), "num_seqs": 1, "value_size": 1})
        teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.eval()
    mean = teacher_checkpoint["running_mean_std"]["running_mean"].float().cpu()
    var = teacher_checkpoint["running_mean_std"]["running_var"].float().cpu()
    window_id = None
    try:
        tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True)
        match = re.search(r'(0x[0-9a-f]+) "MuJoCo[^"]*": \("MuJoCo" "MuJoCo"\)', tree)
        if match:
            window_id = match.group(1)
            x11 = ctypes.CDLL("libX11.so.6")
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            x11.XFlush.argtypes = [ctypes.c_void_p]
            x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            display = x11.XOpenDisplay(None)
            if display:
                x11.XRaiseWindow(display, int(window_id, 16))
                x11.XFlush(display)
                x11.XCloseDisplay(display)
    except (OSError, subprocess.SubprocessError):
        pass

    def title():
        text = "ULTRA student {} | {} | {} | 1/2/3/4 modes, SPACE pause, R reset, G reference".format(checkpoint_label, MODES[state["mode"]], args.physics)
        print(text, flush=True)
        if window_id:
            for name, kind in [("WM_NAME", "8s"), ("_NET_WM_NAME", "8u")]:
                subprocess.run(["xprop", "-id", window_id, "-f", name, kind, "-set", name, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    report_path = ROOT / "output/reproduction/viewer_rollouts.jsonl"
    episode = 0
    try:
        while env.viewer.is_running() and not state["quit"]:
            if state["reset"]:
                torch.manual_seed(42)
                np.random.seed(42)
                # The native reset adds a pelvis-height offset in-place. Start
                # each comparison from a fresh copy instead of accumulating it.
                env.hoi_ref = initial_reference.clone()
                env.reset_human_and_object_from_ref()
                student_obs = StudentObs(env.model, env.object_name, env.max_episode_length,
                                         env.hoi_data, env.object_points, env.object_corners,
                                         history_step=10, goal_phase_dim=4, obj_goal_decouple="off",
                                         point_fixed_grid_sampling=True)
                student_obs.configure_task_mode("full_track" if state["mode"] == 1 else "object_obs", obj_obs="points")
                student_obs.set_command_goal(env.commands)
                teacher_obs = TeacherObs(env.model, env.object_name, env.max_episode_length,
                                         env.hoi_data, env.object_points, history_step=10,
                                         keep_task_obs=args.teacher_observations == "full",
                                         keep_ig_obs=args.teacher_observations == "full")
                last_action = torch.zeros(1, 29)
                last_torque = torch.zeros(1, 29)
                timestep = 0
                episode += 1
                minimum_height = float(env.data.qpos[2])
                initial_box = env.data.body(env.object_name).xpos.copy()
                highest_box = float(initial_box[2])
                hand_contact_controls = 0
                both_hand_contact_controls = 0
                peak_hand_normal_force = 0.0
                box_trajectory = []
                deadline = time.perf_counter()
                state["reset"] = False
                title()
            if state["paused"]:
                env.viewer.sync()
                time.sleep(0.02)
                deadline = time.perf_counter()
                continue
            mode = state["mode"]
            with torch.inference_mode():
                if mode == 4:
                    raw = teacher_obs._compute_observations(env.data, timestep, last_action, last_torque,
                                                            student_obs=False, episode_length=timestep)
                    normalized = ((raw - mean) / torch.sqrt(var + 1e-5)).clamp(-5, 5)
                    action = teacher({"is_train": False, "obs": normalized, "prev_actions": None,
                                      "rnn_states": None})["mus"].clamp(-1, 1)
                else:
                    obs = student_obs._compute_observations(env.data, timestep, last_action, last_torque,
                                                           student_obs=True, episode_length=timestep)
                    obs = obs.clamp(-5, 5)  # Match VecTaskDAggerWrapper.
                    inputs = {"is_train": False, "prev_actions": None, "obs": obs, "rnn_states": None,
                              "with_vae": mode == 3, "with_encoder": mode == 3,
                              "vae_noise": torch.zeros(1, 64), "skip_critic": True}
                    if mode == 3:
                        privileged = teacher_obs._compute_observations(
                            env.data, timestep, last_action, last_torque, student_obs=False, episode_length=timestep)
                        inputs["teacher_obs"] = ((privileged - mean) / torch.sqrt(var + 1e-5)).clamp(-5, 5)
                    action = env.policy(inputs)["mus"].clamp(-1, 1)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Policy returned a nonfinite action")
            target = action[0].numpy() * 3.0
            last_action = action
            for _ in range(env.sim_decimation):
                q, qd = _gather_dof_slices(env.data, student_obs.dof_qpos_ids, student_obs.dof_qvel_ids)
                torque = np.clip((target - q[0].numpy()) * env.stiffness - qd[0].numpy() * env.damping,
                                 -env.torque_limits, env.torque_limits)
                env.data.ctrl[:] = torque
                last_torque = torch.from_numpy(torque).float().unsqueeze(0)
                mujoco.mj_step(env.model, env.data)
            minimum_height = min(minimum_height, float(env.data.qpos[2]))
            highest_box = max(highest_box, float(env.data.body(env.object_name).xpos[2]))
            box_trajectory.append(env.data.body(env.object_name).xpos.copy())
            touching = set()
            for contact_index in range(env.data.ncon):
                contact = env.data.contact[contact_index]
                body1 = int(env.model.geom_bodyid[contact.geom1])
                body2 = int(env.model.geom_bodyid[contact.geom2])
                hand = body2 if body1 == box_body else body1 if body2 == box_body else None
                if hand in hand_bodies:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, contact_index, force)
                    if abs(float(force[0])) > 0.1:
                        touching.add(hand)
                    peak_hand_normal_force = max(peak_hand_normal_force, abs(float(force[0])))
            hand_contact_controls += bool(touching)
            both_hand_contact_controls += len(touching) == 2
            with env.viewer.lock():
                env.viewer.cam.lookat[:] = env.data.qpos[:3]
                scene = env.viewer.user_scn
                scene.ngeom = 0
                if state["ghost"]:
                    reference = env.hoi_data[min(timestep + 1, env.max_episode_length - 1), 84:201].reshape(39, 3).numpy()
                    for pos in reference:
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.018, 0.018, 0.018]), pos.astype(float), np.eye(3).ravel(),
                                            np.array([0.15, 0.55, 1.0, 0.55], dtype=np.float32))
                        scene.ngeom += 1
            env.viewer.sync()
            timestep += 1
            end = min(len(source) - 1, int(15 / env.control_dt))
            if timestep >= end:
                box = env.data.body(env.object_name).xpos.copy()
                box_path = np.asarray(box_trajectory)
                reference_box = source[1:timestep+1, 71:74].numpy()
                record = {"viewer_version": 5, "physics": args.physics,
                          "observation_encoding": "gym_camera_points_raw_goal_all_modalities_full_track",
                          "checkpoint": str(args.checkpoint),
                          "initial_state": args.initial_state,
                          "teacher_observations": args.teacher_observations,
                          "episode": episode, "mode": mode, "description": MODES[mode],
                          "controls": timestep, "minimum_pelvis_height_m": minimum_height,
                          "maximum_box_lift_m": highest_box - float(initial_box[2]),
                          "box_xy_travel_m": float(np.linalg.norm(box[:2] - initial_box[:2])),
                          "box_position_rmse_m": float(np.sqrt(np.square(box_path - reference_box).sum(-1).mean())),
                          "final_box_xy_error_m": float(np.linalg.norm(box[:2] - reference_box[-1, :2])),
                          "hand_box_contact_controls": hand_contact_controls,
                          "both_hand_box_contact_controls": both_hand_contact_controls,
                          "peak_hand_box_normal_force_n": peak_hand_normal_force,
                          "note": "Diagnostic rollout only; these values do not establish task success."}
                with report_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                until = time.perf_counter() + 1.5
                while time.perf_counter() < until and env.viewer.is_running() and not state["reset"]:
                    env.viewer.sync()
                    time.sleep(0.02)
                state["reset"] = True
            deadline += env.control_dt
            time.sleep(max(0, deadline - time.perf_counter()))
            deadline = max(deadline, time.perf_counter() - env.control_dt)
    finally:
        env.viewer.close()


if __name__ == "__main__":
    main()
