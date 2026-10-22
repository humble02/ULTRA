#!/usr/bin/env python3
"""Audit compiled MuJoCo properties against a native Gym actor export."""
import contextlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_sim_gap import create_env, set_state, StudentObs
import mujoco
import numpy as np
import torch
from utils import obs as teacher_obs, obs_vae as student_obs


def main():
    os.chdir(ROOT); torch.set_num_threads(2)
    out = ROOT / "output/reproduction/gap/recheck"
    native = json.loads((out / "native_capture/native_actor_properties.json").read_text())
    with (out / "physics_build.log").open("w") as log, contextlib.redirect_stdout(log):
        env = create_env()
    def compare(prop, bid):
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, env.model.body_iquat[bid])
        rot = rot.reshape(3, 3)
        inertia = rot @ np.diag(env.model.body_inertia[bid]) @ rot.T
        return {"name": prop["name"], "gym_mass_kg": prop["mass"],
                "mujoco_mass_kg": float(env.model.body_mass[bid]),
                "com_max_error_m": float(np.max(np.abs(env.model.body_ipos[bid] - prop["com"]))),
                "inertia_matrix_max_error_kg_m2": float(np.max(np.abs(inertia - prop["inertia"]))),
                "gym_inertia": prop["inertia"], "mujoco_inertia": inertia.tolist()}
    robot = [compare(p, env.model.body(p["name"]).id) for p in native["humanoid_bodies"]]
    box = compare(native["object_bodies"][0], env.model.body(env.object_name).id)
    capture = torch.load(out / "native_capture/full_track_snapshots.pt", map_location="cpu")
    gym_corners = capture["object_corners"].reshape(-1, 3)
    mj_corners = env.object_corners.reshape(-1, 3)
    geometry = {"corner_set_hausdorff_m": float(torch.cdist(gym_corners, mj_corners).min(-1).values.max()),
                "gym_point_extents_m": (gym_corners.max(0).values - gym_corners.min(0).values).tolist(),
                "mujoco_point_extents_m": (mj_corners.max(0).values - mj_corners.min(0).values).tolist(),
                "same_seed_surface_point_max_error_m": (capture["object_points"].reshape(-1, 3) - env.object_points.reshape(-1, 3)).abs().max().item()}
    obs = StudentObs(env.model, env.object_name, env.max_episode_length, env.hoi_data,
                     env.object_points, env.object_corners, goal_phase_dim=4)
    velocity = []
    for index in (1, 30, 60):
        snap = capture["snapshots"][index]
        state = snap["_curr_obs"].clone()
        root_delta = (state[:13] - snap["root_actor_states"]).abs().max().item()
        state[:13] = snap["root_actor_states"]
        state[71:84] = snap["object_actor_states"]
        set_state(env, state)
        linear, angular = student_obs._body_world_velocities(env.model, env.data, obs.body_ids)
        positions, _ = student_obs._gather_body_poses(env.model, env.data, obs.body_ids)
        ref_linear = state[357:474].reshape(1, 39, 3)
        ref_angular = state[474:591].reshape(1, 39, 3)
        velocity.append({"control": index, "root_actor_vs_hoi_max_error": root_delta,
                         "body_position_max_error_m": (positions - state[84:201].reshape(1, 39, 3)).abs().max().item(),
                         "linear_velocity_rmse_m_s": (linear - ref_linear).square().mean().sqrt().item(),
                         "angular_velocity_rmse_rad_s": (angular - ref_angular).square().mean().sqrt().item()})
    # A settled 1 kg body should have ~9.81 N upward contact force, zero torque.
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.001"/><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .101"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    forces = {}
    for name, module in (("teacher_helper", teacher_obs), ("student_helper", student_obs)):
        force = module._gather_contact_forces(model, data, [1])[0, 0].numpy()
        assert np.allclose(force, [0, 0, 9.81], atol=.02), force
        forces[name] = force.tolist()
    result = {"object": box, "object_geometry": geometry, "robot_bodies": robot,
              "robot_total_mass_gym": sum(p["gym_mass_kg"] for p in robot),
              "robot_total_mass_mujoco": sum(p["mujoco_mass_kg"] for p in robot),
              "native_shapes": {k: native[k] for k in ("object_shapes", "humanoid_shapes")},
              "mujoco_geoms": [{"name": env.model.geom(i).name, "body": env.model.body(env.model.geom_bodyid[i]).name,
                                  "friction": env.model.geom_friction[i].tolist(),
                                  "margin": float(env.model.geom_margin[i]), "gap": float(env.model.geom_gap[i])}
                                 for i in range(env.model.ngeom) if env.model.geom_contype[i]],
              "same_state_velocity_checks": velocity, "settled_box_contact_force_N": forces,
              "scope": "Nominal no-DR Gym actor export, one 100-percent largebox asset. Inertias expressed in each body frame. Velocity checks transfer actor root states and joint positions/velocities; engine velocity-reporting differences are unresolved."}
    (out / "physics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"object": box, "robot_total_mass_gym": result["robot_total_mass_gym"],
                      "robot_total_mass_mujoco": result["robot_total_mass_mujoco"],
                      "worst_robot_inertia": sorted(robot, key=lambda p:p["inertia_matrix_max_error_kg_m2"], reverse=True)[:3],
                      "same_state_velocity_checks": velocity, "force_test": forces}, indent=2))


if __name__ == "__main__":
    main()
