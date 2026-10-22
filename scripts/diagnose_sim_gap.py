#!/usr/bin/env python3
"""CPU-only MuJoCo ablations and Gym state/observation comparisons."""
import argparse
import ast
import contextlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from learning import ultra_models, ultra_network_builder
from sim2sim_vae import HumanoidEnv, _set_freejoint_state, _write_hinge_qpos_qvel_by_names, JOINT_NAMES_29
from utils import obs as teacher_obs_module
from utils.obs import MujocoObs as TeacherObs
from utils.obs_vae import MujocoObs as StudentObs

MOTION = "InterAct/OMOMO_retarget_aug/sub3_largebox_028_082_084_079_100_100_100.pt"
CHECKPOINT = "output/student/g1_student_vae/nn/g1_student_vae_00008000.pth"


def create_env():
    dummy = SimpleNamespace(cam=SimpleNamespace(), opt=SimpleNamespace(sitegroup=np.zeros(6)),
                            user_scn=SimpleNamespace(ngeom=0))
    launch = mujoco.viewer.launch_passive
    mujoco.viewer.launch_passive = lambda *a, **kw: dummy
    try:
        return HumanoidEnv(CHECKPOINT, device="cpu", motion_path=MOTION, point_viz_sites=0, goal_phase_dim=4,
                           align_training_dynamics=False, match_training_reset=False)
    finally:
        mujoco.viewer.launch_passive = launch


def training_joints(env):
    tree = ast.parse((ROOT / "ultra/env/tasks/humanoid_g1.py").read_text())
    build = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_build_env")
    values = {}
    for n in ast.walk(build):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ("armature", "effort"):
                values[n.targets[0].id] = ast.literal_eval(n.value)
    ids = np.array([int(env.model.joint(n).dofadr[0]) for n in JOINT_NAMES_29])
    env.model.dof_armature[ids] = values["armature"]
    env.model.dof_damping[ids] = 0
    env.model.dof_frictionloss[ids] = 0
    env.torque_limits = np.asarray(values["effort"]) * 0.8


def set_state(env, state):
    mujoco.mj_resetData(env.model, env.data)
    _set_freejoint_state(env.model, env.data, "pelvis", state[:3], state[3:7], state[7:10], state[10:13])
    _write_hinge_qpos_qvel_by_names(env.model, env.data, JOINT_NAMES_29, state[13:42], state[42:71])
    _set_freejoint_state(env.model, env.data, env.object_name + "_free", state[71:74], state[74:78], state[78:81], state[81:84])
    mujoco.mj_forward(env.model, env.data)


def teacher_model():
    ck = torch.load("ultra/weights/teacher_ultra_inference.pth", map_location="cpu")
    cfg = yaml.safe_load(Path("ultra/data/cfg/train/rlg/g1_teacher.yaml").read_text())["params"]["network"]
    b = ultra_network_builder.UltraBuilder(); b.load(cfg)
    m = ultra_models.ModelUltraContinuous(b).build({"actions_num": 29, "input_shape": (4052,), "num_seqs": 1, "value_size": 1})
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["running_mean_std"]["running_mean"].float(), ck["running_mean_std"]["running_var"].float()


def rollout(env, teacher, mean, var, profile, policy, frames, student_mode="full_track"):
    torch.manual_seed(42); np.random.seed(42)
    source = torch.load(MOTION, map_location="cpu")
    if profile in ("joints", "unmasked", "init", "mass"):
        training_joints(env)
    if profile == "mass":
        bid = env.model.body(env.object_name).id
        mass = json.loads(Path("output/reproduction/isaac_asset_properties.json").read_text())["box_nominal_mass"]
        env.model.body_inertia[bid] *= mass / env.model.body_mass[bid]
        env.model.body_mass[bid] = mass
    # mj_setConst changes data.qpos; run it before resetting the episode.
    mujoco.mj_setConst(env.model, env.data)
    if profile in ("init", "mass"):
        state = source[0].clone(); state[73] += 0.1
        set_state(env, state)
    else:
        env.hoi_ref = source[:, :84].clone()
        env.reset_human_and_object_from_ref()
    enabled = profile in ("released-unmasked", "unmasked", "init", "mass")
    tobs = TeacherObs(env.model, env.object_name, env.max_episode_length, env.hoi_data, env.object_points,
                      history_step=10, keep_task_obs=enabled, keep_ig_obs=enabled)
    sobs = StudentObs(env.model, env.object_name, env.max_episode_length, env.hoi_data, env.object_points,
                      env.object_corners, history_step=10, goal_phase_dim=4, obj_goal_decouple="off",
                      point_fixed_grid_sampling=True)
    sobs.configure_task_mode(student_mode, "points")
    action, torque = torch.zeros(1,29), torch.zeros(1,29)
    qids, vids = sobs.dof_qpos_ids, sobs.dof_qvel_ids
    poses, boxes, actions = [], [], []
    hand_bodies = {env.model.body(n).id for n in ("left_rubber_hand", "right_rubber_hand")}
    box_body = env.model.body(env.object_name).id
    hand_contacts = bilateral_contacts = 0
    peak_hand_force = 0.0
    for index in range(frames):
        with torch.inference_mode():
            if policy == "teacher":
                obs = tobs._compute_observations(env.data,index,action,torque,episode_length=index)
                action = teacher({"is_train":False,"obs":((obs-mean)/torch.sqrt(var+1e-5)).clamp(-5,5),"prev_actions":None,"rnn_states":None})["mus"].clamp(-1,1)
            else:
                obs = sobs._compute_observations(env.data,index,action,torque,student_obs=True,episode_length=index).clamp(-5,5)
                action = env.policy({"is_train":False,"obs":obs,"prev_actions":None,"rnn_states":None,"with_vae":False,"with_encoder":False,"vae_noise":torch.zeros(1,64),"skip_critic":True})["mus"].clamp(-1,1)
        target = action[0].numpy()*3
        for _ in range(17):
            tau = np.clip((target-env.data.qpos[qids].astype(np.float32))*env.stiffness-env.data.qvel[vids].astype(np.float32)*env.damping,-env.torque_limits,env.torque_limits)
            env.data.ctrl[:] = tau
            torque = torch.from_numpy(tau).float().unsqueeze(0)
            mujoco.mj_step(env.model,env.data)
        poses.append(env.data.qpos[:3].copy());boxes.append(env.data.body(env.object_name).xpos.copy());actions.append(action[0].clone())
        touching = set()
        for contact_index in range(env.data.ncon):
            contact = env.data.contact[contact_index]
            b1, b2 = (int(env.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            hand = b2 if b1 == box_body else b1 if b2 == box_body else None
            if hand in hand_bodies:
                force = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, contact_index, force)
                peak_hand_force = max(peak_hand_force, abs(float(force[0])))
                if abs(force[0]) > 0.1:
                    touching.add(hand)
        hand_contacts += bool(touching)
        bilateral_contacts += len(touching) == 2
    pos=np.asarray(poses);box=np.asarray(boxes);ref=source[1:frames+1,71:74].numpy()
    return {"profile":profile,"policy":policy,"student_mode":student_mode,"controls":frames,"minimum_pelvis_height_m":float(pos[:,2].min()),
            "hand_box_force_contact_controls": hand_contacts, "bilateral_hand_box_force_contact_controls": bilateral_contacts,
            "peak_hand_normal_force_N": peak_hand_force,
            "maximum_box_lift_m":float(box[:,2].max()-source[0,73]),"box_xy_travel_m":float(np.linalg.norm(box[-1,:2]-source[0,71:73].numpy())),
            "final_box_xy_error_m":float(np.linalg.norm(box[-1,:2]-ref[-1,:2])),"box_position_rmse_m":float(np.sqrt(np.square(box-ref).sum(-1).mean()))}, {"pelvis":pos,"box":box,"actions":torch.stack(actions)}


def frozen(env, path):
    capture=torch.load(path,map_location="cpu")
    results=[]
    for index in (0,1,30,60):
        snap=capture["snapshots"][index];state=snap["_curr_obs"]
        set_state(env,state)
        tobs=TeacherObs(env.model,env.object_name,env.max_episode_length,env.hoi_data,env.object_points,
                        history_step=10,keep_task_obs=True,keep_ig_obs=True)
        actual=env.data.xpos[tobs.body_ids]
        expected=state[84:201].reshape(39,3).numpy()
        error=np.linalg.norm(actual-expected,axis=-1)
        tobs.last_dof_pos=snap["last_dof_pos"].unsqueeze(0)
        tobs.last_dof_vel=snap["last_dof_vel"].unsqueeze(0)
        raw=tobs._compute_observations(env.data,index,snap["previous_action"].unsqueeze(0),snap["torques"].unsqueeze(0),episode_length=index)[0]
        delta=(raw-snap["obs_buf"]).abs()
        results.append({"control":index,"body_position_max_error_m":float(error.max()),
                        "body_position_errors":dict(zip(capture["body_names"],error.tolist())),
                        "teacher_obs_max_abs_error":delta.max().item(),"teacher_obs_mean_abs_error":delta.mean().item(),
                        "teacher_top_error_indices":torch.topk(delta,20).indices.tolist(),
                        "teacher_top_error_values":torch.topk(delta,20).values.tolist()})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles",nargs="+",choices=["released","released-unmasked","joints","unmasked","init","mass"],default=["joints","unmasked","init","mass"])
    parser.add_argument("--policies",nargs="+",default=["teacher","student"])
    parser.add_argument("--frames",type=int,default=411)
    parser.add_argument("--student-mode",choices=["full_track","object_obs"],default="full_track")
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--output-dir",type=Path,default=Path("output/reproduction/gap/mujoco"))
    args=parser.parse_args();os.chdir(ROOT);torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/"build.log").open("w") as log,contextlib.redirect_stdout(log):
        env=create_env();env.policy.eval();teacher,mean,var=teacher_model()
    if args.frozen:
        result=frozen(env,args.frozen)
        (args.output_dir/"frozen.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)
        return
    model_arrays={n:getattr(env.model,n).copy() for n in ("dof_armature","dof_damping","dof_frictionloss","body_mass","body_inertia")}
    limits=env.torque_limits.copy();reports=[]
    for policy in args.policies:
        for profile in args.profiles:
            for n,v in model_arrays.items():getattr(env.model,n)[:]=v
            env.torque_limits=limits.copy()
            report,trajectory=rollout(env,teacher,mean,var,profile,policy,args.frames,args.student_mode)
            reports.append(report);torch.save(trajectory,args.output_dir/f"{policy}_{profile}.pt")
            (args.output_dir/"metrics.json").write_text(json.dumps(reports,indent=2)+"\n")
            print(json.dumps(report),flush=True)


if __name__=="__main__":main()
