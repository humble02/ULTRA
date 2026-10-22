#!/usr/bin/env python3
"""Compare the original Gym and MuJoCo student observation math on identical tensors."""
import contextlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
sys.path.insert(0, str(ROOT / "scripts"))
import isaacgym  # noqa: F401
import torch
from env.tasks.ultra_g1_distill_obj_v2vae import UltraDistillObjV2Point
from diagnose_sim_gap import create_env, TeacherObs, StudentObs


def main():
    os.chdir(ROOT); torch.set_num_threads(2)
    output = ROOT / "output/reproduction/gap"
    with (output / "observation_test_build.log").open("w") as log, contextlib.redirect_stdout(log):
        env = create_env()
    capture = torch.load(output / "gym_snapshots/full_track_snapshots.pt", map_location="cpu")
    reports = []
    for index in (1, 30, 60):
        snap = capture["snapshots"][index]; state = snap["_curr_obs"].unsqueeze(0)
        ts = int(snap["progress_buf"]); horizon = int(snap["long_term_t"])
        ref = env.hoi_data[[min(ts+horizon,env.max_episode_length-1)]]
        local = env.hoi_data[[ts+1]]; ref16 = env.hoi_data[[ts+16]]
        pos=state[:,84:201].reshape(1,39,3);rot=state[:,201:357].reshape(1,39,4)
        vel=state[:,357:474].reshape(1,39,3);ang=state[:,474:591].reshape(1,39,3)
        contact=torch.zeros(1,39,3);contact[:,:,0]=state[:,591:630]
        action=snap["previous_action"].view(1,1,29)
        q,qd=state[:,13:42],state[:,42:71]
        torque=snap["torques"].unsqueeze(0)
        previous_q=snap["last_dof_pos"].unsqueeze(0);previous_qd=snap["last_dof_vel"].unsqueeze(0)
        history=capture["snapshots"][index-1]["obs_history_buf"].unsqueeze(0)
        fake=SimpleNamespace(cfg={"env":{"history_len":10},"noise":{"add_noise":False}},
                             _rand_vec=lambda x,s:x, reindex=lambda x:x, num_envs=1,
                             obs_history_buf=history.clone(),episode_length_buf=snap["episode_length_buf"].view(1))
        expected=UltraDistillObjV2Point.compute_humanoid_observations_max(
            fake,pos,rot,vel,ang,False,True,contact,None,ref,None,None,None,action,q,qd,torque,
            previous_q,previous_qd,state[:,:13],torch.tensor([0]),student_obs=True,local_ref_obs=local)
        obs=StudentObs(env.model,env.object_name,env.max_episode_length,env.hoi_data,env.object_points,
                       env.object_corners,history_step=10,goal_phase_dim=4,obj_goal_decouple="off")
        obs.configure_task_mode("full_track","points");obs.obs_history_buf=history.clone()
        actual,_=obs.compute_humanoid_observations_max(pos,rot,vel,ang,False,True,contact,ref,ref16,
                  action,q,qd,torque,previous_q,previous_qd,state[:,:13],int(snap["episode_length_buf"]),
                  student_obs=True,local_ref_obs=local)
        delta=(actual-expected).abs()
        reports.append({"control":index,"compared_dimensions":actual.shape[-1],
                        "student_humanoid_observation_max_abs_error":delta.max().item(),
                        "student_history_max_abs_error":(obs.obs_history_buf-fake.obs_history_buf).abs().max().item()})
        assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6),reports[-1]
        assert torch.allclose(obs.obs_history_buf,fake.obs_history_buf,atol=2e-6,rtol=2e-6)
    # Each teacher reference horizon must see the same previous-step DOF state.
    obs=TeacherObs(env.model,env.object_name,env.max_episode_length,env.hoi_data,env.object_points,history_step=10)
    seen=[]
    def reference_probe(*args,**kwargs):
        seen.append((obs.last_dof_pos.clone(),obs.last_dof_vel.clone()))
        obs.last_dof_pos=torch.ones(1,29);obs.last_dof_vel=torch.ones(1,29)*2
        return torch.zeros(1,2026),{}
    obs._compute_observations_iter=reference_probe
    obs.last_dof_pos=torch.ones(1,29)*3;obs.last_dof_vel=torch.ones(1,29)*4
    obs._compute_observations(env.data,0,torch.zeros(1,29),torch.zeros(1,29))
    assert all(torch.equal(seen[0][i],seen[1][i]) for i in (0,1))
    result={"student_checks":reports,"teacher_two_horizons_share_previous_state":True,
            "scope":"Humanoid/global/local goal and history math only; point-cloud sampling/contact dynamics are excluded."}
    (output/"observation_equivalence.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
