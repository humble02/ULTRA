#!/usr/bin/env python3
"""Native Gym benchmark: five motions sharing one collision asset per process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "ultra"), str(ROOT / "scripts")]
import isaacgym  # noqa: F401
import numpy as np
import torch
import yaml
from learning import ultra_players_distill
from student_benchmark_metrics import measure_trajectory
from student_benchmark_policy import verify_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, choices=["largebox", "plasticbox", "smallbox", "suitcase"])
    parser.add_argument("--base", type=Path, default=ROOT / "output/reproduction/student_10000_benchmark")
    args = parser.parse_args(); os.chdir(ROOT)
    base = args.base.resolve()
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest.get("policy_architecture") == "v3_rl":
        # Keep the original V2 benchmark task/physics/goal scheduler; only the
        # policy builder changes to load the complete RL network and FiLM.
        from learning import ultra_network_builder_obj_v2, ultra_network_builder_obj_v3
        ultra_network_builder_obj_v2.UltraBuilder = ultra_network_builder_obj_v3.UltraBuilder
    pca_sha = hashlib.sha256((ROOT / 'ultra/data/assets/objects/point_pca_frames.json').read_bytes()).hexdigest()
    rows = [r for r in manifest["motions"] if r["object"] == args.object]
    n = len(rows)
    out = base / "gym" / args.object; out.mkdir(parents=True, exist_ok=True)
    sources = [torch.load(ROOT / r["motion"], map_location="cpu") for r in rows]
    lengths = [len(s) - 1 for s in sources]
    config = yaml.safe_load((ROOT / "ultra/data/cfg/g1_student_vae.yaml").read_text())
    clean = yaml.safe_load((ROOT / "ultra/data/cfg/g1_teacher_no_dr.yaml").read_text())
    config["domain_rand"] = clean["domain_rand"]; config["noise"] = clean["noise"]
    config["env"].pop("dataSub", None)
    config["env"].update(numEnvs=n, stateInit="Start", enableEarlyTermination=False, enableDebugVis=False,
                          obs_mask_noise_std=0., envSpacing=8.)
    for k in ("pointNoiseStd", "pointDropoutProb", "pointOutlierProb", "pointDepthNoiseScale", "pointClusterNoiseStd",
              "pointTranslationNoise", "pointOcclusionProb", "cameraRotNoise", "cameraPosNoise"):
        config["env"][k] = 0.
    for k in ("pointDensityMin", "pointDensityMax", "pointScaleMin", "pointScaleMax"):
        config["env"][k] = 1.
    config["sim"]["physx"].update(max_gpu_contact_pairs=131072, default_buffer_size_multiplier=2.)
    cfg = out / "evaluation.yaml"; cfg.write_text(yaml.safe_dump(config))

    def run(player):
        task = player.env.task
        policy_identity = verify_policy(player.model, manifest)
        assert [Path(p).name for p in task.motion_file] == [Path(r["motion"]).name for r in rows]
        assert len(task.object_name) == 1, task.object_name
        # Export the real actor inertial properties, including PhysX recomputation.
        def vec(v):
            return [float(v.x), float(v.y), float(v.z)]
        prop_list = []
        for eid in range(n):
            props = task.gym.get_actor_rigid_body_properties(task.envs[eid], task._target_handles[eid])
            assert len(props) == 1
            p = props[0]
            prop_list.append({"asset_key": rows[eid]["asset_key"], "mass": float(p.mass), "com": vec(p.com),
                              "inertia": [vec(p.inertia.x), vec(p.inertia.y), vec(p.inertia.z)]})
        for p in prop_list[1:]:
            assert np.allclose(p["inertia"], prop_list[0]["inertia"], atol=1e-9)
            assert np.allclose(p["com"], prop_list[0]["com"], atol=1e-9)
            assert abs(p["mass"] - prop_list[0]["mass"]) < 1e-9
        (out / "object_inertia.json").write_text(json.dumps(prop_list[0], indent=2) + "\n")
        task.rollout_length = max(lengths) + 100
        task._enable_early_termination = False
        encoder_calls = [0]
        def reject(module, inputs):
            encoder_calls[0] += 1
            raise RuntimeError("Privileged encoder must not run during deploy evaluation")
        hook = player.model.a2c_network.encoder_net.register_forward_pre_hook(reject)
        reports = []
        try:
            for mode in manifest["modes"]:
                for name in ("actions", "last_actions", "last_dof_pos", "last_dof_vel", "torques", "action_history_buf",
                             "obs_history_buf", "episode_length_buf", "last_contact_forces"):
                    value = getattr(task, name, None)
                    if isinstance(value, torch.Tensor):
                        value.zero_()
                task.configure_task_mode(mode, "points")
                torch.manual_seed(manifest["rollout_seed"]); np.random.seed(manifest["rollout_seed"])
                obs = player.env_reset()
                assert task.data_id.cpu().tolist() == list(range(n))
                assert not task.is_stand_still.any()
                player.get_batch_size(obs["obs"], n)
                histories = [[] for _ in rows]; contacts = [[] for _ in rows]; supports = [[] for _ in rows]
                hand_ids = [28, 38]
                for step in range(max(lengths)):
                    player._last_expert = None
                    player._vae_noise = torch.zeros(n, 64, device=player.device)
                    action = player.get_action(obs, True)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite policy action")
                    next_obs, _, done, _ = player.env_step(player.env, action)
                    obs = next_obs if isinstance(next_obs, dict) else {"obs": next_obs}
                    state = task._curr_obs.detach().cpu().numpy()
                    hand_force = task._contact_forces[:, hand_ids].detach().cpu().numpy()
                    tar_force = task._tar_contact_forces.detach().cpu().numpy()
                    for eid, limit in enumerate(lengths):
                        if step >= limit:
                            continue
                        assert int(task.data_id[eid]) == eid
                        histories[eid].append(state[eid, :84].copy())
                        # Net hand forces proxy hand-box contact; in this ground-only scene,
                        # target upward force with released hands proxies ground support.
                        hand = bool(np.any(np.abs(hand_force[eid]) > .1))
                        contacts[eid].append(hand)
                        supports[eid].append(bool(tar_force[eid, 2] > .1 and not hand))
                    if (step + 1) % 100 == 0:
                        print(args.object, mode, step + 1, max(lengths), flush=True)
                for eid, row in enumerate(rows):
                    actual = np.asarray(histories[eid])
                    reference = sources[eid][1:, :84].numpy()
                    result = measure_trajectory(actual, reference, contacts[eid], supports[eid], lengths[eid])
                    result.update(engine="gym", mode=mode, motion=row["motion"], object=args.object,
                                  **policy_identity,
                                  checkpoint=manifest["checkpoint"], seed=manifest["rollout_seed"],
                                  privileged_encoder_calls=encoder_calls[0],
                                  point_pca_frame_policy='training4096_reference_v1', point_pca_frame_registry_sha256=pca_sha,
                                  contact_measurement="hand net force >.1 N; target upward net contact force when hands released")
                    reports.append(result)
                    np.savez_compressed(out / (Path(row["motion"]).stem + f"_{mode}.npz"),
                                        states=actual, reference=reference, hand_contact=contacts[eid], support_contact=supports[eid])
                    print(json.dumps(result), flush=True)
                (out / "metrics.json").write_text(json.dumps(reports, indent=2) + "\n")
        finally:
            hook.remove()

    ultra_players_distill.UltraPlayerContinuousDistill.run = run
    sys.argv = [str(ROOT / "ultra/run_distill.py"), "--task", "UltraDistillObjV2Point", "--cfg_env", str(cfg),
                "--cfg_train", manifest.get("policy_training_config", "ultra/data/cfg/train/rlg/g1_student_vae.yaml"), "--test", "--num_envs", str(n),
                "--seed", str(manifest["rollout_seed"]), "--checkpoint", str(ROOT / manifest["checkpoint"]),
                "--motion_file", str(base / "motions" / args.object), "--task_mode", "full_track", "--obj_obs", "points",
                "--output_path", str(out / "rlgames"), "--headless"]
    runpy.run_path(str(ROOT / "ultra/run_distill.py"), run_name="__main__")


if __name__ == "__main__":
    main()
