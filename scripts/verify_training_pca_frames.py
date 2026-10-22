#!/usr/bin/env python3
"""Check runtime point sampling against the unmodified training eigensolver."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ultra'))
import isaacgym  # noqa: F401
import mujoco
import numpy as np
import torch
import trimesh
from env.tasks.ultra_g1_distill_obj_v2vae import compute_obj_observations_pca_corners as native
from env.tasks.ultra_g1_distill_obj_v3rl import compute_obj_observations_pca_corners as rl_points
from sim2sim_vae import merge_mjcf, _set_freejoint_state, _write_hinge_qpos_qvel_by_names, JOINT_NAMES_29
from utils.obs_vae import MujocoObs, wxyz_to_xyzw
from utils.box_pca import box_pca_axes


def point_distance(a, b):
    a, b = a.reshape(-1, 3).double().cpu().numpy(), b.reshape(-1, 3).double().cpu().numpy()
    distances = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return float(max(distances.min(0).max(), distances.min(1).max()))


def main():
    os.chdir(ROOT); torch.set_num_threads(2)
    base = ROOT / 'output/reproduction/student_10000_benchmark'
    out = base / 'pca_fix'; out.mkdir(exist_ok=True)
    manifest = json.loads((base / 'manifest.json').read_text())
    registry = json.loads(Path('ultra/data/assets/objects/point_pca_frames.json').read_text())
    frame_checks = []
    # Compare every stored frame to fresh GPU solves, including training batches.
    for frame in registry['frames']:
        corners = torch.tensor(frame['corners'], dtype=torch.float32)
        runtime = box_pca_axes(corners)
        for batch in (1, 5, 4096):
            vertices = corners.cuda()[None].repeat(batch, 1, 1)
            centered = vertices - vertices.mean(1, keepdim=True)
            cov = centered.transpose(1, 2) @ centered / 8.
            _, raw = torch.linalg.eigh(cov)
            selected = box_pca_axes(vertices, cov)
            assert torch.equal(selected[0].cpu(), runtime[0])
            if batch > 1:
                assert torch.equal(selected, raw), (frame['asset'], batch)
            frame_checks.append({'asset': frame['asset'], 'batch': batch,
                                 'runtime_axes_equal_cpu': True,
                                 'raw_eigensolver_max_error': float((selected - raw).abs().max())})
    checks, rl_checks = [], []
    for kind in ('largebox', 'plasticbox', 'smallbox', 'suitcase'):
        path = f'/tmp/ultra_training_pca_{kind}.xml'
        merge_mjcf('ultra/data/assets/g1/g1_29dof.xml', f'ultra/data/assets/objects/{kind}.xml', path, viz_sites=0)
        model = mujoco.MjModel.from_xml_path(path); data = mujoco.MjData(model)
        mesh = trimesh.load(f'ultra/data/assets/objects/objects/{kind}/{kind}.obj', process=False, force='mesh')
        corners = torch.tensor(mesh.bounding_box_oriented.vertices - mesh.vertices.mean(0), dtype=torch.float32)
        gpu_vertices = corners.cuda()[None].repeat(5, 1, 1)
        centered = gpu_vertices - gpu_vertices.mean(1, keepdim=True)
        _, raw_batch_axes = torch.linalg.eigh(centered.transpose(1, 2) @ centered / 8.)
        obs = MujocoObs(model, kind, 1000, torch.zeros(1000, 747), torch.zeros(256, 3), corners,
                        history_step=10, point_fixed_grid_sampling=True)
        for row in (r for r in manifest['motions'] if r['object'] == kind):
            trajectory = np.load(base / 'gym' / kind / (Path(row['motion']).stem + '_full_track.npz'))
            ref = torch.tensor(trajectory['reference'][-1:], dtype=torch.float32)
            indices = sorted(set([0, 29, 59] + np.linspace(0, len(trajectory['states']) - 1, 10).astype(int).tolist()))
            for index in indices:
                state = torch.tensor(trajectory['states'][index], dtype=torch.float32)
                _set_freejoint_state(model, data, 'pelvis', state[:3], state[3:7], state[7:10], state[10:13])
                _write_hinge_qpos_qvel_by_names(model, data, JOINT_NAMES_29, state[13:42], state[42:71])
                _set_freejoint_state(model, data, kind + '_free', state[71:74], state[74:78], state[78:81], state[81:84])
                mujoco.mj_forward(model, data)
                camera = model.body('d435_link').id
                pos = torch.tensor(data.xpos[camera][None], dtype=torch.float32)
                rot = torch.tensor(wxyz_to_xyzw(data.xquat[camera])[None], dtype=torch.float32)
                args = (state[None, :13], state[None, 71:84], corners[None], ref, pos, rot)
                # Reference axes are freshly solved with the original GPU batch
                # algorithm, independent of the exported frame registry.
                reference, rp = native(*(a.cuda().repeat(5, *([1] * (a.ndim - 1))) for a in args),
                                       pca_axes=raw_batch_axes, fixed_surface_sampling=True, disable_geometry_noise=True)
                actual, ap, _ = obs.compute_obj_observations_pca_corners(*args, fixed_surface_sampling=True, disable_geometry_noise=True)
                single, sp = native(*(a.cuda() for a in args), fixed_surface_sampling=True, disable_geometry_noise=True)
                record = {'object': kind, 'motion': row['motion'], 'control': index + 1,
                          'cpu_world_point_set_error_m': point_distance(ap, rp[:1]),
                          'cpu_camera_point_set_error_m': point_distance(actual[:, 12:], reference[:1, 12:]),
                          'single_env_gpu_world_point_set_error_m': point_distance(sp, rp[:1]),
                          'single_env_gpu_camera_point_set_error_m': point_distance(single[:, 12:], reference[:1, 12:])}
                checks.append(record)
                if index in (0, 29, 59):
                    _, rl_world = rl_points(*(a.cuda() for a in args), fixed_surface_sampling=True, disable_geometry_noise=True)
                    rl_checks.append({'object': kind, 'control': index + 1,
                                      'world_point_set_error_m': point_distance(rl_world, rp[:1])})
    # A changed mesh must not silently receive a named object's saved frame.
    unknown = corners.clone(); unknown[0, 0] += .001
    centered = unknown[None] - unknown[None].mean(1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / 8.
    _, fallback = np.linalg.eigh(cov.numpy())
    assert np.allclose(box_pca_axes(unknown).numpy(), fallback)
    maxima = {field: max(r[field] for r in checks) for field in checks[0] if field.endswith('_m')}
    result = {'frame_checks': frame_checks, 'point_checks': checks, 'rl_point_checks': rl_checks, 'maxima': maxima,
              'pca_fix_applied': True,
              'all_point_sets_match_within_2e_5_m': all(value < 2e-5 for value in maxima.values()),
              'unknown_geometry_retains_original_pca': True,
              'scope': 'Eight 080/100 geometry PCA frames checked at batch sizes 1, 5, 4096. Point math checked at poses from all 20 benchmark motions for the four supported 100 assets. Raw 5-env CUDA eigenvectors match 4096-env training exactly and serve as the reference. No physics success-rate claims.'}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'point_checks': len(checks), 'maxima': maxima}, indent=2))
    assert all(value < 2e-5 for value in maxima.values()), maxima
    assert all(r['world_point_set_error_m'] < 2e-5 for r in rl_checks)
    (base / 'point_encoding_verification.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
