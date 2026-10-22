#!/usr/bin/env python3
"""Export the original 4096-env Gym GPU PCA frames for known object geometries.

The sampler truncates six face grids to 512 points, so the eigenvector signs
are part of the observation distribution seen by existing checkpoints.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert torch.cuda.is_available(), 'Export once on the training GPU; inference uses the saved CPU-readable data.'
    paths = sorted((ROOT / 'ultra/data/assets/objects/diverse').glob('*/*.obj'))
    rows, corners = [], []
    for path in paths:
        mesh = trimesh.load(path, process=False, force='mesh')
        points = np.asarray(mesh.bounding_box_oriented.vertices - mesh.vertices.mean(0), dtype=np.float32)
        rows.append({'asset': path.stem, 'corners': points.tolist(),
                     'mesh_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        corners.append(points)
    n = len(rows)
    batch = 4096
    ids = torch.arange(batch, device='cuda') % n
    vertices = torch.tensor(np.asarray(corners), device='cuda')[ids]
    centered = vertices - vertices.mean(1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / float(vertices.shape[1])
    values, axes = torch.linalg.eigh(cov)
    for i, row in enumerate(rows):
        assert torch.allclose(axes[ids == i], axes[i].expand_as(axes[ids == i]), atol=0, rtol=0)
        row['eigenvalues'] = values[i].cpu().tolist()
        row['axes'] = axes[i].cpu().tolist()
    result = {'version': 1, 'reference_batch_size': batch, 'torch_version': torch.__version__,
              'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
              'note': 'Columns are the original Gym training eigenvectors, including signs. Geometry match is required; do not choose frames by object name alone.',
              'frames': rows}
    out = ROOT / 'ultra/data/assets/objects/point_pca_frames.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('Exported', n, 'training PCA frames to', out)


if __name__ == '__main__':
    main()
