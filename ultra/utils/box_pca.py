"""Checkpoint-compatible PCA frames shared by Gym and MuJoCo point sampling."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch


@lru_cache(maxsize=1)
def _frames():
    path = Path(__file__).resolve().parents[1] / 'data/assets/objects/point_pca_frames.json'
    document = json.loads(path.read_text())
    if document.get('version') != 1:
        raise ValueError(f'Unsupported PCA frame registry: {path}')
    frames = document['frames']
    corners = np.asarray([f['corners'] for f in frames], dtype=np.float32)
    axes = np.asarray([f['axes'] for f in frames], dtype=np.float32)
    values = np.asarray([f['eigenvalues'] for f in frames], dtype=np.float32)
    centered = corners - corners.mean(1, keepdims=True)
    cov = centered.transpose(0, 2, 1) @ centered / corners.shape[1]
    if not np.allclose(axes.transpose(0, 2, 1) @ axes, np.eye(3), atol=2e-6):
        raise ValueError('Invalid training PCA axes: not orthonormal')
    if not np.allclose(cov @ axes, axes * values[:, None, :], atol=2e-7):
        raise ValueError('Training PCA axes do not match the stored geometry')
    return torch.from_numpy(corners), torch.from_numpy(axes)


@lru_cache(maxsize=16)
def _device_frames(device, dtype):
    corners, axes = _frames()
    return corners.to(device=device, dtype=dtype), axes.to(device=device, dtype=dtype)


def box_pca_axes(corners, cov=None):
    """Keep the training basis for registered geometry; preserve PCA for new assets.

    Eigenvector signs change with CPU/GPU backend and GPU batch size. A new
    canonical sign rule would change the sampled faces seen by old weights.
    Match the exact float32 OBB corners instead and reuse their training basis.
    Unknown geometries retain the original eigensolver behavior.
    """
    if corners.ndim == 2:
        corners = corners.unsqueeze(0)
    known_corners, known_axes = _device_frames(str(corners.device), corners.dtype)
    matches = (corners[:, None] == known_corners[None]).all(dim=-1).all(dim=-1)
    found = matches.any(dim=-1)
    selected = known_axes[matches.to(torch.int64).argmax(dim=-1)]
    if bool(found.all()):
        return selected
    if cov is None:
        centered = corners - corners.mean(dim=1, keepdim=True)
        cov = centered.transpose(1, 2) @ centered / float(corners.shape[1])
    if cov.device.type == 'cpu' and not getattr(torch._C, 'has_lapack', True):
        _, axes = np.linalg.eigh(cov.detach().numpy())
        fallback = torch.from_numpy(axes).to(dtype=cov.dtype)
    else:
        _, fallback = torch.linalg.eigh(cov)
    return torch.where(found[:, None, None], selected, fallback)
