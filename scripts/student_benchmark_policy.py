"""Strict policy identity checks shared by Gym and MuJoCo benchmarks."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_policy(policy, manifest):
    import torch
    saved = torch.load(ROOT / manifest["checkpoint"], map_location="cpu")["model"]
    current = policy.state_dict()
    assert current.keys() == saved.keys(), "Policy architecture/state keys differ from checkpoint"
    for key, value in saved.items():
        assert torch.equal(current[key].detach().cpu(), value), key
    architecture = manifest.get("policy_architecture", "v2_distill")
    has_film = hasattr(policy.a2c_network, "rl_finetune_film")
    assert has_film == (architecture == "v3_rl"), (architecture, has_film)
    film = hashlib.sha256()
    for key in sorted(saved):
        if ".rl_finetune_film." in key:
            film.update(key.encode())
            film.update(saved[key].detach().cpu().numpy().tobytes())
    return {"policy_architecture": architecture, "strict_all_policy_weights_verified": True,
            "rl_film_sha256": film.hexdigest() if has_film else None,
            "rl_finetune_epochs": manifest.get("rl_finetune_epochs", 0)}
