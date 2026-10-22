#!/usr/bin/env python3
"""Choose a fixed, outcome-independent set of supported pick-and-place motions."""
import json
from pathlib import Path
import random
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/reproduction/student_10000_benchmark"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261019)
    selected = []
    for kind in ("largebox", "plasticbox", "smallbox", "suitcase"):
        candidates = sorted((ROOT / "InterAct/OMOMO_retarget_aug").glob(f"*_{kind}_*_100_100_100.pt"))
        rng.shuffle(candidates)
        picked = []
        for path in candidates:
            data = torch.load(path, map_location="cpu")
            lift = float(data[:, 73].max() - data[0, 73])
            travel = float(torch.linalg.vector_norm(data[-1, 71:73] - data[0, 71:73]))
            height_change = float(data[-1, 73] - data[0, 73])
            if len(data) < 120 or len(data) > 900 or lift < .2 or travel < .5 or abs(height_change) > .12:
                continue
            row = {"motion": str(path.relative_to(ROOT)), "object": kind,
                   "asset_key": f"{kind}_100_100_100", "raw_frames": len(data),
                   "reference_lift_m": lift, "reference_xy_displacement_m": travel,
                   "reference_final_height_change_m": height_change}
            picked.append(row)
            directory = OUT / "motions" / kind
            directory.mkdir(parents=True, exist_ok=True)
            link = directory / path.name
            if not link.exists():
                link.symlink_to(path)
            if len(picked) == 5:
                break
        assert len(picked) == 5, (kind, len(picked))
        selected.extend(sorted(picked, key=lambda row: row["motion"]))
    manifest = {"selection_seed": 20261019, "rollout_seed": 42,
                "checkpoint": "output/student_prior_fixed/g1_student_vae/nn/g1_student_vae_00010000.pth",
                "modes": ["full_track", "object_obs"], "object_input": "points", "latent_noise": "zero",
                "selection": "Five motions per object, unscaled 100 assets, 120..900 frames, reference lift >=.2 m, terminal XY travel >=.5 m, absolute terminal height change <=.12 m. Chosen before observing policy outcomes.",
                "task_success": "finite complete rollout, pelvis never below .25 m, terminal object position 3D distance <.3 m",
                "strict_placement": "task success plus final 15 controls all within .3 m and vertical error <.1 m, object speed <.2 m/s, hand contact force absent and object-ground support present throughout final 15 controls",
                "limitations": "Small in-distribution augmented training-dataset sample, one seed per motion. Other known sim differences retained. Not a held-out paper benchmark or an estimate for 080-size objects.",
                "motions": selected}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
