"""Freeze 100 reference-qualified clips, disjoint from development motion families."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401
import torch


def family(path):
    return "_".join(Path(path).stem.split("_")[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Cohort is frozen: use a new explicit output for any new experiment"
    torch.set_num_threads(2)
    old = json.loads((ROOT / "output/reproduction/student_10000_benchmark/manifest.json").read_text())
    excluded = {family(row["motion"]) for row in old["motions"]}
    rng = random.Random(20261023)
    selected, inspected, eligible = [], 0, {}
    for kind in ("largebox", "plasticbox", "smallbox", "suitcase"):
        groups = {}
        for path in sorted((ROOT / "InterAct/OMOMO_retarget_aug").glob(f"*_{kind}_*_100_100_100.pt")):
            key = family(path)
            if key not in excluded:
                groups.setdefault(key, []).append(path)
        families = sorted(groups); rng.shuffle(families)
        rows = []
        for key in families:
            paths = groups[key]; rng.shuffle(paths)
            for path in paths:
                reference = torch.load(path, map_location="cpu"); inspected += 1
                frames = len(reference)
                if not 120 <= frames <= 900:
                    continue
                lift = float(reference[:, 73].max() - reference[0, 73])
                travel = float((reference[-1, 71:73] - reference[0, 71:73]).norm())
                height = float(reference[-1, 73] - reference[0, 73])
                if lift < .2 or travel < .5 or abs(height) > .12:
                    continue
                rows.append(dict(motion=str(path.relative_to(ROOT)), object=kind,
                                 asset_key=f"{kind}_100_100_100", raw_frames=frames,
                                 reference_lift_m=lift, reference_xy_displacement_m=travel,
                                 reference_final_height_change_m=height))
                break
        eligible[kind] = rows
    assert sum(map(len, eligible.values())) >= 100, {k: len(v) for k, v in eligible.items()}
    # Balance objects while retaining distinct original families. Some objects
    # have fewer than 25 reference-qualified families; never relax the filters.
    counts = {kind: 0 for kind in eligible}
    while len(selected) < 100:
        for kind, rows in eligible.items():
            if counts[kind] < len(rows) and len(selected) < 100:
                selected.append(rows[counts[kind]])
                counts[kind] += 1
    selected.sort(key=lambda row: (row["object"], row["motion"]))
    assert len({family(row["motion"]) for row in selected}) == 100
    assert not excluded & {family(row["motion"]) for row in selected}
    manifest = copy.deepcopy(old)
    manifest.update(motions=selected, selection_seed=20261023,
                    selection="100 distinct original motion families disjoint from development. Balanced round-robin across objects subject to eligible family availability. Reference geometry/length filters only; no policy outcomes inspected.",
                    limitations="Independent acceptance cohort for this experiment, not guaranteed unseen during historical distillation/RL. One deterministic rollout per case and setting.",
                    development_original_families=sorted(excluded), inspected_reference_clips=inspected,
                    cases_by_object=counts, eligible_families_by_object={k: len(v) for k, v in eligible.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"motions": len(selected), "original_families": 100,
                      "development_family_overlap": 0, "inspected_reference_clips": inspected, "cases_by_object": counts,
                      "path": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
