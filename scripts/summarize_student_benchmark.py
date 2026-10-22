#!/usr/bin/env python3
"""Recompute shared metrics from all saved trajectories, then summarize/plot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from student_benchmark_metrics import measure_trajectory

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/reproduction/student_10000_benchmark"


def main():
    global BASE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    BASE = args.base.resolve()
    manifest = json.loads((BASE / "manifest.json").read_text())
    results = []
    for engine in ("gym", "mujoco"):
        for row in manifest["motions"]:
            for mode in manifest["modes"]:
                directory = BASE / engine / row["object"] if engine == "gym" else BASE / engine
                path = directory / (Path(row["motion"]).stem + f"_{mode}.npz")
                assert path.is_file(), path
                data = np.load(path)
                assert len(data["states"]) == row["raw_frames"] - 1, path
                report = measure_trajectory(data["states"], data["reference"], data["hand_contact"],
                                            data["support_contact"], row["raw_frames"] - 1)
                report.update(engine=engine, object=row["object"], mode=mode, motion=row["motion"],
                              seed=manifest["rollout_seed"], checkpoint=manifest["checkpoint"],
                              trajectory=str(path.relative_to(ROOT)))
                results.append(report)
    assert len(results) == 80
    assert len({(r["engine"], r["motion"], r["mode"]) for r in results}) == 80
    assert all(r["all_finite"] for r in results)
    for row in manifest["motions"]:
        for mode in manifest["modes"]:
            name = Path(row["motion"]).stem + f"_{mode}.npz"
            gym_ref = np.load(BASE / "gym" / row["object"] / name)["reference"]
            mj_ref = np.load(BASE / "mujoco" / name)["reference"]
            assert np.array_equal(gym_ref, mj_ref), (row["motion"], mode)
    def summary(rows):
        return {"episodes": len(rows), "task_successes": sum(r["task_success"] for r in rows),
                "task_success_rate": float(np.mean([r["task_success"] for r in rows])),
                "strict_placement_successes": sum(r["strict_placement_success"] for r in rows),
                "carry_and_place_successes": sum(r["carry_and_place_success"] for r in rows),
                "falls": sum(r["fell"] for r in rows),
                "mean_final_box_error_m": float(np.mean([r["final_box_position_error_m"] for r in rows])),
                "median_final_box_error_m": float(np.median([r["final_box_position_error_m"] for r in rows])),
                "mean_box_trajectory_rmse_m": float(np.mean([r["box_trajectory_rmse_m"] for r in rows])),
                "mean_joint_trajectory_rmse_rad": float(np.mean([r["joint_trajectory_rmse_rad"] for r in rows]))}
    groups, by_object = {}, {}
    for engine in ("gym", "mujoco"):
        groups[engine] = {}; by_object[engine] = {}
        for mode in manifest["modes"]:
            groups[engine][mode] = summary([r for r in results if r["engine"] == engine and r["mode"] == mode])
        for kind in sorted({r["object"] for r in results}):
            by_object[engine][kind] = {mode: summary([r for r in results if r["engine"] == engine and
                                                     r["mode"] == mode and r["object"] == kind])
                                     for mode in manifest["modes"]}
    # Confirm all native and MuJoCo evaluations used the deploy prior exclusively.
    originals = [json.loads((BASE / "gym" / kind / "metrics.json").read_text())
                 for kind in sorted({r["object"] for r in results})]
    originals.append(json.loads((BASE / "mujoco/metrics.json").read_text()))
    assert all(r.get('point_fixed_grid_sampling') is True for r in originals[-1])
    import yaml
    assert all(yaml.safe_load((BASE / 'gym' / kind / 'evaluation.yaml').read_text())['env']['pointFixedGridSampling'] is True
               for kind in sorted({r['object'] for r in results}))
    assert sum(map(len, originals)) == 80
    assert all(r["privileged_encoder_calls"] == 0 for rows in originals for r in rows)
    if manifest.get("policy_architecture") == "v3_rl":
        assert all(r.get("policy_architecture") == "v3_rl" and r.get("strict_all_policy_weights_verified") is True
                   for rows in originals for r in rows)
        film_hashes = {r["rl_film_sha256"] for rows in originals for r in rows}
        assert len(film_hashes) == 1 and None not in film_hashes
    rollout_pca_fixed = all(r.get('point_pca_frame_policy') == 'training4096_reference_v1' for rows in originals for r in rows)
    alignment = json.loads((BASE / "mujoco/inertia_verification.json").read_text())
    assert len(alignment) == 4
    assert all(r["verified_equal_to_native_actor_export"] for r in alignment)
    for actual in alignment:
        native = json.loads((BASE / "gym" / actual["object"] / "object_inertia.json").read_text())
        for key in ("mass", "com", "inertia"):
            assert np.allclose(actual[key], native[key], atol=1e-8, rtol=0), (actual["object"], key)
    checkpoint = ROOT / manifest["checkpoint"]
    checkpoint_metadata = json.loads((BASE / "checkpoint_metadata.json").read_text())
    assert checkpoint_metadata['epoch'] == manifest.get('epoch', 10000)
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    output = {"manifest": manifest, "summary": groups, "by_object": by_object,
              "policy_architecture": manifest.get("policy_architecture", "v2_distill"),
              "rl_finetune_epochs": manifest.get("rl_finetune_epochs", 0),
              "checkpoint_metadata": checkpoint_metadata,
              "unique_original_motion_families": len({'_'.join(Path(r['motion']).stem.split('_')[:3]) for r in manifest['motions']}),
              "checkpoint_sha256": sha(checkpoint), "manifest_sha256": sha(BASE / "manifest.json"),
              "verified_deploy_prior_only": True, "inertia_verification": alignment,
              "point_fixed_grid_sampling_verified_both_engines": True,
              "point_sampling_correction": json.loads((BASE / 'point_sampling_correction.json').read_text()),
              "point_encoding_verification": json.loads((BASE / 'point_encoding_verification.json').read_text()),
              "rollouts_use_training_pca_frames": rollout_pca_fixed,
              "point_encoding_limitation": "Use the point encoding verification for current code status. Rollout rates without point_pca_frame_policy metadata predate the PCA frame correction and must not be presented as measured after that correction.",
              "strict_placement_contact_limitation": "Gym uses net hand forces and upward target force as release/support proxies; MuJoCo uses resolved object contact pairs. Compare the common no-fall+terminal-distance metric first.",
              "other_differences_retained": "friction, collision/solver response, body velocity conventions, goal/phase call order, reset first-frame state, RNG streams, original engine timestep settings. No additional sim gap fixes in this benchmark.",
              "source_sha256": {p:sha(ROOT / p) for p in ("scripts/student_benchmark_metrics.py",
                                    "scripts/benchmark_student_gym.py", "scripts/benchmark_student_mujoco.py")}}
    (BASE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (BASE / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with (BASE / "results.csv").open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    labels = ['Full tracking', 'Points + goal']
    x = np.arange(2)
    for engine, shift, color in [('gym', -.18, '#3579b5'), ('mujoco', .18, '#d6862e')]:
        a = [groups[engine][m] for m in manifest['modes']]
        axes[0].bar(x + shift, [r['task_success_rate'] * 100 for r in a], .34, label=engine, color=color)
        axes[1].bar(x + shift, [r['carry_and_place_successes'] / r['episodes'] * 100 for r in a], .34, label=engine, color=color)
        axes[2].bar(x + shift, [r['mean_box_trajectory_rmse_m'] for r in a], .34, label=engine, color=color)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set(title='Task success (n=20 per bar)', ylabel='Percent', ylim=(0, 105))
    axes[1].set(title='Lift, release and stable placement', ylabel='Percent', ylim=(0, 105))
    axes[2].set(title='Mean box trajectory RMSE', ylabel='Meters')
    axes[0].legend()
    label = (f"RL finetune +{manifest['rl_finetune_epochs']} (checkpoint epoch {checkpoint_metadata['epoch']})"
             if manifest.get("policy_architecture") == "v3_rl" else f"Student epoch {checkpoint_metadata['epoch']}")
    fig.suptitle(f"{label} | matched nominal object inertia | other sim differences retained")
    fig.savefig(BASE / 'comparison.png', dpi=180); plt.close(fig)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
