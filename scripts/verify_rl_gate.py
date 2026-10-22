"""Regression tests for deployment masks and evaluation barriers (no simulation)."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra"))
import isaacgym  # noqa: F401
import torch
from env.tasks.ultra_g1_distill_obj_v3rl import UltraDistillObjV3RL
from rl_experiment_gate import EvaluationGate, assess, validate_report
from run_gated_rl_experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    names = ("keep_obj_point_mask", "keep_obj_trans_mask", "keep_obj_rot_mask", "keep_obj_pos_mask",
             "keep_human_move_mask", "keep_global_goal_mask", "keep_local_goal_mask", "keep_goal_mask")
    task = SimpleNamespace(num_envs=4096, distill_full_track_fraction=.25, distill_point_goal_fraction=.25)
    for name in names:
        setattr(task, name, torch.rand(4096) > .5)
    before = {name: getattr(task, name).clone() for name in names}
    UltraDistillObjV3RL._apply_distillation_modes(task, torch.arange(4096))
    for name in names:
        v = getattr(task, name)
        assert v[:256].all()
        expected = name in ("keep_obj_point_mask", "keep_obj_trans_mask", "keep_human_move_mask", "keep_goal_mask")
        assert (v[256:512] == expected).all()
        assert torch.equal(v[512:], before[name][512:]), "Random distillation/PPO masks changed"
    ids = torch.tensor([4095, 3, 256, 900, 12, 400, 1025])
    for name in names:
        setattr(task, name, before[name].clone())
    UltraDistillObjV3RL._apply_distillation_modes(task, ids)
    outside = torch.ones(4096, dtype=torch.bool); outside[ids] = False
    assert all(torch.equal(getattr(task, name)[outside], before[name][outside]) for name in names)
    task.distill_full_track_fraction = task.distill_point_goal_fraction = 0
    before_zero = {name: getattr(task, name).clone() for name in names}
    UltraDistillObjV3RL._apply_distillation_modes(task, torch.arange(4096))
    assert all(torch.equal(getattr(task, name), before_zero[name]) for name in names)

    def read(path):
        return json.loads((ROOT / path).read_text())
    base_dir = ROOT / "output/reproduction/finetune_fixed_6000_20261022/benchmarks/epoch_6500/attempt_1"
    baseline = read(str(base_dir / "summary.json"))
    original_checkpoint = ROOT / baseline["manifest"]["checkpoint"]
    validate_report(base_dir, original_checkpoint, 6500)
    historical_failures = {}
    for epoch in (6700, 7000, 7500, 8000):
        report = read(f"output/reproduction/finetune_resetfix_from_rl500_20261022/benchmarks/epoch_{epoch}/attempt_1/summary.json")
        decision = assess(report, baseline, baseline)
        assert not decision["passed"]
        historical_failures[str(epoch)] = decision["failures"]
    assert assess(baseline, baseline, baseline)["passed"]
    assert not assess(baseline, baseline, baseline, final=True)["passed"], "Plateau cannot pass 1000-update gate"
    improved = copy.deepcopy(baseline)
    for engine in ("gym", "mujoco"):
        group = improved["summary"][engine]["object_obs"]
        group["task_successes"] += 2
        group["carry_and_place_successes"] += 2
    assert assess(improved, baseline, baseline, final=True)["passed"]
    assert not assess(baseline, baseline, improved)["passed"], "Regression from a better previous checkpoint was missed"
    assert assess(improved, baseline, baseline, final=True)["long_training_allowed"] is False

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        saved, evaluated, statuses = [], [], []
        gate = EvaluationGate.__new__(EvaluationGate)
        gate.run, gate.start, gate.interval, gate.limit = directory, 6500, 200, 1000
        gate.config = {"required_final_gain": .1}
        gate.baseline = gate.previous = baseline
        gate.decisions = []
        failure = read("output/reproduction/finetune_resetfix_from_rl500_20261022/benchmarks/epoch_6700/attempt_1/summary.json")
        def evaluate(agent, checkpoint):
            evaluated.append(agent.epoch_num)
            return failure
        gate.evaluate = evaluate
        gate.status = lambda agent, state, checkpoint, **details: statuses.append(state)
        agent = SimpleNamespace(epoch_num=6699, nn_dir=str(directory), config={"name": "test"},
                                writer=SimpleNamespace(flush=lambda: None), save=lambda path: saved.append(path))
        gate.after_update(agent)
        assert not saved and not evaluated
        agent.epoch_num = 6700
        try:
            gate.after_update(agent)
            raise AssertionError("A failed checkpoint returned to training")
        except SystemExit as error:
            assert error.code == 0
        assert evaluated == [6700] and len(saved) == 1
        assert statuses == ["stopped_by_evaluation_gate"]
        # Acceptance at 1000 is still a stop, never a long-training launch.
        gate.decisions = []; gate.previous = improved
        gate.evaluate = lambda agent, checkpoint: improved
        agent.epoch_num = 7500
        try:
            gate.after_update(agent)
            raise AssertionError("1000-update cap failed")
        except SystemExit as error:
            assert error.code == 0
        assert statuses[-1] == "short_run_passed_pending_acceptance"

    # Exercise the real orchestration with stubbed workloads: acceptance and
    # replication must both pass before long training can become eligible.
    pipeline_checks = {}
    for scenario in ("short_failure", "acceptance_failure", "replication_failure", "all_pass"):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); (path / "acceptance").mkdir()
            experiment = Experiment.__new__(Experiment)
            experiment.run, experiment.long_windows = path, 0
            experiment.request = {"initial_checkpoint": str(original_checkpoint),
                                  "experiment_profile": str(ROOT / "experiments/local_5090/gated_mode_coverage")}
            experiment.state = {"long_training_allowed": False}
            calls = []
            def train(run, phase):
                calls.append(phase)
                if scenario == "short_failure" or (scenario == "replication_failure" and phase == "seed43_replication"):
                    return None
                return path / "g1_student_finetune_00007500.pth"
            def acceptance(checkpoint, directory, restart):
                calls.append(directory.name)
                return baseline if directory.name == "baseline" or scenario == "acceptance_failure" else improved
            experiment.train, experiment.acceptance = train, acceptance
            experiment.prepare = lambda *args: path / "replication"
            experiment.run_all()
            eligible = (path / "long_training_eligibility.json").exists()
            assert eligible == (scenario == "all_pass")
            assert experiment.state["long_training_allowed"] == eligible
            if scenario in ("short_failure", "acceptance_failure"):
                assert "seed43_replication" not in calls
            pipeline_checks[scenario] = {"expected_eligibility": eligible, "stages": calls}

    result = {"passed": True, "full_track_envs": 256, "point_goal_envs": 256,
              "random_distillation_envs_unchanged": 512, "PPO_envs_unchanged": 3072,
              "noncontiguous_and_disabled_mode_checks": True,
              "historical_regressions_rejected": historical_failures,
              "checkpoint_at_exact_200_boundary": True, "failed_gate_exits_training": True,
              "flat_1000_update_run_rejected": True, "successful_short_run_stops_for_acceptance": True,
              "acceptance_and_replication_pipeline_tests": pipeline_checks,
              "synthetic_success_cases_are_gate_tests_not_measured_policy_improvements": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "historical_regressions_rejected"}, indent=2))


if __name__ == "__main__":
    main()
