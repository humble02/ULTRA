"""Run a registered candidate, independent acceptance and seed replication.

Long training is reachable only after both short runs and both independent
acceptance comparisons pass. Every long window retains the 200-update barriers
and the 1000-update improvement requirement. Failures end this experiment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from rl_experiment_gate import assess, validate_report
from run_student_benchmark import sha, source_hashes, write_json

ROOT = Path(__file__).resolve().parents[1]


class Experiment:
    def __init__(self, run, unit, long_windows):
        self.run, self.unit, self.long_windows = run, unit, long_windows
        self.request = json.loads((run / "training_request.json").read_text())
        self.expected = json.loads((run / "source_sha256.json").read_text())
        self.stop = False
        self.state = {"pid": os.getpid(), "initial_run": str(run), "started_at": self.now(),
                      "long_training_allowed": False, "events": []}
        signal.signal(signal.SIGTERM, lambda *args: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *args: setattr(self, "stop", True))

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def status(self, phase, **fields):
        if phase.endswith("rejected") or phase in ("stopped_by_user", "failed"):
            self.state["long_training_allowed"] = False
        self.state.update(state=phase, updated_at=self.now(), **fields)
        write_json(self.run / "experiment_status.json", self.state)

    def check_sources(self):
        assert source_hashes() == self.expected, "Code changed since experiment registration"

    def command(self, command, log, environment=None, timeout=43200):
        self.check_sources()
        assert not self.stop, "Stop requested"
        env = dict(os.environ); env.update(environment or {})
        started, signalled = time.monotonic(), False
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            self.status(self.state["state"], active_child_pid=process.pid, active_log=str(log))
            while process.poll() is None:
                if self.stop and not signalled:
                    os.killpg(process.pid, signal.SIGTERM)
                    signalled = True
                if time.monotonic() - started > timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=300)
                    raise TimeoutError(f"Stage exceeded time limit: {log}")
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.stop:
            self.status("stopped_by_user")
            raise SystemExit(0)
        assert process.returncode == 0, f"Stage failed ({process.returncode}): {log}"
        self.check_sources()

    def point_current_run(self, run):
        request = json.loads((run / "training_request.json").read_text())
        write_json(ROOT / "output/reproduction/current_training.json", {
            "stage": "gated_rl_experiment", "candidate_profile": Path(request["experiment_profile"]).name,
            "run_dir": str(run.relative_to(ROOT)),
            "training_status": str((run / "memory_status.json").relative_to(ROOT)),
            "evaluation_status": str((run / "evaluation_status.json").relative_to(ROOT)),
            "experiment_status": str((self.run / "experiment_status.json").relative_to(ROOT)),
            "output_path": str(Path(request["output_path"]).relative_to(ROOT)),
            "initial_student_epoch": request["initial_student_epoch"],
            "initial_checkpoint_epoch": request["initial_checkpoint_epoch"],
            "training": self.unit, "periodic_evaluation": self.unit,
            "evaluation_mode": "synchronous_training_barrier", "periodic_evaluation_interval": 200,
            "source_commit": request["source_commit"], "started_at": self.now()})
        write_json(run / "services.json", {"experiment": self.unit, "training": self.unit,
                                           "evaluation": "synchronous child of training"})

    def train(self, run, phase):
        self.point_current_run(run)
        self.status(phase, active_run=str(run))
        launch = json.loads((run / "launch.json").read_text())
        self.command(launch["command"], run / "training.log", launch["environment"])
        result = json.loads((run / "memory_status.json").read_text())
        self.state["events"].append({"run": str(run), "phase": phase, "result": result["state"],
                                     "epoch": result.get("epoch"), "at": self.now()})
        if result["state"] not in ("short_run_passed_pending_acceptance", "completed_long_window"):
            self.status("candidate_rejected", rejected_run=str(run), training_result=result)
            return None
        return Path(result["checkpoint"])

    def acceptance(self, checkpoint, directory, restart_epoch):
        assert not directory.exists()
        directory.mkdir(parents=True)
        # Metadata is loaded on CPU; this driver never creates a CUDA context.
        import torch
        torch.set_num_threads(2)
        saved = torch.load(checkpoint, map_location="cpu"); epoch = int(saved["epoch"]); del saved
        self.status("expanded_acceptance", acceptance_checkpoint=str(checkpoint))
        self.command([sys.executable, str(ROOT / "scripts/run_student_benchmark.py"),
                      "--epoch", str(epoch), "--checkpoint", str(checkpoint), "--base", str(directory),
                      "--policy-architecture", "v3_rl", "--initial-student-epoch", "6000",
                      "--finetune-restart-epoch", str(restart_epoch),
                      "--cohort-manifest", self.request["acceptance_cohort_manifest"],
                      "--training-log", str(self.run / "training.log")], directory / "runner.log", timeout=21600)
        return validate_report(directory, checkpoint, epoch, expected_motions=100)

    def prepare(self, label, checkpoint, seed, config_dir):
        run = ROOT / "output/reproduction" / label
        self.status("preparing_next_stage", next_run=str(run))
        self.command([sys.executable, str(ROOT / "scripts/prepare_rl_continuation.py"), "--label", label,
                      "--checkpoint", str(checkpoint), "--config-dir", str(config_dir), "--seed", str(seed)],
                     self.run / f"prepare_{label}.log")
        return run

    def run_all(self):
        main_checkpoint = self.train(self.run, "seed42_short_training")
        if main_checkpoint is None:
            return
        source = Path(self.request["initial_checkpoint"])
        reference = self.acceptance(source, self.run / "acceptance/baseline", 6500)
        candidate = self.acceptance(main_checkpoint, self.run / "acceptance/seed42", 6500)
        first_decision = assess(candidate, reference, reference, final=True)
        write_json(self.run / "acceptance/seed42_decision.json", first_decision)
        if not first_decision["passed"]:
            self.status("acceptance_rejected", decision=first_decision)
            return
        profile = Path(self.request["experiment_profile"])
        replicate = self.prepare(self.run.name + "_seed43", source, 43, profile)
        replicate_checkpoint = self.train(replicate, "seed43_replication")
        if replicate_checkpoint is None:
            return
        repeated = self.acceptance(replicate_checkpoint, self.run / "acceptance/seed43", 6500)
        second_decision = assess(repeated, reference, reference, final=True)
        write_json(self.run / "acceptance/seed43_decision.json", second_decision)
        if not second_decision["passed"]:
            self.status("replication_acceptance_rejected", decision=second_decision)
            return
        eligibility = {"passed": True, "training_seeds": [42, 43], "independent_cases_per_setting": 100,
                       "seed42_decision": first_decision, "seed43_decision": second_decision,
                       "main_checkpoint_selected_by_preregistration": str(main_checkpoint),
                       "long_window_limit": self.long_windows, "at": self.now()}
        write_json(self.run / "long_training_eligibility.json", eligibility)
        self.state["long_training_allowed"] = True
        previous_run = self.run
        for window in range(1, self.long_windows + 1):
            # New windows inherit full model/Adam/normalizer/curriculum state.
            # Simulator episodes reset once per window, never at the 200-update gates.
            config = self.run / f"long_window_{window:02d}_config"
            shutil.copytree(profile, config)
            gate = json.loads((config / "gate.json").read_text())
            epoch = int(main_checkpoint.stem.split("_")[-1])
            gate.update(stage="long_window", historical_baseline_report=str(
                (previous_run / f"benchmarks/epoch_{epoch}/attempt_1/summary.json").relative_to(ROOT)))
            write_json(config / "gate.json", gate)
            next_run = self.prepare(self.run.name + f"_long{window:02d}", main_checkpoint, 42, config)
            main_checkpoint = self.train(next_run, f"long_window_{window}")
            if main_checkpoint is None:
                return
            previous_run = next_run
        self.status("completed_registered_long_windows", final_checkpoint=str(main_checkpoint))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--long-windows", type=int, default=10)
    parser.add_argument("--next-profile", type=Path)
    parser.add_argument("--next-label")
    args = parser.parse_args()
    assert 0 <= args.long_windows <= 10
    experiment = Experiment(args.run_dir.resolve(), args.unit, args.long_windows)
    try:
        experiment.run_all()
        if args.next_profile is not None and experiment.state["state"].endswith("rejected") and not experiment.stop:
            assert args.next_label
            previous = dict(experiment.state)
            next_run = experiment.prepare(args.next_label, Path(experiment.request["initial_checkpoint"]),
                                          42, args.next_profile.resolve())
            experiment.status("candidate_rejected_next_candidate_prepared", previous_result=previous,
                              next_run=str(next_run))
            experiment = Experiment(next_run, args.unit, args.long_windows)
            experiment.state["previous_candidate"] = previous
            experiment.run_all()
    except Exception as error:
        experiment.status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
