"""Synchronous, fail-closed evaluation barriers for short RL experiments.

The simulator, optimizer, rollout state and RNG stay in the training process.
Only separate benchmark processes run while this hook waits. No next PPO
update can start until the benchmark and its acceptance checks have passed.
"""
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from run_student_benchmark import source_hashes, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("task_successes", "carry_and_place_successes")


def now():
    return datetime.now(timezone.utc).isoformat()


def assess(report, baseline, previous, final=False, required_gain=0.10):
    """Compare integer counts on the same cases, separately for both engines."""
    assert report["manifest"]["motions"] == baseline["manifest"]["motions"] == previous["manifest"]["motions"]
    failures, comparisons = [], []
    for engine in ("gym", "mujoco"):
        for mode in ("full_track", "object_obs"):
            current = report["summary"][engine][mode]
            start = baseline["summary"][engine][mode]
            last = previous["summary"][engine][mode]
            n = current["episodes"]
            assert n > 0 and n == start["episodes"] == last["episodes"]
            assert 0 <= current["carry_and_place_successes"] <= current["task_successes"] <= n
            for metric in METRICS:
                floor = start[metric] if mode == "full_track" else last[metric]
                if final and mode == "object_obs":
                    floor = max(floor, start[metric] + math.ceil(required_gain * n - 1e-9))
                row = dict(engine=engine, mode=mode, metric=metric, episodes=n,
                           baseline=start[metric], previous=last[metric], actual=current[metric],
                           minimum=floor, passed=current[metric] >= floor)
                comparisons.append(row)
                if not row["passed"]:
                    failures.append(row)
    return {"passed": not failures, "final_short_run_check": final,
            "required_final_gain_percentage_points": required_gain * 100,
            "comparisons": comparisons, "failures": failures,
            "long_training_allowed": False}


def validate_report(directory, checkpoint, epoch, expected_motions=20):
    state = json.loads((directory / "status.json").read_text())
    report = json.loads((directory / "summary.json").read_text())
    assert state["state"] == "completed"
    assert report["checkpoint_metadata"]["epoch"] == epoch
    assert report["checkpoint_sha256"] == sha(checkpoint)
    assert report["verified_deploy_prior_only"] and report["rollouts_use_training_pca_frames"]
    assert not report["source_changed_during_evaluation"]
    assert not report["source_changed_since_scheduled"]
    cases = report["manifest"]["motions"]
    assert len(cases) == expected_motions and len({x["motion"] for x in cases}) == expected_motions
    records = json.loads((directory / "results.json").read_text())
    expected_records = expected_motions * 4
    assert len(records) == expected_records and all(x["all_finite"] for x in records)
    assert all(x["controls"] == x["expected_controls"] for x in records)
    assert len({(x["engine"], x["mode"], x["motion"]) for x in records}) == expected_records
    for engine in ("gym", "mujoco"):
        for mode in ("full_track", "object_obs"):
            rows = [x for x in records if x["engine"] == engine and x["mode"] == mode]
            summary = report["summary"][engine][mode]
            assert len(rows) == summary["episodes"] == expected_motions
            assert sum(x["task_success"] for x in rows) == summary["task_successes"]
            assert sum(x["carry_and_place_success"] for x in rows) == summary["carry_and_place_successes"]
    return report


class EvaluationGate:
    def __init__(self, run, request, stopping):
        self.run, self.request, self.stopping = run, request, stopping
        self.config = request["evaluation_gate"]
        self.interval = int(self.config["interval"])
        self.limit = int(self.config["max_new_updates"])
        assert self.interval == 200 and self.limit == 1000
        assert self.config["required_final_gain"] == 0.10
        self.start = request["initial_checkpoint_epoch"]
        self.expected = json.loads((run / "source_sha256.json").read_text())
        self.config_hashes = json.loads((run / "config_sha256.json").read_text())
        self.lock = (run / "evaluation_gate.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.completed, self.baseline, self.previous = [], None, None
        self.decisions = []

    def verify_sources(self):
        actual = source_hashes()
        drift = [p for p in set(self.expected) | set(actual) if self.expected.get(p) != actual.get(p)]
        assert not drift, f"Source changed during experiment: {sorted(drift)}"
        assert all(sha(self.run / p) == digest for p, digest in self.config_hashes.items())

    def status(self, agent, state, checkpoint, **details):
        path = self.run / "memory_status.json"
        data = json.loads(path.read_text())
        data.update(state=state, epoch=agent.epoch_num, log_epoch=agent.epoch_num,
                    relative_rl_epochs=agent.epoch_num - self.request["initial_student_epoch"],
                    updates_since_restart=agent.epoch_num - self.start,
                    checkpoint=str(checkpoint), updated_at=now(), **details)
        write_json(path, data)
        write_json(self.run / "evaluation_status.json", {
            "state": state, "pid": os.getpid(), "updated_at": now(),
            "completed_epochs": self.completed,
            "target_epoch": agent.epoch_num,
            "target_updates_since_restart": agent.epoch_num - self.start,
            "interval": self.interval, "max_new_updates": self.limit,
            "long_training_allowed": False, **details})

    def terminate_child(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)

    def evaluate(self, agent, checkpoint):
        import numpy as np
        import torch
        from watch_student_benchmarks import update_history
        self.verify_sources()
        epoch = agent.epoch_num
        directory = self.run / "benchmarks" / f"epoch_{epoch}" / "attempt_1"
        assert not directory.exists(), "Do not overwrite or silently rerun a previous evaluation"
        directory.mkdir(parents=True)
        command = [sys.executable, str(ROOT / "scripts/run_student_benchmark.py"),
                   "--epoch", str(epoch), "--checkpoint", str(checkpoint), "--base", str(directory),
                   "--training-log", str(self.run / "training.log"), "--policy-architecture", "v3_rl",
                   "--initial-student-epoch", str(self.request["initial_student_epoch"]),
                   "--finetune-restart-epoch", str(self.start),
                   "--cohort-manifest", self.request["development_cohort_manifest"]]
        phase = "evaluating_baseline" if self.baseline is None else "waiting_evaluation"
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        frame, steps = agent.frame, agent.vec_env.env.task.common_step_counter
        task = agent.vec_env.env.task
        snapshot = {name: getattr(task, name).clone() for name in (
            "_humanoid_root_states", "_target_states", "progress_buf", "obs_buf_student")}
        # Release unused allocator blocks for the evaluation subprocess; all live
        # simulator, optimizer and policy tensors stay in this process.
        torch.cuda.empty_cache()
        process = None
        try:
            with (directory / "runner.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                started = time.monotonic()
                while process.poll() is None:
                    self.status(agent, phase, checkpoint, report_dir=str(directory), evaluation_pid=process.pid)
                    if self.stopping:
                        self.terminate_child(process)
                        self.status(agent, "stopped_after_checkpoint", checkpoint, reason="user_requested_stop")
                        raise SystemExit(0)
                    if time.monotonic() - started > self.config.get("timeout_seconds", 7200):
                        raise TimeoutError("Evaluation exceeded its time limit; training remains stopped")
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            if process.returncode:
                raise RuntimeError(f"Evaluation failed ({process.returncode}); see {directory / 'runner.log'}")
            if self.stopping:
                self.status(agent, "stopped_after_checkpoint", checkpoint, reason="user_requested_stop")
                raise SystemExit(0)
            self.verify_sources()
            report = validate_report(directory, checkpoint, epoch)
            self.completed.append(epoch)
            update_history(self.run, report, directory)
            assert agent.frame == frame and agent.vec_env.env.task.common_step_counter == steps
            assert all(torch.equal(value, getattr(task, name)) for name, value in snapshot.items())
            write_json(directory / "training_pause_verification.json", {
                "passed": True, "epoch_before_and_after": epoch, "frames_before_and_after": frame,
                "environment_steps_before_and_after": steps, "unchanged_tensors": list(snapshot),
                "training_process_preserved": True, "RNG_restored_on_return": True})
            return report
        except Exception as error:
            if process is not None:
                self.terminate_child(process)
            self.status(agent, "evaluation_failed", checkpoint, error=repr(error), report_dir=str(directory))
            raise
        finally:
            random.setstate(rng[0]); np.random.set_state(rng[1])
            torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])

    def initialize(self, agent):
        checkpoint = Path(self.request["initial_checkpoint"])
        assert agent.epoch_num == self.start
        self.baseline = self.evaluate(agent, checkpoint)
        self.previous = self.baseline
        reference = json.loads((ROOT / self.config["historical_baseline_report"]).read_text())
        assert reference["checkpoint_sha256"] == self.baseline["checkpoint_sha256"]
        # A changed baseline must be investigated before any candidate update.
        for engine in ("gym", "mujoco"):
            for mode in ("full_track", "object_obs"):
                for key in (*METRICS, "falls"):
                    assert reference["summary"][engine][mode][key] == self.baseline["summary"][engine][mode][key], (
                        "Baseline reproduction differs", engine, mode, key)
        self.status(agent, "training", checkpoint, next_evaluation_epoch=self.start + self.interval,
                    baseline_reproduced=True, live_simulator_state_preserved=True)

    def after_update(self, agent):
        updates = agent.epoch_num - self.start
        assert updates <= self.limit, "Short-run budget exceeded"
        if updates % self.interval:
            return
        path = Path(agent.nn_dir) / f"{agent.config['name']}_{agent.epoch_num:08d}"
        agent.save(str(path)); agent.writer.flush()
        checkpoint = path.with_suffix(".pth")
        report = self.evaluate(agent, checkpoint)
        decision = assess(report, self.baseline, self.previous, updates == self.limit,
                          self.config["required_final_gain"])
        decision.update(epoch=agent.epoch_num, updates_since_restart=updates,
                        checkpoint=str(checkpoint), checked_at=now())
        self.decisions.append(decision)
        write_json(self.run / "gate_decisions.json", self.decisions)
        if not decision["passed"]:
            self.status(agent, "stopped_by_evaluation_gate", checkpoint, decision=decision,
                        last_accepted_epoch=self.previous["checkpoint_metadata"]["epoch"])
            raise SystemExit(0)
        self.previous = report
        if updates == self.limit:
            is_long = self.config.get("stage") == "long_window"
            self.status(agent, "completed_long_window" if is_long else "short_run_passed_pending_acceptance",
                        checkpoint, decision=decision,
                        remaining_requirements=[] if is_long else ["independent expanded acceptance cases", "second training seed"])
            raise SystemExit(0)
        self.status(agent, "training", checkpoint, next_evaluation_epoch=agent.epoch_num + self.interval,
                    live_simulator_state_preserved=True)
