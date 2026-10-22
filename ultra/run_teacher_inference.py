"""Run the included tracking teacher on one G1 motion."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import time

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "ultra"))

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch.
import torch
from learning import ultra_players


def run_rollout(self):
    task = self.env.task
    source = Path(task.motion_file[0])
    source_frames = int(task.max_episode_length[0]) - 20  # loader pads the last frame
    task._enable_early_termination = False
    task.rollout_length = int(task.max_episode_length[0]) + 1

    output = Path(os.environ["ULTRA_TEACHER_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    result = output / "rollout.pt"
    while True:
        obs = self.env_reset()
        self.get_batch_size(obs["obs"], 1)
        observed, reference, actions = [], [], []
        for _ in range(source_frames - 1):
            started = time.perf_counter()
            action = self.get_action(obs, True)
            next_obs, _, done, _ = self.env_step(self.env, action)
            obs = next_obs if isinstance(next_obs, dict) else {"obs": next_obs}
            observed.append(task._curr_obs[0].detach().cpu().clone())
            reference.append(task._curr_ref_obs[0].detach().cpu().clone())
            actions.append(action[0].detach().cpu().clone())
            if task.viewer:
                time.sleep(max(0.0, task.dt - (time.perf_counter() - started)))
            if bool(done[0]):
                break
        torch.save({"observed_hoi": torch.stack(observed),
                    "reference_hoi": torch.stack(reference),
                    "actions": torch.stack(actions),
                    "motion_file": str(source)}, result)
        print(f"Saved {len(actions)} teacher actions to {result}", flush=True)
        if not task.viewer:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, required=True,
                        help="directory containing one released [T,630] G1 motion")
    parser.add_argument("--checkpoint", type=Path,
                        default=repo / "ultra/weights/teacher_ultra_inference.pth")
    parser.add_argument("--output-dir", type=Path, default=repo / "output/teacher_inference")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-env", type=Path, default=repo / "ultra/data/cfg/g1_teacher_no_dr.yaml",
                        help="evaluation environment configuration")
    parser.add_argument("--viewer", action="store_true", help="show live PhysX teacher rollouts in a loop")
    args = parser.parse_args()
    motion_dir = args.motion_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    motion_files = sorted(motion_dir.glob("*.pt"))
    if len(motion_files) != 1:
        parser.error("--motion-dir must contain exactly one .pt motion")
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["ULTRA_TEACHER_OUTPUT_DIR"] = str(output_dir)
    os.chdir(repo)
    ultra_players.UltraPlayerContinuous.run = run_rollout
    sys.argv = [str(repo / "ultra/run.py"),
                "--task", "UltraG1Retarget",
                "--cfg_env", str(args.cfg_env.resolve()),
                "--cfg_train", "ultra/data/cfg/train/rlg/g1_teacher.yaml",
                "--test", "--num_envs", "1", "--seed", str(args.seed),
                "--checkpoint", str(checkpoint), "--motion_file", str(motion_dir),
                "--output_path", str(output_dir / "rlgames")]
    if not args.viewer:
        sys.argv.append("--headless")
    runpy.run_path(str(repo / "ultra/run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
