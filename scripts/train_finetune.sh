#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
student_checkpoint="$1"
shift
finetune_entry="ultra/run_distill.py"
if [[ "${ULTRA_RL_ACTIVATION_CHECKPOINTING:-0}" == "1" ]]; then
  finetune_entry="scripts/run_finetune_checkpointed.py"
fi
python "$finetune_entry" --task UltraDistillObjV3RL \
  --cfg_env ultra/data/cfg/g1_student_finetune.yaml \
  --cfg_train ultra/data/cfg/train/rlg/g1_student_finetune.yaml \
  --resume_from "$student_checkpoint" \
  --headless \
  --output_path output/finetune \
  "$@"
