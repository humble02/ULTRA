#!/usr/bin/env bash
# Run the released finetuning configuration on a distilled checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ultra_env.sh
exec bash scripts/train_finetune.sh "$@"
