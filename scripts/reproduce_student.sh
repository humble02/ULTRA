#!/usr/bin/env bash
# Run the released student configuration without hyperparameter overrides.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ultra_env.sh
exec bash scripts/train_student.sh "$@"
