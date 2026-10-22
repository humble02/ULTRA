#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/ultra_env.sh
exec python scripts/view_checkpoint.py "$@"
