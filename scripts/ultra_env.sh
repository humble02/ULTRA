#!/usr/bin/env bash
# Source this file before using the local RTX 5090 / Isaac Gym environment.
ultra_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ultra_base_python="$("$ultra_repo_root/.venv/bin/python" -c 'import sys; print(sys.base_prefix)')"
export PATH="$ultra_repo_root/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$ultra_base_python/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$ultra_repo_root/.cache/torch_extensions"
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
unset ultra_base_python ultra_repo_root
