#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-jax}"
if [[ "$backend" != jax && "$backend" != torch ]]; then
  echo "Usage: bash scripts/setup.sh [jax|torch]" >&2
  exit 2
fi
conda_bin="${CONDA_EXE:-$(command -v conda)}"
env_name="doublemdp-$backend"
spec="environments/jax-tested.yml"
[[ "$backend" == torch ]] && spec="environments/dppo-torch-tested.yml"
export PIP_CONFIG_FILE=/dev/null
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
if "$conda_bin" run -n "$env_name" python -c 'import sys' >/dev/null 2>&1; then
  echo "Using existing project environment $env_name. Tested pins are in $spec."
else
  "$conda_bin" env create --solver libmamba -f "$spec"
fi
"$conda_bin" run -n "$env_name" python -m pip install --no-deps -e .
"$conda_bin" run -n "$env_name" python -m pip check
bash scripts/fetch_references.sh
output_root="${DOUBLEMDP_OUTPUTS:-$HOME/doublemdp_outputs}"
mkdir -p "$output_root"
"$conda_bin" run -n "$env_name" python scripts/check_runtime.py --backend "$backend" --output "$output_root/${backend}_runtime.json"
