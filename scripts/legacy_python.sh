#!/usr/bin/env bash
set -euo pipefail
legacy_env="${DOUBLEMDP_LEGACY_ENV:-/home/dmsdmswns/miniforge3/envs/doublemdp-legacy}"
legacy_output="${DOUBLEMDP_OUTPUTS:-/home/dmsdmswns/doublemdp_outputs}"
export MUJOCO_PY_MUJOCO_PATH="$legacy_output/runtime/mujoco210"
export CPATH="$legacy_env/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$legacy_env/bin/python" "$@"
