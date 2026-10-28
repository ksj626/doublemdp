#!/usr/bin/env bash
# Isolated original Gym/D4RL/DPPO runtime. No host libraries/drivers are changed.
set -euo pipefail
legacy_conda="${CONDA_EXE:-/home/dmsdmswns/miniforge3/bin/conda}"
legacy_env="${DOUBLEMDP_LEGACY_ENV:-/home/dmsdmswns/miniforge3/envs/doublemdp-legacy}"
legacy_output="${DOUBLEMDP_OUTPUTS:-/home/dmsdmswns/doublemdp_outputs}"
legacy_repo="$(cd "$(dirname "$0")/.." && pwd)"
export PIP_CONFIG_FILE=/dev/null
legacy_spec="$(mktemp --suffix=.yml)"
trap 'rm -f "$legacy_spec"' EXIT
# D4RL's unused kitchen/bullet extras require a different modern MuJoCo stack.
# The tested locomotion runtime installs its wheel without those dependencies.
sed '/^      - d4rl==/d' "$legacy_repo/environments/dppo-legacy-tested.yml" > "$legacy_spec"
"$legacy_conda" env create --solver libmamba -p "$legacy_env" -f "$legacy_spec"
"$legacy_env/bin/python" -m pip install --no-deps d4rl==1.1
mkdir -p "$legacy_output/runtime"
# MuJoCo 2.1's public binary ran without a private activation key in our check.
# The archive's THIRD_PARTY_NOTICES are retained with the extracted runtime.
curl -fL --retry 3 -o "$legacy_output/runtime/mujoco210-linux-x86_64.tar.gz" \
  https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf "$legacy_output/runtime/mujoco210-linux-x86_64.tar.gz" -C "$legacy_output/runtime"
export MUJOCO_PY_MUJOCO_PATH="$legacy_output/runtime/mujoco210"
export CPATH="$legacy_env/include${CPATH:+:$CPATH}"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$legacy_env/bin/python" -c 'import gym; env=gym.make("Hopper-v2"); env.reset(); print(env.step(env.action_space.sample())); env.close()'
printf 'Legacy Python: %s/bin/python\nMuJoCo runtime: %s\n' "$legacy_env" "$MUJOCO_PY_MUJOCO_PATH"
