# doublemdp

A compact native JAX/Flax implementation of RQL and BDPO, a separate native PyTorch DPPO online track, and saved-data diagnostics for diffusion/flow policies. It includes upstream parity tests, complete-state resume, a 50-task RQL OGBench manifest, an experiment queue, and regenerable figures/videos.

Read [RESULTS.md](RESULTS.md) for what actually ran, [PROVENANCE.md](PROVENANCE.md) for fidelity decisions, and the component audits in `docs/`. A configured or active experiment is not a completed reproduction.

## Setup and verification

```bash
cd /home/dmsdmswns/doublemdp
bash scripts/setup.sh jax
bash scripts/setup.sh torch
conda activate doublemdp-jax
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
python scripts/check_runtime.py --backend jax --require-gpu --output ../doublemdp_outputs/jax_runtime.json
python scripts/check_ogbench.py --output ../doublemdp_outputs/ogbench_runtime.json
python -m pytest tests/test_data_and_toys.py tests/test_diagnostics.py tests/test_queue.py tests/test_training_resume.py tests/test_rql.py tests/test_bdpo.py -q
```

Run `tests/test_dppo*.py` in `doublemdp-torch`. The checked exports in `environments/` pin the tested runtimes. Setup affects project-specific conda environments only. `third_party/` is a reference area; the maintained package does not import those frameworks for ordinary runs. Parity tests intentionally import pinned upstream files. Original DPPO source experiments use their pinned source launcher separately.

Original Gym/D4RL/DPPO source checks use an isolated legacy environment and scoped MuJoCo runtime:

```bash
bash scripts/setup_legacy.sh
CUDA_VISIBLE_DEVICES=2 scripts/legacy_python.sh -m pytest tests/test_dppo_original_bc_resume.py -q
```

The original BC resume test needs this legacy environment and is skipped by the torch-only suite. `DOUBLEMDP_LEGACY_ENV`, `DOUBLEMDP_OUTPUTS`, and `CONDA_EXE` override setup locations. The public MuJoCo 2.1 binary runs without a private activation key; its supplied notices are retained.

## Saved-data toys

```bash
python -m doublemdp.toy bandit --count 50000 --seed 2026 --output ../doublemdp_outputs/datasets/bandit.npz
python -m doublemdp.toy maze --count 600 --seed 2026 --output ../doublemdp_outputs/datasets/maze.npz
python -m doublemdp.train --config configs/rql_bandit.json
python -m doublemdp.train --config configs/rql_maze.json
python -m doublemdp.train --config configs/bdpo_bandit.json
python -m doublemdp.train --config configs/bdpo_maze.json
```

Use `--gap` when generating a separate coverage-gap dataset. `--reward dense` selects already saved potential-shaped rewards; training never queries a reward grid. Add `--seed 1 --output <new-directory>` for another training seed. The RQL maze preset explicitly selects the `transition_h1` data adapter. Dense-shaping discount mismatches are rejected.

```bash
python -m doublemdp.train --config configs/rql_maze.json --resume
python -m doublemdp.evaluate ../doublemdp_outputs/runs/rql_maze_transition_seed0
python -m doublemdp.visualize ../doublemdp_outputs/runs/rql_maze_transition_seed0 \
  --output ../doublemdp_outputs/figures/rql_maze --continuation
python -m doublemdp.visualize --output ../doublemdp_outputs/figures/rql_maze --regenerate
```

`--resume` restores the saved RNG, sampler, optimizer and counters; use the same run configuration. Changing only the target step budget is supported but does not retroactively redefine a learning-rate schedule. Final and numbered checkpoints, `metrics.jsonl`, CSV, `dataset.json`, configuration and device information are saved. Figures regenerate from `plot_inputs.npz` without retraining or model inference. A small static `index.html` links PNG/PDF and MP4 (GIF fallback).

The bandit's finite-support mixture distinguishes a truly unsupported mode from a rare supported one. Its endpoint-KL quadrature is an objective-specific reference, not RQL's target or a claim that path-regularized BDPO must match it exactly. The continuous maze checks whole action segments against walls. Sparse and dense variants share transitions. DP grid values are approximate; policy-specific continuation rollouts report simulation uncertainty and finite-rollout limitations. Native heads are never mislabeled unregularized returns.

## OGBench and the queue

```bash
python scripts/download_ogbench.py
python -m doublemdp.train --config configs/rql_antmaze-large_official_code.json
python -m doublemdp.train --config configs/rql_scene_official_code.json
python scripts/build_sweep.py --queue ../doublemdp_outputs/rql_sweep.sqlite
python -m doublemdp.sweep work --queue ../doublemdp_outputs/rql_sweep.sqlite --gpu 0
python -m doublemdp.sweep status --queue ../doublemdp_outputs/rql_sweep.sqlite
python -m doublemdp.sweep resume --queue ../doublemdp_outputs/rql_sweep.sqlite --retry-failed
```

Use one worker per chosen GPU; workers claim jobs atomically. The queue exposes `pending/running/failed/complete/blocked`, PIDs, output locations, costs and resume commands. `--max-jobs 1` runs one representative job. Use `--profile paper_budget` with `build_sweep.py` for the separate 2M-update budget; the official-code profile uses 1M. Four independently trained seeds for each of 50 single-task goals make 200 runs per budget profile. The source's ten default-task commands are not a 50-task reproduction.

```bash
python scripts/download_100m.py puzzle-4x4
python scripts/download_100m.py cube-quadruple
```

These resumable downloads create verified 100M shard manifests. Until all required shards exist, their jobs cannot run as faithful large-data experiments. Downloaded files remain outside the repository. The source shard schedule is retained; no smaller substitute is made.

BDPO uses `configs/bdpo_ogbench_port.json` and the same task/data/evaluation manifest. Its frozen development-only tuning budget is in `configs/bdpo_ogbench_manifest.json`; the transferred default is explicitly untuned. `bdpo_ogbench_integration.json` is a short integration check and is not its full training budget. Candidate count, chunk length and inference cost remain method-specific.

## Original BDPO and online DPPO

```bash
python -m doublemdp.agents.bdpo_experiment --task 8gaussians \
  --config configs/bdpo_8gaussians_official_code.json \
  --out ../doublemdp_outputs/runs/bdpo-original-8gaussians-official-code-seed0 --seed 0
python -m doublemdp.agents.bdpo_experiment --task halfcheetah-medium-v2 \
  --config configs/bdpo_halfcheetah_medium_v2.json \
  --out ../doublemdp_outputs/runs/bdpo-halfcheetah-full-seed0 --seed 0
```

Original toy eta .05/current budget, eta .06/paper setting, short sanity checks and paper-budget candidates have distinct configs. The original D4RL loader preserves dataset versions, timeout removal, action clipping, reward normalization and scan sampling. Its legacy simulator evaluation is isolated from modern Gymnasium/OGBench. See `docs/bdpo_audit.md` for the inherited zero-variance `rho=0` gradient corner and separately labeled numerical profile.

```bash
conda activate doublemdp-torch
CUDA_VISIBLE_DEVICES=3 python scripts/train_dppo.py --config configs/dppo_toy.json \
  --dataset ../doublemdp_outputs/datasets/maze.npz --reward dense \
  --output ../doublemdp_outputs/runs/dppo_maze_seed0 --seed 0 \
  --bc-epochs 50 --online-iterations 50 --eval-every 25 --eval-episodes 10
```

DPPO performs real BC initialization followed by real online rollouts/PPO. Every executed training interaction is counted. Evaluation trajectories are kept out of training. Use `--bc-only --strict-offline` for the offline BC reference; `dppo_online` itself rejects strict-offline settings. Resume with `--resume <run>/latest.pt`. Original Hopper uses `scripts/dppo_original.py` and the native v0.6 configs; see `docs/dppo_audit.md` for exact original versus port behavior.

```bash
python scripts/evaluate_dppo.py ../doublemdp_outputs/runs/dppo_maze_seed0/latest.pt \
  --episodes 50 --output ../doublemdp_outputs/runs/dppo_maze_seed0/reevaluation.json
python scripts/dppo_atlas.py --run ../doublemdp_outputs/runs/dppo_maze_seed0 \
  --output ../doublemdp_outputs/figures/dppo_atlas --device cpu
python scripts/dppo_atlas.py --output ../doublemdp_outputs/figures/dppo_atlas --regenerate
```

Evaluation verifies the saved artifact/task identity and uses true sparse maze return, including for policies trained with shaping. It infers the BC/online label from the checkpoint phase; online checkpoints cannot be relabeled as offline BC. The atlas uses one candidate for both remaining generation and environment continuation and saves its common noise draws. Original-source BC supports exact epoch-boundary resume; original-source online resume is explicitly rejected because its checkpoint omits rollout-local and asynchronous worker state. Complete online resume is supported by the maintained DPPO runner.

## Results and limitations

```bash
python scripts/summarize.py
```

This writes a machine-readable report from completed final checkpoints, grouping identical task/data/reward/budget/adapter conditions. Confidence intervals use independent training seeds, not evaluation episodes. Online DPPO is kept separate. Raw run evidence and any failed or superseded trial remain available. The tested implementation is state-input only; original vision/UNet DPPO profiles are not implemented by the compact wrapper and are explicitly rejected rather than silently replaced.
