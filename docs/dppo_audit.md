# DPPO implementation and evidence

The maintained implementation is a native PyTorch, state-input DDPM track. It performs offline diffusion behavior cloning, loads the EMA actor, freezes that actor, and collects **new environment transitions and their actual denoising chains** before PPO. `dppo_online` is rejected in strict-offline mode. The BC checkpoint is the separate offline reference. DPPO has an environment value critic; no native inner value head is available.

Reference: [paper v3, 2024-12-09](https://arxiv.org/html/2409.00588v3), especially Appendix B and E.2; [author project](https://diffusion-ppo.github.io/); [official v0.6 source](https://github.com/irom-princeton/dppo/tree/dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc). The repository was cloned and `git rev-parse HEAD` verified as `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`. Upstream is isolated in `third_party/dppo`, with its MIT license retained. The extracted kernels' additional scheduler license is in `docs/dppo_LICENSE.txt`.

| Component | Pinned upstream source | Local symbol/config | Fidelity or disclosed difference |
|---|---|---|---|
| DDPM schedule, BC noise objective | `model/diffusion/{sampling,diffusion}.py` | `agents/dppo_kernels.py:DiffusionModel` | Exact selected source bodies, consolidated imports only. Cosine schedule and epsilon prediction retained. |
| Actor/critic architectures | `model/diffusion/mlp_diffusion.py:DiffusionMLP`, `model/common/{mlp,critic}.py` | Same symbols in `dppo_kernels.py` | State-input MLP/residual MLP only; no vision, UNet, learned DDIM eta CLI support. |
| Frozen actor, trainable tail, stochastic sampling | `diffusion_vpg.py:VPGDiffusion` | Same class | Exact source; noise clipping, final action convention, deterministic DDPM intermediate noise preserved. |
| PPO likelihood and losses | `diffusion_ppo.py:PPODiffusion.loss` | Same class | Per-coordinate log probabilities clamp to `[-5,2]`, slice reward horizon, then **mean** over action horizon and dimension. No generic summed Gaussian likelihood substitution. |
| Sampling / likelihood variance floors | `VPGDiffusion.forward/get_logprobs*` | Same class | Distinct floor settings; likelihood uses Gaussian even though sampling clips standard-normal draws. This upstream approximation is retained. |
| Denoising weighting/clipping | `PPODiffusion.loss` | Same class | Normalize advantage per minibatch with PyTorch sample std, quantile clipping, denoising discount, exponential clipping schedule retained. |
| BC sequences | `agent/dataset/sequence.py:StitchedSequenceDataset` | `BCSequences` | Uniform valid starts, shuffled once per epoch; chunks never cross boundaries. Compact rollout adapter supports observation history length one. No transition-uniform padded-chunk substitute. |
| BC optimizer/EMA | `agent/pretrain/{train_agent,train_diffusion_agent}.py` | `pretrain_update`, `finish_bc_epoch` | AdamW, epoch-based cosine schedule, EMA every five epochs starting at epoch ten retained by default. EMA is **not** updated per minibatch. |
| PPO minibatches/order | `train_ppo_diffusion_agent.py:run` | `ppo_update` | Flatten environment-step × trained-denoising-step indices, shuffle each epoch, discard remainder when larger than one minibatch, joint backward followed by actor then critic optimizer steps. KL stop after step retained. |
| Reward scaling | `util/reward_scaling.py` | Same selected classes | Exact running-return scaling including upstream sample-variance convention; optional constant scale applied before GAE. |
| GAE | `train_ppo_diffusion_agent.py:run` | `generalized_advantages` | Original source uses terminated-only recursion and next collected states after truncations. Explicit `official_code_v0.6` switch retains this; **toy/OGBench ports** bootstrap from timeout final states and stop recursion across resets. |
| Action chunk execution | `env/gym_utils/wrapper/multi_step.py` | `OnlineRunner.collect` | Port executes up to configured chunk length, stopping at first boundary, sums rewards, counts each physical step. Original Hopper source wrapper auto-resets before returning a terminal chunk; original launcher uses that source directly. Source counters overstate steps when a chunk ends early; observation-only telemetry counts physical calls. |
| Task normalization | `script/dataset/get_d4rl_dataset.py`, `mujoco_locomotion_lowdim.py` | Original source launcher | Source min/max maps observations/actions to `[-1,1]`; paper E.2 describes `[0,1]`. Source convention wins for the official-code run. Ports use saved shared observations/actions without this task-specific min/max transform. |
| Original Hopper | `cfg/gym/pretrain/hopper-medium-v2/pre_diffusion_mlp.yaml`, `cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp.yaml` | `configs/dppo_hopper_official.json`, `scripts/dppo_original.py` | 20 DDPM steps, 10 trained steps, predicted/executed horizon four, actor512³, critic256³, BC3000epochs, 40 environments, 500 chunk steps/iteration, 1000 iterations, minibatch50000, five epochs. Full reproduction budget is distinct from short source integration. |
| New ports | No original-paper OGBench/PointMaze profile | `dppo_toy.json`, `dppo_ogbench_port.json` | Declared new diagnostics/ports. Toy actor/critic128³, horizon1; OGBench actor512³/critic256³, horizon1. These are not original-paper results. |
| Checkpoint/resume | Upstream saves primarily model states | `DPPOAgent.save/load`, `OnlineRunner.state_dict` | Infrastructure extension saves all model/EMA/optimizer/scheduler/RNG/sampler/phase/counters and exact supported environment state, atomically. Dataset checksum checked by CLI. |
| Native traces | `VPGDiffusion.forward` | `sample_actions(return_trace=True)`, `continue_generation` | Collects all native DDPM states, including frozen prefix, with actual indices. RNG/action equality verified; no interpolation of fake states. `native_inner_values=None`. |

Known upstream edge cases are explicit: fine-tuning a single denoising step makes the v0.6 clipping schedule undefined, so the harness rejects it. Upstream `return_chain=False` can fail with all denoising steps trainable; the wrapper collects and discards chains, preserving sampled actions and RNG. DDIM kernels remain in the extracted source for attribution/parity, but this maintained runner exposes only DDPM and rejects unsupported observation history. The original-profile CLI calls the pinned source runner rather than silently applying port-specific timeout/chunk semantics.

Runtime modernization: tested Python3.10.22, PyTorch2.6.0+cu124, CUDA runtime12.4, NumPy1.26.4; upstream package metadata requests torch2.4.0. Both oracle and local kernels run on the tested runtime. This establishes fixed-input implementation parity on that runtime, not cross-version bitwise parity. Fresh environments and complete tested package exports are in `environments/dppo-torch-tested.{yml,txt}`. No system CUDA, host drivers, or global Python packages were changed. Scoped `PIP_CONFIG_FILE=/dev/null` avoided an inherited unavailable NVIDIA extra index; the Aliyun PyPI mirror supplied interrupted/resumed package downloads.

GPU evidence is `/home/dmsdmswns/doublemdp_outputs/dppo_runtime.json`: physical GPU3, NVIDIA RTX A6000, `cuda:0` within `CUDA_VISIBLE_DEVICES=3`, GPU matrix multiplication, finite gradients, changed Adam parameters, real BC and real PPO updates, and synchronization. GPU matmul loss128.782974, PPO unchanged-old-policy ratio1.0, eight real training transitions in the validation test. Peak allocated memory of this **small test** was18,620,416bytes; it is not a full-training memory measurement.

Validation command:

```bash
CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl OMP_NUM_THREADS=1 \
  /home/dmsdmswns/miniforge3/envs/doublemdp-torch/bin/python -m pytest tests/test_dppo*.py -q
```

The final maintained DPPO suite completed **27 tests** (21.02 seconds), with the original-source BC resume test executed separately in the legacy environment. It checks independent pinned-source schedule/sampler/logprob/loss/gradient/AdamW parity, true old-policy chain ratios, frozen-actor gradients, distinct variance floors, trace RNG invariance, sequence boundaries, timeout GAE, shaping-discount mismatch rejection, offline rejection, full checkpoint replay on both CPU and GPU (including optimizer tensor devices), GPU computation and actual PPO. Evaluation/continuation tests also verify artifact identity, BC/online labels, exact replay with supplied native noise, and sparse terminal return. OGBench tests restore a second independently initialized PointMaze and cube-double simulator, reproduce five actions, and reproduce the subsequent reset. Batched evaluation restores Python, NumPy and torch RNG states and never increments training-interaction counters. Tolerances: exact same-runtime source calculations where possible, gradients `rtol=1e-6,atol=1e-7`; simulator observations `atol=1e-10`.

The simulator checks exposed two real resume hazards and drove fixes: OGBench can consume global NumPy RNG, and MuJoCo's cached kinematics may correspond to the preceding integration substep. The DPPO adapter records task variables, all writable model/data arrays, integration state, wrapper counters, and RNG. Calling `mj_forward` and restoring qpos/qvel alone changed the next manipulation action. Exact resume is validated only for the tested state-observation adapters and MuJoCo version; unsupported pixels are rejected.

Completed toy budget: seeds0,1,2, the shared saved maze dataset, dense potential shaping, 50BC epochs (4400updates), then 50PPO iterations (2500minibatch updates), **12,800 new physical training transitions per seed**. Evaluation uses ten separate episodes at fixed seed10000; these never enter training. Final-update checkpoint selection, no best-seed/checkpoint selection. BC success `[.6,.4,.6]`; final DPPO success `[.6,.9,1.0]`; across-training-seed mean `.8333`, sample standard deviation `.2082`. Dense returns are shaping-dependent and must not be called sparse return. Files: `/home/dmsdmswns/doublemdp_outputs/runs/dppo_maze_seed{0,1,2}/` (`bc.pt`, `latest.pt`, `metrics.jsonl`, `evaluation_trajectories.npz`, `traces.npz`, `completed.json`). This supports an executable online diagnostic, not a full original-paper reproduction or an offline leaderboard score.

One complete toy command (repeat with seeds1and2):

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
  /home/dmsdmswns/miniforge3/envs/doublemdp-torch/bin/python scripts/train_dppo.py \
  --dataset /home/dmsdmswns/doublemdp_outputs/datasets/maze.npz \
  --output /home/dmsdmswns/doublemdp_outputs/runs/dppo_maze_seed0 \
  --seed 0 --reward dense --bc-epochs 50 --online-iterations 50 --eval-every 25 --eval-episodes 10
```

Resume by adding `--resume <run>/latest.pt` with the same dataset, environment and reward mode. `--bc-only --strict-offline` ends with BC; evaluation interactions remain separately labeled. Native sampler inference calls the frozen actor at each of20steps and the trainable actor at the final10steps:30actor-network calls per candidate in the v0.6 implementation, not20. Candidate count is one.

The strict-offline guard also checks resumed phase: an online checkpoint cannot be relabeled as BC using `--bc-only --strict-offline`. `scripts/evaluate_dppo.py` provides standalone read-only evaluation, validates checkpoint SHA/task/dataset identity, and infers method labels from the saved phase. It reports true sparse maze reward even when training used shaping. A fresh evaluation of dense-maze seed0's final checkpoint with ten episodes batched ten scored .8 success in624 physical evaluation steps and zero training steps. The original sequential ten-episode score was .6; the explicitly changed batching changes the stochastic noise allocation, so this extra check is not a replacement selected score. Artifact: `runs/dppo_maze_seed0/reevaluation.json`.

Original Hopper integration completed in the isolated `doublemdp-legacy` environment. Public author-linked assets were downloaded, checksummed in per-run `original_manifest.json`, and used with the pinned source launch configurations. A from-scratch BC integration trained one full epoch on 999,906 normalized transitions from 2,186 episodes: loss 0.2493, 46.86 seconds, `dppo_original/bc_integration/checkpoint/state_1.pt`. Fine-tuning separately started from the authors' published 3,000-epoch checkpoint. Three source iterations comprise one initial evaluation and two training iterations, with the original 40 environments, 500 chunk steps, horizon four, 50,000-element minibatches, and five PPO epochs. The counted run completed **159,452 physical training transitions** and **79,787 initial evaluation transitions**; the source nominal counter reports 160,000. It performed 40 PPO minibatch updates. Paths are under `/home/dmsdmswns/doublemdp_outputs/dppo_original/ppo_integration_counted/`.

A separate evaluation of its final trained actor and frozen prefix completed with the original 40-environment, 500-chunk-step evaluation loop: mean raw Hopper return **1454.0601** over **156 completed episodes**, using another **79,769 physical evaluation transitions**. This is the source evaluator's completed-episode aggregation, which differs from a fixed 40-episode paper interpretation. Its reported threshold-based “success” is not a meaningful Hopper task success metric. Initial source evaluation return was 1444.2329. These are single-seed integration observations, not a stochastic improvement claim or a completed paper-budget reproduction. Source wrapper seeding calls global NumPy seeding without propagating to every underlying old-Gym environment generator; repeated source runs need not have identical environment trajectories. The port's reset seeding is explicit.

Full budgets remain uncompleted: 3,000 BC epochs and 1,000 source fine-tuning iterations. The one-epoch BC timing extrapolates to about 39 hours; source training iterations took roughly 40–47 seconds on this shared machine (about 11–13 hours for 1,000 iterations, before evaluation differences). Upstream weight-only checkpoints remain insufficient for full resume. The maintained port supports full online state; the explicit source BC extension described below supports exact exhausted-epoch boundaries. Original online `--resume` is rejected: the source `run()` retains rollout/GAE locals, and its asynchronous workers do not expose the simulator, wrapper and RNG state needed for exact continuation. Original integration completion is not mislabeled as a full reproduction.

Legacy setup and original commands:

```bash
bash scripts/setup_legacy.sh
# Explicit short original-task checks:
CUDA_VISIBLE_DEVICES=3 scripts/legacy_python.sh scripts/dppo_original.py \
  --phase pretrain --output /home/dmsdmswns/doublemdp_outputs/dppo_original/bc_integration --epochs 1
CUDA_VISIBLE_DEVICES=3 scripts/legacy_python.sh scripts/dppo_original.py \
  --phase finetune --output /home/dmsdmswns/doublemdp_outputs/dppo_original/ppo_integration_counted --iterations 3
# Full original source fine-tuning budget (not already completed):
CUDA_VISIBLE_DEVICES=3 scripts/legacy_python.sh scripts/dppo_original.py \
  --phase finetune --output /home/dmsdmswns/doublemdp_outputs/dppo_original/full_hopper
```

`--prepare-only` fetches the needed public source assets and records hashes without training. `--phase evaluate --checkpoint <state_2.pt> --output <evaluation_dir>` evaluates the actual fine-tuned actor/prefix with the original evaluation loop. Tested exports are `environments/dppo-legacy-tested.{yml,txt}`. D4RL 1.1 is installed without unused kitchen/Bullet dependencies to keep the old MuJoCo-py locomotion stack isolated. Setup downloads the public MuJoCo 2.1 binary into the external runtime directory, retaining its `THIRD_PARTY_NOTICES`; the binary ran here without an activation key. It installs GLEW in the conda environment and scopes CPATH/LD_LIBRARY_PATH in `legacy_python.sh`, with no host changes.

Original BC runs now apply the reviewed `scripts/dppo_bc_resume.patch` to a separate pinned worktree; the reference checkout stays unchanged. `doublemdp/agents/dppo_original_bc.py` saves `checkpoint/full_bc_latest.pt` after each exhausted epoch: model, EMA, AdamW, scheduler, epoch/update counters, dataset/configuration signature, and Python/NumPy/CPU/CUDA RNG. No iterator remains live at that boundary; restored torch RNG reproduces the next source RandomSampler permutation. A GPU test compares uninterrupted unmodified source for two epochs with one patched epoch plus resume and requires bit-exact model/EMA/optimizer/scheduler/RNG equality. It also rejects changed data. Run it with `CUDA_VISIBLE_DEVICES=3 scripts/legacy_python.sh -m pytest tests/test_dppo_original_bc_resume.py -q`. Resume to an absolute total epoch target with:

```sh
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 scripts/legacy_python.sh scripts/dppo_original.py \
  --phase pretrain --epochs 3000 --resume <previous_run>/checkpoint/full_bc_latest.pt \
  --assets <author_asset_directory> --output <resumed_run_directory>
```

`--assets` controls the downloaded data/model location; it defaults to `$DOUBLEMDP_OUTPUTS/dppo_original`. Setup also accepts `CONDA_EXE`, `DOUBLEMDP_LEGACY_ENV`, and `DOUBLEMDP_OUTPUTS`. The launcher records the exact patch SHA and rejects a worktree differing from the reviewed source hash. This extension does not change BC losses, EMA timing, updates, schedule or source weight checkpoints.

The actual original Hopper CLI also completed epoch one and resumed through epoch two in `dppo_original/bc_full_resume_epoch{1,2}`. Its losses were 0.2493 then 0.2010; the resumed invocation performed only epoch two. Both retain full checkpoints and source manifests. This validates the original data/configuration launch path in addition to the focused exact-state test.

The new OGBench port smoke also completed (`runs/dppo_ogbench_integration`): one BC epoch, 7,813 updates, 34.77 seconds; 64 real training environment steps and 4,000 separate evaluation steps on `pointmaze-medium-navigate-singletask-v0`. Final success was 0/2 and raw return -1000. It verifies end-to-end integration and exact checkpoint infrastructure, not learned task performance. The dedicated integration profile uses 128-wide networks, EMA every epoch, and a tiny two-iteration online budget, all saved explicitly.

The task-matched RQL-scope integration also completed on `antmaze-large-navigate-singletask-task1-v0` in `runs/dppo_antmaze_task1_matched`. The shared loader produced exactly the same one-million-row task relabeling, observations, action clipping and masks as the other tracks, preserving the default reward-minus-one convention (`sparse_conversion=false`). Saved dataset `datasets/ogbench/dppo_antmaze_task1_matched.npz` has file SHA256 `71a7d6f1709024d814817c277cfb446bf9bfb2299a4939bd60d48e935e8d5ee2` and array SHA256 `94632775fc0fa7b886c629b9bb9277613cd450e102191ab517ecdb80e60cd74c`. One BC epoch took 33.50 seconds (7,813 updates), followed by 64 actual online training transitions. BC and final policies each received 50 held-out episodes; both achieved 0/50 success and raw return -1000, with 100,000 total evaluation transitions. `--eval-envs 50` batches policy inference over independent evaluation environments and is explicitly recorded because it changes the evaluation random-number tape shape. This is a dataset-matched port integration check, not a full training result or paper reproduction.

```sh
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 \
  /home/dmsdmswns/miniforge3/envs/doublemdp-torch/bin/python scripts/train_dppo.py \
  --config configs/dppo_ogbench_integration.json \
  --env antmaze-large-navigate-singletask-task1-v0 \
  --dataset /home/dmsdmswns/doublemdp_outputs/datasets/ogbench/dppo_antmaze_task1_matched.npz \
  --output /home/dmsdmswns/doublemdp_outputs/runs/dppo_antmaze_task1_matched \
  --eval-episodes 50 --eval-envs 50 --eval-every 2
```

The paired **sparse-reward** maze runs also completed for seeds 0, 1, 2 with the same saved transitions, architectures and budgets: final success `[.7,1.0,1.0]`, across-seed mean `.9`, sample standard deviation `.1732`. Each used 12,800 actual online interactions. Their logs/checkpoints are in `runs/dppo_maze_sparse_seed{0,1,2}` and regenerated figures in `figures/dppo_sparse`. These three-seed observations do not establish a statistically reliable difference between sparse and dense shaping.

Regenerated native traces, PNG/PDF figures, MP4/GIF, and measured timing are in `/home/dmsdmswns/doublemdp_outputs/figures/dppo/`. Batch-one stochastic inference measured 15.39 ms on CUDA with synchronization (100 samples after ten warmups); one candidate uses 30 actor network calls. The latent ranges remain visible beyond action bounds. Plot inputs contain actual native indices and persistent particle ordering.

The completed synchronized continuation atlas is in `/home/dmsdmswns/doublemdp_outputs/figures/dppo_atlas/index.html`. It contains four physical probes, saved-data coverage, policy-specific sparse V/Q Monte Carlo grids, actual native particles with persistent IDs, terminal/horizon-censored destination colors, and remaining-generation continuation estimates. Its history compares the actual saved BC and final online checkpoints; no intermediate checkpoint is invented. The same single-candidate source evaluation sampler defines generation and outer continuation (`deterministic=True` still samples intermediate denoising noise). Supplied Gaussian draws are saved and shared across inner steps/checkpoints. Exact continuation with each original path's remaining draws reproduced endpoints with maximum error0 on both checkpoints.

The atlas uses eight independent repetitions and a100-step rollout horizon. Mean unfinished fractions across probe/time cells are .60 for BC and .30 for final online, so zero-tail estimates have material truncation bias; error bars only measure conditional Monte Carlo uncertainty. DPPO's native environment V(s) is displayed separately and is not equated with the sparse-return estimate; no inner-head calibration is available. Numerical collection used physical GPU2, took16.31 seconds, and executed180,547 evaluation transitions with zero training transitions. It saves all noise, arrays, checkpoint hashes, native indices and grid-refinement evidence. `scripts/dppo_atlas.py --output <atlas_dir> --regenerate` rebuilds PNG/PDF/MP4/GIF without model inference or training, checking the saved plot-input hash.
