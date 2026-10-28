# Measured results

Snapshot: **2026-10-28 07:10 KST**. This document is generated from saved artifacts by `python scripts/report_results.py`; run `python scripts/summarize.py` first to refresh toy aggregates.

The native implementations, focused parity checks, diagnostic suite and execution infrastructure are working. Full research reproduction is **not** declared complete. Original-paper checks, new ports, offline toy diagnostics and online references are separated below. Checkpoint selection is the final requested update; failed/superseded runs remain on disk.

## Original RQL OGBench: representative code-budget runs

These use the complete public architecture and **1M-update official-code budget**, not the separate 2M paper-budget candidate. Each row is one trained seed, evaluated for 50 episodes. Paper means/intervals use four trained seeds; a single local seed cannot supply a comparable training-seed confidence interval.

| Task | Status / measured updates | Final success | Paper success | Difference | Wall time |
|---|---:|---:|---:|---:|---:|
| [antmaze-large-navigate-singletask-task1-v0](/home/dmsdmswns/doublemdp_outputs/runs/rql_antmaze-large-navigate-singletask-task1-v0_official_code_seed0) | complete; 1,000,000/1,000,000 | 82.0% | 84% [83, 85] | -2.0 pp | 1.13 h |
| [scene-play-singletask-task2-v0](/home/dmsdmswns/doublemdp_outputs/runs/rql_scene-play-singletask-task2-v0_official_code_seed0) | complete; 1,000,000/1,000,000 | 60.0% | 72% [69, 76] | -12.0 pp | 1.20 h |

Intermediate evaluations are retained in each run directory and are not substituted for final scores. The public code versus paper budget discrepancy, independent evaluation RNG, current tested simulator version, and unspecified paper checkpoint/CI estimator are documented in [PROVENANCE.md](PROVENANCE.md) and [RQL audit](docs/rql_audit.md).

Recorded curves: [PNG](/home/dmsdmswns/doublemdp_outputs/figures/reproduction/rql_representatives.png), [PDF](/home/dmsdmswns/doublemdp_outputs/figures/reproduction/rql_representatives.pdf), [raw inputs](/home/dmsdmswns/doublemdp_outputs/figures/reproduction/rql_representatives.npz). Local error bars describe evaluation-episode uncertainty conditional on one trained policy, separately from published training-seed intervals.

All completed per-task/seed records and compatible training-seed aggregates are saved in [the machine-readable report](/home/dmsdmswns/doublemdp_outputs/summary.json); pending and blocked task/seed entries remain in the live queue below.

The full fifty-task/four-seed official-code queue currently has **2 complete, 7 running, 91 pending, 100 blocked, 0 failed**. Its complete manifest is `configs/rql_ogbench_manifest.json`. A launched sweep is not completed reproduction. The 2M-budget sweep is a separate unrun profile.

## Offline toy diagnostics

Each comparison below uses three fixed training seeds and the same saved dataset within a condition. RQL uses 10k updates; BDPO uses 1k behavior pretraining plus 10k RL updates. The explicitly declared toy architectures/inner steps are different. These are diagnostic results, not a matched-compute ranking or paper-score reproduction. Maze metrics are true sparse success even when training uses dense shaping.

| Method | Task / saved-data condition | Training reward | Mean score | 95% interval across training seeds |
|---|---|---|---:|---:|
| BDPO | bandit / gap | sparse | 0.188 | [0.154, 0.222] |
| BDPO | bandit / main | sparse | 0.236 | [0.186, 0.285] |
| BDPO | maze / main | dense | 51.667% | [-7.031, 110.365] |
| BDPO | maze / gap | sparse | 43.333% | [-12.675, 99.341] |
| BDPO | maze / main | sparse | 40.000% | [7.138, 72.862] |
| RQL | bandit / gap | sparse | 0.280 | [0.270, 0.291] |
| RQL | bandit / main | sparse | 0.281 | [0.272, 0.290] |
| RQL | maze / main | dense | 100.000% | [100.000, 100.000] |
| RQL | maze / gap | sparse | 93.333% | [86.162, 100.504] |
| RQL | maze / main | sparse | 100.000% | [100.000, 100.000] |

Intervals are Student-t intervals over independent training-seed scores (n=3), left unclipped. They can extend outside the score range. Identical observed 20/20 successes across three seeds produce a degenerate between-seed interval; this does **not** establish certain success on all future episodes. Episode-level Monte Carlo uncertainty is a separate quantity. No held-out result was used to tune these presets.

Context conditioning matters: the saved behavior rewards are approximately .1256 / .3137 / .4103 across the three bandit contexts (equal-context mean .2832). Neither short diagnostic policy establishes an overall improvement over that behavior baseline. In the rare-high-reward context 0, RQL averages .1054 and BDPO .3075 on the main dataset; the gap control gives .0972 and .1282. BDPO still assigns about 19.7% of context 0 samples to the missing high-reward mode in the gap condition, exposing unsupported extrapolation rather than proving recovery. See the per-context reward/support/mode evidence below.

The gap bandit removes a finite-support mode exactly. The maze gap removes the shortcut connector while retaining the supported detour; success under that control is not evidence of recovering unsupported dynamics. Dense shaping supplies extra geometric information and changes reward scale relative to regularization.

Artifacts: [all run/seed metrics](/home/dmsdmswns/doublemdp_outputs/summary.json), [learning curves](/home/dmsdmswns/doublemdp_outputs/figures/comparison/learning_curves.png), [context-conditioned bandit evidence](/home/dmsdmswns/doublemdp_outputs/figures/comparison/bandit_contexts.json), [fragment-start and empirical support diagnostics](/home/dmsdmswns/doublemdp_outputs/figures/comparison/fragment_start_support.json). The fragment-start test measures held-out continuation success; it is not a proof of compositional stitching. Nearest-neighbor distances are empirical diagnostics, not exact continuous support membership.

## Original BDPO tasks

- **8gaussians sanity:** three full-current-architecture runs, each 2k BC + 5k RL, 50 denoising steps, eta .06, rho 0, batch 512, no clipping. Rewards **0.8255 / 0.6664 / 0.8933**; mean **0.7951**, training-seed SD **0.1164**. These shortened runs are not exact figure reproduction; the paper illustration uses DDIM while these pinned-code traces use DDPM.

- **HalfCheetah-medium-v2 sanity:** full architecture, 1k BC + 1k RL with declared short warmup; actual original Gym/D4RL/MuJoCo evaluation on 10 episodes gives normalized score **2.4411**, versus paper **71.2 ± 0.9**. This confirms integration, not reproduction success. Evaluation used 10,000 simulator interactions and zero training interactions.

| Full original-task job | Status | Completed BC / RL updates | Required BC / RL |
|---|---|---:|---:|
| [bdpo-original-8gaussians-official-code-seed0](/home/dmsdmswns/doublemdp_outputs/runs/bdpo-original-8gaussians-official-code-seed0) | complete | 100,000 / 100,000 | 100,000 / 100,000 |
| [bdpo-original-halfcheetah-medium-v2-official-code-seed0](/home/dmsdmswns/doublemdp_outputs/runs/bdpo-original-halfcheetah-medium-v2-official-code-seed0) | running | 1,927,900 / 0 | 2,000,000 / 2,000,000 |

**Completed current-code 8gaussians budget:** reward **0.9884**, grid JS **0.03530**, 8,192 held-out single-candidate samples; 1.14 h wall time. This is one trained seed with eta .05, separate from the eta .06 short checks and the paper figure. Final evidence: [metrics and compute](/home/dmsdmswns/doublemdp_outputs/runs/bdpo-original-8gaussians-official-code-seed0/run.json), [endpoint evaluation](/home/dmsdmswns/doublemdp_outputs/runs/bdpo-original-8gaussians-official-code-seed0/toy_evaluation.json).

The current official toy profile uses eta .05 and 100k+100k; the eta .06/paper-budget candidates remain separately configured. Full D4RL requires 2M+2M and final simulator evaluation; a completion watcher runs the latter only after both counters reach their budgets. Full original five-seed D4RL reproduction remains unfinished.

Evidence: [original toy aggregate](/home/dmsdmswns/doublemdp_outputs/bdpo_original_toy_aggregate.json), [original D4RL evaluation](/home/dmsdmswns/doublemdp_outputs/runs/bdpo-original-halfcheetah-medium-v2-sanity-seed0/original_evaluation.json), [commands and checkpoint locations](/home/dmsdmswns/doublemdp_outputs/bdpo_active_jobs.json).

## New OGBench ports

- **BDPO scene task2:** 500 BC + 500 RL updates, native best-of-10 action selection, 50 evaluation episodes, **0% success**. This is an integration budget, not the configured 2M+2M port. Reward scale, chunk length one and the unused development-only tuning budget are explicit in `configs/bdpo_ogbench_manifest.json`.

- **DPPO matched AntMaze-large task1:** the exact shared 1M-row preprocessing and array hash were reused for initialization. One BC epoch (7,813 updates), **64 actual online training steps**, and 50 BC plus 50 final evaluation episodes completed. Both evaluated checkpoints scored **0/50 success**, with raw return -1000. The 100,000 evaluation interactions never entered training. This is a new online port integration, not an offline result.

- A separate small OGBench PointMaze integration also completed; it is outside the fifty-task RQL comparison. Full port budgets and tuning/final-seed sweeps remain pending.

Evidence: [BDPO port](/home/dmsdmswns/doublemdp_outputs/runs/bdpo_ogbench_scene_integration_seed0), [matched DPPO port](/home/dmsdmswns/doublemdp_outputs/runs/dppo_antmaze_task1_matched), `configs/dppo_ogbench_matched_manifest.json`.

## Online references and offline BC

DPPO results are kept outside the offline comparison. Six maze runs completed: three sparse and three shaped-reward seeds, each 50 BC epochs / 4,400 BC updates, then **12,800 actual online training interactions** and 2,500 PPO minibatch updates. Each final score uses ten held-out episodes.

| Condition | BC success by seed | Final DPPO success by seed | Final mean ± training-seed SD |
|---|---|---|---|
| Sparse maze | .6, .4, .6 | .7, 1.0, 1.0 | .900 ± .173 |
| Shaped maze | .6, .4, .6 | .6, .9, 1.0 | .833 ± .208 |

The BC checkpoints are eligible as separate offline references. These are different training budgets/evaluation protocols from the offline methods above; no method superiority claim is made.

**Original DPPO Hopper integration:** one complete original BC epoch ran (loss .2493); exact epoch-boundary resume also reached epoch two (loss .2010). Fine-tuning started from the author-provided 3,000-epoch checkpoint and ran two PPO update iterations (40 minibatches). Actual training interactions were **159,452**, not the source nominal counter of 160,000. Initial evaluation used 79,787 physical steps; final held-out evaluation used 79,769 and produced **1454.0601 mean return over 156 completed episodes**. These are original-source integration results; the 3,000-epoch BC and 1,000-iteration fine-tuning budgets were not rerun.

Maintained DPPO checkpoints resume the whole online process, including tested simulator state. Original-source BC now has exact epoch-boundary resume. Original-source online resume is explicitly rejected because that runner omits rollout-local and asynchronous worker state; weights-only restart is not represented as faithful resume.

Evidence: [dense online gallery](/home/dmsdmswns/doublemdp_outputs/figures/dppo/index.html), [sparse online gallery](/home/dmsdmswns/doublemdp_outputs/figures/dppo_sparse/index.html), [original source outputs](/home/dmsdmswns/doublemdp_outputs/dppo_original); details in [DPPO audit](docs/dppo_audit.md).

## Diagnostics, validation and compute

- Fresh JAX and PyTorch environments both passed actual GPU matrix multiplication, differentiation, optimizer updates and synchronization on RTX A6000 GPUs. Hardware: eight 48GB A6000s, driver 550.100; approximately 503GiB RAM and 24TiB free disk at inspection. Tested CUDA runtime is 12.4. No host driver, global Python or other job was modified.

- Fixed-input upstream checks passed for RQL (10 CPU and 10 GPU tests), BDPO (6 core GPU parity tests and expanded 10-test CPU suite), and DPPO (27 maintained tests plus exact original BC resume). DPPO CPU/GPU resume checks cover optimizer tensor devices and actual subsequent rollout/update equality. Shared regression/integrity tests include dataset boundaries versus masks, terminal rewards, immutable arrays, metadata/snapshot/shard identity, offline isolation, complete-state resume, queue behavior and real trace/action equivalence. The latest shared test log is authoritative as tests are extended.

- Full RQL architecture timings on GPU: AntMaze **3.80 ms/update**, scene **3.01 ms/update**, excluding compilation/data/evaluation; first compile/update **17.22s / 14.62s**. JAX allocator peak in those isolated profiles was about **481MB / 471MB**. End-to-end run cost is separately recorded above.

- Final 10k toy reverse/forward cycle RMSE: bandit **.08627**, corrected maze **.09518**; source duration-mixture RMSE **.04293 / .05235**. These measured finite-Euler residuals are not zero. Ten-checkpoint cycle curves and reconstruction-versus-generation latent comparisons are saved.

- Native action particles, physical trajectories, policy-specific Q/V rollouts, approximate DP values with refinement, and continuation estimates are saved in the atlases. Invalid/terminal starting states are handled explicitly. Native heads are separately labeled. RQL expectiles are not directly calibrated against unregularized MC; BDPO objective-matched probes include remaining current-generation and future regularization costs, with finite-tail and source nominal-KL caveats.

- Checkpoint-history probes also completed. RQL uses fixed noise across 1k/5k/10k snapshots; finishing each deterministic inner state gives endpoints agreeing within 1.79e-7 and identical common-noise continuation returns across inner steps. BDPO native inner TD RMSE at 1k/6k/11k is .16407/.03848/.007205, while regularized 100-step MC calibration RMSE is 1.51865/1.77139/1.25650. Unfinished fractions .556/.750/.650 imply substantial truncation bias. A small native TD residual does not establish accurate long-horizon calibration.

- [The DPPO continuation atlas](/home/dmsdmswns/doublemdp_outputs/figures/dppo_atlas/index.html) contains physical trajectories, sparse policy-specific Q/V grids, actual native particles, conditional continuation estimates and BC-to-final history. It used **180,547 evaluation-only transitions**, eight Monte Carlo repetitions and a 100-step horizon in **16.31s** on GPU 2. Mean unfinished fractions .60/.30 for BC/final imply tail bias; saved-path endpoint replay error is zero. Only the two saved stages are compared. Its environment value head uses a different training objective and is not calibrated as an inner head. All supplied noise and plotting inputs are saved; regeneration needs no inference.

Evidence: [JAX GPU](/home/dmsdmswns/doublemdp_outputs/jax_runtime.json), [PyTorch GPU](/home/dmsdmswns/doublemdp_outputs/dppo_runtime.json), [shared tests](/home/dmsdmswns/doublemdp_outputs/shared_tests_latest.log), [navigation profile](/home/dmsdmswns/doublemdp_outputs/rql_profile_antmaze.json), [manipulation profile](/home/dmsdmswns/doublemdp_outputs/rql_profile_scene.json), [artifact gallery](/home/dmsdmswns/doublemdp_outputs/index.html).

## Failures, corrections and unfinished work

The final audit closed a DPPO resume-labeling gap: an online-phase checkpoint can no longer be resumed or evaluated as strict-offline BC. Checkpoint phase and online counters are checked before changing output metadata. The saved online experiments remain in the online track.

The initial RQL maze sampler excluded the saved terminal success transition; those five old runs are marked `SCIENTIFIC_LIMITATIONS.json` and excluded from credit-assignment claims. The corrected toy-only `transition_h1` adapter was tested and all three-seed main/dense/gap conditions rerun. The OGBench sampler was unchanged.

A reduced two-critic BDPO original-toy trial became non-finite. Its endpoint results were invalidated. The pinned zero-rho LCB expression can evaluate an undefined standard-deviation gradient at identical ensemble values. Official kernels remain unchanged; the direct-mean numerical alternative requires an explicit `stable_zero_rho` profile. Full-architecture short original toy runs were finite.

Both 100M datasets are being downloaded in full with resumable range requests; verified shard manifests appear only after all 100 train/100 validation files, hashes and actual counts pass. At the observed server rate these downloads take many hours. Jobs are proactively blocked while required standard files or full verified shard manifests are unavailable; the supervisor re-enables them after byte/count verification. No 1M substitution is used.

The remaining fifty-task/four-seed sweep requires hundreds of GPU-hours once evaluation/data costs are included. Full original BDPO D4RL and original DPPO budgets also exceed this interactive session. Original DPPO full BC is roughly 39 GPU-hours from measured epoch cost; original fine-tuning is approximately 11–13 hours. These are estimates, not completed compute. Pending runs, unavailable aggregate uncertainty, and incomplete original-budget comparisons are retained explicitly.

Live execution state: [sweep supervisor](/home/dmsdmswns/doublemdp_outputs/supervisor/supervisor.json), [per-run queue and exact resume commands](/home/dmsdmswns/doublemdp_outputs/supervisor/queue_status.json), [BDPO run commands](/home/dmsdmswns/doublemdp_outputs/bdpo_active_jobs.json), [D4RL completion/evaluation watcher](/home/dmsdmswns/doublemdp_outputs/runs/bdpo-original-halfcheetah-medium-v2-official-code-seed0/postprocess.json). The supervisor unlocks remaining RQL tasks only after both representatives complete, uses idle assigned GPUs, and does not silently retry failed algorithms.

```bash
python -m doublemdp.sweep status --queue ../doublemdp_outputs/rql_sweep.sqlite
python -m doublemdp.sweep resume --queue ../doublemdp_outputs/rql_sweep.sqlite --retry-failed
python scripts/continue_reproduction.py --queue ../doublemdp_outputs/rql_sweep.sqlite --output ../doublemdp_outputs/supervisor
python scripts/summarize.py
python scripts/report_results.py
```

Inspect the saved error before retrying a failed run. Do not start a second supervisor while its recorded PID is alive. Each run contains the actual interpreter/environment command; checkpoint resume preserves its frozen task/configuration and sampler identity.
