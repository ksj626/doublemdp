#!/usr/bin/env python
"""Refresh the reviewable results document from measured run artifacts."""
import argparse
from datetime import datetime
import html
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from doublemdp.sweep import connect,status

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-root',type=Path,default=Path.home()/'doublemdp_outputs')
p.add_argument('--document',type=Path,default=Path('RESULTS.md'))
args=p.parse_args();root=args.output_root.resolve()
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def link(path,label):return f'[{label}]({Path(path).resolve()})'
def last_metrics(run):
    path=run/'metrics.jsonl'
    if not path.exists():return []
    lines=path.read_text().splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer can be part-way through its final JSONL record.
            if i != len(lines)-1:raise
    return rows
def ci_text(group,scale=1):
    bounds=group.get('ci95')
    return 'unavailable' if bounds is None else f"[{bounds[0]*scale:.3f}, {bounds[1]*scale:.3f}]"

summary=read(root/'summary.json',{})
manifest=read('configs/rql_ogbench_manifest.json',{})
tasks={t['env']:t for t in manifest.get('tasks',[])}
queue=status(connect(root/'rql_sweep.sqlite'))
counts={s:sum(r['status']==s for r in queue) for s in ('pending','running','failed','complete','blocked')}
now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M %Z')
lines=['# Measured results',f'\nSnapshot: **{now}**. This document is generated from saved artifacts by `python scripts/report_results.py`; run `python scripts/summarize.py` first to refresh toy aggregates.',
       '\nThe native implementations, focused parity checks, diagnostic suite and execution infrastructure are working. Full research reproduction is **not** declared complete. Original-paper checks, new ports, offline toy diagnostics and online references are separated below. Checkpoint selection is the final requested update; failed/superseded runs remain on disk.']
lines+=['\n## Original RQL OGBench: representative code-budget runs',
        '\nThese use the complete public architecture and **1M-update official-code budget**, not the separate 2M paper-budget candidate. Each row is one trained seed, evaluated for 50 episodes. Paper means/intervals use four trained seeds; a single local seed cannot supply a comparable training-seed confidence interval.',
        '\n| Task | Status / measured updates | Final success | Paper success | Difference | Wall time |',
        '|---|---:|---:|---:|---:|---:|']
for env in ('antmaze-large-navigate-singletask-task1-v0','scene-play-singletask-task2-v0'):
    run=root/'runs'/f'rql_{env}_official_code_seed0';state=read(run/'status.json',{});rows=last_metrics(run)
    latest=rows[-1]['step'] if rows else state.get('step',0)
    complete=state.get('status')=='complete' and state.get('step')==1000000
    final=read(run/'evaluation_1000000.json') if complete else None
    ref=tasks[env]['paper_reference'];paper=ref['score_percent'];bounds=ref['ci95_percent']
    score=f"{100*final['success']:.1f}%" if final else 'pending'
    delta=f"{100*final['success']-paper:+.1f} pp" if final else 'pending'
    elapsed=state.get('elapsed_seconds',0)/3600
    lines.append(f"| {link(run,env)} | {'complete' if complete else state.get('status','pending')}; {latest:,}/1,000,000 | {score} | {paper}% [{bounds[0]}, {bounds[1]}] | {delta} | {elapsed:.2f} h |")
lines+=['\nIntermediate evaluations are retained in each run directory and are not substituted for final scores. The public code versus paper budget discrepancy, independent evaluation RNG, current tested simulator version, and unspecified paper checkpoint/CI estimator are documented in [PROVENANCE.md](PROVENANCE.md) and [RQL audit](docs/rql_audit.md).',
        f'\nRecorded curves: {link(root/"figures/reproduction/rql_representatives.png","PNG")}, {link(root/"figures/reproduction/rql_representatives.pdf","PDF")}, {link(root/"figures/reproduction/rql_representatives.npz","raw inputs")}. Local error bars describe evaluation-episode uncertainty conditional on one trained policy, separately from published training-seed intervals.',
        f'\nAll completed per-task/seed records and compatible training-seed aggregates are saved in {link(root/"summary.json","the machine-readable report")}; pending and blocked task/seed entries remain in the live queue below.',
        f'\nThe full fifty-task/four-seed official-code queue currently has **{counts["complete"]} complete, {counts["running"]} running, {counts["pending"]} pending, {counts["blocked"]} blocked, {counts["failed"]} failed**. Its complete manifest is `configs/rql_ogbench_manifest.json`. A launched sweep is not completed reproduction. The 2M-budget sweep is a separate unrun profile.']

lines+=['\n## Offline toy diagnostics',
        '\nEach comparison below uses three fixed training seeds and the same saved dataset within a condition. RQL uses 10k updates; BDPO uses 1k behavior pretraining plus 10k RL updates. The explicitly declared toy architectures/inner steps are different. These are diagnostic results, not a matched-compute ranking or paper-score reproduction. Maze metrics are true sparse success even when training uses dense shaping.',
        '\n| Method | Task / saved-data condition | Training reward | Mean score | 95% interval across training seeds |',
        '|---|---|---|---:|---:|']
records=summary.get('completed_offline_runs',[])
for g in summary.get('training_seed_aggregates',[]):
    if g['n_seeds']!=3 or g['env'] not in ('bandit','maze'):continue
    matching=[r for r in records if r['method']==g['method'] and r['dataset_arrays_sha256']==g['dataset_arrays_sha256'] and r['reward']==g['reward']]
    condition='gap' if matching and '_gap' in str(matching[0].get('dataset')) else 'main'
    scale=100 if g['env']=='maze' else 1;unit='%' if scale==100 else ''
    lines.append(f"| {g['method'].upper()} | {g['env']} / {condition} | {g['reward']} | {g['mean']*scale:.3f}{unit} | {ci_text(g,scale)} |")
lines+=['\nIntervals are Student-t intervals over independent training-seed scores (n=3), left unclipped. They can extend outside the score range. Identical observed 20/20 successes across three seeds produce a degenerate between-seed interval; this does **not** establish certain success on all future episodes. Episode-level Monte Carlo uncertainty is a separate quantity. No held-out result was used to tune these presets.',
        '\nContext conditioning matters: the saved behavior rewards are approximately .1256 / .3137 / .4103 across the three bandit contexts (equal-context mean .2832). Neither short diagnostic policy establishes an overall improvement over that behavior baseline. In the rare-high-reward context 0, RQL averages .1054 and BDPO .3075 on the main dataset; the gap control gives .0972 and .1282. BDPO still assigns about 19.7% of context 0 samples to the missing high-reward mode in the gap condition, exposing unsupported extrapolation rather than proving recovery. See the per-context reward/support/mode evidence below.',
        '\nThe gap bandit removes a finite-support mode exactly. The maze gap removes the shortcut connector while retaining the supported detour; success under that control is not evidence of recovering unsupported dynamics. Dense shaping supplies extra geometric information and changes reward scale relative to regularization.',
        f'\nArtifacts: {link(root/"summary.json","all run/seed metrics")}, {link(root/"figures/comparison/learning_curves.png","learning curves")}, {link(root/"figures/comparison/bandit_contexts.json","context-conditioned bandit evidence")}, {link(root/"figures/comparison/fragment_start_support.json","fragment-start and empirical support diagnostics")}. The fragment-start test measures held-out continuation success; it is not a proof of compositional stitching. Nearest-neighbor distances are empirical diagnostics, not exact continuous support membership.']

lines+=['\n## Original BDPO tasks',
        '\n- **8gaussians sanity:** three full-current-architecture runs, each 2k BC + 5k RL, 50 denoising steps, eta .06, rho 0, batch 512, no clipping. Rewards **0.8255 / 0.6664 / 0.8933**; mean **0.7951**, training-seed SD **0.1164**. These shortened runs are not exact figure reproduction; the paper illustration uses DDIM while these pinned-code traces use DDPM.',
        '\n- **HalfCheetah-medium-v2 sanity:** full architecture, 1k BC + 1k RL with declared short warmup; actual original Gym/D4RL/MuJoCo evaluation on 10 episodes gives normalized score **2.4411**, versus paper **71.2 ± 0.9**. This confirms integration, not reproduction success. Evaluation used 10,000 simulator interactions and zero training interactions.',
        '\n| Full original-task job | Status | Completed BC / RL updates | Required BC / RL |',
        '|---|---|---:|---:|']
completed_original=[]
for name in ('bdpo-original-8gaussians-official-code-seed0','bdpo-original-halfcheetah-medium-v2-official-code-seed0'):
    run=root/'runs'/name;meta=read(run/'run.json',{});cfg=meta.get('config',{})
    rows=last_metrics(run);latest=rows[-1] if rows else {}
    bc=max(meta.get('completed_pretrain_steps',0),latest.get('pretrain_steps',0))
    rl=max(meta.get('completed_train_steps',0),latest.get('train_steps',0))
    lines.append(f"| {link(run,name)} | {meta.get('status','pending')} | {bc:,} / {rl:,} | {cfg.get('pretrain_steps',0):,} / {cfg.get('train_steps',0):,} |")
    if meta.get('status')=='complete' and 'evaluation' in meta:
        ev=meta['evaluation']
        completed_original.append(f"\n**Completed current-code 8gaussians budget:** reward **{ev['reward_mean']:.4f}**, grid JS **{ev['grid_JS']:.5f}**, {ev['sample_count']:,} held-out single-candidate samples; {meta['elapsed_seconds']/3600:.2f} h wall time. This is one trained seed with eta .05, separate from the eta .06 short checks and the paper figure. Final evidence: {link(run/'run.json','metrics and compute')}, {link(run/'toy_evaluation.json','endpoint evaluation')}.")
    original_eval=read(run/'original_evaluation.json')
    if meta.get('status')=='complete' and original_eval:
        completed_original.append(f"\nFull-budget original D4RL simulator evaluation is available in {link(run/'original_evaluation.json','final evaluation')}; the 1k+1k sanity score above is a separate checkpoint.")
lines+=completed_original
lines+=['\nThe current official toy profile uses eta .05 and 100k+100k; the eta .06/paper-budget candidates remain separately configured. Full D4RL requires 2M+2M and final simulator evaluation; a completion watcher runs the latter only after both counters reach their budgets. Full original five-seed D4RL reproduction remains unfinished.',
        f'\nEvidence: {link(root/"bdpo_original_toy_aggregate.json","original toy aggregate")}, {link(root/"runs/bdpo-original-halfcheetah-medium-v2-sanity-seed0/original_evaluation.json","original D4RL evaluation")}, {link(root/"bdpo_active_jobs.json","commands and checkpoint locations")}.']

lines+=['\n## New OGBench ports',
        '\n- **BDPO scene task2:** 500 BC + 500 RL updates, native best-of-10 action selection, 50 evaluation episodes, **0% success**. This is an integration budget, not the configured 2M+2M port. Reward scale, chunk length one and the unused development-only tuning budget are explicit in `configs/bdpo_ogbench_manifest.json`.',
        '\n- **DPPO matched AntMaze-large task1:** the exact shared 1M-row preprocessing and array hash were reused for initialization. One BC epoch (7,813 updates), **64 actual online training steps**, and 50 BC plus 50 final evaluation episodes completed. Both evaluated checkpoints scored **0/50 success**, with raw return -1000. The 100,000 evaluation interactions never entered training. This is a new online port integration, not an offline result.',
        '\n- A separate small OGBench PointMaze integration also completed; it is outside the fifty-task RQL comparison. Full port budgets and tuning/final-seed sweeps remain pending.',
        f'\nEvidence: {link(root/"runs/bdpo_ogbench_scene_integration_seed0","BDPO port")}, {link(root/"runs/dppo_antmaze_task1_matched","matched DPPO port")}, `configs/dppo_ogbench_matched_manifest.json`.']

lines+=['\n## Online references and offline BC',
        '\nDPPO results are kept outside the offline comparison. Six maze runs completed: three sparse and three shaped-reward seeds, each 50 BC epochs / 4,400 BC updates, then **12,800 actual online training interactions** and 2,500 PPO minibatch updates. Each final score uses ten held-out episodes.',
        '\n| Condition | BC success by seed | Final DPPO success by seed | Final mean ± training-seed SD |',
        '|---|---|---|---|',
        '| Sparse maze | .6, .4, .6 | .7, 1.0, 1.0 | .900 ± .173 |',
        '| Shaped maze | .6, .4, .6 | .6, .9, 1.0 | .833 ± .208 |',
        '\nThe BC checkpoints are eligible as separate offline references. These are different training budgets/evaluation protocols from the offline methods above; no method superiority claim is made.',
        '\n**Original DPPO Hopper integration:** one complete original BC epoch ran (loss .2493); exact epoch-boundary resume also reached epoch two (loss .2010). Fine-tuning started from the author-provided 3,000-epoch checkpoint and ran two PPO update iterations (40 minibatches). Actual training interactions were **159,452**, not the source nominal counter of 160,000. Initial evaluation used 79,787 physical steps; final held-out evaluation used 79,769 and produced **1454.0601 mean return over 156 completed episodes**. These are original-source integration results; the 3,000-epoch BC and 1,000-iteration fine-tuning budgets were not rerun.',
        '\nMaintained DPPO checkpoints resume the whole online process, including tested simulator state. Original-source BC now has exact epoch-boundary resume. Original-source online resume is explicitly rejected because that runner omits rollout-local and asynchronous worker state; weights-only restart is not represented as faithful resume.',
        f'\nEvidence: {link(root/"figures/dppo/index.html","dense online gallery")}, {link(root/"figures/dppo_sparse/index.html","sparse online gallery")}, {link(root/"dppo_original","original source outputs")}; details in [DPPO audit](docs/dppo_audit.md).']

lines+=['\n## Diagnostics, validation and compute',
        '\n- Fresh JAX and PyTorch environments both passed actual GPU matrix multiplication, differentiation, optimizer updates and synchronization on RTX A6000 GPUs. Hardware: eight 48GB A6000s, driver 550.100; approximately 503GiB RAM and 24TiB free disk at inspection. Tested CUDA runtime is 12.4. No host driver, global Python or other job was modified.',
        '\n- Fixed-input upstream checks passed for RQL (10 CPU and 10 GPU tests), BDPO (6 core GPU parity tests and expanded 10-test CPU suite), and DPPO (27 maintained tests plus exact original BC resume). DPPO CPU/GPU resume checks cover optimizer tensor devices and actual subsequent rollout/update equality. Shared regression/integrity tests include dataset boundaries versus masks, terminal rewards, immutable arrays, metadata/snapshot/shard identity, offline isolation, complete-state resume, queue behavior and real trace/action equivalence. The latest shared test log is authoritative as tests are extended.',
        '\n- Full RQL architecture timings on GPU: AntMaze **3.80 ms/update**, scene **3.01 ms/update**, excluding compilation/data/evaluation; first compile/update **17.22s / 14.62s**. JAX allocator peak in those isolated profiles was about **481MB / 471MB**. End-to-end run cost is separately recorded above.',
        '\n- Final 10k toy reverse/forward cycle RMSE: bandit **.08627**, corrected maze **.09518**; source duration-mixture RMSE **.04293 / .05235**. These measured finite-Euler residuals are not zero. Ten-checkpoint cycle curves and reconstruction-versus-generation latent comparisons are saved.',
        '\n- Native action particles, physical trajectories, policy-specific Q/V rollouts, approximate DP values with refinement, and continuation estimates are saved in the atlases. Invalid/terminal starting states are handled explicitly. Native heads are separately labeled. RQL expectiles are not directly calibrated against unregularized MC; BDPO objective-matched probes include remaining current-generation and future regularization costs, with finite-tail and source nominal-KL caveats.',
        '\n- Checkpoint-history probes also completed. RQL uses fixed noise across 1k/5k/10k snapshots; finishing each deterministic inner state gives endpoints agreeing within 1.79e-7 and identical common-noise continuation returns across inner steps. BDPO native inner TD RMSE at 1k/6k/11k is .16407/.03848/.007205, while regularized 100-step MC calibration RMSE is 1.51865/1.77139/1.25650. Unfinished fractions .556/.750/.650 imply substantial truncation bias. A small native TD residual does not establish accurate long-horizon calibration.',
        f'\n- {link(root/"figures/dppo_atlas/index.html","The DPPO continuation atlas")} contains physical trajectories, sparse policy-specific Q/V grids, actual native particles, conditional continuation estimates and BC-to-final history. It used **180,547 evaluation-only transitions**, eight Monte Carlo repetitions and a 100-step horizon in **16.31s** on GPU 2. Mean unfinished fractions .60/.30 for BC/final imply tail bias; saved-path endpoint replay error is zero. Only the two saved stages are compared. Its environment value head uses a different training objective and is not calibrated as an inner head. All supplied noise and plotting inputs are saved; regeneration needs no inference.',
        f'\nEvidence: {link(root/"jax_runtime.json","JAX GPU")}, {link(root/"dppo_runtime.json","PyTorch GPU")}, {link(root/"shared_tests_latest.log","shared tests")}, {link(root/"rql_profile_antmaze.json","navigation profile")}, {link(root/"rql_profile_scene.json","manipulation profile")}, {link(root/"index.html","artifact gallery")}.']

lines+=['\n## Failures, corrections and unfinished work',
        '\nThe final audit closed a DPPO resume-labeling gap: an online-phase checkpoint can no longer be resumed or evaluated as strict-offline BC. Checkpoint phase and online counters are checked before changing output metadata. The saved online experiments remain in the online track.',
        '\nThe initial RQL maze sampler excluded the saved terminal success transition; those five old runs are marked `SCIENTIFIC_LIMITATIONS.json` and excluded from credit-assignment claims. The corrected toy-only `transition_h1` adapter was tested and all three-seed main/dense/gap conditions rerun. The OGBench sampler was unchanged.',
        '\nA reduced two-critic BDPO original-toy trial became non-finite. Its endpoint results were invalidated. The pinned zero-rho LCB expression can evaluate an undefined standard-deviation gradient at identical ensemble values. Official kernels remain unchanged; the direct-mean numerical alternative requires an explicit `stable_zero_rho` profile. Full-architecture short original toy runs were finite.',
        '\nBoth 100M datasets are being downloaded in full with resumable range requests; verified shard manifests appear only after all 100 train/100 validation files, hashes and actual counts pass. At the observed server rate these downloads take many hours. Jobs are proactively blocked while required standard files or full verified shard manifests are unavailable; the supervisor re-enables them after byte/count verification. No 1M substitution is used.',
        '\nThe remaining fifty-task/four-seed sweep requires hundreds of GPU-hours once evaluation/data costs are included. Full original BDPO D4RL and original DPPO budgets also exceed this interactive session. Original DPPO full BC is roughly 39 GPU-hours from measured epoch cost; original fine-tuning is approximately 11–13 hours. These are estimates, not completed compute. Pending runs, unavailable aggregate uncertainty, and incomplete original-budget comparisons are retained explicitly.',
        f'\nLive execution state: {link(root/"supervisor/supervisor.json","sweep supervisor")}, {link(root/"supervisor/queue_status.json","per-run queue and exact resume commands")}, {link(root/"bdpo_active_jobs.json","BDPO run commands")}, {link(root/"runs/bdpo-original-halfcheetah-medium-v2-official-code-seed0/postprocess.json","D4RL completion/evaluation watcher")}. The supervisor unlocks remaining RQL tasks only after both representatives complete, uses idle assigned GPUs, and does not silently retry failed algorithms.',
        '\n```bash\npython -m doublemdp.sweep status --queue ../doublemdp_outputs/rql_sweep.sqlite\npython -m doublemdp.sweep resume --queue ../doublemdp_outputs/rql_sweep.sqlite --retry-failed\npython scripts/continue_reproduction.py --queue ../doublemdp_outputs/rql_sweep.sqlite --output ../doublemdp_outputs/supervisor\npython scripts/summarize.py\npython scripts/report_results.py\n```',
        '\nInspect the saved error before retrying a failed run. Do not start a second supervisor while its recorded PID is alive. Each run contains the actual interpreter/environment command; checkpoint resume preserves its frozen task/configuration and sampler identity.']
args.document.write_text('\n'.join(lines)+'\n')

gallery=[('RQL bandit',root/'runs/rql_bandit_seed0/visualizations/index.html'),('RQL maze continuation atlas',root/'runs/rql_maze_transition_seed0/visualizations/index.html'),
         ('Full-budget RQL representative curves',root/'figures/reproduction/rql_representatives.png'),
         ('RQL continuation across checkpoints',root/'runs/rql_maze_transition_seed0/continuation_history/index.html'),
         ('BDPO objective-matched calibration history',root/'diagnostics/bdpo_probe_history_seed0/calibration_history.png'),
         ('Full current-code BDPO 8gaussians',root/'runs/bdpo-original-8gaussians-official-code-seed0/index.html'),
         ('BDPO bandit',root/'figures/bdpo_bandit/index.html'),('BDPO maze continuation atlas',root/'figures/bdpo_maze/index.html'),
         ('DPPO dense online',root/'figures/dppo/index.html'),('DPPO sparse online',root/'figures/dppo_sparse/index.html')]
gallery.append(('DPPO continuation atlas and checkpoint history',root/'figures/dppo_atlas/index.html'))
page="<!doctype html><meta charset='utf-8'><title>doublemdp evidence</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.5}img{max-width:100%}a{color:#175b96}</style><h1>doublemdp measured evidence</h1>"
page+=f'<p>Updated {html.escape(now)}. Separate offline diagnostics, original-task checks, and online references. Active jobs are not completed reproduction.</p><ul>'
for title,path in gallery:
    if path.exists():page+=f'<li><a href="{html.escape(os.path.relpath(path,root))}">{html.escape(title)}</a></li>'
page+='</ul><p><a href="summary.json">Machine-readable results and queue</a> · <a href="supervisor/queue_status.json">Live queue</a></p><h2>Declared toy learning curves</h2><img src="figures/comparison/learning_curves.png" alt="Three independent training seeds; method-specific compute">'
page+='<p>Native inner heads have method-specific objectives. Sampling latents are action-space quantities, not physical trajectories. Online results include additional training interactions.</p>'
(root/'index.html').write_text(page)
print(args.document.resolve()); print(root/'index.html')
