"""Fixed-input BDPO continuation calibration across saved training snapshots."""
import json
from pathlib import Path
import numpy as np


def plot_history(out):
    """Regenerate both figures from saved inputs, without loading any checkpoint."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import t
    out=Path(out)
    with np.load(out/'probe_history.npz') as data:
        d={key:data[key] for key in data.files}
    names=['Native vs regularized MC RMSE','Native inner TD RMSE','Unfinished rollout fraction']
    error=d['native_minus_probe'].reshape(len(d['checkpoints']),4,-1)
    maps=[np.sqrt(np.mean(error**2,axis=1)),
          np.sqrt(np.mean(d.get('td_residual',d.get('td_td_residual')).reshape(len(d['checkpoints']),4,-1)**2,axis=1)),
          d['unfinished_fraction'].reshape(len(d['checkpoints']),4,-1).mean(1)]
    fig,axes=plt.subplots(1,3,figsize=(14,4.1))
    for ax,title,values in zip(axes,names,maps):
        picture=ax.imshow(values,origin='lower',aspect='auto',cmap='viridis')
        ax.set_xticks(range(values.shape[1]),[f'{p:.2f}\nt={time}' for p,time in zip(d['progress'][:values.shape[1]],d['selected_times'][:values.shape[1]])])
        ax.set_yticks(range(len(d['checkpoints'])),[str(x) for x in d['checkpoints']])
        ax.set(title=title,xlabel='Actual generation progress / native time',ylabel='Saved training update')
        fig.colorbar(picture,ax=ax)
    report=json.loads((out/'probe_history.json').read_text())
    fig.suptitle(f'Fixed final-checkpoint latents; nominal path costs | {report["horizon"]}-step zero tail: truncation bias remains')
    fig.tight_layout();fig.savefig(out/'calibration_history.png',dpi=160);fig.savefig(out/'calibration_history.pdf');plt.close(fig)
    # The unregularized q_probe is obtained from the exact same held-out
    # environment trajectories, before subtracting source path penalties.
    fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True)
    critical=float(t.ppf(.975,report['repetitions']-1))
    for row,(key,label) in enumerate([('samples','Nominal regularized continuation'),('reward_samples','Unregularized q_probe')]):
        values=d[key].reshape(len(d['checkpoints']),4,len(d['progress']),-1)
        for state in range(4):
            ax=axes[row,state]
            for checkpoint_index,step in enumerate(d['checkpoints']):
                samples=values[checkpoint_index,state]
                se=samples.std(-1,ddof=1)/np.sqrt(report['repetitions'])
                ax.errorbar(d['progress'],samples.mean(-1),yerr=critical*se,fmt='-o',markersize=3,label=f'update {step}')
            ax.set_title(f'Probe {state}: {d["probe_states"][state]}')
            if state==0:ax.set_ylabel(label)
            if row==1:ax.set_xlabel('Actual generation progress')
    axes[0,-1].legend(fontsize=8)
    fig.suptitle('Same states, latents and held-out noise; 95% Monte Carlo t intervals (no tail-bias correction)')
    fig.tight_layout();fig.savefig(out/'continuation_history.png',dpi=160);fig.savefig(out/'continuation_history.pdf');plt.close(fig)


def evaluate_history(run, checkpoint_steps, out, *, repetitions=8, horizon=100, reward_mode='sparse'):
    from doublemdp.evaluate import load_run
    from doublemdp.toy import PointMaze
    from doublemdp.diagnostics import native_progress_indices
    from .bdpo_probe import regularized_maze_continuation,native_inner_td_residual
    run,out=Path(run),Path(out);out.mkdir(parents=True,exist_ok=True)
    steps=sorted(set(checkpoint_steps))
    paths=[run/f'checkpoint_{step:09d}.pkl' for step in steps]
    if len(steps)<2 or any(not path.exists() for path in paths):
        raise ValueError('At least two existing saved checkpoint files are required')
    reference,_,_,reference_checkpoint=load_run(paths[-1])
    states=PointMaze.probes.copy()
    trace=reference.sample_trace(states,seed=771,num_samples=1)
    indices=native_progress_indices(trace['progress'])
    progress=np.asarray(trace['progress'])[indices]
    selected_times=np.asarray(trace['native_times'])[indices]
    latents=np.asarray(trace['latents'])[indices,:,0].swapaxes(0,1).reshape(-1,2)
    times=np.tile(selected_times,len(states)); repeated_states=np.repeat(states,len(indices),axis=0)
    valid=times>0
    arrays={};summaries=[]
    for requested_step,path in zip(steps,paths):
        agent,config,metadata,checkpoint=load_run(path)
        if int(checkpoint['step'])!=requested_step:
            raise ValueError('Checkpoint filename and saved training counter disagree')
        result=regularized_maze_continuation(agent,repeated_states,latents,times,
            repetitions=repetitions,horizon=horizon,seed=281,reward_mode=reward_mode)
        td=native_inner_td_residual(agent,repeated_states[valid],latents[valid],times[valid],repetitions=32,seed=311)
        for key,value in result.items():
            if isinstance(value,np.ndarray):arrays.setdefault(key,[]).append(value)
        for key,value in td.items():
            arrays.setdefault(key if key.startswith('td_') else 'td_'+key,[]).append(value)
        entry={'training_update':requested_step,'pretrain_steps':agent._n_pretraining_steps,
            'rl_steps':agent._n_training_steps,'checkpoint':str(path),
            'checkpoint_identity':checkpoint.get('artifact_identity'),
            'regularized_native_vs_mc_rmse':float(np.sqrt(np.mean(result['native_minus_probe']**2))),
            'native_inner_td_rmse':float(np.sqrt(np.mean(td['td_residual']**2))),
            'unregularized_probe_mean':float(result['reward_samples'].mean()),
            'unfinished_fraction_mean':float(result['unfinished_fraction'].mean()),
            'unfinished_fraction_max':float(result['unfinished_fraction'].max())}
        summaries.append(entry);print(json.dumps(entry),flush=True)
    arrays={key:np.stack(value) for key,value in arrays.items()}
    rng=np.random.RandomState(281)
    arrays.update(checkpoints=np.asarray(steps),probe_states=states,fixed_states=repeated_states,
        fixed_latents=latents,fixed_native_times=times,progress=progress,selected_times=selected_times,
        reference_full_latents=np.asarray(trace['latents']),reference_trace_seed=np.array(771),
        continuation_keys=np.asarray([rng.randint(2**31-1) for _ in np.unique(times)]),
        future_generation_keys=np.asarray([rng.randint(2**31-1) for _ in range(horizon-1)]),
        td_seed=np.array(311))
    np.savez_compressed(out/'probe_history.npz',**arrays)
    report={'checkpoints':summaries,'horizon':horizon,'repetitions':repetitions,'reward_mode':reward_mode,
        'reference_checkpoint':str(paths[-1]),'reference_identity':reference_checkpoint.get('artifact_identity'),
        'fixed_inputs':'Four physical states, twenty native latents sampled at the final checkpoint, native times and all continuation/rollout random keys held fixed.',
        'latent_distribution':'Actual final-checkpoint sampler states. Earlier-checkpoint queries are post-hoc probes on this fixed reference distribution, not claims of earlier native trajectories.',
        'policy':'Each checkpoint frozen actor_target, single candidate; its own frozen behavior_target for nominal regularization.',
        'unregularized_quantity':'q_probe from discounted environmental rewards on the same held-out paths, with all generation costs removed. Not equated with native Vt/Q0.',
        'regularized_quantity':'Current remaining nominal path cost plus discounted future generation costs subtracted from rollout return; target-critic ensemble means; no inner gamma.',
        'noise_note':'Within-method common random draws; same PRNG keys and batch shapes. No cross-method noise equivalence implied.',
        'uncertainty':f't intervals across {repetitions} Monte Carlo continuations at each fixed input, not uncertainty across training seeds.',
        'limitations':'One trained seed; checkpoint-specific behavior reference; finite rollout horizon with zero tail and potentially substantial bias; source epsilon penalty is not strict sampler KL at deterministic or clipped steps.'}
    (out/'probe_history.json').write_text(json.dumps(report,indent=2))
    plot_history(out)
    return report
