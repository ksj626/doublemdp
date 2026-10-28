"""Fixed-noise RQL continuation history, with native heads shown separately.

The reference is the single-candidate EMA policy's unregularized, truncated
Q^pi continuation. Native expectile-head errors against this reference are
explicitly incomparable and are not reported as calibration errors.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np


def collect(run,output,steps=(1000,5000,10000),repetitions=16,horizon=100,seed=731):
    import jax
    from doublemdp.evaluate import load_run
    from doublemdp.diagnostics import native_progress_indices,rollout_continuations
    from doublemdp.toy import PointMaze
    from doublemdp.logging import write_json
    if repetitions<2 or horizon<1:
        raise ValueError('Need at least two Monte Carlo repetitions and a positive horizon')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    probes=PointMaze.probes.copy()
    rng=np.random.RandomState(seed)
    initial_noise=rng.normal(size=(len(probes),2)).astype('float32')
    rollout_seeds=rng.randint(2**31-1,size=len(probes))
    policy_seed_tapes=np.stack([np.random.RandomState(int(s)).randint(2**31-1,size=horizon-1) for s in rollout_seeds])
    history=[];traces=[];heads=[];endpoints=[];returns=[];unfinished=[]
    checkpoint_identities=[]
    for step in steps:
        path=Path(run)/f'checkpoint_{int(step):09d}.pkl'
        agent,config,_,checkpoint=load_run(path)
        if (config['method']!='rql' or config['env']!='maze' or int(agent.config['h'])!=1
                or config.get('rql_sequence_adapter')!='transition_h1'):
            raise ValueError('Use corrected h=1 RQL point-maze transition_h1 checkpoints')
        if checkpoint['step']!=step:
            raise ValueError('Checkpoint filename and saved update counter differ')
        gamma=float(agent.config['discount'])
        checkpoint_identities.append(checkpoint['artifact_identity'])
        step_traces=[];step_heads=[];step_endpoints=[];step_returns=[];step_unfinished=[]
        for probe_id,probe in enumerate(probes):
            trace=agent.sample_trace(probe,jax.random.PRNGKey(seed),noise=initial_noise[probe_id:probe_id+1])
            indices=native_progress_indices(trace['progress'])
            native_times=np.asarray(trace['native_times'])
            probe_endpoints=[];probe_returns=[];probe_unfinished=[]
            observations=np.repeat(probe[None],repetitions,axis=0)
            for native_step in indices:
                latent=np.repeat(np.asarray(trace['latents'])[native_step,0][None],repetitions,axis=0)
                final=np.asarray(agent.continue_flow(observations,latent,int(native_step)))[:,0]
                # The same seed tape is reused for every inner step and checkpoint.
                # This controls simulation noise; it is not a training-seed CI.
                result=rollout_continuations(agent,'rql',observations,final,repetitions=1,
                                             horizon=horizon,gamma=gamma,seed=int(rollout_seeds[probe_id]))
                probe_endpoints.append(final)
                probe_returns.append(result['mean'])
                probe_unfinished.append(result['unfinished_fraction'])
            probe_endpoints=np.stack(probe_endpoints)
            deviation=float(np.max(np.abs(probe_endpoints-np.asarray(trace['final_actions'])[0,0])))
            if deviation>5e-5:
                raise AssertionError(f'Native continuation changed its deterministic endpoint by {deviation}')
            samples=np.stack(probe_returns)
            native=np.asarray(trace['native_values'])[indices,:,0]
            means=samples.mean(-1);ses=samples.std(-1,ddof=1)/np.sqrt(repetitions)
            history.append({'checkpoint_step':int(step),'probe_id':probe_id,'state':probe.tolist(),
                            'native_steps':indices.tolist(),'native_times':native_times[indices].tolist(),
                            'mc_mean':means.tolist(),'mc_se':ses.tolist(),
                            'unfinished_fraction':np.stack(probe_unfinished).mean(-1).tolist(),
                            'native_head_mean':native.mean(-1).tolist(),
                            'native_ensemble_std':native.std(-1).tolist(),
                            'endpoint_max_numerical_deviation':deviation})
            step_traces.append(np.asarray(trace['latents'])[:,0]);step_heads.append(native)
            step_endpoints.append(probe_endpoints);step_returns.append(samples)
            step_unfinished.append(np.stack(probe_unfinished))
        traces.append(step_traces);heads.append(step_heads);endpoints.append(step_endpoints)
        returns.append(step_returns);unfinished.append(step_unfinished)
    np.savez_compressed(output/'raw_history.npz',checkpoint_steps=np.asarray(steps),probe_states=probes,
                        initial_noise=initial_noise,rollout_seeds=rollout_seeds,future_policy_seed_tapes=policy_seed_tapes,
                        native_indices=indices,native_times=native_times,native_latents=np.asarray(traces),
                        native_heads=np.asarray(heads),decoded_continuations=np.asarray(endpoints),
                        mc_returns=np.asarray(returns),unfinished=np.asarray(unfinished))
    report={'method':'rql','run':str(Path(run).resolve()),'checkpoint_steps':list(steps),
            'checkpoints':checkpoint_identities,'training_seed':config['seed'],'diagnostic_seed':seed,
            'policy':'single-candidate EMA actor, h=1; all future environment decisions use that same policy',
            'quantity':'unregularized finite-rollout q_probe, with zero tail estimate',
            'horizon':horizon,'gamma':gamma,'repetitions':repetitions,
            'mc_intervals':'mean +/- 1.96 MC SE; conditional simulation uncertainty only, with possible truncation bias',
            'noise_protocol':'identical initial Gaussian samples and future-policy RNG tapes across inner steps/checkpoints',
            'native_comparison':{'status':'incomparable','error':None,
                'reason':'Native online expectile/reversal heads are not equated to unregularized EMA-policy Monte Carlo returns; ensemble spread is not a confidence interval.'},
            'rows':history,'device':str(jax.devices()[0])}
    write_json(output/'report.json',report)
    return report


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    report=json.loads((output/'report.json').read_text())
    with np.load(output/'raw_history.npz') as data:
        returns=data['mc_returns'];heads=data['native_heads'];probes=data['probe_states']
        times=data['native_times'][data['native_indices']];steps=data['checkpoint_steps']
    means=returns.mean(-1);ses=returns.std(-1,ddof=1)/np.sqrt(returns.shape[-1])
    native_means=heads.mean(-1);spread=heads.std(-1)
    fig,axes=plt.subplots(len(probes),2,figsize=(12,3*len(probes)),layout='constrained')
    labels=['fork','shortcut entrance','detour','near goal']
    for i,probe in enumerate(probes):
        for c,step in enumerate(steps):
            axes[i,0].errorbar(times,means[c,i],yerr=1.96*ses[c,i],marker='o',capsize=3,label=f'update {step}')
            line=axes[i,1].plot(times,native_means[c,i],marker='o',label=f'update {step}')[0]
            axes[i,1].fill_between(times,native_means[c,i]-spread[c,i],native_means[c,i]+spread[c,i],
                                   color=line.get_color(),alpha=.12)
        axes[i,0].set(title=f'{labels[i]} {tuple(np.round(probe,2))}: unregularized continuation',
                      xlabel='native generation time',ylabel='truncated q_probe')
        axes[i,1].set(title='Native expectile head (separate objective)',xlabel='native generation time',
                      ylabel='native value')
        axes[i,0].legend(fontsize=8);axes[i,1].legend(fontsize=8)
    fig.suptitle(f"RQL continuation history: fixed probe/noise tapes, {report['repetitions']} rollouts, horizon {report['horizon']}\n"
                 'MC bars: +/- 1.96 SE; native shading: ensemble std (not a confidence interval)\n'
                 'Native-minus-MC calibration error is incomparable and is not reported',fontsize=12)
    fig.savefig(output/'continuation_history.png',dpi=150)
    fig.savefig(output/'continuation_history.pdf')
    plt.close(fig)
    page="<!doctype html><meta charset='utf-8'><title>RQL continuation history</title>"
    page+="<h1>Fixed-noise continuation across training checkpoints</h1>"
    page+="<p>Single-candidate EMA policy; unregularized truncated Monte Carlo continuation and native expectile heads are separate quantities.</p>"
    page+="<img src='continuation_history.png' style='max-width:1200px;width:100%'><p><a href='continuation_history.pdf'>PDF</a> | <a href='raw_history.npz'>Raw samples and noise tapes</a> | <a href='report.json'>Protocol and snapshot identities</a></p>"
    (output/'index.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',nargs='?');p.add_argument('--output',required=True)
    p.add_argument('--steps',nargs='+',type=int,default=[1000,5000,10000])
    p.add_argument('--repetitions',type=int,default=16);p.add_argument('--horizon',type=int,default=100)
    p.add_argument('--seed',type=int,default=731);p.add_argument('--regenerate',action='store_true')
    args=p.parse_args()
    if not args.regenerate:
        if not args.run:p.error('run required unless --regenerate')
        collect(args.run,args.output,args.steps,args.repetitions,args.horizon,args.seed)
    render(args.output)
    print(Path(args.output)/'index.html')

if __name__=='__main__':main()
