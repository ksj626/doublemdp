"""Evaluation-only comparison of actual generated and RQL training latents.

Distances pool the same saved-state cohort; they are descriptive marginal
statistics, not state-conditional calibration or path correspondence claims.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
from doublemdp.data import OfflineDataset
from doublemdp.evaluate import load_run
from doublemdp.diagnostics import native_progress_indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run');p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=256);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args()
    agent,config,_,checkpoint=load_run(args.run)
    if config['method']!='rql' or int(agent.config['h'])!=1 or not config.get('dataset'):
        raise ValueError('Requires h=1 RQL and a local saved toy dataset')
    dataset=OfflineDataset.load(config['dataset'],reward=config.get('reward','sparse'))
    rng=np.random.RandomState(args.seed)
    chosen=rng.randint(len(dataset),size=args.count)
    obs=dataset.arrays['observations'][chosen];actions=dataset.arrays['actions'][chosen]
    noise=rng.normal(size=actions.shape).astype('float32')
    online=agent.sample_trace(obs,jax.random.PRNGKey(args.seed),num_samples=args.count,temperature=1.,noise=noise)
    ema=agent.sample_trace(obs,jax.random.PRNGKey(args.seed),num_samples=args.count,temperature=0.,noise=noise)
    indices=native_progress_indices(online['progress'])
    times=np.asarray(online['native_times'])[indices]
    latent={'generated_online':np.asarray(online['latents'])[indices],
            'generated_ema':np.asarray(ema['latents'])[indices],
            'reconstructed':np.stack([np.asarray(agent.reverse_cycle(obs,actions,1.-time)['reconstructed']) for time in times]),
            'bc_interpolation':np.stack([(1.-time)*noise+time*actions for time in times])}
    directions=rng.normal(size=(32,actions.shape[-1]));directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
    rows=[];native={}
    for name,states in latent.items():
        native[name]=[]
        for k,time in enumerate(times):
            query=jnp.concatenate([jnp.asarray(obs),jnp.asarray(states[k]),jnp.full((len(obs),1),time)],-1)
            native[name].append(np.asarray(agent.network.select('value')(query)))
            a=np.sort(states[k]@directions.T,axis=0)
            b=np.sort(latent['generated_online'][k]@directions.T,axis=0)
            rows.append({'population':name,'native_time':float(time),
                         'sliced_wasserstein_2_vs_generated_online':float(np.sqrt(np.mean((a-b)**2))),
                         'native_head_mean':float(native[name][-1].mean())})
        native[name]=np.stack(native[name])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'raw_latents.npz',indices=chosen,observations=obs,endpoints=actions,noise=noise,
                        native_times=times,directions=directions,**latent,**{k+'_native_value':v for k,v in native.items()})
    report={'checkpoint_step':checkpoint['step'],'run':args.run,'count':args.count,'seed':args.seed,
            'claim':'Pooled same-state marginal latent shift; no pairing/conditional calibration/optimality claim',
            'actor':'Reversal uses online actor; actual native generated states saved for online and EMA actors',
            'native_head':'Online expectile/pessimistic value ensemble, explicitly not unregularized MC return','rows':rows}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(times),figsize=(15,6),layout='constrained')
    for k,time in enumerate(times):
        for row,name in enumerate(['reconstructed','bc_interpolation']):
            ax=axes[row,k]
            ax.scatter(*latent['generated_online'][k,:,:2].T,s=8,alpha=.5,label='Actual online generation')
            ax.scatter(*latent[name][k,:,:2].T,s=8,alpha=.4,label=name.replace('_',' '))
            lim=max(1.,float(np.abs(latent[name][k]).max()),float(np.abs(latent['generated_online'][k]).max()))*1.05
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),title=f'Native time {time:.2f}',xlabel='action latent x',ylabel='action latent y')
            if k==0:ax.legend(fontsize=7)
    fig.suptitle('Training latents and actual generated latents at the same saved-state cohort (unclipped)')
    fig.savefig(output/'latent_shift.png',dpi=150);fig.savefig(output/'latent_shift.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in latent:
        records=[r for r in rows if r['population']==name]
        axes[0].plot(times,[r['sliced_wasserstein_2_vs_generated_online'] for r in records],marker='o',label=name)
        axes[1].plot(times,[r['native_head_mean'] for r in records],marker='o',label=name)
    axes[0].set(xlabel='native time',ylabel='pooled sliced Wasserstein-2',title='Marginal shift from actual online generation')
    axes[1].set(xlabel='native time',ylabel='native expectile-head mean',title='Native objective output; not MC return calibration')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
    fig.savefig(output/'native_head_and_shift.png',dpi=150);fig.savefig(output/'native_head_and_shift.pdf');plt.close(fig)
    print(output)

if __name__=='__main__':main()
