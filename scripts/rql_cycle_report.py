"""Measure trained RQL finite-Euler cycles at fixed saved dataset endpoints.

This is evaluation only. It uses the online training actor (not the EMA actor)
for both directions, with no clipping. It does not assert exact invertibility.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import numpy as np
from doublemdp.evaluate import load_run
from doublemdp.data import OfflineDataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=256)
    parser.add_argument('--seed',type=int,default=731)
    args=parser.parse_args()
    run=Path(args.run)
    files=sorted(run.glob('checkpoint_[0-9]*.pkl')) or [run/'checkpoint.pkl']
    agent,config,metadata,_=load_run(files[-1])
    if config['method']!='rql' or int(agent.config['h'])!=1 or not config.get('dataset'):
        raise ValueError('This diagnostic currently requires an h=1 RQL saved toy dataset')
    dataset=OfflineDataset.load(config['dataset'],reward=config.get('reward','sparse'))
    rng=np.random.RandomState(args.seed)
    indices=rng.randint(len(dataset),size=args.count)
    observations=dataset.arrays['observations'][indices]
    endpoints=dataset.arrays['actions'][indices]
    steps=int(agent.config['flow_steps'])
    durations=np.r_[rng.uniform(size=args.count//2),rng.randint(steps+1,size=args.count-args.count//2)/steps].astype('float32')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    np.savez(output/'fixed_inputs.npz',indices=indices,observations=observations,endpoints=endpoints,durations=durations)
    rows=[]
    raw={}
    for path in files:
        agent,_,_,checkpoint=load_run(path)
        row={'step':checkpoint['step'],'checkpoint':str(path)}
        for name,duration in [('full',1.),('source_duration_mixture',durations)]:
            cycle=jax.device_get(agent.reverse_cycle(observations,endpoints,duration))
            row[name+'_rmse']=float(cycle['rmse'])
            row[name+'_max_abs']=float(cycle['max_abs'])
            raw[f"step{checkpoint['step']}_{name}_reconstructed"]=np.asarray(cycle['reconstructed'])
            raw[f"step{checkpoint['step']}_{name}_cycled"]=np.asarray(cycle['cycled'])
            raw[f"step{checkpoint['step']}_{name}_residual"]=np.asarray(cycle['residual'])
        rows.append(row)
    np.savez_compressed(output/'cycles.npz',**raw)
    report={'run':str(run),'profile':config['profile'],'seed':args.seed,'count':args.count,
            'device':str(jax.devices()[0]),'endpoint_sampling':'uniform saved transitions; fixed across checkpoints',
            'actor':'online actor for both directions; no clipping','rows':rows}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for name in ['full','source_duration_mixture']:
        ax.plot([r['step'] for r in rows],[r[name+'_rmse'] for r in rows],marker='o',label=name)
    ax.set(xlabel='training updates',ylabel='reverse-forward RMSE',title='Finite Euler cycle residual at saved actions')
    ax.legend()
    fig.savefig(output/'cycle_error.png',dpi=160);fig.savefig(output/'cycle_error.pdf')
    plt.close(fig)
    print(json.dumps(rows[-1],indent=2))

if __name__=='__main__':
    main()
