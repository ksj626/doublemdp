#!/usr/bin/env python
"""Final-checkpoint toy curves and held-out fragment-start/support diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from doublemdp.logging import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--output-root",default=str(Path.home()/"doublemdp_outputs"))
p.add_argument("--continuations",action="store_true")
args=p.parse_args(); root=Path(args.output_root); output=root/'figures'/'comparison'; output.mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
series={}
for method,color in [('rql','#3477b3'),('bdpo','#d77c32')]:
    for col,env in enumerate(['bandit','maze']):
        prefix='rql_maze_transition' if method=='rql' and env=='maze' else f'{method}_{env}'
        for seed in range(3):
            run=root/'runs'/f'{prefix}_seed{seed}'
            rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            evals=[r for r in rows if r['phase']=='evaluation']; field='reward_mean' if env=='bandit' else 'success'
            steps=np.array([r['step'] for r in evals]); wall=np.array([r['time']-rows[0]['time'] for r in evals]); score=np.array([r[field] for r in evals])
            series[f'{method}_{env}_{seed}_steps']=steps;series[f'{method}_{env}_{seed}_seconds']=wall;series[f'{method}_{env}_{seed}_score']=score
            axes[0,col].plot(steps,score,color=color,alpha=.6,label=method if seed==0 else None)
            axes[1,col].plot(wall,score,color=color,alpha=.6,label=method if seed==0 else None)
for col,env in enumerate(['bandit','maze']):
    for row,label in enumerate(['Training updates (includes BDPO pretraining)','Elapsed seconds (includes compilation/evaluation)']):
        axes[row,col].set(xlabel=label,ylabel='Reward' if env=='bandit' else 'Success fraction',title=env)
        axes[row,col].legend(); axes[row,col].grid(alpha=.2)
fig.suptitle('Declared toy configurations, three training seeds; not a matched-compute or reproduction ranking')
for ext in ('png','pdf'):fig.savefig(output/f'learning_curves.{ext}',dpi=150)
plt.close(fig); np.savez_compressed(output/'learning_curves.npz',**series)

if args.continuations:
    from scipy.spatial import cKDTree
    from doublemdp.evaluate import load_run,policy_actions
    from doublemdp.toy import PointMaze
    from doublemdp.data import OfflineDataset
    results=[]
    for method in ('rql','bdpo'):
        prefix='rql_maze_transition' if method=='rql' else 'bdpo_maze'
        for condition in ('','_dense','_gap'):
            run=root/'runs'/f'{prefix}{condition}_seed0'
            if not (run/'checkpoint.pkl').exists():continue
            agent,config,meta,checkpoint=load_run(run)
            ds=OfflineDataset.load(config['dataset'])
            env=PointMaze(); rng=np.random.RandomState(577)
            features=np.concatenate([ds.arrays['observations']/env.action_scale,ds.arrays['actions']],-1)
            tree=cKDTree(features)
            reference_dist=tree.query(features[::max(1,len(features)//2000)],k=2)[0][:,1]
            threshold=float(np.quantile(reference_dist,.95))
            probes=PointMaze.probes
            positions=np.repeat(probes,32,axis=0); alive=np.ones(len(positions),bool); success=np.zeros(len(positions),bool)
            distances=[]; traces=[positions.copy()]
            for step in range(150):
                actions=policy_actions(agent,positions,method,int(rng.randint(2**31-1)),single_candidate=True)[:,0]
                distances.extend(tree.query(np.concatenate([positions[alive]/env.action_scale,actions[alive]],-1))[0].tolist())
                nxt,_,_,terminal,_=env.transition(positions,actions)
                positions=np.where(alive[:,None],nxt,positions);success|=alive&terminal;alive&=~terminal;traces.append(positions.copy())
                if not alive.any():break
            record={'method':method,'condition':condition or 'sparse','run':str(run),'policy':'single_candidate',
                'probe_states':probes.tolist(),'success_by_start':success.reshape(len(probes),32).mean(-1).tolist(),
                'episodes_per_start':32,'horizon':150,'support_nn_distance_mean':float(np.mean(distances)),
                'empirical_nn_threshold':threshold,'fraction_beyond_threshold':float(np.mean(np.array(distances)>threshold)),
                'interpretation':'Held-out fragment-start continuation test; not proof of compositional stitching. Nearest-neighbor distance is an empirical support diagnostic, not an exact density.'}
            results.append(record)
            np.savez_compressed(output/f'{method}{condition}_fragment_continuations.npz',trajectories=np.array(traces),success=success,distances=distances)
    write_json(output/'fragment_start_support.json',results)
print(output)
