"""Measure native RQL compute at explicit observation/action dimensions.

Synthetic batches measure compute only and are never reported as task scores.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import numpy as np
from doublemdp.agents.rql import RQLAgent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--observations',type=int,required=True)
    parser.add_argument('--actions',type=int,required=True)
    parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())['agent']
    b,h=config['batch_size'],config['h']
    rng=np.random.RandomState(0)
    batch=dict(observations=rng.normal(size=(h+1,b,args.observations)).astype('float32'),
               actions=np.tanh(rng.normal(size=(h+1,b,args.actions))).astype('float32'),
               rewards=-np.ones((h+1,b),dtype='float32'),terminals=np.zeros((h+1,b),dtype='float32'),
               masks=np.ones((h+1,b),dtype='float32'))
    agent=RQLAgent.create(0,batch['observations'][0],batch['actions'][0],config)
    start=time.perf_counter()
    agent,metrics=agent.update(batch)
    jax.block_until_ready(metrics)
    compile_and_first=time.perf_counter()-start
    for _ in range(5):
        agent,metrics=agent.update(batch)
    jax.block_until_ready(metrics)
    start=time.perf_counter()
    for _ in range(args.updates):
        agent,metrics=agent.update(batch)
    jax.block_until_ready(metrics)
    steady=(time.perf_counter()-start)/args.updates
    obs=batch['observations'][0,0]
    key=jax.random.PRNGKey(0)
    action=agent.sample_actions(obs,key)
    jax.block_until_ready(action)
    start=time.perf_counter()
    for i in range(100):
        action=agent.sample_actions(obs,jax.random.fold_in(key,i))
    jax.block_until_ready(action)
    inference=(time.perf_counter()-start)/100
    stats=jax.devices()[0].memory_stats()
    report=dict(kind='synthetic_shape_compute_only',device=str(jax.devices()[0]),
                config=config,observation_dim=args.observations,action_dim=args.actions,
                first_compile_and_update_seconds=compile_and_first,steady_update_seconds=steady,
                timed_updates=args.updates,sample_chunk_seconds=inference,candidate_count=1,
                inference_actor_evaluations=config['flow_steps'],
                parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(agent.network.params)),
                device_memory_stats=stats,estimated_1m_training_hours=steady*1_000_000/3600,
                estimate_excludes_environment_evaluation=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
