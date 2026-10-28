"""Resumable original-task BDPO execution, without a legacy D4RL import dependency.

python -m doublemdp.agents.bdpo_experiment --task 8gaussians --profile toy_paper_eta --out /external/run
python -m doublemdp.agents.bdpo_experiment --task halfcheetah-medium-v2 --data-path /external/raw.hdf5 --out /external/run
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

from .bdpo_data import ScanSampler, generate_8gaussians, load_d4rl_hdf5, file_sha256


def atomic_pickle(path, payload):
    temporary = path.with_suffix('.tmp')
    with open(temporary,'wb') as stream:
        pickle.dump(payload,stream,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def evaluate_toy(agent, out, count=8192):
    """Evaluation-only oracle and complete native trace/plot artifact export."""
    import jax
    from .bdpo_data import reward_8gaussians, density_8gaussians, centers_8gaussians
    actions=np.concatenate([np.asarray(agent.sample_actions(np.zeros((256,1),np.float32),seed=50000+i,num_samples=1)) for i in range((count+255)//256)])[:count]
    if not np.isfinite(actions).all():
        raise FloatingPointError('Non-finite policy samples; evaluation is invalid')
    trace=agent.sample_trace(np.zeros((128,1),np.float32),seed=2029,num_samples=1)
    arrays={key:np.asarray(value) for key,value in trace.items() if not isinstance(value,str)}
    np.savez_compressed(out/'traces.npz',**arrays,observations=np.zeros((128,1),np.float32),seed=np.array(2029))
    np.save(out/'policy_samples.npy',actions)
    modes=((actions[:,None]-centers_8gaussians())**2).sum(-1).argmin(-1)
    mass=np.bincount(modes,minlength=8)/len(modes)
    x=np.linspace(-5,5,161); mid=(x[:-1]+x[1:])/2
    xx,yy=np.meshgrid(mid,mid,indexing='ij'); grid=np.stack([xx,yy],-1)
    beta=density_8gaussians(grid); reward=reward_8gaussians(grid)
    eta=agent.cfg.critic.eta
    reference=beta*np.exp((reward-1)/eta); reference/=reference.sum()
    empirical=np.histogram2d(actions[:,0],actions[:,1],bins=(x,x))[0]
    empirical/=max(empirical.sum(),1)
    mix=(reference+empirical)/2
    def kl(p,q):
        keep=p>0
        return float((p[keep]*np.log(p[keep]/q[keep])).sum())
    metrics={'reward_mean':float(reward_8gaussians(actions).mean()),
             'mode_mass':mass.tolist(),'grid_JS':.5*(kl(reference,mix)+kl(empirical,mix)),
             'outside_grid_mass':float((np.abs(actions)>5).any(-1).mean()),
             'sample_count':len(actions),'selection':'single_candidate',
             'reference':'analytic behavior density times exp(reward/eta), quadrature grid [-5,5]^2 160x160',
             'eta':eta}
    (out/'toy_evaluation.json').write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(out/'plot_inputs.npz',grid=grid,behavior_density=beta,reward=reward,
                        reference_mass=reference,policy_mass=empirical,mode_mass=mass)
    from .bdpo_artifacts import render_toy_artifacts
    render_toy_artifacts(out, training_updates=agent._n_training_steps)
    return metrics


def validate_resume(saved, previous, *, requested_config, task, dataset, seed):
    """Validate the scientific identity before restoring any mutable run state."""
    experiment=saved.get('experiment',previous)
    if not experiment.get('task') or not experiment.get('dataset'):
        raise ValueError('Checkpoint lacks experiment identity; keep its original run.json alongside it')
    if task!=experiment['task']:
        raise ValueError('Cannot resume with a different task')
    # Dataset files may move; contents and preprocessing must be identical.
    identity=lambda value:{k:v for k,v in value.items() if k!='path'}
    if identity(dataset)!=identity(experiment['dataset']):
        raise ValueError('Cannot resume with different dataset contents or preprocessing')
    if seed!=experiment['seed']:
        raise ValueError('Cannot relabel the training seed on resume')
    if requested_config is not None and requested_config!=saved['agent']['config']:
        raise ValueError('Cannot resume with a different algorithm configuration or budget')


def resume_elapsed(saved, previous, metrics_path):
    """Recover accumulated active training time, excluding interruption downtime."""
    if 'elapsed_seconds' in saved:
        return float(saved['elapsed_seconds'])
    # Migration for checkpoints created by this runner before elapsed time was
    # embedded. Use only log rows at or before the saved optimizer counters.
    wanted=saved['agent']['pretrain_counter']+saved['agent']['train_counter']
    elapsed=None
    if metrics_path.exists():
        with metrics_path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['step']<=wanted:
                    elapsed=max(float(row['elapsed_seconds']),elapsed or 0.)
    return float(previous.get('elapsed_seconds',0.) if elapsed is None else elapsed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',default='8gaussians')
    parser.add_argument('--profile',default=None)
    parser.add_argument('--config')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data-path',type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--dataset-seed',type=int,default=0)
    parser.add_argument('--data-size',type=int,default=1000000)
    parser.add_argument('--pretrain-steps',type=int)
    parser.add_argument('--train-steps',type=int)
    parser.add_argument('--checkpoint-interval',type=int,default=10000)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--max-seconds',type=float)
    args=parser.parse_args()
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    from .bdpo import BDPOAgent, default_config
    args.out.mkdir(parents=True,exist_ok=True)
    checkpoint=args.out/'checkpoint.pkl'
    previous={}
    saved=None
    if args.resume or args.eval_only:
        with checkpoint.open('rb') as stream:
            saved=pickle.load(stream)
        if (args.out/'run.json').exists():
            previous=json.loads((args.out/'run.json').read_text())
    elif checkpoint.exists():
        raise FileExistsError('Run already has a checkpoint; use --resume or another output path')
    is_toy=args.task=='8gaussians'
    profile=args.profile or ('toy_paper_eta' if is_toy else 'official_code')
    cfg=json.loads(Path(args.config).read_text()) if args.config else default_config(profile,args.task)
    cfg=cfg.get('agent_config',cfg)
    for name in ('pretrain_steps','train_steps'):
        if getattr(args,name) is not None:
            cfg[name]=getattr(args,name)
    if is_toy:
        dataset_path=args.data_path or args.out.parent/f'8gaussians-{args.data_size}-seed{args.dataset_seed}.npz'
        if not dataset_path.exists():
            np.savez_compressed(dataset_path,**generate_8gaussians(args.data_size,args.dataset_seed))
        with np.load(dataset_path) as stored:
            dataset={key:np.array(stored[key]) for key in stored.files}
        data_metadata={'sha256':file_sha256(dataset_path),'size':len(dataset['actions']),
                       'seed':args.dataset_seed,'task':'8gaussians','path':str(dataset_path),
                       'sampling':'Fixed saved dataset. Scan-with-reshuffle, upstream >= batch boundary rule.'}
    else:
        if args.data_path is None:
            raise ValueError('--data-path must name the original D4RL HDF5')
        dataset,data_metadata=load_d4rl_hdf5(args.data_path,
            norm_reward='antmaze100' if args.task.startswith('antmaze') else 'iql_mujoco')
        data_metadata['path']=str(args.data_path)
    for value in dataset.values():
        value.setflags(write=False)
    sampler=ScanSampler(dataset,args.seed)
    elapsed_before=0.
    if saved is not None:
        explicit_config=args.config or args.profile or args.pretrain_steps is not None or args.train_steps is not None
        validate_resume(saved,previous,requested_config=cfg if explicit_config else None,
                        task=args.task,dataset=data_metadata,seed=args.seed)
        elapsed_before=resume_elapsed(saved,previous,args.out/'metrics.jsonl')
        agent=BDPOAgent.from_state_dict(saved['agent'])
        sampler.load_state_dict(saved['sampler']); cfg=agent.config_dict
    else:
        agent=BDPOAgent.create(args.seed,dataset['observations'][:1],dataset['actions'][:1],cfg)
    metadata={'task':args.task,'seed':args.seed,'config':cfg,'dataset':data_metadata,
              'device':str(jax.devices()[0]),'pid':os.getpid(),'status':'running',
              'resume_count':int(previous.get('resume_count',0))+(saved is not None),
              'elapsed_seconds':elapsed_before,
              'online_training_interactions':0,'oracle_training_queries':0,
              'commit':'bae22f4cad3f9ce0a220aceb877b0e979de89808',
              'budget_label':'explicit configuration; shorter overrides are sanity runs, not original-paper reproduction'}
    (args.out/'run.json').write_text(json.dumps(metadata,indent=2))
    start=time.monotonic(); durations={}; interval=args.checkpoint_interval
    def total_elapsed():
        return elapsed_before+time.monotonic()-start
    def save_progress():
        atomic_pickle(checkpoint,{'agent':agent.state_dict(),'sampler':sampler.state_dict(),
            'experiment':{'task':args.task,'seed':args.seed,'dataset':data_metadata},
            'elapsed_seconds':total_elapsed(),'resume_count':metadata['resume_count']})
    try:
        if not args.eval_only:
            with open(args.out/'metrics.jsonl','a',buffering=1) as log:
                while agent._is_pretraining or agent._n_training_steps < cfg['train_steps']:
                    phase='pretrain' if agent._is_pretraining else 'train'
                    t=time.perf_counter()
                    _,metrics=agent.update(sampler.sample(cfg['batch_size']))
                    jax.block_until_ready(metrics)
                    if not all(np.isfinite(np.asarray(value)).all() for value in metrics.values()):
                        raise FloatingPointError(f'Non-finite BDPO metrics at pretrain={agent._n_pretraining_steps}, train={agent._n_training_steps}')
                    elapsed=time.perf_counter()-t
                    key=phase+('_actor' if 'loss/actor_loss' in metrics else '')
                    durations.setdefault(key,[]).append(elapsed)
                    step=agent._n_pretraining_steps+agent._n_training_steps
                    if step%100==0 or len(durations[key])==1:
                        entry={'step':step,'pretrain_steps':agent._n_pretraining_steps,
                               'train_steps':agent._n_training_steps,'phase':phase,
                               'elapsed_seconds':total_elapsed(),
                               'update_seconds':elapsed,
                               **{k:float(v) for k,v in jax.device_get(metrics).items()}}
                        log.write(json.dumps(entry)+'\n'); print(json.dumps(entry),flush=True)
                    if step%interval==0:
                        save_progress()
                        metadata.update(completed_pretrain_steps=agent._n_pretraining_steps,
                            completed_train_steps=agent._n_training_steps,elapsed_seconds=total_elapsed())
                        (args.out/'run.json').write_text(json.dumps(metadata,indent=2))
                    if args.max_seconds and time.monotonic()-start >= args.max_seconds:
                        break
        save_progress()
        finished=not agent._is_pretraining and agent._n_training_steps>=cfg['train_steps']
        metadata['status']='complete' if finished else 'pending'
        metadata['completed_pretrain_steps']=agent._n_pretraining_steps
        metadata['completed_train_steps']=agent._n_training_steps
        metadata['timing']={key:{'first_call_seconds':values[0],
             'steady_mean_seconds':float(np.mean(values[1:])) if len(values)>1 else None,
             'measured_updates':len(values)} for key,values in durations.items()}
        metadata['elapsed_seconds']=total_elapsed()
        metadata['device_memory_stats']=jax.devices()[0].memory_stats()
        if is_toy:
            metadata['evaluation']=evaluate_toy(agent,args.out)
        else:
            # This is a policy inference/finite-update sanity check. Original
            # D4RL environment evaluation must be recorded separately.
            obs=dataset['observations'][:64]
            t=time.perf_counter(); actions=agent.sample_actions(obs,seed=999)
            jax.block_until_ready(actions)
            np.save(args.out/'sanity_actions.npy',np.asarray(actions))
            metadata['evaluation_status']='Original D4RL simulator evaluation not yet run; offline update/inference only.'
            metadata['inference_seconds_batch64']=time.perf_counter()-t
            metadata['actions_finite']=bool(np.isfinite(actions).all())
    except Exception as error:
        metadata.update(status='failed',error=repr(error))
        raise
    finally:
        (args.out/'run.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
