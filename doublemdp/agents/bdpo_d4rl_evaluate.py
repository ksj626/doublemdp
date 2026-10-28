"""Evaluate native BDPO through an isolated original Gym/D4RL process.

The learner stays in its JAX environment; only JSON observations/actions cross
stdin/stdout. Evaluations never update the policy or any training data.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def worker(task, episodes, seed):
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import d4rl
        # D4RL groups unrelated mjrl imports with locomotion registrations;
        # import the requested official family explicitly after optional failures.
        import d4rl.gym_mujoco
        import gym
        env=gym.make(task)
        env.seed(seed)
    returns=[]; normalized=[]; interactions=0
    for episode in range(episodes):
        obs=env.reset(); done=False; total=0.
        while not done:
            print(json.dumps({'kind':'observation','observation':obs.tolist(),
                              'episode':episode}),flush=True)
            message=json.loads(sys.stdin.readline())
            obs,reward,done,info=env.step(np.asarray(message['action'], dtype=np.float32))
            total+=float(reward); interactions+=1
        returns.append(total)
        normalized.append(float(env.get_normalized_score(total)*100))
    env.close()
    print(json.dumps({'kind':'results','returns':returns,
                      'normalized_scores':normalized,'evaluation_interactions':interactions}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--task',default='halfcheetah-medium-v2')
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--legacy-python',default='/home/dmsdmswns/miniforge3/envs/doublemdp-legacy/bin/python')
    parser.add_argument('--runtime',default='/home/dmsdmswns/doublemdp_outputs/runtime/mujoco210')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.task,args.episodes,args.seed); return
    if args.checkpoint is None or args.output is None:
        parser.error('--checkpoint and --output are required for evaluation')
    import jax
    import numpy as np
    from doublemdp.agents.bdpo import BDPOAgent
    with open(args.checkpoint,'rb') as stream:
        saved=pickle.load(stream)
    agent=BDPOAgent.from_state_dict(saved['agent'])
    environment=os.environ.copy()
    prefix=str(Path(args.legacy_python).resolve().parent.parent)
    environment.update(MUJOCO_PY_MUJOCO_PATH=args.runtime,
        LD_LIBRARY_PATH=args.runtime+'/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu',
        CPATH=prefix+'/include')
    command=[args.legacy_python,'-u',str(Path(__file__).resolve()),'--worker',
             '--task',args.task,'--episodes',str(args.episodes),'--seed',str(args.seed)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); inference=[]; step=0; result=None
    with open(args.output.with_suffix('.worker.log'),'w') as error_log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                 stderr=error_log,text=True,bufsize=1,env=environment)
        try:
            for line in process.stdout:
                message=json.loads(line)
                if message['kind']=='results':
                    result=message; break
                obs=np.asarray(message['observation'],np.float32)
                begin=time.perf_counter()
                action=agent.sample_actions(obs,seed=args.seed*1000000+10000+step)
                jax.block_until_ready(action)
                inference.append(time.perf_counter()-begin)
                if not np.isfinite(action).all():
                    raise FloatingPointError('Nonfinite evaluation action')
                process.stdin.write(json.dumps({'action':np.asarray(action).tolist()})+'\n')
                process.stdin.flush(); step+=1
            process.wait(timeout=30)
            if process.returncode or result is None:
                raise RuntimeError(f'Legacy environment failed: exit={process.returncode}; see worker log')
        finally:
            if process.poll() is None:
                process.terminate()
    result.update(task=args.task,seed=args.seed,episodes=args.episodes,
        profile=agent.config_dict['profile'],checkpoint=str(args.checkpoint),
        checkpoint_selection='final requested-budget checkpoint',
        pretrain_steps=agent._n_pretraining_steps,train_steps=agent._n_training_steps,
        num_candidates=agent.eval_num_samples,diffusion_steps=agent.cfg.diffusion.steps,
        training_environment_interactions=0,policy_updates_during_evaluation=0,
        mean_normalized_score=float(np.mean(result['normalized_scores'])),
        episode_std_normalized_score=float(np.std(result['normalized_scores'],ddof=1)),
        training_seed_uncertainty='unavailable: one trained seed; episodes are not independent trained agents',
        elapsed_seconds=time.monotonic()-start,
        first_inference_seconds=inference[0],
        steady_inference_mean_seconds=float(np.mean(inference[1:])),
        device=str(jax.devices()[0]))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
