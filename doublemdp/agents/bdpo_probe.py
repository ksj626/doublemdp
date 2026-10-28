"""Frozen-policy continuation estimates matched to BDPO's nominal path cost.

Q0(s,a) = E[sum_j gamma^j r_j - eta sum_{j>=1} gamma^j c_j]
Vt(s,x) = E[-eta c_remaining(s,x) + Q0(s,a_final)]

Here c is the source epsilon-space path penalty. Due to deterministic terminal
noise and optional sample clipping, it is not always the strict sampler KL.
Only mean backups with no pessimism/max-Q match this rollout objective.
"""
import numpy as np


def regularized_maze_continuation(agent, states, latents, native_times, *,
                                  repetitions=16, horizon=150, seed=281,
                                  reward_mode='sparse', environment=None):
    """Estimate native-objective Vt/Q0 with Monte Carlo SE and truncation flags.

    `states`, `latents`, and `native_times` identify M fixed probe conditions.
    All remaining and subsequent generation is single-candidate actor_target;
    behavior_target and the environment reward transform are frozen as well.
    Terminal heads (native time 0) represent Q0 and have no current path cost.
    The horizon is a reporting cutoff, with zero tail estimate and possible bias.
    """
    c=agent.cfg.critic
    if c.rho != 0 or c.maxQ or c.q_target not in ('lcb','mean'):
        raise ValueError('Objective-matched rollout requires rho=0, maxQ=False, mean/zero-rho LCB backup')
    if agent._is_pretraining:
        raise ValueError('BDPO critics are not trained during behavior pretraining')
    if repetitions < 2 or horizon < 1:
        raise ValueError('At least two repetitions and a positive horizon are required')
    if reward_mode not in ('sparse','dense'):
        raise ValueError('Choose the same sparse/dense rewards used during training')
    from doublemdp.toy import PointMaze
    env=environment or PointMaze(gamma=c.discount,reward=reward_mode)
    states=np.atleast_2d(states).astype(np.float32)
    latents=np.atleast_2d(latents).astype(np.float32)
    times=np.broadcast_to(np.asarray(native_times,dtype=np.int32),len(states))
    if len(latents)!=len(states):
        raise ValueError('A latent is required for every physical probe state')
    count=len(states)
    positions=np.repeat(states,repetitions,axis=0)
    repeated_latents=np.repeat(latents,repetitions,axis=0)
    repeated_times=np.repeat(times,repetitions)
    rng=np.random.RandomState(seed)
    first_actions=np.zeros_like(repeated_latents)
    costs=np.zeros(len(positions))
    for time in np.unique(times):
        selected=repeated_times==time
        info=agent.continue_from_latent(positions[selected],repeated_latents[selected],
            int(time),int(rng.randint(2**31-1)),return_info=True)
        first_actions[selected]=np.asarray(info['actions'])
        costs[selected]=np.asarray(info['remaining_path_kl'])
    returns=-c.eta*costs
    discounted_rewards=np.zeros(len(positions))
    discounted_costs=costs.copy()
    active=np.ones(len(positions),bool)
    for step in range(horizon):
        if not active.any():
            break
        if step==0:
            actions=first_actions
        else:
            trace=agent.sample_trace(positions,seed=int(rng.randint(2**31-1)),num_samples=1)
            actions=np.asarray(trace['actions'])
            path_cost=np.asarray(trace['step_path_kl']).sum(axis=0)[:,0]
            discounted_costs+=c.discount**step*path_cost*active
            returns-=c.eta*c.discount**step*path_cost*active
        nxt,sparse,dense,done,_=env.transition(positions,actions)
        reward=sparse if reward_mode=='sparse' else dense
        contribution=c.discount**step*np.asarray(reward)*active
        returns+=contribution; discounted_rewards+=contribution
        positions=np.where(active[:,None],nxt,positions).astype(np.float32)
        active &= ~np.asarray(done)
    q=np.asarray(agent.q0_target(states,latents))[...,0]
    v=np.asarray(agent.vt_target(states,latents,times[:,None]))[...,0]
    native=np.where(times[None]==0,q,v)
    samples=returns.reshape(count,repetitions)
    return {'mean':samples.mean(-1),'se':samples.std(-1,ddof=1)/np.sqrt(repetitions),
        'samples':samples,'reward_samples':discounted_rewards.reshape(count,repetitions),
        'path_penalty_samples':discounted_costs.reshape(count,repetitions),
        'native_ensemble':native,'native_mean':native.mean(0),
        'native_minus_probe':native.mean(0)-samples.mean(-1),
        'discounted_reward_mean':discounted_rewards.reshape(count,repetitions).mean(-1),
        'discounted_path_penalty_mean':discounted_costs.reshape(count,repetitions).mean(-1),
        'unfinished_fraction':active.reshape(count,repetitions).mean(-1),
        'native_times':times,'states':states,'latents':latents,
        'policy':'frozen actor_target, one candidate; fixed behavior_target',
        'quantity':'nominal BDPO epsilon-path-cost regularized truncated continuation',
        'native_head':'target critic ensemble mean; t=0 Q0, t>0 Vt',
        'reward_mode':reward_mode,'eta':c.eta,'gamma':c.discount,'horizon':horizon,
        'caveat':'zero tail estimate; remaining active trajectories imply truncation bias; final deterministic/clipped transitions prevent interpreting all epsilon costs as strict sampler KL'}


def native_inner_td_residual(agent, states, latents, native_times, repetitions=32, seed=311):
    """Frozen native TD residual; sample means and ensemble means, no inner gamma."""
    import jax
    import jax.numpy as jnp
    from .bdpo import get_penalty
    cfg=agent.cfg.critic
    if cfg.rho!=0 or cfg.maxQ or cfg.q_target not in ('lcb','mean'):
        raise ValueError('Only mean/zero-rho LCB non-maxQ heads have this nominal objective')
    obs=jnp.asarray(states); x=jnp.asarray(latents)
    times=jnp.asarray(native_times,dtype=jnp.int32).reshape(-1,1)
    if not bool(jnp.all(times>0)):
        raise ValueError('Inner TD residual requires native_time>0; Q0 has an outer Bellman target')
    model=agent.actor_target
    _,next_x,_,next_t,_,eps=model.onestep_sample(jax.random.PRNGKey(seed),x,obs,
        False,repetitions,agent.cfg.diffusion.solver,False,times)
    repeated_obs=jnp.repeat(obs[:,None],repetitions,axis=1)
    next_values=jnp.where(next_t[None]==0,
        agent.q0_target(repeated_obs,next_x),agent.vt_target(repeated_obs,next_x,next_t))
    penalty=get_penalty(eps,agent.behavior_target(x,times,obs),times,model.steps,
                        model.alphas,model.alpha_hats,model.betas).sum(-1)
    samples=np.asarray(next_values.mean(axis=0)[...,0]-cfg.eta*penalty[:,None])
    current=np.asarray(agent.vt_target(obs,x,times).mean(axis=0)[:,0])
    return {'native_mean':current,'td_target_mean':samples.mean(-1),
            'td_target_se':samples.std(-1,ddof=1)/np.sqrt(repetitions),
            'td_residual':current-samples.mean(-1),'native_times':np.asarray(times[:,0])}


def main():
    import argparse,json
    from pathlib import Path
    from doublemdp.evaluate import load_run
    from doublemdp.toy import PointMaze
    from doublemdp.diagnostics import native_progress_indices
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=16)
    parser.add_argument('--horizon',type=int,default=150)
    parser.add_argument('--reward-mode',default='sparse',choices=['sparse','dense'])
    parser.add_argument('--checkpoint-steps',type=int,nargs='+',help='Compare saved snapshots on fixed final-checkpoint latents and noise')
    args=parser.parse_args()
    if args.checkpoint_steps:
        from .bdpo_probe_history import evaluate_history
        evaluate_history(args.run,args.checkpoint_steps,args.out,repetitions=args.repetitions,
                         horizon=args.horizon,reward_mode=args.reward_mode)
        return
    agent,config,metadata,checkpoint=load_run(args.run)
    states=PointMaze.probes
    trace=agent.sample_trace(states,seed=771,num_samples=1)
    indices=native_progress_indices(trace['progress'])
    latents=np.asarray(trace['latents'])[indices,:,0].swapaxes(0,1).reshape(-1,2)
    times=np.tile(np.asarray(trace['native_times'])[indices],len(states))
    repeated_states=np.repeat(states,len(indices),axis=0)
    results=regularized_maze_continuation(agent,repeated_states,latents,times,
        repetitions=args.repetitions,horizon=args.horizon,reward_mode=args.reward_mode)
    valid=times>0
    td=native_inner_td_residual(agent,repeated_states[valid],latents[valid],times[valid])
    args.out.mkdir(parents=True,exist_ok=True)
    arrays={key:value for key,value in results.items() if isinstance(value,np.ndarray)}
    arrays.update({'td_'+key:value for key,value in td.items()})
    np.savez_compressed(args.out/'regularized_probe.npz',**arrays)
    report={key:value for key,value in results.items() if not isinstance(value,np.ndarray)}
    report.update(checkpoint=str(args.run),training_step=checkpoint['step'],
        reward_transform=metadata.get('reward_key',args.reward_mode),
        native_vs_truncated_mc_rmse=float(np.sqrt(np.mean(results['native_minus_probe']**2))),
        native_inner_td_rmse=float(np.sqrt(np.mean(td['td_residual']**2))),
        max_unfinished_fraction=float(results['unfinished_fraction'].max()))
    (args.out/'regularized_probe.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15,3.5),sharey=True)
    for i,ax in enumerate(axes):
        block=slice(i*len(indices),(i+1)*len(indices))
        progress=np.asarray(trace['progress'])[indices]
        ax.plot(progress,results['native_mean'][block],'-o',label='Native target head')
        ax.errorbar(progress,results['mean'][block],yerr=1.96*results['se'][block],fmt='-s',label='Regularized MC (95% MC error)')
        ax.set(title=f'Probe {i}: {states[i]}',xlabel='Generation progress')
        for t,x,y in zip(times[block],progress,results['native_mean'][block]):
            ax.annotate(f't={t}',(x,y),fontsize=7)
    axes[0].set_ylabel('Nominal regularized continuation value');axes[-1].legend(fontsize=7)
    fig.suptitle(f'Single candidate | frozen checkpoint | {args.horizon}-step zero tail; truncation bias remains')
    fig.tight_layout();fig.savefig(args.out/'regularized_probe.png',dpi=160);fig.savefig(args.out/'regularized_probe.pdf')
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
