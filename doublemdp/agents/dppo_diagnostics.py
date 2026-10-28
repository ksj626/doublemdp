"""Evaluation-only DPPO single-candidate continuation with explicit noise tapes."""
import numpy as np
from ..toy import PointMaze,segments_hit_boxes


def noise_tape(seed,horizon,groups,steps):
    """Initial Gaussian plus every native transition draw at each physical step."""
    return np.random.RandomState(seed).normal(size=(horizon,steps+1,groups,1,2)).astype(np.float32)


def rollouts(agent,states,tape,*,first_actions=None,groups=None,gamma=None):
    """Unregularized discounted sparse return, zero tail at the recorded horizon.

    Policy = one candidate, native source evaluation sampler (deterministic=True,
    which remains stochastic before the final denoising step). Repeated group
    IDs reuse the exact future policy noise, enabling paired inner-time probes.
    Finished particles keep their noise allocation but execute no transitions.
    """
    if agent.config.horizon_steps!=1 or agent.config.act_steps!=1 or agent.action_dim!=2:
        raise ValueError("Maze diagnostic requires action chunk length one and two actions")
    gamma=agent.config.gamma if gamma is None else gamma
    positions=np.asarray(states,dtype=np.float32).copy()
    groups=np.arange(len(positions)) if groups is None else np.asarray(groups)
    valid=np.isfinite(positions).all(-1)&(positions>=0).all(-1)&(positions<=1).all(-1)
    valid &= ~segments_hit_boxes(positions,positions,PointMaze.boxes)
    terminal=np.linalg.norm(positions-PointMaze.goal,axis=-1)<.055
    active=valid&~terminal
    returns=np.zeros(len(positions))
    collisions=np.zeros(len(positions))
    counts=np.zeros(len(positions),dtype=int)
    paths=[positions.copy()]
    environment=PointMaze(gamma=gamma)
    for t,noise in enumerate(tape):
        if not active.any():
            break
        if t==0 and first_actions is not None:
            actions=np.asarray(first_actions)
        else:
            # Fixed batch/groups across stopping times and checkpoint comparisons.
            actions=agent.continue_generation(positions,noise[0,groups],agent.config.denoising_steps-1,
                deterministic=True,noise_tape=noise[1:,groups])[:,0]
        idx=np.flatnonzero(active)
        nxt,reward,_,done,collision=environment.transition(positions[idx],actions[idx])
        positions[idx]=nxt.astype(np.float32)  # exactly the scalar PointMaze.step cast
        returns[idx]+=gamma**t*reward
        collisions[idx]+=collision
        counts[idx]+=1
        active[idx]&=~done
        paths.append(positions.copy())
    returns[~valid]=np.nan
    return {"returns":returns,"unfinished":active,"collisions":collisions,"counts":counts,
        "valid":valid,"terminal":terminal,"paths":np.stack(paths)}


def summarize_samples(values):
    values=np.asarray(values)
    return {"mean":values.mean(-1),"se":values.std(-1,ddof=1)/np.sqrt(values.shape[-1]) if values.shape[-1]>1 else np.zeros(values.shape[:-1])}


def q_probe(agent,probes,trace,indices,remaining_noise,outer_noise):
    """Conditional return at a selected real latent; no native inner head needed.

    Each probe uses particle zero. Remaining generation and future environment
    draws are identical across native steps and checkpoints, with independent
    repetitions. No monotonicity claim or native-head calibration is implied.
    """
    repetitions=remaining_noise.shape[1]//len(probes)
    states=np.repeat(probes,repetitions,axis=0)
    endpoints=[]
    for index in indices:
        latents=np.repeat(trace["latents"][::trace["particles_per_probe"],index],repetitions,axis=0)
        endpoints.append(agent.continue_generation(states,latents,int(trace["native_times"][index]),
            deterministic=True,noise_tape=remaining_noise)[:,0])
    # Reuse future environment draws across every inner generation index.
    stacked_states=np.tile(states,(len(indices),1))
    result=rollouts(agent,stacked_states,outer_noise,first_actions=np.concatenate(endpoints),
        groups=np.tile(np.arange(len(states)),len(indices)))
    samples=result["returns"].reshape(len(indices),len(probes),repetitions).transpose(1,0,2)
    unfinished=result["unfinished"].reshape(len(indices),len(probes),repetitions).transpose(1,0,2)
    return {**summarize_samples(samples),"samples":samples,"unfinished":unfinished,
        "endpoints":np.stack(endpoints),"evaluation_steps":int(result["counts"].sum())}
