"""Evaluation references with explicit objectives and Monte Carlo uncertainty."""
from __future__ import annotations
import numpy as np
from .toy import PointMaze, segments_hit_boxes


def native_progress_indices(progress, requested=(0,.25,.5,.75,1)):
    """Choose actual native sampler states; never fabricate interpolated states."""
    progress=np.asarray(progress)
    return np.array([np.argmin(np.abs(progress-z)) for z in requested])


def grid_dynamic_programming(resolution=41, gamma=.99, tolerance=1e-9):
    """Nearest-grid state projection and nine actions: approximate continuous oracle."""
    env=PointMaze(gamma=gamma)
    axis=np.linspace(0,1,resolution)
    x,y=np.meshgrid(axis,axis)
    states=np.stack([x.ravel(),y.ravel()],-1)
    u,v=np.meshgrid(np.linspace(-1,1,3),np.linspace(-1,1,3))
    actions=np.stack([u.ravel(),v.ravel()],-1)
    valid=~segments_hit_boxes(states,states,env.boxes)
    nxt,reward,_,done,collision=env.transition(states[:,None,:], actions[None,:,:])
    ij=np.clip(np.rint(nxt*(resolution-1)).astype(int),0,resolution-1)
    next_indices=ij[...,1]*resolution+ij[...,0]
    allowed=valid[next_indices]
    absorbing=np.linalg.norm(states-env.goal,axis=-1)<.055
    values=np.zeros(len(states))
    residual=np.inf
    for iterations in range(10000):
        qs=reward+gamma*(~done)*values[next_indices]
        qs=np.where(allowed,qs,-np.inf)
        new=np.max(qs,axis=1)
        new[~valid|absorbing]=0
        residual=float(np.max(np.abs(new-values)))
        values=new
        if residual<tolerance:
            break
    return {"axis":axis,"states":states,"actions":actions,"values":values.reshape(resolution,resolution),
            "q_values":qs.reshape(resolution,resolution,len(actions)),"valid":valid.reshape(resolution,resolution),
            "bellman_residual":np.array(residual),"iterations":np.array(iterations+1),
            "resolution":np.array(resolution),"gamma":np.array(gamma)}


def refinement_check(resolutions=(21,41,81), gamma=.99):
    results=[]
    for resolution in resolutions:
        oracle=grid_dynamic_programming(resolution,gamma)
        x,y=np.rint(PointMaze.start*(resolution-1)).astype(int)
        results.append({"resolution":resolution,"start_value":float(oracle["values"][y,x]),
                        "bellman_residual":float(oracle["bellman_residual"])})
    return {"reference":"approximate state/action-grid oracle, not exact continuous optimum", "results":results}


def rollout_continuations(agent, method, states, first_actions=None, *, repetitions=16, horizon=150, gamma=.99, seed=123):
    """Single-candidate policy-specific truncated Monte Carlo V^pi/Q^pi.

    All simulator access belongs to evaluation. The finite rollout horizon is a
    reporting truncation with zero tail estimate, not a change to the task MDP.
    It has MC intervals plus potentially nonzero truncation bias.
    """
    from .evaluate import policy_actions
    states=np.atleast_2d(states)
    positions=np.repeat(states,repetitions,axis=0)
    valid_start=(np.isfinite(states).all(-1) & (states>=0).all(-1) & (states<=1).all(-1)
                 & ~segments_hit_boxes(states,states,PointMaze.boxes))
    terminal_start=np.linalg.norm(states-PointMaze.goal,axis=-1)<.055
    active=np.repeat(valid_start & ~terminal_start,repetitions)
    policy_indices=np.flatnonzero(active)
    returns=np.zeros(len(positions))
    collision_counts=np.zeros(len(positions))
    env=PointMaze(gamma=gamma)
    rng=np.random.RandomState(seed)
    for t in range(horizon):
        if not np.any(active):
            break
        active_indices=np.flatnonzero(active)
        if t==0 and first_actions is not None:
            actions=np.repeat(np.atleast_2d(first_actions),repetitions,axis=0)[active_indices]
        else:
            # Keep one fixed batch shape and per-particle random-key allocation.
            # Finished particles may receive an unused network query, but only
            # active particles execute an environment transition below.
            candidates=policy_actions(agent,positions[policy_indices],method,int(rng.randint(2**31-1)),single_candidate=True)[:,0]
            actions=candidates[active[policy_indices]]
        nxt,reward,_,done,collision=env.transition(positions[active_indices],actions)
        returns[active_indices] += gamma**t * reward
        collision_counts[active_indices] += collision
        positions[active_indices]=nxt
        active[active_indices] &= ~done
    values=returns.reshape(len(states),repetitions)
    # Terminal states have zero continuation; invalid physical states have no
    # defined value and must not masquerade as legitimate zero-valued states.
    values[~valid_start]=np.nan
    return {"mean":values.mean(-1),"se":values.std(-1,ddof=1)/np.sqrt(repetitions) if repetitions>1 else np.zeros(len(states)),
            "samples":values,"unfinished_fraction":active.reshape(len(states),repetitions).mean(-1),
            "collisions":collision_counts.reshape(len(states),repetitions).mean(-1),
            "valid_start":valid_start,"terminal_start":terminal_start}


def continuation_probe(agent, method, observation, trace, *, repetitions=16, seed=731, horizon=150):
    """q_probe for the explicitly single-candidate policy; no native-head equivalence."""
    indices=native_progress_indices(trace["progress"])
    if method=="dppo_online":
        from .agents.dppo_diagnostics import q_probe,noise_tape
        native_trace=dict(trace,particles_per_probe=len(trace["latents"]))
        result=q_probe(agent,np.atleast_2d(observation),native_trace,indices,
            noise_tape(seed,1,repetitions,agent.config.denoising_steps)[0,1:],
            noise_tape(seed+1,horizon,repetitions,agent.config.denoising_steps))
        return {"indices":indices,"times":np.asarray(trace["native_times"])[indices],
            "mean":result["mean"][0],"se":result["se"][0],"unfinished_fraction":result["unfinished"][0].mean(-1),
            "policy":"single_candidate_native_evaluation_sampler",
            "quantity":"unregularized truncated Monte Carlo q_probe; DPPO has no inner head to calibrate"}
    latents=np.asarray(trace["latents"])
    values=[]
    rng=np.random.RandomState(seed)
    for index in indices:
        if method=="rql":
            latent=latents[index,0]
            obs=np.repeat(np.atleast_2d(observation),repetitions,axis=0)
            repeated=np.repeat(latent[None],repetitions,axis=0)
            endpoints=np.asarray(agent.continue_flow(obs,repeated,int(index))).reshape(repetitions,-1,2)[:,0]
        elif method=="bdpo":
            latent=latents[index,0,0]
            obs=np.repeat(np.atleast_2d(observation),repetitions,axis=0)
            repeated=np.repeat(latent[None],repetitions,axis=0)
            endpoints=np.asarray(agent.continue_from_latent(obs,repeated,int(np.asarray(trace["native_times"])[index]),int(rng.randint(2**31-1))))
        else:
            raise NotImplementedError(f"No continuation evaluator for method {method}")
        result=rollout_continuations(agent,method,obs,endpoints,repetitions=1,horizon=horizon,seed=int(rng.randint(2**31-1)))
        samples=result["mean"]
        values.append((float(samples.mean()),float(samples.std(ddof=1)/np.sqrt(repetitions))))
    return {"indices":indices,"times":np.asarray(trace["native_times"])[indices],
            "mean":np.array(values)[:,0],"se":np.array(values)[:,1],
            "policy":"single_candidate", "quantity":"unregularized truncated Monte Carlo q_probe; native critics are not equated"}


def aggregate_training_seeds(run_results):
    """Treat independently trained seeds as the uncertainty unit."""
    if len({r["seed"] for r in run_results}) != len(run_results):
        raise ValueError("Repeated training seed; evaluation episodes are not independent agents")
    values=np.array([r["score"] for r in run_results],float)
    n=len(values)
    if n<2:
        return {"n_seeds":n,"mean":float(values.mean()),"ci95":None,"reason":"Need independent training seeds"}
    from scipy.stats import t
    width=float(t.ppf(.975,n-1)*values.std(ddof=1)/np.sqrt(n))
    return {"n_seeds":n,"mean":float(values.mean()),"ci95":[float(values.mean()-width),float(values.mean()+width)],"unit":"training seed"}
