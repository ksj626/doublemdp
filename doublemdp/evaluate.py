"""Evaluation-only reward queries, environment rollouts and checkpoint loading."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .logging import load_checkpoint, write_json


def policy_actions(agent, observations, method, seed=0, *, single_candidate=False):
    import jax
    import jax.numpy as jnp
    obs = np.atleast_2d(observations)
    if method == "rql":
        if len(obs)==1:
            return np.asarray(agent.sample_actions(obs[0],seed=jax.random.PRNGKey(seed)))[None]
        keys = jax.random.split(jax.random.PRNGKey(seed), len(obs))
        result = jax.vmap(lambda o,k: agent.sample_actions(o, seed=k))(jnp.asarray(obs), keys)
        return np.asarray(result)
    actions = agent.sample_actions(obs, seed=seed, num_samples=1 if single_candidate else None)
    return np.asarray(actions)[:, None, :]


def evaluate_bandit(agent, config, metadata, seed=1729):
    from .toy import Bandit
    bandit = Bandit(gap=metadata.get("gap", False))
    count = config.get("eval_samples", 2048)
    context = np.arange(count) % bandit.contexts
    obs = np.eye(bandit.contexts, dtype=np.float32)[context]
    start = time.perf_counter()
    policy_actions(agent, obs, config["method"], seed)
    warmup_latency = time.perf_counter()-start
    start = time.perf_counter()
    actions = policy_actions(agent, obs, config["method"], seed)[:, 0]
    latency = (time.perf_counter()-start) / count
    reward = bandit.reward(actions, context)
    modes = np.argmin(np.linalg.norm(actions[:,None]-bandit.centers,axis=-1),axis=-1)
    result = {"reward_mean":float(reward.mean()), "reward_mc_se":float(reward.std(ddof=1)/np.sqrt(count)),
              "support_violation":float(np.mean(bandit.density(actions)==0)), "sample_latency_seconds":latency,
              "sampling_warmup_seconds":warmup_latency}
    result.update({f"mode_mass_{i}":float(np.mean(modes==i)) for i in range(8)})
    for c in range(bandit.contexts):
        selected = context == c
        result[f"context_{c}_reward_mean"] = float(reward[selected].mean())
        result[f"context_{c}_support_violation"] = float(np.mean(bandit.density(actions[selected]) == 0))
        result[f"context_{c}_highest_reward_mode_mass"] = float(np.mean(modes[selected] == (7+c) % 8))
    # Endpoint KL optimum is explicitly an objective-specific diagnostic, never
    # asserted to be RQL's target or the path-regularized BDPO optimum.
    if config["method"] == "bdpo":
        eta = agent.config_dict["critic"]["eta"]
        distances=[]
        for c in range(bandit.contexts):
            ref=bandit.reference_grid(c, eta, resolution=41)
            axis=ref["axis"]
            edges=np.r_[axis-.025,axis[-1]+.025]
            hist,_,_=np.histogram2d(actions[context==c,1],actions[context==c,0], bins=(edges,edges))
            # Include out-of-grid mass as an extra bin, not renormalized away.
            p=np.r_[hist.ravel(),np.sum(context==c)-hist.sum()]/np.sum(context==c)
            q=np.r_[ref["reference_mass"].ravel(),0.]
            m=(p+q)/2
            def kl(a,b):
                mask=a>0
                return np.sum(a[mask]*np.log(a[mask]/b[mask]))
            distances.append(.5*(kl(p,m)+kl(q,m)))
        result["endpoint_kl_reference_js_diagnostic"] = float(np.mean(distances))
    return result


def evaluate_rollouts(agent, config, env, seed=1729, episodes=None, *, save_path=None):
    episodes = episodes or config.get("eval_episodes",50)
    returns, successes, collisions, lengths, shortcuts, detours, trajectories = [],[],[],[],[],[],[]
    first_shortcut, first_detour = [], []
    rng = np.random.RandomState(seed)
    for episode in range(episodes):
        obs, info = env.reset(seed=seed+episode)
        ret, success, collision, length = 0., 0., 0., 0
        queue=[]
        trajectory=[np.asarray(obs).copy()]
        while True:
            # RQL public wrapper draws a new chunk at every environment step,
            # even while executing an existing queue; retain this RNG convention.
            if config["method"] == "rql" or not queue:
                chunk=policy_actions(agent, obs, config["method"], int(rng.randint(2**31-1)))[0]
                if not queue:
                    queue.extend(chunk)
            action=np.clip(queue.pop(0),-1,1)
            obs,reward,term,trunc,info=env.step(action)
            trajectory.append(np.asarray(obs).copy())
            ret += reward
            success=max(success,float(info.get("success",0.)))
            collision+=float(info.get("collision",0.))
            length+=1
            if term or trunc:
                break
        trajectory=np.array(trajectory)
        trajectories.append(trajectory)
        returns.append(ret); successes.append(success); collisions.append(collision); lengths.append(length)
        if config.get("env")=="maze":
            crossed=(trajectory[:,0]>.48)
            shortcuts.append(bool(np.any(crossed & (trajectory[:,1]>.36)&(trajectory[:,1]<.46))))
            detours.append(bool(np.any(crossed & (trajectory[:,1]>.78))))
            first_y = trajectory[np.flatnonzero(crossed)[0],1] if crossed.any() else np.nan
            first_shortcut.append(.36 < first_y < .46)
            first_detour.append(first_y > .78)
    result={"return_mean":float(np.mean(returns)),"success":float(np.mean(successes)),
            "return_mc_se":float(np.std(returns,ddof=1)/np.sqrt(episodes)) if episodes>1 else 0.,
            "collisions_mean":float(np.mean(collisions)),"episode_length_mean":float(np.mean(lengths)),
            "eval_interactions":float(np.sum(lengths))}
    if shortcuts:
        result.update(shortcut_visit_fraction=float(np.mean(shortcuts)),detour_visit_fraction=float(np.mean(detours)),
                      shortcut_first_crossing=float(np.mean(first_shortcut)),detour_first_crossing=float(np.mean(first_detour)))
    if save_path:
        np.savez(save_path, **{f"episode_{i}":t for i,t in enumerate(trajectories)},
                 returns=returns,successes=successes,collisions=collisions)
    return result


def evaluate_agent(agent, config, metadata, step=0, env=None):
    # Evaluation RNG and transitions are isolated from training RNG/sampler.
    seed = config.get("evaluation_seed",1729)
    if metadata.get("kind")=="bandit" or config.get("env")=="bandit":
        return evaluate_bandit(agent,config,metadata,seed)
    if metadata.get("kind")=="bdpo_original_toy":
        # Original toy runner owns its exact evaluation; this generic harness
        # cannot silently substitute a different reward convention.
        raise ValueError("Use scripts/run_bdpo_reference.py for the original toy")
    own_env = env is None
    if own_env:
        from .envs import make_env
        kwargs={"reward":"sparse"} if config["env"]=="maze" else {}
        env=make_env(config["env"],**kwargs)
    result=evaluate_rollouts(agent,config,env,seed)
    if own_env:
        env.close()
    return result


def load_run(path):
    from .data import OfflineDataset
    from .train import restore_agent
    path=Path(path)
    checkpoint,identity=load_checkpoint(path if path.suffix==".pkl" else path/"checkpoint.pkl",with_identity=True)
    checkpoint["artifact_identity"]=identity
    config=checkpoint["config"]
    if config.get("dataset"):
        dataset=OfflineDataset.load(config["dataset"],reward=config.get("reward","sparse"))
        dataset.sequence_adapter=config.get("rql_sequence_adapter","official")
        dataset.load_state_dict(checkpoint["sampler"])
        metadata=dataset.metadata
    else:
        # Reconstruct only the model's shape, without reloading million-row data.
        cfg=checkpoint["agent"]["config"]
        if config["method"]=="rql":
            obs_dim=cfg["observation_dim"] if "observation_dim" in cfg else None
            if obs_dim is None:
                from .envs import make_env
                env=make_env(config["env"])
                obs_dim=env.observation_space.shape[-1]
                env.close()
            action_dim=cfg["action_dim"]//cfg["h"]
        else:
            obs_dim,action_dim=checkpoint["agent"]["obs_dim"],checkpoint["agent"]["act_dim"]
        dataset=OfflineDataset({"observations":np.zeros((1,obs_dim)),"next_observations":np.zeros((1,obs_dim)),
                                "actions":np.zeros((1,action_dim)),"rewards":np.zeros(1),"masks":np.zeros(1),"terminals":np.ones(1)})
        metadata={"kind":"ogbench"}
    agent=restore_agent(checkpoint["agent"],config["method"],dataset)
    return agent,config,metadata,checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--episodes",type=int)
    p.add_argument("--output")
    args=p.parse_args()
    agent,config,metadata,checkpoint=load_run(args.run)
    if args.episodes:
        config["eval_episodes"]=args.episodes
    result=evaluate_agent(agent,config,metadata,checkpoint["step"])
    write_json(args.output or Path(config["output"])/"reevaluation.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
