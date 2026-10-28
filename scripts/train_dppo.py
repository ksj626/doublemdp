#!/usr/bin/env python3
"""BC initialization and a separate, explicitly online DPPO training track."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import random
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from doublemdp.agents.dppo_online import DPPOAgent, DPPOConfig, BCSequences, OnlineRunner
from doublemdp.data import OfflineDataset
from doublemdp.envs import make_env


@contextmanager
def preserve_host_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def evaluate(agent, env_factory, episodes=20, seed=10000, parallel_envs=1):
    returns, successes, lengths, collisions = [], [], [], []
    trajectories = []
    devices = [agent.device] if agent.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), preserve_host_rng():
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        for first in range(0, episodes, parallel_envs):
            count = min(parallel_envs, episodes-first)
            envs = [env_factory() for _ in range(count)]
            observations = [env.reset(seed=seed+first+i)[0] for i,env in enumerate(envs)]
            paths = [[obs.copy()] for obs in observations]
            reward_sums, collision_sums = np.zeros(count), np.zeros(count)
            steps, succeeded, done = np.zeros(count,dtype=int), np.zeros(count), np.zeros(count,dtype=bool)
            while not done.all():
                active = np.flatnonzero(~done)
                actions = agent.sample_actions(np.asarray([observations[i] for i in active]), deterministic=True)
                for i, chunk in zip(active, actions):
                    for action in chunk[:agent.config.act_steps]:
                        obs, reward, term, trunc, info = envs[i].step(action)
                        observations[i] = obs
                        reward_sums[i] += reward
                        steps[i] += 1
                        agent.counters["evaluation_env_steps"] += 1
                        collision_sums[i] += float(info.get("collision", 0))
                        paths[i].append(obs.copy())
                        done[i] = term or trunc
                        if done[i]:
                            succeeded[i] = float(info.get("success", term))
                            break
            returns.extend(reward_sums.tolist())
            successes.extend(succeeded.tolist())
            lengths.extend(steps.tolist())
            collisions.extend(collision_sums.tolist())
            trajectories.extend(np.asarray(path) for path in paths)
            for env in envs:
                env.close()
    return {"return_mean": float(np.mean(returns)), "success_rate": float(np.mean(successes)),
            "episode_returns": returns, "episode_successes": successes, "episode_lengths": lengths,
            "collision_mean": float(np.mean(collisions)), "evaluation_episodes": episodes,
            "evaluation_parallel_envs": parallel_envs}, trajectories


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/dppo_toy.json")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--env", default="maze")
    p.add_argument("--reward", choices=["sparse", "dense"], default="sparse")
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    p.add_argument("--resume")
    p.add_argument("--bc-epochs", type=int)
    p.add_argument("--online-iterations", type=int)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-envs", type=int, default=1, help="Batch independent evaluation policies; changes the evaluation RNG tape shape")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--strict-offline", action="store_true")
    p.add_argument("--bc-only", action="store_true", help="Produce an offline BC reference, no PPO or online training transitions")
    p.add_argument("--torch-threads", type=int, default=1)
    args = p.parse_args()
    if args.eval_envs < 1:
        p.error("--eval-envs must be positive")
    if args.strict_offline and not args.bc_only:
        p.error("dppo_online cannot run in strict-offline mode; use --bc-only for an offline BC reference")
    torch.set_num_threads(args.torch_threads)
    config = json.loads(Path(args.config).read_text())
    for field, value in (("device", args.device), ("seed", args.seed), ("bc_epochs", args.bc_epochs),
                         ("online_iterations", args.online_iterations)):
        if value is not None:
            config[field] = value
    c = DPPOConfig(**config)
    c.validate()
    if c.profile == "official_code_v0.6":
        p.error("Original-paper runs must use scripts/dppo_original.py with pinned upstream environment wrappers/configs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = OfflineDataset.load(args.dataset, seed=c.seed, reward=args.reward)
    if args.reward == "dense":
        shaping_gamma = dataset.metadata.get("gamma")
        if shaping_gamma is None or not np.isclose(float(shaping_gamma), c.gamma, rtol=0, atol=1e-12):
            raise ValueError("Dense shaping requires saved dataset gamma to equal DPPO gamma")
    sampler = BCSequences(dataset.arrays["observations"], dataset.arrays["actions"], dataset.arrays["terminals"], c.horizon_steps, c.seed)
    agent = DPPOAgent.create(dataset.arrays["observations"].shape[-1], dataset.arrays["actions"].shape[-1], c)
    payload = None
    if args.resume:
        agent, payload = DPPOAgent.load(args.resume, device=c.device, sampler=sampler)
        if (args.bc_only or args.strict_offline) and agent.phase != "bc":
            raise ValueError("An online-trained checkpoint cannot become an offline BC reference; resume a BC-phase checkpoint")
        c = agent.config
        if payload["metadata"].get("dataset_sha256") != dataset.digest:
            raise ValueError("Checkpoint dataset checksum differs")
        for key,value in (("env",args.env),("reward",args.reward)):
            if payload["metadata"].get(key) != value:
                raise ValueError(f"Checkpoint {key} differs; resuming cannot silently change the task")
        if args.online_iterations is not None:
            # A resumed larger budget is a labeled extension, preserving scheduler state.
            c.online_iterations = args.online_iterations
        if args.bc_epochs is not None:
            c.bc_epochs = args.bc_epochs
    metadata = {"method": "bc" if args.bc_only else "dppo_online", "profile": c.profile,
                "dataset_path": str(Path(args.dataset).resolve()), "dataset_sha256": dataset.digest,
                "env": args.env, "reward": args.reward, "seed": c.seed,
                "checkpoint_selection": "final training update", "online_interactions": not args.bc_only,
                "upstream_commit": "dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc"}
    (output/"config.json").write_text(json.dumps(asdict(c), indent=2))
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2))
    log = output/"metrics.jsonl"
    def record(metrics):
        with log.open("a") as f:
            f.write(json.dumps(metrics)+"\n")
        print(json.dumps(metrics), flush=True)
    while agent.phase == "bc" and agent.counters["bc_epochs"] < c.bc_epochs:
        obs, acts = sampler.next_batch(c.bc_batch_size)
        result = agent.pretrain_update(obs, acts)
        if sampler.epoch_finished:
            agent.finish_bc_epoch()
            record({"phase": "bc", **result, **agent.counters, "bc_seconds": agent.bc_seconds})
            agent.save(output/"latest.pt", sampler=sampler, metadata=metadata)
    dataset.verify_immutable()
    def env_factory():
        kwargs = {"gamma": c.gamma, "reward": args.reward} if args.env == "maze" else {}
        env = make_env(args.env, **kwargs)
        if args.env != "maze":
            from doublemdp.agents.dppo_env_state import StatefulOGBench
            env = StatefulOGBench(env)
        return env
    if agent.phase == "bc":
        result, _ = evaluate(agent, env_factory, args.eval_episodes, parallel_envs=args.eval_envs)
        record({"phase": "bc_evaluation", **result, **agent.counters})
        agent.save(output/"bc.pt", sampler=sampler, metadata=metadata)
    if not args.bc_only:
        agent.begin_online()
        runner = OnlineRunner(agent, [env_factory() for _ in range(c.n_envs)])
        if payload is not None and payload["runner"] is not None:
            runner.load_state_dict(payload["runner"])
            # Runner construction resets environments, not policy RNG; restoring
            # env state above is sufficient because reset uses per-env generators.
        while agent.counters["online_iterations"] < c.online_iterations:
            if agent.device.type == "cuda":
                torch.cuda.synchronize(agent.device)
            started = time.perf_counter()
            rollout = runner.collect()
            rollout_seconds = time.perf_counter()-started
            update_started = time.perf_counter()
            result = agent.ppo_update(rollout)
            if agent.device.type == "cuda":
                torch.cuda.synchronize(agent.device)
            result.update(rollout_seconds=rollout_seconds, update_seconds=time.perf_counter()-update_started)
            record({"phase": "online", **result, **agent.counters, "raw_reward_mean": rollout["raw_reward_mean"],
                    "completed_training_episodes": rollout["completed_episodes"]})
            if agent.counters["online_iterations"] % args.eval_every == 0 or agent.counters["online_iterations"] == c.online_iterations:
                result, trajectories = evaluate(agent, env_factory, args.eval_episodes, parallel_envs=args.eval_envs)
                record({"phase": "online_evaluation", **result, **agent.counters})
                np.savez_compressed(output/"evaluation_trajectories.npz", **{f"episode_{i}": t for i, t in enumerate(trajectories)})
            agent.save(output/"latest.pt", sampler=sampler, runner=runner, metadata=metadata)
        for env in runner.envs:
            env.close()
    if args.env == "maze":
        from doublemdp.toy import PointMaze
        devices = [agent.device] if agent.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(7000)
            probes = np.repeat(PointMaze.probes, 128, axis=0)
            _, trace = agent.sample_actions(probes, return_trace=True)
            np.savez_compressed(output/"traces.npz", observations=probes, seed=7000,
                **{k:v for k,v in trace.items() if isinstance(v, np.ndarray)})
    dataset.verify_immutable()
    (output/"completed.json").write_text(json.dumps({**metadata, **agent.counters, "bc_seconds": agent.bc_seconds,
        "status": "complete", "claim": "Completed configured diagnostic budget, not an original-paper reproduction"}, indent=2))


if __name__ == "__main__":
    main()
