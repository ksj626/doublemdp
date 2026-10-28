"""One offline training loop around separate native JAX algorithm kernels."""
from __future__ import annotations
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
from .data import OfflineDataset
from .logging import Logger, save_checkpoint, load_checkpoint, write_json


def validate_config(config):
    method = config.get("method", "rql")
    if method == "dppo_online":
        if config.get("strict_offline", True):
            raise ValueError("dppo_online requires actual online interactions; strict_offline=True is invalid")
        raise ValueError("Use scripts/train_dppo.py for the separate actual on-policy DPPO track")
    if method not in ("rql", "bdpo"):
        raise ValueError(f"Unknown method {method}")
    if config.get("steps", 1) < 1:
        raise ValueError("steps must be positive")
    if any(config.get(k,1) <= 0 for k in ("eval_interval", "save_interval", "log_interval")):
        raise ValueError("Evaluation, save and logging intervals must be positive")
    if not config.get("strict_offline", True):
        raise ValueError("The offline harness only supports strict_offline=True")
    profile = config.get("profile", "toy_diagnostic")
    if profile in ("official_code", "paper_budget") and config.get("env") in ("bandit", "maze"):
        raise ValueError("Proposed toy diagnostics must use a distinct toy_diagnostic profile")
    if "singletask" in config.get("env", "") and method == "bdpo" and profile != "ogbench_port":
        raise ValueError("BDPO-on-OGBench is a new benchmark port; set profile=ogbench_port")
    adapter=config.get("rql_sequence_adapter","official")
    if adapter != "official" and not (method=="rql" and config.get("env")=="maze"
                                      and profile=="toy_diagnostic" and adapter=="transition_h1"):
        raise ValueError("The transition_h1 sequence adapter is restricted to the explicit RQL point-maze toy diagnostic")
    if method == "bdpo" and config.get("agent", {}).get("critic", {}).get("q_target") == "mean" and "stable_zero_rho" not in profile:
        raise ValueError("The direct-mean numerical correction requires an explicit stable_zero_rho profile")
    if any(x in config.get("env", "") for x in ("puzzle-4x4", "cube-quadruple")) and not config.get("shard_manifest"):
        raise ValueError("100M task requires a verified shard_manifest; local reduced-data overrides are not permitted")


def create_agent(config, dataset):
    import copy
    method, agent_config = config["method"], copy.deepcopy(config.get("agent", {}))
    obs, actions = dataset.arrays["observations"][:1], dataset.arrays["actions"][:1]
    if method == "rql":
        from .agents.rql import RQLAgent
        agent_config["batch_size"] = config["batch_size"]
        return RQLAgent.create(config["seed"], obs, actions, agent_config)
    from .agents.bdpo import BDPOAgent, default_config
    defaults = default_config(agent_config.pop("config_profile", "toy_official_code")) if "diffusion" not in agent_config else {}
    defaults.update(agent_config)
    return BDPOAgent.create(config["seed"], obs, actions, defaults)


def serialize_agent(agent, method):
    import jax
    if method == "rql":
        return {"config":dict(agent.config), "state":jax.device_get(agent.state_dict())}
    return agent.state_dict()


def restore_agent(state, method, dataset):
    if method == "rql":
        from .agents.rql import RQLAgent
        agent = RQLAgent.create(0, dataset.arrays["observations"][:1], dataset.arrays["actions"][:1], state["config"])
        return agent.restore_state_dict(state["state"])
    from .agents.bdpo import BDPOAgent
    return BDPOAgent.from_state_dict(state)


def run(config, resume=None):
    validate_config(config)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    from .evaluate import evaluate_agent
    config = dict(config)
    config.setdefault("seed", 0)
    config.setdefault("batch_size", 256)
    output = Path(config["output"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pkl").exists() and not resume:
        raise FileExistsError("Run already contains a checkpoint; use --resume or a new output path")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    shard = None
    env = None
    if config.get("shard_manifest"):
        from .envs import ShardedOGBench
        shard = ShardedOGBench(config["env"],config["shard_manifest"], seed=config["seed"], sparse=config.get("sparse",False))
        dataset, env = shard.dataset, shard.env
    elif config.get("dataset"):
        dataset = OfflineDataset.load(config["dataset"], seed=config["seed"], reward=config.get("reward", "sparse"))
    else:
        if any(name in config["env"] for name in ("puzzle-4x4", "cube-quadruple")):
            raise FileNotFoundError("100M task requires --shard-manifest; refusing the default 1M dataset")
        from .envs import load_ogbench
        env, dataset = load_ogbench(config["env"], config["dataset_dir"], seed=config["seed"], sparse=config.get("sparse", False))
    step, elapsed_before = 0, 0.
    dataset.sequence_adapter = config.get("rql_sequence_adapter", "official")
    if config.get("reward") == "dense":
        agent_config = config.get("agent", {})
        gamma = agent_config.get("discount", .99) if config["method"] == "rql" else agent_config.get("critic", {}).get("discount", .99)
        if dataset.metadata.get("gamma") != gamma:
            raise ValueError("Dense-shaping dataset gamma must equal the agent environment discount")
    if resume:
        checkpoint = load_checkpoint(resume)
        previous = checkpoint["config"]
        for key in ("method", "seed", "env", "reward", "profile", "agent", "batch_size", "sparse", "rql_sequence_adapter"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"Cannot change {key} when resuming")
        if shard:
            shard.load_state_dict(checkpoint["sampler"])
            dataset = shard.dataset
        else:
            dataset.load_state_dict(checkpoint["sampler"])
        agent = restore_agent(checkpoint["agent"], config["method"], dataset)
        step, elapsed_before = checkpoint["step"], checkpoint["elapsed_seconds"]
        if step > config["steps"]:
            raise ValueError("Requested training budget is below the restored checkpoint step")
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    else:
        agent = create_agent(config, dataset)
    write_json(output / "config.json", config)
    write_json(output / "dataset.json", dataset.metadata | {"arrays_sha256":dataset.digest,"count":len(dataset),"rql_sequence_adapter":dataset.sequence_adapter})
    write_json(output / "runtime.json", {"pid":os.getpid(),"devices":[str(d) for d in jax.devices()],
                                       "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"), "strict_offline":True})
    logger = Logger(output)
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.perf_counter()
    window_start, window_steps = started, 0
    compile_seconds = None
    def checkpoint_now():
        elapsed = elapsed_before + time.perf_counter() - started
        save_checkpoint(output / "checkpoint.pkl", {"format":1,"config":config,"step":step,
            "agent":serialize_agent(agent, config["method"]),
            "sampler":shard.state_dict() if shard else dataset.state_dict(),
            "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
            "elapsed_seconds":elapsed,"online_interactions":0})
        if config.get("keep_checkpoints", True):
            snapshot = output / f"checkpoint_{step:09d}.pkl"
            if not snapshot.exists():
                os.link(output / "checkpoint.pkl", snapshot)
        memory = jax.devices()[0].memory_stats() or {}
        write_json(output / "status.json", {"status":"complete" if step >= config["steps"] else "paused" if interrupted else "running",
            "step":step,"total_steps":config["steps"],"elapsed_seconds":elapsed,"pid":os.getpid(),
            "online_interactions":0,"compile_first_update_seconds":compile_seconds,"device_memory_stats":memory})
    try:
        # The previous checkpoint may be complete under a smaller budget. Do
        # not leave that stale completion visible if extension fails early.
        write_json(output / "status.json", {"status":"running" if step < config["steps"] else "complete",
            "step":step,"total_steps":config["steps"],"elapsed_seconds":elapsed_before,"pid":os.getpid(),
            "online_interactions":0,"compile_first_update_seconds":None})
        while step < config["steps"] and not interrupted:
            step += 1
            if shard:
                shard.prepare_step(step)
                dataset = shard.dataset
            batch = dataset.sample_rql(config["batch_size"], int(agent.config["h"])) if config["method"] == "rql" else dataset.sample(config["batch_size"])
            update_start = time.perf_counter()
            agent, metrics = agent.update(batch)
            if step in range(2, 12):
                jax.block_until_ready(metrics)
                logger.log(step, {"synchronized_update_seconds":time.perf_counter()-update_start}, phase="profile")
            window_steps += 1
            if compile_seconds is None:
                jax.block_until_ready(metrics)
                compile_seconds = time.perf_counter()-window_start
                logger.log(step, {"first_update_seconds":compile_seconds}, phase="runtime")
                window_start, window_steps = time.perf_counter(), 0
            if step % config.get("log_interval", 100) == 0 or step == config["steps"]:
                metrics = jax.device_get(metrics)
                elapsed = time.perf_counter()-window_start
                logger.log(step, {**metrics, "synchronized_window_seconds_per_update":elapsed / max(window_steps,1),
                                  "online_interactions":0})
                window_start, window_steps = time.perf_counter(), 0
            if step % config.get("eval_interval", 1000) == 0 or step == config["steps"] or (step == 1 and config.get("profile") in ("official_code", "paper_budget")):
                result = evaluate_agent(agent, config, dataset.metadata, step=step, env=env)
                logger.log(step, result, phase="evaluation")
                write_json(output / f"evaluation_{step}.json", result)
                print(json.dumps({"step":step,**result}), flush=True)
                window_start, window_steps = time.perf_counter(), 0
            if step % config.get("save_interval", 1000) == 0 or step == config["steps"]:
                checkpoint_now()
        dataset.verify_immutable()
        checkpoint_now()
        logger.export_csv()
        return output
    finally:
        for sig,handler in previous_handlers.items():
            signal.signal(sig,handler)
        if env is not None:
            env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--resume", nargs="?", const="auto")
    p.add_argument("--method", choices=["rql", "bdpo", "dppo_online"])
    for field in ("dataset", "dataset-dir", "shard-manifest", "env", "profile", "output", "reward"):
        p.add_argument("--"+field)
    for field in ("steps", "seed", "batch-size", "eval-interval", "eval-episodes", "save-interval"):
        p.add_argument("--"+field, type=int)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    config.update({k:v for k,v in vars(args).items() if v is not None and k not in ("config","resume")})
    config.setdefault("method", "rql")
    config.setdefault("strict_offline", True)
    config.setdefault("steps", 10000)
    config.setdefault("profile", "toy_diagnostic")
    config.setdefault("dataset_dir", str(Path.home()/"doublemdp_outputs"/"datasets"/"ogbench"))
    config.setdefault("output", str(Path.home()/"doublemdp_outputs"/"runs"/f"{config['method']}_{int(time.time())}"))
    resume = str(Path(config["output"])/"checkpoint.pkl") if args.resume == "auto" else args.resume
    run(config, resume)


if __name__ == "__main__":
    main()
