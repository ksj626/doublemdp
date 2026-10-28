#!/usr/bin/env python3
"""Evaluate a saved native DPPO/BC checkpoint without training or rewriting it."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from doublemdp.agents.dppo_artifacts import load_evaluation_checkpoint
from doublemdp.envs import make_env
from scripts.train_dppo import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--output",required=True,help="JSON report; trajectories are saved alongside it")
    p.add_argument("--device",default="cpu")
    p.add_argument("--episodes",type=int,default=50)
    p.add_argument("--parallel-envs",type=int,default=1)
    p.add_argument("--seed",type=int,default=10000)
    p.add_argument("--env",help="Optional expected task identity")
    p.add_argument("--expected-sha256")
    p.add_argument("--dataset",help="Optional saved dataset to verify against the checkpoint digest")
    p.add_argument("--strict-offline",action="store_true",help="Accept only a BC-phase policy; evaluation interactions are still counted")
    args=p.parse_args()
    if args.episodes<1 or args.parallel_envs<1:
        p.error("episodes and parallel-envs must be positive")
    torch.set_num_threads(1)
    agent,identity=load_evaluation_checkpoint(args.checkpoint,device=args.device,
        expected_sha256=args.expected_sha256,env=args.env,strict_offline=args.strict_offline)
    metadata=identity["metadata"]
    if args.dataset:
        from doublemdp.data import OfflineDataset
        if OfflineDataset.load(args.dataset,reward=metadata["reward"]).digest!=metadata["dataset_sha256"]:
            raise ValueError("Evaluation dataset identity differs from the checkpoint")
    # Report true sparse task return even when the policy trained with shaping.
    def factory():
        return make_env(metadata["env"],**({"reward":"sparse","gamma":agent.config.gamma} if metadata["env"]=="maze" else {}))
    before=agent.counters["evaluation_env_steps"]
    metrics,trajectories=evaluate(agent,factory,args.episodes,args.seed,args.parallel_envs)
    result={"checkpoint":identity,"method":identity["method"],"phase":identity["phase"],
        "env":metadata["env"],"training_reward":metadata["reward"],"evaluation_reward":"true sparse task return",
        "evaluation_seed":args.seed,"policy":"one candidate; source deterministic=True evaluation sampler (stochastic intermediate denoising)",
        "new_training_interactions":0,"evaluation_interactions":agent.counters["evaluation_env_steps"]-before,**metrics}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix(".trajectories.npz"),**{f"episode_{i}":path for i,path in enumerate(trajectories)})
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
