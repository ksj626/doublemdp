#!/usr/bin/env python
"""Expand the frozen fifty-task manifest into four separately trained seeds."""
import argparse
import json
from pathlib import Path
from doublemdp.sweep import connect, enqueue

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest",default="configs/rql_ogbench_manifest.json")
p.add_argument("--queue",required=True)
p.add_argument("--output-root",default=str(Path.home()/"doublemdp_outputs"))
p.add_argument("--profile",choices=["official_code","paper_budget"],default="official_code")
args=p.parse_args()
manifest=json.loads(Path(args.manifest).read_text())
root=Path(args.output_root).resolve(); config_dir=root/"sweep_configs"; config_dir.mkdir(parents=True,exist_ok=True)
db=connect(args.queue)
for task in manifest["tasks"]:
    for seed in task["seeds"]:
        env=task.get("env",task["task_id"])
        run_id=f"rql_{env}_{args.profile}_seed{seed}"
        budget=task.get("training_budgets",{}).get(args.profile,1000000 if args.profile=="official_code" else 2000000)
        if isinstance(budget,dict):
            budget=budget.get("updates",budget.get("steps"))
        config={"method":"rql","strict_offline":True,"profile":args.profile,"env":env,"seed":seed,
                "agent":task["agent"],"batch_size":256,"steps":budget,"sparse":task.get("sparse",False),
                "eval_interval":100000,"eval_episodes":50,"evaluation_seed":1729,"save_interval":100000,"log_interval":5000,
                "output":str(root/"runs"/run_id),"dataset_dir":str(root/"datasets"/"ogbench"),"run_id":run_id}
        if "puzzle-4x4" in env or "cube-quadruple" in env:
            family="puzzle-4x4" if "puzzle-4x4" in env else "cube-quadruple"
            config["shard_manifest"]=str(root/"datasets"/"ogbench"/(family+"-100M-shards.json"))
        priority=0 if seed==0 and env in ("antmaze-large-navigate-singletask-task1-v0","scene-play-singletask-task2-v0") else 10
        config["priority"]=priority
        (config_dir/(run_id+".json")).write_text(json.dumps(config,indent=2)+"\n")
        enqueue(db,run_id,config,priority)
print(json.dumps({"jobs":db.execute("SELECT count(*) FROM jobs").fetchone()[0],"queue":args.queue,"configs":str(config_dir)},indent=2))
