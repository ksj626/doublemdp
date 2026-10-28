#!/usr/bin/env python
"""Actual pinned OGBench registry, reset/step and headless rendering check."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--output",required=True)
p.add_argument("--manifest",default="configs/rql_ogbench_manifest.json")
args=p.parse_args()
os.environ.setdefault("MUJOCO_GL","egl")
import gymnasium
import ogbench
from doublemdp.envs import make_env

manifest=json.loads(Path(args.manifest).read_text())
registered=[]
for task in manifest["tasks"]:
    name=task["env"]
    pieces=name.split("-"); pos=pieces.index("singletask")
    env_id="-".join(pieces[:pos-1]+pieces[pos:])
    gymnasium.spec(env_id)
    registered.append(env_id)
report={"ogbench":importlib.metadata.version("ogbench"),"gymnasium":gymnasium.__version__,
        "mujoco":importlib.metadata.version("mujoco"),"render_backend":os.environ["MUJOCO_GL"],
        "registered_tasks":registered,"checked":[]}
for name in ("antmaze-large-navigate-singletask-task1-v0","scene-play-singletask-task2-v0"):
    env=make_env(name)
    obs,info=env.reset(seed=0)
    nxt,reward,term,trunc,info=env.step(np.zeros(env.action_space.shape,dtype=np.float32))
    frame=env.render()
    assert frame.ndim==3 and frame.shape[-1]==3 and frame.max()>frame.min()
    report["checked"].append({"task":name,"observation_shape":list(obs.shape),"action_shape":list(env.action_space.shape),
        "reward":float(reward),"terminated":bool(term),"truncated":bool(trunc),"frame_shape":list(frame.shape)})
    import imageio.v2 as imageio
    imageio.imwrite(Path(args.output).parent/(name+".png"),frame)
    env.close()
Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
