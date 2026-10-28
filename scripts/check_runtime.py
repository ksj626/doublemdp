#!/usr/bin/env python
"""Record actual synchronized computation, separately for each backend."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["jax", "torch"], required=True)
p.add_argument("--output", required=True)
p.add_argument("--require-gpu", action="store_true")
args = p.parse_args()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
report = {"backend":args.backend, "python":platform.python_version(), "platform":platform.platform(),
          "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
          "ld_library_path":os.environ.get("LD_LIBRARY_PATH"), "packages":{}}
for cmd in (["nvidia-smi"], ["free","-h"], ["df","-h","."], ["ps","-eo","pid,comm,pcpu,pmem","--sort=-pcpu"]):
    result = subprocess.run(cmd, capture_output=True, text=True)
    report[cmd[0]] = result.stdout[:16000] or result.stderr
for package in ("jax", "jaxlib", "flax", "optax", "torch", "ogbench", "gymnasium", "mujoco"):
    try:
        report["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
t = time.perf_counter()
if args.backend == "jax":
    import jax
    import jax.numpy as jnp
    import optax
    x = jnp.ones((1024,1024), dtype=jnp.float32) / 1024
    w = jnp.eye(1024)
    optimizer = optax.adam(.001)
    state = optimizer.init(w)
    @jax.jit
    def update(w, state):
        loss, grad = jax.value_and_grad(lambda z: jnp.mean((x @ z - .5)**2))(w)
        updates, state = optimizer.update(grad, state, w)
        return optax.apply_updates(w, updates), state, loss, jnp.linalg.norm(grad)
    new, state, loss, gradient = update(w,state)
    new.block_until_ready()
    report.update(device=str(new.device), devices=[str(d) for d in jax.devices()], loss=float(loss),
                  gradient_norm=float(gradient), optimizer_delta=float(jnp.linalg.norm(new-w)),
                  is_gpu=new.device.platform == "gpu")
else:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.ones((1024,1024), device=device) / 1024
    w = torch.eye(1024, device=device, requires_grad=True)
    original = w.detach().clone()
    optimizer = torch.optim.Adam([w], lr=.001)
    loss = ((x @ w-.5)**2).mean()
    loss.backward()
    gradient = w.grad.norm().item()
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    report.update(device=str(w.device), loss=loss.item(), gradient_norm=gradient,
                  optimizer_delta=(w-original).norm().item(), is_gpu=device=="cuda")
report["seconds"] = time.perf_counter()-t
assert report["gradient_norm"] > 0 and report["optimizer_delta"] > 0
Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("ps","nvidia-smi","free","df")},indent=2))
if args.require_gpu and not report["is_gpu"]:
    raise SystemExit("GPU validation failed: computation actually ran on CPU")
