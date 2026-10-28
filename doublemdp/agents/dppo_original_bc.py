"""Explicit epoch-boundary checkpoint hooks for pinned DPPO BC source.

No active DataLoader iterator survives an exhausted epoch. Restoring its global
torch RNG reproduces the next RandomSampler permutation and denoising draws.
This intentionally does not claim to resume the upstream online vector runner.
"""
import copy
import hashlib
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import torch


def prepare_bc_source(reference, destination, patch):
    """Apply a small reviewed patch to a separate pinned worktree."""
    reference, destination, patch = map(Path, (reference, destination, patch))
    commit = "dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "worktree", "add", "--detach", str(destination), commit], cwd=reference, check=True, capture_output=True)
        subprocess.run(["git", "apply", str(patch.resolve())], cwd=destination, check=True, capture_output=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip() != commit:
        raise ValueError("BC resume source is not pinned v0.6")
    # Reject an unrelated or partial modification instead of following it.
    subprocess.run(["git", "apply", "--reverse", "--check", str(patch.resolve())], cwd=destination, check=True, capture_output=True)
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=destination, text=True).splitlines()
    if changed != ["agent/pretrain/train_diffusion_agent.py"]:
        raise ValueError("Unexpected modifications in BC resume source worktree")
    expected = "918ec91355a5a8bf9fa45bf2b0df9c22bf0ebfe34b2afb5d524c836a8015c9b0"
    if hashlib.sha256((destination / changed[0]).read_bytes()).hexdigest() != expected:
        raise ValueError("BC resume source differs from the reviewed patch")
    return destination


def initialize_bc_checkpoint(agent, cfg):
    from omegaconf import OmegaConf
    configuration = OmegaConf.to_container(cfg, resolve=True)
    selected = {key: copy.deepcopy(configuration[key]) for key in ("model", "ema", "train_dataset", "train", "seed")}
    for key in ("n_epochs", "save_model_freq", "log_freq"):
        selected["train"].pop(key, None)
    selected["train_dataset"].pop("dataset_path", None)
    path = Path(agent.dataset_train.dataset_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    selected["dataset_sha256"] = digest.hexdigest()
    selected["valid_sequences"] = len(agent.dataset_train)
    selected["batches_per_epoch"] = len(agent.dataloader_train)
    agent._doublemdp_bc_signature = selected


def restore_bc_epoch(agent):
    path = os.environ.get("DPPO_BC_RESUME")
    if not path:
        return 1
    # AdamW's scalar step lives on CPU unless capturable/fused is enabled.
    # Loading onto CUDA indiscriminately changes that source optimizer state.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("kind") != "dppo_v06_bc_epoch_boundary_v1":
        raise ValueError("Expected a full BC epoch-boundary checkpoint; source-only weights cannot resume")
    if saved["signature"] != agent._doublemdp_bc_signature:
        raise ValueError("BC resume dataset or algorithm configuration differs")
    for name in ("model", "ema_model", "optimizer", "lr_scheduler"):
        getattr(agent, name).load_state_dict(saved[name])
    random.setstate(saved["rng"]["python"])
    np.random.set_state(saved["rng"]["numpy"])
    torch.set_rng_state(saved["rng"]["torch"].cpu())
    if saved["rng"]["cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["rng"]["cuda"]])
    return saved["next_epoch"]


def save_bc_epoch(agent):
    saved = {"kind": "dppo_v06_bc_epoch_boundary_v1", "phase": "bc",
        "next_epoch": agent.epoch, "completed_epochs": agent.epoch-1,
        "bc_updates": (agent.epoch-1)*len(agent.dataloader_train),
        "signature": agent._doublemdp_bc_signature,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    for name in ("model", "ema_model", "optimizer", "lr_scheduler"):
        saved[name] = getattr(agent, name).state_dict()
    destination = Path(agent.checkpoint_dir)/"full_bc_latest.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(saved, temporary)
    temporary.replace(destination)

