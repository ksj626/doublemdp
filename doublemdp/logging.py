"""Local durable logs and atomic complete-state checkpoints; no remote service."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import pickle
import time
from pathlib import Path
import numpy as np


def json_default(x):
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(type(x).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")
    os.replace(tmp, path)


class Logger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "metrics.jsonl"

    def log(self, step, metrics, *, phase="train"):
        row = {"step": int(step), "phase": phase, "time": time.time()}
        row.update({k: float(np.asarray(v)) for k, v in metrics.items()})
        if any(not np.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError(f"Nonfinite metric at step {step}")
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        return row

    def export_csv(self):
        if not self.path.exists():
            return
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with (self.directory / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, *, with_identity=False):
    """Load only locally created/trusted checkpoints (Python pickle)."""
    with open(path, "rb") as f:
        state=pickle.load(f)
        if not with_identity:
            return state
        # Hash the same open snapshot, even if the trainer atomically replaces
        # checkpoint.pkl while evaluation is loading it.
        f.seek(0)
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(8<<20),b""):
            digest.update(block)
        return state,{"path":str(Path(path).resolve()),"sha256":digest.hexdigest()}
