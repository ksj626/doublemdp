"""Immutable offline transitions and the pinned RQL sequence sampling protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

FIELDS = ("observations", "actions", "rewards", "masks", "terminals", "next_observations")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arrays_sha256(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def metadata_sha256(metadata):
    """Identity for semantic sidecar fields such as reward layout and shaping."""
    payload=json.dumps(metadata,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


class OfflineDataset:
    def __init__(self, arrays, *, seed=0, metadata=None):
        missing = set(FIELDS) - arrays.keys()
        if missing:
            raise ValueError(f"Missing dataset fields: {sorted(missing)}")
        self.arrays = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
        n = len(self.arrays["observations"])
        if not n or any(len(v) != n for v in self.arrays.values()):
            raise ValueError("All fields must have the same nonzero leading dimension")
        for k in ("rewards", "masks", "terminals"):
            self.arrays[k] = self.arrays[k].reshape(n)
        if not np.all(np.isin(self.arrays["masks"], [0, 1])):
            raise ValueError("Bootstrap masks must be binary")
        if self.arrays["terminals"][-1] != 1:
            raise ValueError("Last row must close a dataset episode (independent of bootstrap mask)")
        if any(not np.isfinite(v).all() for v in self.arrays.values()):
            raise ValueError("Dataset contains nonfinite values")
        for v in self.arrays.values():
            v.setflags(write=False)
        self.metadata = metadata or {}
        self.sequence_adapter = "official"
        self.rng = np.random.RandomState(seed)
        self.ends = np.flatnonzero(self.arrays["terminals"])
        self.starts = np.r_[0, self.ends[:-1] + 1]
        self.eligible = np.flatnonzero(self.ends - self.starts)
        self.digest = arrays_sha256(self.arrays)

    def __len__(self):
        return len(self.arrays["observations"])

    def sample(self, batch_size):
        indices = self.rng.randint(len(self), size=batch_size)
        return {k: v[indices] for k, v in self.arrays.items() if k in FIELDS}

    def sample_rql(self, batch_size, h=1):
        """Episode uniform, random start excluding terminal, clamp h+1 rows.

        The documented terminal-bandit extension duplicates the sole terminal
        row for h=1. It retains one reward and zero bootstrap in the native loss.
        """
        if self.sequence_adapter == "transition_h1":
            if h != 1:
                raise ValueError("The explicit transition-successor adapter requires h=1")
            episodes = self.rng.choice(len(self.starts), batch_size, replace=batch_size >= len(self.starts))
            indices = self.rng.randint(self.starts[episodes], self.ends[episodes] + 1)
            first = {k:self.arrays[k][indices] for k in FIELDS}
            # Conceptually append a successor row to each trajectory. The
            # previously final transition can now be a start, preserving its
            # saved success reward. The target uses the actual saved successor,
            # including at a continuing-task timeout. No loss changes.
            return {"observations":np.stack([first["observations"],first["next_observations"]]),
                    "actions":np.stack([first["actions"],first["actions"]]),
                    "rewards":np.stack([first["rewards"],np.zeros(batch_size,dtype=np.float32)]),
                    "terminals":np.stack([first["terminals"],first["terminals"]]),
                    "masks":np.stack([first["masks"],first["masks"]])}
        if self.sequence_adapter != "official":
            raise ValueError(f"Unknown sequence adapter: {self.sequence_adapter}")
        if self.metadata.get("terminal_bandit"):
            if h != 1:
                raise ValueError("Terminal-bandit adapter requires h=1")
            if not np.all(self.arrays["terminals"] == 1) or np.any(self.arrays["masks"]):
                raise ValueError("A terminal bandit must have no bootstrapping")
            i = self.rng.randint(len(self), size=batch_size)
            indices = np.stack([i, i])
        else:
            if not len(self.eligible):
                raise ValueError("No length>=2 episodes; use the explicit terminal-bandit adapter")
            # Upstream bases replacement on all episodes. The only edge-case
            # fix is using eligible count when singleton episodes coexist.
            replace = batch_size >= len(self.starts) or batch_size > len(self.eligible)
            episodes = self.rng.choice(self.eligible, batch_size, replace=replace)
            start = self.rng.randint(self.starts[episodes], self.ends[episodes])
            indices = np.minimum(np.arange(h + 1)[:, None] + start, self.ends[episodes])
        return {k: self.arrays[k][indices] for k in FIELDS if k != "next_observations"}

    def state_dict(self):
        return {"rng": self.rng.get_state(), "sha256": self.digest, "sequence_adapter": self.sequence_adapter,
                "metadata_sha256":metadata_sha256(self.metadata)}

    def load_state_dict(self, state):
        if state["sha256"] != self.digest:
            raise ValueError("Resume dataset differs from checkpoint")
        if state.get("metadata_sha256") not in (None,metadata_sha256(self.metadata)):
            raise ValueError("Dataset semantic metadata differs from checkpoint")
        if state.get("sequence_adapter", "official") != self.sequence_adapter:
            raise ValueError("Resume sequence adapter differs from checkpoint")
        self.rng.set_state(state["rng"])

    def verify_immutable(self):
        if arrays_sha256(self.arrays) != self.digest:
            raise RuntimeError("Offline dataset mutated during training")

    @classmethod
    def load(cls, path, *, seed=0, reward="sparse"):
        path = Path(path)
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get("file_sha256") and file_sha256(path) != metadata["file_sha256"]:
            raise ValueError("Dataset checksum mismatch")
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if reward == "dense":
            if "rewards_dense" not in arrays:
                raise ValueError("Dense rewards were not saved in this dataset")
            arrays["rewards"] = arrays["rewards_dense"].copy()
        return cls(arrays, seed=seed, metadata=metadata)


def save_dataset(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    meta = dict(metadata, count=len(arrays["observations"]), file_sha256=file_sha256(path),
                arrays_sha256=arrays_sha256(arrays))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
