"""Reward-based single-task OGBench adapters, preserving boundaries and masks."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from .data import OfflineDataset, file_sha256, metadata_sha256


def make_env(name, **kwargs):
    if name == "maze":
        from .toy import PointMaze
        return PointMaze(**kwargs)
    if "singletask" not in name:
        raise ValueError("OGBench ports require a reward-based singletask task ID")
    import ogbench
    if importlib.metadata.version("ogbench") != "1.2.0":
        raise RuntimeError("This profile requires OGBench 1.2.0 (current-state reward timing)")
    return ogbench.make_env_and_datasets(name, env_only=True, **kwargs)


def load_ogbench(name, dataset_dir, *, seed=0, sparse=False, dataset_path=None, clip_actions=True):
    import ogbench
    if "singletask" not in name:
        raise ValueError("Goal-conditioned OGBench is outside this reward-based protocol")
    if importlib.metadata.version("ogbench") != "1.2.0":
        raise RuntimeError("Expected pinned OGBench 1.2.0")
    env, train, _ = ogbench.make_env_and_datasets(
        name, dataset_dir=str(dataset_dir), dataset_path=str(dataset_path) if dataset_path else None)
    if sparse:
        train["rewards"] = -(train["rewards"] != 0).astype(np.float32)
    if clip_actions:
        train["actions"] = np.clip(train["actions"], -1+1e-5, 1-1e-5)
    if dataset_path:
        files = [Path(dataset_path)]
    else:
        parts = name.split("-")
        pos = parts.index("singletask")
        base = "-".join(parts[:pos] + parts[-1:])
        files = [Path(dataset_dir) / f"{base}.npz"]
    metadata = {"kind":"ogbench", "task":name, "version":"1.2.0", "reward_timing":"r(s), not r(s_next)",
                "sparse_conversion":sparse, "action_clip_eps":1e-5 if clip_actions else None,
                "sources":[{"path":str(f),"size_bytes":f.stat().st_size,"sha256":file_sha256(f)} for f in files]}
    return env, OfflineDataset(train, seed=seed, metadata=metadata)


class ShardedOGBench:
    """Sorted cyclic source shard schedule, replacement before every 1000th update.

    A required shard manifest certifies the dataset size; a 1M substitute cannot
    accidentally run under a 100M experiment label. Each shard has an upstream
    train/validation NPZ pair. Sampler RNG continues across replacements.
    """
    def __init__(self, name, manifest_path, *, seed=0, sparse=False, interval=1000):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("total_transitions", 0) < 100_000_000:
            raise ValueError("100M profile requires a manifest certifying at least 100M transitions")
        self.paths = sorted(Path(p) if Path(p).is_absolute() else manifest_path.parent / p for p in manifest["shards"])
        if not self.paths or any(not p.exists() for p in self.paths):
            raise FileNotFoundError("100M shard files unavailable; no reduced-data substitution allowed")
        if len(self.paths) != 100 or len({p.resolve() for p in self.paths}) != 100:
            raise ValueError("100M source protocol requires 100 distinct training shards")
        records = manifest.get("records", [])
        expected = {str((Path(r["path"]) if Path(r["path"]).is_absolute() else manifest_path.parent/Path(r["path"])).resolve()):r for r in records}
        if len(expected) != 100 or any(str(p.resolve()) not in expected for p in self.paths):
            raise ValueError("All 100 shards need count/checksum records")
        actual_count, hashes, verified_records = 0, set(), []
        for path in self.paths:
            record = expected[str(path.resolve())]
            checksum = file_sha256(path)
            if checksum != record["sha256"]:
                raise ValueError(f"Shard checksum differs: {path}")
            if checksum in hashes:
                raise ValueError("Duplicate shard contents cannot certify 100M distinct source transitions")
            hashes.add(checksum)
            with np.load(path, allow_pickle=False) as file:
                count = int(np.sum(1-file["terminals"]))
            if count != record["transitions"]:
                raise ValueError(f"Shard transition count differs: {path}")
            actual_count += count
            verified_records.append({"path":str(path.resolve()),"sha256":checksum,"transitions":count})
            if not path.with_name(path.stem+"-val.npz").exists():
                raise FileNotFoundError(f"Validation partner missing for {path}")
        if actual_count != manifest["total_transitions"] or actual_count < 100_000_000:
            raise ValueError("Verified processed transition count does not match the 100M claim")
        self.shards_digest=metadata_sha256(verified_records)
        self.name, self.seed, self.sparse, self.interval = name, seed, sparse, interval
        self.index = 0
        self.env, self.dataset = self._load(0)

    def _load(self, index):
        return load_ogbench(self.name, self.paths[index].parent, seed=self.seed, sparse=self.sparse,
                            dataset_path=self.paths[index], clip_actions=False)

    def prepare_step(self, step):
        if step % self.interval == 0:
            rng = self.dataset.rng.get_state()
            self.index = (self.index+1) % len(self.paths)
            new_env, self.dataset = self._load(self.index)
            new_env.close()
            self.dataset.rng.set_state(rng)

    def state_dict(self):
        return {"index":self.index,"paths":[str(p) for p in self.paths],"dataset":self.dataset.state_dict(),
                "shards_sha256":self.shards_digest}

    def load_state_dict(self, state):
        if state["paths"] != [str(p) for p in self.paths]:
            raise ValueError("Shard manifest changed on resume")
        if state.get("shards_sha256") not in (None,self.shards_digest):
            raise ValueError("Shard contents changed on resume, including a currently inactive shard")
        self.index = state["index"]
        env, self.dataset = self._load(self.index)
        env.close()
        self.dataset.load_state_dict(state["dataset"])
