"""Read-only identity checks shared by DPPO evaluation and diagnostics."""
import hashlib
from pathlib import Path
from .dppo_online import DPPOAgent


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_checkpoint(path, *, device="cpu", expected_sha256=None, env=None, strict_offline=False):
    path=Path(path).resolve()
    digest=file_sha256(path)
    if expected_sha256 is not None and digest!=expected_sha256:
        raise ValueError("Checkpoint SHA256 differs from the requested artifact identity")
    agent,payload=DPPOAgent.load(path,device=device)
    if file_sha256(path)!=digest:
        raise ValueError("Checkpoint changed while being loaded; retry with an immutable artifact")
    metadata=payload["metadata"]
    if not metadata.get("env") or not metadata.get("dataset_sha256") or metadata.get("reward") not in ("dense","sparse"):
        raise ValueError("Evaluation requires saved task, reward and dataset identity")
    if env is not None and env!=metadata["env"]:
        raise ValueError("Requested environment differs from checkpoint identity")
    if strict_offline and agent.phase!="bc":
        raise ValueError("An online-trained checkpoint is not an offline BC reference")
    if agent.phase=="bc" and agent.counters["online_env_steps"]:
        raise ValueError("Inconsistent BC checkpoint with nonzero online training history")
    identity={"path":str(path),"sha256":digest,"bytes":path.stat().st_size,
        "phase":agent.phase,"method":"bc" if agent.phase=="bc" else "dppo_online",
        "training_counters":dict(agent.counters),"metadata":metadata}
    return agent,identity
