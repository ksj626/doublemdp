"""Full harness resume must reproduce uninterrupted optimizer and RNG state."""
import numpy as np
import pytest


def test_resume_matches_uninterrupted_updates(tmp_path):
    jax=pytest.importorskip("jax")
    from doublemdp.data import save_dataset
    from doublemdp.toy import Bandit
    from doublemdp.train import run
    from doublemdp.logging import load_checkpoint
    path=tmp_path/"bandit.npz"
    save_dataset(path,Bandit().sample(64,10),{"kind":"bandit","terminal_bandit":True})
    config={"method":"rql","profile":"toy_diagnostic","env":"bandit","strict_offline":True,"seed":9,
        "dataset":str(path),"batch_size":16,"steps":4,"eval_samples":16,"eval_interval":10,
        "save_interval":2,"log_interval":2,"agent":{"h":1,"flow_steps":2,"ensemble_ct":2,
        "actor_hidden_dims":[16,16],"value_hidden_dims":[16,16]}}
    run(config|{"output":str(tmp_path/"full")})
    partial=config|{"output":str(tmp_path/"resumed"),"steps":2}
    run(partial)
    run(partial|{"steps":4},str(tmp_path/"resumed"/"checkpoint.pkl"))
    full=load_checkpoint(tmp_path/"full"/"checkpoint.pkl")
    resumed=load_checkpoint(tmp_path/"resumed"/"checkpoint.pkl")
    for a,b in zip(jax.tree_util.tree_leaves(full["agent"]),jax.tree_util.tree_leaves(resumed["agent"])):
        np.testing.assert_array_equal(a,b)
    assert full["step"]==resumed["step"]==4
    assert full["online_interactions"]==resumed["online_interactions"]==0
    for a,b in zip(full["sampler"]["rng"],resumed["sampler"]["rng"]):
        np.testing.assert_array_equal(a,b)
