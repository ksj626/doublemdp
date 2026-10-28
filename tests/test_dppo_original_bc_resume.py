"""Run with scripts/legacy_python.sh -m pytest (original source dependencies)."""
import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("wandb")
from omegaconf import OmegaConf
from doublemdp.agents.dppo_original_bc import prepare_bc_source

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=ROOT/"third_party/dppo"
sys.path.insert(0,str(UPSTREAM))


def source_class(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainDiffusionAgent


def assert_tree_equal(left,right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:
            assert_tree_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):
            assert_tree_equal(a,b)
    else:
        assert left==right


@pytest.mark.skipif(not torch.cuda.is_available(),reason="Official GPU dataset path required by source training loop")
def test_original_epoch_resume_matches_unmodified_source(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    reference=source_class(UPSTREAM/"agent/pretrain/train_diffusion_agent.py","original_bc_reference")
    output_root=Path(os.environ.get("DOUBLEMDP_OUTPUTS", "/home/dmsdmswns/doublemdp_outputs"))
    worktree=prepare_bc_source(UPSTREAM,output_root/"runtime/dppo_bc_resume_v1",ROOT/"scripts/dppo_bc_resume.patch")
    resumed_class=source_class(worktree/"agent/pretrain/train_diffusion_agent.py","resumable_bc_source")
    rng=np.random.RandomState(4)
    dataset=tmp_path/"train.npz"
    np.savez(dataset,states=rng.randn(20,3).astype('f'),actions=rng.randn(20,2).astype('f'),traj_lengths=np.array([5,5,5,5]))
    def configuration(name,epochs):
        cfg=OmegaConf.load(UPSTREAM/"cfg/gym/pretrain/hopper-medium-v2/pre_diffusion_mlp.yaml")
        cfg.logdir=str(tmp_path/name)
        cfg.wandb=None
        cfg.train_dataset_path=str(dataset)
        cfg.device="cuda:0"
        cfg.obs_dim,cfg.action_dim,cfg.horizon_steps,cfg.denoising_steps=3,2,1,4
        cfg.model.network.mlp_dims=[16,16,16]
        cfg.train.n_epochs=epochs
        cfg.train.batch_size=4
        cfg.train.lr_scheduler.first_cycle_steps=5
        cfg.train.epoch_start_ema=1
        cfg.train.update_ema_freq=1
        cfg.train.save_model_freq=1
        return cfg
    OmegaConf.register_new_resolver("eval",eval,replace=True)
    monkeypatch.delenv("DPPO_BC_RESUME",raising=False)
    uninterrupted=reference(configuration("reference",2))
    uninterrupted.run()
    expected={name:copy.deepcopy(getattr(uninterrupted,name).state_dict()) for name in ("model","ema_model","optimizer","lr_scheduler")}
    expected_cpu_rng=torch.get_rng_state()
    expected_cuda_rng=torch.cuda.get_rng_state_all()
    expected_python_rng=random.getstate()
    expected_numpy_rng=np.random.get_state()
    interrupted=resumed_class(configuration("interrupted",1))
    interrupted.run()
    checkpoint=tmp_path/"interrupted/checkpoint/full_bc_latest.pt"
    monkeypatch.setenv("DPPO_BC_RESUME",str(checkpoint))
    resumed=resumed_class(configuration("resumed",2))
    resumed.run()
    for name in expected:
        assert_tree_equal(expected[name],getattr(resumed,name).state_dict())
    torch.testing.assert_close(expected_cpu_rng,torch.get_rng_state(),rtol=0,atol=0)
    assert_tree_equal(expected_cuda_rng,torch.cuda.get_rng_state_all())
    assert random.getstate()==expected_python_rng
    actual_numpy_rng=np.random.get_state()
    assert actual_numpy_rng[0]==expected_numpy_rng[0]
    np.testing.assert_array_equal(actual_numpy_rng[1],expected_numpy_rng[1])
    assert actual_numpy_rng[2:]==expected_numpy_rng[2:]
    state=torch.load(tmp_path/"resumed/checkpoint/full_bc_latest.pt",weights_only=False)
    assert state["completed_epochs"]==2 and state["bc_updates"]==10 and state["next_epoch"]==3
    # Changing saved data is a scientific resume error, even with same shapes.
    np.savez(dataset,states=np.zeros((20,3),'f'),actions=np.zeros((20,2),'f'),traj_lengths=np.array([5,5,5,5]))
    invalid=resumed_class(configuration("invalid",2))
    with pytest.raises(ValueError,match="dataset or algorithm configuration differs"):
        invalid.run()
