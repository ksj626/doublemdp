import numpy as np
import pytest
from doublemdp.data import OfflineDataset, save_dataset
from doublemdp.toy import Bandit, PointMaze, segments_hit_boxes, maze_dataset
from doublemdp.train import validate_config


def dataset():
    return OfflineDataset(dict(observations=np.arange(8).reshape(4,2),next_observations=np.arange(8).reshape(4,2)+1,
        actions=np.zeros((4,2)),rewards=np.array([0,1,0,0]),terminals=np.array([0,1,0,1]),masks=np.array([1,0,1,1])))


def test_timeout_boundary_retains_bootstrap_and_padding():
    ds=dataset()
    batch=ds.sample_rql(256,h=4)
    # Both episodes occur equally despite different bootstrap semantics.
    assert set(batch["masks"][-1])=={0,1}
    np.testing.assert_array_equal(batch["observations"][2:],np.broadcast_to(batch["observations"][1],(3,256,2)))
    assert np.all(batch["terminals"][1:]==1)


def test_terminal_bandit_reward_adapter():
    ds=OfflineDataset(Bandit().sample(32,4),metadata={"terminal_bandit":True})
    batch=ds.sample_rql(32,1)
    assert not batch["masks"].any()
    shifted=np.concatenate([np.zeros((1,32)),batch["terminals"][:-1]],axis=0)
    # Native loss's shifted-terminal guard preserves exactly the single reward.
    np.testing.assert_allclose((batch["rewards"]*(1-shifted)).sum(0),batch["rewards"][0])
    with pytest.raises(ValueError,match="h=1"):
        ds.sample_rql(32,2)


def test_transition_adapter_preserves_terminal_reward_and_timeout_successor():
    ds=dataset()
    ds.sequence_adapter="transition_h1"
    batch=ds.sample_rql(1000,h=1)
    # Last transition reward must actually reach the loss, and next observation
    # must be the saved transition successor rather than a clamped current row.
    assert batch["rewards"][0].sum()>100
    np.testing.assert_array_equal(batch["observations"][1],batch["observations"][0]+1)
    timeout=(batch["terminals"][0]==1)&(batch["masks"][0]==1)
    assert timeout.any()
    assert np.all(batch["rewards"][1]==0)


def test_resume_rejects_changed_sequence_semantics():
    ds=dataset(); state=ds.state_dict()
    ds.sequence_adapter="transition_h1"
    with pytest.raises(ValueError,match="sequence adapter"):
        ds.load_state_dict(state)


def test_sampler_resume_and_immutable_data():
    ds=dataset(); ds.sample(5)
    state=ds.state_dict(); expected=ds.sample_rql(8,3)
    clone=dataset(); clone.load_state_dict(state)
    for key,value in clone.sample_rql(8,3).items():
        np.testing.assert_array_equal(value,expected[key])
    with pytest.raises(ValueError):
        ds.arrays["actions"][0]=1
    ds.verify_immutable()


def test_checksum_rejects_modified_saved_reward(tmp_path):
    path=tmp_path/"data.npz"
    arrays=Bandit().sample(32)
    save_dataset(path,arrays,{"terminal_bandit":True})
    arrays["rewards"][0]+=1
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match="checksum"):
        OfflineDataset.load(path)


def test_finite_support_gap_is_exact():
    b=Bandit(gap=True)
    assert b.density(b.centers[-1])==0
    assert Bandit().density(b.centers[-1])>0
    sample=b.sample(10000,12)
    assert np.all(b.density(sample["actions"])>0)
    assert b.reward(b.centers[-1],0)>.9


def test_segment_collision_prevents_tunneling_through_thin_wall():
    assert segments_hit_boxes([0,.5],[1,.5],[[.499,0,.501,1]])
    assert not segments_hit_boxes([0,.2],[1,.2],[[.499,.3,.501,1]])
    env=PointMaze()
    nxt,_,_,_,collision=env.transition(np.array([.40,.60]),np.array([1.,0.]))
    assert collision
    np.testing.assert_allclose(nxt,[.4,.6])


def test_shaping_telescopes_and_timeout_keeps_potential():
    env=PointMaze(max_steps=2)
    initial,_=env.reset()
    sparse=dense=0.
    for i in range(2):
        obs,r,term,trunc,info=env.step([0,1])
        sparse+=env.gamma**i*info["sparse_reward"]
        dense+=env.gamma**i*info["reward_dense"]
    assert trunc and not term
    np.testing.assert_allclose(dense-sparse,-env.potential(initial)+env.gamma**2*env.potential(obs),atol=1e-7)
    env.reset(options={"position":[.85,.55]})
    obs,r,term,trunc,info=env.step([1,0])
    assert term
    np.testing.assert_allclose(info["reward_dense"],1-env.potential([.85,.55]),atol=1e-7)


def test_maze_dataset_has_success_and_timeout_and_both_routes():
    arrays=maze_dataset(60,18)
    assert arrays["rewards"].sum()>5
    boundaries=arrays["terminals"]==1
    assert set(arrays["masks"][boundaries])=={0,1}
    assert {0,1}.issubset(set(arrays["route"]))
    ds=OfflineDataset(arrays)
    assert len(ds)>100


def test_strict_offline_rejects_ppo_before_importing_backend():
    with pytest.raises(ValueError,match="actual online"):
        validate_config({"method":"dppo_online","strict_offline":True})


def test_offline_update_has_no_oracle_or_environment_callback():
    # The complete training path accepts only saved numeric arrays. Monkeypatch
    # oracles to fail while loading and drawing both native batch types.
    from unittest.mock import patch
    arrays=Bandit().sample(64,3)
    with patch.object(Bandit,"reward",side_effect=AssertionError("oracle accessed")), \
         patch.object(PointMaze,"step",side_effect=AssertionError("online step")):
        ds=OfflineDataset(arrays,metadata={"terminal_bandit":True})
        ds.sample(16); ds.sample_rql(16,1)
