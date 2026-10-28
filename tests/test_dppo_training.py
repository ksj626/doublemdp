import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from doublemdp.agents.dppo_online import DPPOAgent, DPPOConfig, BCSequences, OnlineRunner, generalized_advantages
from doublemdp.toy import PointMaze


def config(**kwargs):
    options = dict(device="cpu",denoising_steps=4,ft_denoising_steps=2,
        actor_dims=(16,16,16),critic_dims=(16,16,16),n_envs=2,n_steps=4,
        batch_size=16,update_epochs=1,bc_epochs=5)
    options.update(kwargs)
    return DPPOConfig(**options)


def test_reject_offline_and_single_step_upstream_bug():
    with pytest.raises(ValueError,match="strict_offline"):
        DPPOAgent(2,2,config(strict_offline=True))
    with pytest.raises(ValueError,match="undefined clipping"):
        DPPOAgent(2,2,config(ft_denoising_steps=1))
    agent=DPPOAgent(2,2,config())
    with pytest.raises(ValueError,match="actual old-policy chains"):
        agent.ppo_update({"source":"offline_actions_corrupted"})


def test_sequences_do_not_cross_boundaries_or_mutate_data():
    obs=np.arange(8,dtype=np.float32)[:,None]
    acts=obs.copy()
    dataset=BCSequences(obs,acts,[0,1,0,0,1,0,0,1],horizon=2,seed=3)
    assert dataset.starts.tolist()==[0,2,3,5,6]
    observed=[]
    while not dataset.epoch_finished or not len(observed):
        states,actions=dataset.next_batch(2)
        observed.extend(states[:,0].tolist())
        np.testing.assert_array_equal(actions[:,:,0],states+np.arange(2))
    assert sorted(observed)==[0,2,3,5,6]
    np.testing.assert_array_equal(obs,np.arange(8,dtype=np.float32)[:,None])
    assert not dataset.observations.flags.writeable


def test_timeout_bootstrap_but_no_cross_episode_gae():
    rewards=np.array([[1.],[2.]])
    values=np.array([[.3],[.8]])
    nextvalues=np.array([[5.],[.4]])
    term=np.array([[False],[True]])
    trunc=np.array([[True],[False]])
    adv,ret=generalized_advantages(rewards,values,nextvalues,term,trunc,.9,.95)
    assert adv[0,0]==pytest.approx(1+.9*5-.3)
    assert adv[1,0]==pytest.approx(2-.8)
    official,_=generalized_advantages(rewards,values,nextvalues,term,trunc,.9,.95,official=True)
    assert official[0,0]==pytest.approx(adv[0,0]+.9*.95*adv[1,0])


def test_cli_rejects_mismatched_potential_discount_before_training(tmp_path):
    from doublemdp.data import save_dataset
    arrays={"observations":np.zeros((2,2)),"next_observations":np.zeros((2,2)),
        "actions":np.zeros((2,2)),"rewards":np.zeros(2),"rewards_dense":np.zeros(2),
        "masks":np.ones(2),"terminals":np.array([0,1])}
    dataset=tmp_path/"dataset.npz"
    save_dataset(dataset,arrays,{"gamma":.99})
    cfg=tmp_path/"config.json"
    cfg.write_text(json.dumps({"gamma":.95,"device":"cpu"}))
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/"scripts/train_dppo.py"),
        "--config",str(cfg),"--dataset",str(dataset),"--output",str(tmp_path/"run"),"--reward","dense"]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode != 0
    assert "saved dataset gamma to equal DPPO gamma" in result.stderr
    assert not (tmp_path/"run/latest.pt").exists()


def test_real_rollout_counts_and_no_old_chain_bptt():
    agent=DPPOAgent(2,2,config())
    agent.begin_online()
    frozen=copy.deepcopy(agent.model.actor.state_dict())
    runner=OnlineRunner(agent,[PointMaze(max_steps=2) for _ in range(2)])
    rollout=runner.collect()
    assert agent.counters["online_env_steps"]==8
    assert len(rollout["completed_episodes"])==4
    old=rollout["chains"].copy()
    result=agent.ppo_update(rollout)
    assert result["ratio"]==pytest.approx(1.,abs=1e-6)
    assert np.isfinite(list(result.values())).all()
    np.testing.assert_array_equal(old,rollout["chains"])
    for name,value in agent.model.actor.state_dict().items():
        torch.testing.assert_close(value,frozen[name],rtol=0,atol=0)
    assert all(p.grad is None for p in agent.model.actor.parameters())
    assert agent.counters["ppo_updates"]==1
    assert agent.counters["evaluation_env_steps"]==0


def test_evaluation_restores_global_rng_and_never_counts_training_steps():
    from scripts.train_dppo import evaluate
    class GlobalRandomMaze(PointMaze):
        def reset(self,**kwargs):
            random.random()
            np.random.rand()
            return super().reset(**kwargs)
    agent=DPPOAgent(2,2,config())
    random.seed(18)
    np.random.seed(19)
    expected=(random.random(),np.random.rand())
    random.seed(18)
    np.random.seed(19)
    torch_state=torch.get_rng_state()
    result,trajectories=evaluate(agent,lambda:GlobalRandomMaze(max_steps=2),episodes=5,parallel_envs=3)
    assert (random.random(),np.random.rand())==expected
    torch.testing.assert_close(torch_state,torch.get_rng_state(),rtol=0,atol=0)
    assert agent.counters["online_env_steps"]==0
    assert agent.counters["evaluation_env_steps"]==10
    assert result["episode_lengths"]==[2]*5 and len(trajectories)==5


@pytest.mark.parametrize("device",["cpu",pytest.param("cuda:0",marks=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required for GPU resume parity"))])
def test_full_resume_rng_sampler_optimizer_and_environment(tmp_path,device):
    c=config(reward_scale_running=True,device=device)
    agent=DPPOAgent(2,2,c)
    sampler=BCSequences(np.arange(12).reshape(6,2),np.zeros((6,2)),[0,1,0,1,0,1],seed=5)
    obs,actions=sampler.next_batch(2)
    agent.pretrain_update(obs,actions)
    agent.begin_online()
    runner=OnlineRunner(agent,[PointMaze(max_steps=7) for _ in range(2)])
    agent.ppo_update(runner.collect())
    path=tmp_path/"state.pt"
    agent.save(path,sampler=sampler,runner=runner)
    expected_batch=sampler.next_batch(2)
    expected_rng=(random.random(),np.random.rand())
    expected_rollout=runner.collect()
    expected_metrics=agent.ppo_update(expected_rollout)
    expected_params=copy.deepcopy(agent.model.state_dict())
    recovered,payload=DPPOAgent.load(path,sampler=sampler,runner=runner)
    assert (random.random(),np.random.rand())==expected_rng
    actual_batch=sampler.next_batch(2)
    for a,b in zip(actual_batch,expected_batch):
        np.testing.assert_array_equal(a,b)
    actual_rollout=runner.collect()
    for key in ("observations","chains","logprobs","advantages","returns"):
        np.testing.assert_array_equal(actual_rollout[key],expected_rollout[key])
    actual_metrics=recovered.ppo_update(actual_rollout)
    assert actual_metrics==expected_metrics
    for name,value in recovered.model.state_dict().items():
        torch.testing.assert_close(value,expected_params[name],rtol=0,atol=0)
    for optimizer_name in ("bc_optimizer","actor_optimizer","critic_optimizer"):
        expected=getattr(agent,optimizer_name).state_dict()
        actual=getattr(recovered,optimizer_name).state_dict()
        assert actual["param_groups"]==expected["param_groups"]
        for parameter,state in expected["state"].items():
            for key,value in state.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(actual["state"][parameter][key],value,rtol=0,atol=0)
                else:
                    assert actual["state"][parameter][key]==value
    assert recovered.counters==agent.counters


def test_cli_rejects_online_checkpoint_as_strict_offline_bc(tmp_path):
    from dataclasses import asdict
    from doublemdp.data import save_dataset,OfflineDataset
    dataset=tmp_path/"data.npz"
    save_dataset(dataset,{"observations":np.zeros((2,2)),"next_observations":np.zeros((2,2)),
        "actions":np.zeros((2,2)),"rewards":np.zeros(2),"masks":np.ones(2),"terminals":np.array([0,1])},{})
    c=config()
    agent=DPPOAgent(2,2,c)
    agent.begin_online()
    checkpoint=tmp_path/"online.pt"
    agent.save(checkpoint,metadata={"dataset_sha256":OfflineDataset.load(dataset).digest,"env":"maze","reward":"sparse"})
    cfg=tmp_path/"config.json"
    cfg.write_text(json.dumps(asdict(c)))
    output=tmp_path/"output"
    result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"scripts/train_dppo.py"),
        "--config",str(cfg),"--dataset",str(dataset),"--output",str(output),"--resume",str(checkpoint),
        "--bc-only","--strict-offline"],capture_output=True,text=True)
    assert result.returncode!=0 and "cannot become an offline BC reference" in result.stderr
    assert not (output/"metadata.json").exists() and not (output/"completed.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable; never pass CPU as GPU validation")
def test_gpu_matmul_gradient_optimizer_and_actual_dppo_update():
    device="cuda:0"
    torch.manual_seed(7)
    weights=torch.nn.Parameter(torch.randn(128,128,device=device))
    inputs=torch.randn(128,128,device=device)
    original=weights.detach().clone()
    opt=torch.optim.Adam([weights],lr=.01)
    loss=(inputs@weights).square().mean()
    loss.backward()
    assert torch.isfinite(weights.grad).all()
    opt.step()
    torch.cuda.synchronize()
    assert not torch.equal(weights,original)
    agent=DPPOAgent(2,2,config(device=device))
    bc=agent.pretrain_update(np.zeros((4,2),dtype=np.float32),np.zeros((4,1,2),dtype=np.float32))
    agent.begin_online()
    result=agent.ppo_update(OnlineRunner(agent,[PointMaze(max_steps=3) for _ in range(2)]).collect())
    torch.cuda.synchronize()
    evidence={"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
        "device":str(weights.device),"name":torch.cuda.get_device_name(0),
        "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "matmul_loss":loss.item(),"optimizer_parameter_changed":True,
        "bc":bc,"dppo":result,"online_env_steps":agent.counters["online_env_steps"],
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(0)}
    target=os.environ.get("DPPO_RUNTIME_REPORT")
    if target:
        Path(target).write_text(json.dumps(evidence,indent=2)+"\n")

