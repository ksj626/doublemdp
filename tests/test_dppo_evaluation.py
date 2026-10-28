from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from doublemdp.agents.dppo_online import DPPOAgent,DPPOConfig
from doublemdp.agents.dppo_artifacts import load_evaluation_checkpoint


def small_agent():
    return DPPOAgent(2,2,DPPOConfig(denoising_steps=4,ft_denoising_steps=2,actor_dims=(16,16,16),critic_dims=(16,16,16)))


@pytest.mark.parametrize("online",[False,True])
def test_explicit_noise_trace_and_remaining_generation_match_native(online):
    agent=small_agent()
    if online:
        agent.begin_online()
    obs=np.zeros((3,2),np.float32)
    torch.manual_seed(71)
    expected=agent.sample_actions(obs,deterministic=True)
    torch.manual_seed(71)
    initial=torch.randn(3,1,2)
    noise=torch.stack([torch.randn_like(initial) for _ in range(4)])
    before=torch.get_rng_state()
    trace=agent.trace_with_noise(obs,initial,noise)
    np.testing.assert_array_equal(trace["final_actions"],expected)
    for i,t in enumerate(trace["native_times"]):
        resumed=agent.continue_generation(obs,trace["latents"][:,i],t,deterministic=True,noise_tape=noise)
        np.testing.assert_array_equal(resumed,expected)
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)


def test_evaluation_checkpoint_identity_and_phase_are_enforced(tmp_path):
    agent=small_agent()
    path=tmp_path/"policy.pt"
    metadata={"env":"maze","reward":"dense","dataset_sha256":"test-dataset"}
    agent.save(path,metadata=metadata)
    _,identity=load_evaluation_checkpoint(path,strict_offline=True)
    assert identity["method"]=="bc" and identity["phase"]=="bc"
    with pytest.raises(ValueError,match="SHA256"):
        load_evaluation_checkpoint(path,expected_sha256="incorrect")
    with pytest.raises(ValueError,match="environment differs"):
        load_evaluation_checkpoint(path,env="another-task")
    agent.begin_online()
    agent.save(path,metadata=metadata)
    _,identity=load_evaluation_checkpoint(path)
    assert identity["method"]=="dppo_online"
    with pytest.raises(ValueError,match="not an offline BC"):
        load_evaluation_checkpoint(path,strict_offline=True)


def test_continuation_uses_true_sparse_terminal_return_and_preserves_rng():
    from doublemdp.agents.dppo_diagnostics import rollouts,noise_tape,q_probe
    from doublemdp.toy import PointMaze
    agent=small_agent()
    states=np.array([[.84,.55],PointMaze.goal,[.45,.3]],dtype=np.float32)
    before=torch.get_rng_state()
    result=rollouts(agent,states,noise_tape(4,1,3,4),first_actions=np.array([[1.,0.],[0.,0.],[0.,0.]]))
    assert result["returns"][0]==1 and result["returns"][1]==0
    assert np.isnan(result["returns"][2]) and result["counts"].tolist()==[1,0,0]
    probes=PointMaze.probes[:1]
    generation=noise_tape(8,1,2,4)[0]
    trace=agent.trace_with_noise(np.repeat(probes,2,axis=0),generation[0],generation[1:])
    trace["particles_per_probe"]=2
    probe=q_probe(agent,probes,trace,np.array([0,2,4]),noise_tape(9,1,4,4)[0,1:],noise_tape(10,3,4,4))
    assert probe["samples"].shape==(1,3,4)
    np.testing.assert_array_equal(probe["endpoints"][-1],np.repeat(trace["final_actions"][:1,0],4,axis=0))
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
    assert agent.counters["online_env_steps"]==0
    from doublemdp.diagnostics import continuation_probe
    generic=continuation_probe(agent,"dppo_online",probes[0],trace,repetitions=2,horizon=2)
    assert generic["mean"].shape==(5,) and "no inner head" in generic["quantity"]
