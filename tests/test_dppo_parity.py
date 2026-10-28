"""Fixed-input comparisons against the pinned, separately imported v0.6 source."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"third_party/dppo"))
from model.diffusion.diffusion_ppo import PPODiffusion as UpstreamPPO
from model.diffusion.diffusion import DiffusionModel as UpstreamBC
from model.diffusion.sampling import cosine_beta_schedule as upstream_schedule
from doublemdp.agents.dppo_online import DPPOAgent, DPPOConfig
from doublemdp.agents.dppo_kernels import cosine_beta_schedule


def small_config(**kwargs):
    options = dict(device="cpu", denoising_steps=6, ft_denoising_steps=3,
                   actor_dims=(16,16,16), critic_dims=(16,16,16),
                   n_envs=2, n_steps=3, batch_size=18, update_epochs=1)
    options.update(kwargs)
    return DPPOConfig(**options)


def pair(config=None):
    agent = DPPOAgent(3, 2, config or small_config())
    agent.begin_online()
    model, c = agent.model, agent.config
    upstream = UpstreamPPO(actor=copy.deepcopy(model.actor).requires_grad_(True), critic=copy.deepcopy(model.critic),
        ft_denoising_steps=c.ft_denoising_steps, horizon_steps=c.horizon_steps, obs_dim=3, action_dim=2,
        device=c.device, denoising_steps=c.denoising_steps, gamma_denoising=c.gamma_denoising,
        clip_ploss_coef=c.clip_ploss_coef, clip_ploss_coef_base=c.clip_ploss_coef_base,
        clip_ploss_coef_rate=c.clip_ploss_coef_rate,
        min_sampling_denoising_std=float(c.min_sampling_denoising_std),
        min_logprob_denoising_std=c.min_logprob_denoising_std,
        randn_clip_value=c.randn_clip_value, final_action_clip_value=c.final_action_clip_value)
    upstream.load_state_dict(model.state_dict())
    return agent, upstream


@pytest.mark.parametrize("steps", [6,20,100])
def test_schedule_source_parity(steps):
    torch.testing.assert_close(cosine_beta_schedule(steps), upstream_schedule(steps), rtol=0, atol=0)


@pytest.mark.parametrize("ft", [3,6])
def test_actual_old_chains_sampler_logprob_loss_gradient_parity(ft):
    agent, upstream = pair(small_config(ft_denoising_steps=ft, horizon_steps=3, act_steps=2,
        clip_ploss_coef=.2, clip_ploss_coef_base=.001))
    model = agent.model
    obs = {"state": torch.randn(8,1,3)}
    for deterministic in (False, True):
        torch.manual_seed(123)
        actual = model(obs, deterministic=deterministic, return_chain=True)
        torch.manual_seed(123)
        expected = upstream(obs, deterministic=deterministic, return_chain=True)
        torch.testing.assert_close(actual.chains, expected.chains, rtol=0, atol=0)
        torch.testing.assert_close(actual.trajectories, expected.trajectories, rtol=0, atol=0)
        assert actual.chains.shape == (8,ft+1,3,2)
    torch.manual_seed(420)
    sample = model(obs, return_chain=True)
    assert not sample.chains.requires_grad
    old = model.get_logprobs(obs, sample.chains).detach().reshape(8,ft,3,2)
    reference = upstream.get_logprobs(obs, sample.chains).reshape_as(old)
    torch.testing.assert_close(old, reference, rtol=0, atol=0)
    env_idx = torch.arange(8)
    denoise = env_idx % ft
    previous, following = sample.chains[env_idx,denoise], sample.chains[env_idx,denoise+1]
    returns, values, advantage = torch.randn(8), torch.randn(8), torch.randn(8)
    args = (obs, previous, following, denoise, returns, values, advantage, old[env_idx,denoise])
    # Actual chains under an unchanged old policy must give ratio=1.
    result = model.loss(*args, reward_horizon=2)
    assert result[5] == pytest.approx(1., abs=1e-7)
    assert result[4] == pytest.approx(0., abs=1e-7)
    # Perturb both actors identically and verify every component/reduction/gradient.
    with torch.no_grad():
        for param in model.actor_ft.parameters():
            param.add_(.002)
    upstream.load_state_dict(model.state_dict())
    result = model.loss(*args, reward_horizon=2)
    reference = upstream.loss(*args, reward_horizon=2)
    for actual, expected in zip(result, reference):
        torch.testing.assert_close(torch.as_tensor(actual), torch.as_tensor(expected), rtol=1e-6, atol=1e-7)
    (result[0]+.5*result[2]).backward()
    (reference[0]+.5*reference[2]).backward()
    for actual, expected in zip(model.actor_ft.parameters(), upstream.actor_ft.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
    assert all(p.grad is None and not p.requires_grad for p in model.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actor_ft.parameters())


def test_bc_loss_and_adamw_update_source_parity():
    agent = DPPOAgent(3,2,small_config())
    upstream = UpstreamBC(copy.deepcopy(agent.bc.network),1,3,2,device="cpu",denoising_steps=6,randn_clip_value=3)
    upstream.load_state_dict(agent.bc.state_dict())
    opt = torch.optim.AdamW(upstream.parameters(), lr=agent.bc_optimizer.param_groups[0]["lr"], weight_decay=agent.config.bc_weight_decay)
    obs, actions = torch.randn(8,3), torch.randn(8,1,2)
    torch.manual_seed(41)
    result = agent.pretrain_update(obs, actions)
    torch.manual_seed(41)
    loss = upstream.loss(actions, {"state":obs[:,None]})
    loss.backward()
    opt.step()
    assert result["bc_loss"] == pytest.approx(loss.item(), abs=1e-7)
    for actual, expected in zip(agent.bc.parameters(), upstream.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["bc","online"])
@pytest.mark.parametrize("deterministic", [False,True])
def test_trace_does_not_change_sample_or_rng(phase, deterministic):
    agent = DPPOAgent(3,2,small_config())
    if phase == "online":
        agent.begin_online()
    obs = np.zeros((5,3),dtype=np.float32)
    torch.manual_seed(88)
    actions = agent.sample_actions(obs, deterministic=deterministic)
    rng = torch.get_rng_state()
    torch.manual_seed(88)
    traced, trace = agent.sample_actions(obs, deterministic=deterministic,return_trace=True)
    np.testing.assert_array_equal(actions,traced)
    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
    assert trace["latents"].shape == (5,7,1,2)
    assert trace["native_times"].tolist() == [5,4,3,2,1,0,-1]
    assert trace["native_inner_values"] is None


def test_distinct_sampling_and_likelihood_floors():
    agent, upstream = pair(small_config(min_sampling_denoising_std=.2,min_logprob_denoising_std=.1))
    obs={"state":torch.zeros(4,1,3)}
    torch.manual_seed(2)
    sample=agent.model(obs)
    torch.testing.assert_close(agent.model.get_logprobs(obs,sample.chains),upstream.get_logprobs(obs,sample.chains))
    # Last denoising variance is clipped from ~zero independently in likelihood.
    t=torch.zeros(4,dtype=torch.long)
    mu, logvar, _=agent.model.p_mean_var(sample.chains[:,-2],t,obs)
    expected=torch.distributions.Normal(mu,torch.full_like(mu,.1)).log_prob(sample.chains[:,-1])
    actual=agent.model.get_logprobs_subsample(obs,sample.chains[:,-2],sample.chains[:,-1],torch.full((4,),2))
    torch.testing.assert_close(actual,expected)

