"""Native PyTorch DPPO: offline diffusion BC, then real on-policy rollouts.

The numerical kernels are extracted from official v0.6. This small harness
handles phase boundaries, real interaction accounting, and resumable state.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
import numpy as np
import torch

from .dppo_kernels import (CriticObs, DiffusionMLP, DiffusionModel, PPODiffusion,
                           RunningRewardScaler, CosineAnnealingWarmupRestarts,
                           make_timesteps)


@dataclass
class DPPOConfig:
    profile: str = "toy_diagnostic"
    strict_offline: bool = False
    device: str = "cpu"
    seed: int = 0
    horizon_steps: int = 1
    act_steps: int = 1
    cond_steps: int = 1
    denoising_steps: int = 20
    ft_denoising_steps: int = 10
    actor_dims: tuple = (128, 128, 128)
    critic_dims: tuple = (128, 128, 128)
    time_dim: int = 16
    residual_style: bool = True
    bc_lr: float = 1e-3
    bc_weight_decay: float = 1e-6
    bc_epochs: int = 100
    bc_batch_size: int = 128
    bc_min_lr: float = 1e-4
    bc_warmup_epochs: int = 1
    ema_decay: float = .995
    ema_start_epoch: int = 10
    ema_frequency: int = 5
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_weight_decay: float = 0.
    critic_weight_decay: float = 0.
    online_iterations: int = 100
    lr_warmup_iterations: int = 10
    gamma: float = .99
    gae_lambda: float = .95
    gamma_denoising: float = .99
    clip_ploss_coef: float = .01
    clip_ploss_coef_base: float = .01
    clip_ploss_coef_rate: float = 3.
    clip_vloss_coef: float | None = None
    clip_advantage_lower_quantile: float = 0.
    clip_advantage_upper_quantile: float = 1.
    norm_adv: bool = True
    min_sampling_denoising_std: float = .1
    min_logprob_denoising_std: float = .1
    randn_clip_value: float = 3.
    denoised_clip_value: float | None = 1.
    final_action_clip_value: float | None = None
    reward_scale_running: bool = False
    reward_scale_const: float = 1.
    reward_horizon: int | None = None
    n_steps: int = 64
    n_envs: int = 4
    batch_size: int = 256
    update_epochs: int = 5
    vf_coef: float = .5
    target_kl: float | None = 1.
    max_grad_norm: float | None = None
    critic_warmup_iterations: int = 0
    use_bc_loss: bool = False
    bc_loss_coeff: float = 0.

    def validate(self):
        if self.strict_offline:
            raise ValueError("dppo_online requires new environment interactions; strict_offline=True is incompatible. Use the pretrained BC policy for offline comparisons.")
        if self.profile not in {"toy_diagnostic", "ogbench_port", "official_code_v0.6"}:
            raise ValueError("Unknown DPPO profile")
        if not 2 <= self.ft_denoising_steps <= self.denoising_steps:
            raise ValueError("Require 2 <= ft_denoising_steps <= denoising_steps; upstream v0.6 has undefined clipping for one step.")
        if not 1 <= self.act_steps <= self.horizon_steps:
            raise ValueError("act_steps must be within the predicted action horizon")
        if self.cond_steps != 1:
            raise NotImplementedError("The compact rollout adapter supports cond_steps=1; the native kernel supports observation history.")


def _scheduler(optimizer, steps, maximum, minimum, warmup):
    return CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=max(2, steps),
        max_lr=maximum, min_lr=minimum, warmup_steps=min(warmup, max(2, steps)-1))


class BCSequences:
    """v0.6 uniform valid sequence starts; no action padding across boundaries."""
    def __init__(self, observations, actions, terminals, horizon=1, seed=0):
        self.observations = np.array(observations, dtype=np.float32, copy=True)
        self.actions = np.array(actions, dtype=np.float32, copy=True)
        self.observations.setflags(write=False)
        self.actions.setflags(write=False)
        boundaries = np.flatnonzero(terminals).tolist()
        if not boundaries or boundaries[-1] != len(actions)-1:
            boundaries.append(len(actions)-1)
        self.starts = []
        start = 0
        for end in boundaries:
            self.starts.extend(range(start, end-horizon+2))
            start = end+1
        self.starts = np.asarray(self.starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError("No within-episode action sequences fit the horizon")
        self.horizon = horizon
        self.rng = np.random.RandomState(seed)
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.epoch = 0

    def next_batch(self, batch_size):
        if self.cursor == len(self.order):
            self.order = self.rng.permutation(len(self.starts))
            self.cursor = 0
            self.epoch += 1
        indices = self.order[self.cursor:self.cursor+batch_size]
        self.cursor += len(indices)
        starts = self.starts[indices]
        return self.observations[starts], self.actions[starts[:, None]+np.arange(self.horizon)]

    @property
    def epoch_finished(self):
        return self.cursor == len(self.order)

    def state_dict(self):
        return {"rng": self.rng.get_state(), "order": self.order, "cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.rng.set_state(state["rng"])
        self.order, self.cursor, self.epoch = state["order"], state["cursor"], state["epoch"]


class DPPOAgent:
    method = "dppo_online"
    inner_critic_semantics = None  # DPPO learns V(s), never V(s, latent, time).

    def __init__(self, obs_dim, action_dim, config=None):
        self.config = config or DPPOConfig()
        c = self.config
        c.validate()
        random.seed(c.seed)
        np.random.seed(c.seed)
        torch.manual_seed(c.seed)
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.device = torch.device(c.device)
        actor = DiffusionMLP(action_dim, c.horizon_steps, obs_dim*c.cond_steps,
            time_dim=c.time_dim, mlp_dims=list(c.actor_dims), activation_type="ReLU",
            residual_style=c.residual_style)
        self.bc = DiffusionModel(actor, c.horizon_steps, obs_dim, action_dim,
            device=c.device, denoising_steps=c.denoising_steps,
            denoised_clip_value=c.denoised_clip_value, randn_clip_value=c.randn_clip_value,
            final_action_clip_value=c.final_action_clip_value)
        self.ema = copy.deepcopy(self.bc)
        self.bc_optimizer = torch.optim.AdamW(self.bc.parameters(), lr=c.bc_lr,
                                              weight_decay=c.bc_weight_decay)
        self.bc_scheduler = _scheduler(self.bc_optimizer, c.bc_epochs, c.bc_lr, c.bc_min_lr, c.bc_warmup_epochs)
        self.model = None
        self.phase = "bc"
        self.counters = dict(bc_updates=0, bc_epochs=0, online_iterations=0,
                             ppo_updates=0, online_env_steps=0, evaluation_env_steps=0)
        self.bc_seconds = 0.

    @classmethod
    def create(cls, obs_dim, action_dim, config=None):
        return cls(obs_dim, action_dim, config)

    def tensor(self, value):
        return torch.as_tensor(np.asarray(value) if not torch.is_tensor(value) else value,
                               dtype=torch.float32, device=self.device)

    def cond(self, observations):
        observations = self.tensor(observations)
        if observations.ndim == 2:
            observations = observations[:, None, :]
        return {"state": observations}

    def pretrain_update(self, observations, action_chunks):
        if self.phase != "bc":
            raise RuntimeError("BC updates are forbidden after the pretrained actor is frozen")
        started = time.perf_counter()
        action_chunks = self.tensor(action_chunks)
        if action_chunks.ndim == 2:
            action_chunks = action_chunks[:, None]
        self.bc_optimizer.zero_grad(set_to_none=True)
        loss = self.bc.loss(action_chunks, self.cond(observations))
        loss.backward()
        self.bc_optimizer.step()
        self.counters["bc_updates"] += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.bc_seconds += time.perf_counter()-started
        return {"bc_loss": loss.item(), "bc_updates": self.counters["bc_updates"]}

    def finish_bc_epoch(self):
        self.counters["bc_epochs"] += 1
        epoch = self.counters["bc_epochs"]
        c = self.config
        self.bc_scheduler.step()
        if epoch % c.ema_frequency == 0:
            if epoch < c.ema_start_epoch:
                self.ema.load_state_dict(self.bc.state_dict())
            else:
                with torch.no_grad():
                    for ema, current in zip(self.ema.parameters(), self.bc.parameters()):
                        ema.data = ema.data*c.ema_decay+(1-c.ema_decay)*current.data

    def begin_online(self):
        if self.model is not None:
            return
        c = self.config
        critic = CriticObs(self.obs_dim*c.cond_steps, list(c.critic_dims),
                           activation_type="Mish", residual_style=c.residual_style)
        self.model = PPODiffusion(actor=copy.deepcopy(self.ema.network), critic=critic,
            ft_denoising_steps=c.ft_denoising_steps, horizon_steps=c.horizon_steps,
            obs_dim=self.obs_dim, action_dim=self.action_dim, device=c.device,
            denoising_steps=c.denoising_steps, gamma_denoising=c.gamma_denoising,
            clip_ploss_coef=c.clip_ploss_coef, clip_ploss_coef_base=c.clip_ploss_coef_base,
            clip_ploss_coef_rate=c.clip_ploss_coef_rate, clip_vloss_coef=c.clip_vloss_coef,
            clip_advantage_lower_quantile=c.clip_advantage_lower_quantile,
            clip_advantage_upper_quantile=c.clip_advantage_upper_quantile, norm_adv=c.norm_adv,
            min_sampling_denoising_std=float(c.min_sampling_denoising_std),
            min_logprob_denoising_std=c.min_logprob_denoising_std,
            randn_clip_value=c.randn_clip_value, denoised_clip_value=c.denoised_clip_value,
            final_action_clip_value=c.final_action_clip_value)
        self.actor_optimizer = torch.optim.AdamW(self.model.actor_ft.parameters(),
            lr=c.actor_lr, weight_decay=c.actor_weight_decay)
        self.critic_optimizer = torch.optim.AdamW(self.model.critic.parameters(),
            lr=c.critic_lr, weight_decay=c.critic_weight_decay)
        self.actor_scheduler = _scheduler(self.actor_optimizer, c.online_iterations,
            c.actor_lr, c.actor_lr, c.lr_warmup_iterations)
        self.critic_scheduler = _scheduler(self.critic_optimizer, c.online_iterations,
            c.critic_lr, c.critic_lr, c.lr_warmup_iterations)
        self.phase = "online"

    @torch.no_grad()
    def sample_actions(self, observations, *, deterministic=False, return_trace=False):
        cond = self.cond(observations)
        model = self.model if self.phase == "online" else self.ema
        if not return_trace:
            # Upstream v0.6 return_chain=False crashes when all steps fine-tuned.
            # Always collect native chain and discard; this does not alter RNG.
            sample = model(cond, deterministic=deterministic)
            return sample.trajectories.cpu().numpy()
        x = torch.randn((len(cond["state"]), self.config.horizon_steps, self.action_dim), device=self.device)
        states = [x.clone()]
        times = list(reversed(range(model.denoising_steps)))
        for i, t in enumerate(times):
            t_b = make_timesteps(len(x), t, self.device)
            result = model.p_mean_var(x, t_b, cond)
            mean, logvar = result[:2]
            std = torch.exp(.5*logvar)
            if self.phase == "bc" or deterministic:
                std = torch.zeros_like(std) if t == 0 else torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=model.get_min_sampling_denoising_std())
            noise = torch.randn_like(x).clamp_(-model.randn_clip_value, model.randn_clip_value)
            x = mean+std*noise
            if model.final_action_clip_value is not None and i == len(times)-1:
                x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
            states.append(x.clone())
        actions = x.cpu().numpy()
        return actions, {"latents": torch.stack(states, 1).cpu().numpy(),
            "native_times": np.array(times+[-1]), "progress": np.linspace(0, 1, len(states)),
            "final_actions": actions, "native_inner_values": None,
            "native_environment_values": None if self.model is None else self.model.critic(cond).cpu().numpy(),
            "inner_value_semantics": "unavailable: DPPO has an environment state-value critic"}

    @torch.no_grad()
    def continue_generation(self, observations, latents, native_time, deterministic=False, noise_tape=None):
        """Evaluation-only continuation of one DDPM candidate at a native time."""
        model = self.model if self.phase == "online" else self.ema
        x = self.tensor(latents).clone()
        cond = self.cond(observations)
        if noise_tape is not None:
            noise_tape=self.tensor(noise_tape)
            if tuple(noise_tape.shape)!=(model.denoising_steps,*x.shape):
                raise ValueError("noise_tape must contain every native denoising step, then batch/horizon/action axes")
        for t in range(int(native_time), -1, -1):
            noise=torch.randn_like(x) if noise_tape is None else noise_tape[model.denoising_steps-1-t]
            x=self._probe_generation_step(x,t,cond,noise,deterministic)
        return x.cpu().numpy()

    def _probe_generation_step(self,x,t,cond,noise,deterministic):
        """Native DDPM transition; used only by supplied-noise diagnostics."""
        model=self.model if self.phase=="online" else self.ema
        mean,logvar,*_=model.p_mean_var(x,make_timesteps(len(x),t,self.device),cond)
        std=torch.exp(.5*logvar)
        if self.phase=="bc" or deterministic:
            std=torch.zeros_like(std) if t==0 else torch.clip(std,min=1e-3)
        else:
            std=torch.clip(std,min=model.get_min_sampling_denoising_std())
        x=mean+std*noise.clamp(-model.randn_clip_value,model.randn_clip_value)
        if t==0 and model.final_action_clip_value is not None:
            x=x.clamp(-model.final_action_clip_value,model.final_action_clip_value)
        return x

    @torch.no_grad()
    def trace_with_noise(self,observations,initial_noise,noise_tape,*,deterministic=True):
        """Actual native transitions with explicit draws, without global RNG use."""
        model=self.model if self.phase=="online" else self.ema
        x=self.tensor(initial_noise).clone()
        noise_tape=self.tensor(noise_tape)
        if tuple(noise_tape.shape)!=(model.denoising_steps,*x.shape):
            raise ValueError("noise_tape must contain every native denoising step, then batch/horizon/action axes")
        cond=self.cond(observations)
        states=[x.clone()]
        times=list(reversed(range(model.denoising_steps)))
        for i,t in enumerate(times):
            x=self._probe_generation_step(x,t,cond,noise_tape[i],deterministic)
            states.append(x.clone())
        return {"latents":torch.stack(states,1).cpu().numpy(),"native_times":np.array(times+[-1]),
            "progress":np.linspace(0,1,len(states)),"final_actions":x.cpu().numpy(),
            "native_environment_values":None if self.model is None else self.model.critic(cond).cpu().numpy()}

    def ppo_update(self, rollout):
        if self.phase != "online" or rollout.get("source") != "real_on_policy_rollout":
            raise ValueError("DPPO PPO update requires actual old-policy chains collected from online environment rollouts")
        c = self.config
        obs, chains = self.cond(rollout["observations"]), self.tensor(rollout["chains"])
        returns, values, advantages, logprobs = [self.tensor(rollout[k]) for k in ("returns", "values", "advantages", "logprobs")]
        total = len(chains)*c.ft_denoising_steps
        metrics = []
        stop = False
        for _ in range(c.update_epochs):
            inds = torch.randperm(total, device=self.device)
            for batch in range(max(1, total//c.batch_size)):
                idx = inds[batch*c.batch_size:(batch+1)*c.batch_size]
                if len(idx) < 2 and c.norm_adv:
                    raise ValueError("Advantage normalization requires at least two minibatch elements")
                batch_idx, denoise_idx = torch.unravel_index(idx, (len(chains), c.ft_denoising_steps))
                result = self.model.loss({"state": obs["state"][batch_idx]},
                    chains[batch_idx, denoise_idx], chains[batch_idx, denoise_idx+1], denoise_idx,
                    returns[batch_idx], values[batch_idx], advantages[batch_idx], logprobs[batch_idx, denoise_idx],
                    use_bc_loss=c.use_bc_loss, reward_horizon=c.reward_horizon or c.act_steps)
                pg, entropy, value, clipfrac, kl, ratio, bc_loss, eta = result
                loss = pg+c.vf_coef*value+c.bc_loss_coeff*bc_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.counters["online_iterations"] >= c.critic_warmup_iterations:
                    if c.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), c.max_grad_norm)
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.counters["ppo_updates"] += 1
                metrics.append([pg.item(), value.item(), kl, ratio, clipfrac])
                if c.target_kl is not None and kl > c.target_kl:
                    stop = True
                    break
            if stop:
                break
        if self.counters["online_iterations"] >= c.critic_warmup_iterations:
            self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.model.step()
        self.counters["online_iterations"] += 1
        return dict(zip(("policy_loss", "value_loss", "approx_kl", "ratio", "clipfrac"), np.mean(metrics, axis=0).tolist()))

    def save(self, path, *, sampler=None, runner=None, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=1, config=asdict(self.config), obs_dim=self.obs_dim,
            action_dim=self.action_dim, phase=self.phase, counters=self.counters,
            bc_seconds=self.bc_seconds, bc=self.bc.state_dict(), ema=self.ema.state_dict(),
            bc_optimizer=self.bc_optimizer.state_dict(), bc_scheduler=self.bc_scheduler.state_dict(),
            rng={"python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            sampler=None if sampler is None else sampler.state_dict(),
            runner=None if runner is None else runner.state_dict(), metadata=metadata or {})
        if self.model is not None:
            for key in ("model", "actor_optimizer", "critic_optimizer", "actor_scheduler", "critic_scheduler"):
                payload[key] = getattr(self, key).state_dict()
            payload["ft_denoising_steps_cnt"] = self.model.ft_denoising_steps_cnt
        temporary = path.with_suffix(path.suffix+".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, device=None, sampler=None, runner=None):
        # Only load trusted locally produced checkpoints: RNG states include Python objects.
        # Keep non-capturable AdamW scalar step tensors on CPU. Model and
        # optimizer loaders move parameter-shaped state to the selected device.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload["config"]
        if device is not None:
            config["device"] = device
        agent = cls(payload["obs_dim"], payload["action_dim"], DPPOConfig(**config))
        for key in ("bc", "ema", "bc_optimizer", "bc_scheduler"):
            getattr(agent, key).load_state_dict(payload[key])
        if payload["phase"] == "online":
            agent.begin_online()
            for key in ("model", "actor_optimizer", "critic_optimizer", "actor_scheduler", "critic_scheduler"):
                getattr(agent, key).load_state_dict(payload[key])
            agent.model.ft_denoising_steps_cnt = payload["ft_denoising_steps_cnt"]
        agent.counters, agent.bc_seconds = payload["counters"], payload["bc_seconds"]
        if sampler is not None and payload["sampler"] is not None:
            sampler.load_state_dict(payload["sampler"])
        if runner is not None and payload["runner"] is not None:
            runner.agent = agent
            runner.load_state_dict(payload["runner"])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if payload["rng"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in payload["rng"]["cuda"]])
        return agent, payload


def generalized_advantages(rewards, values, next_values, terminated, truncated,
                           gamma, lam, *, official=False):
    """Port: bootstrap timeout final states, stop recursion at reset boundaries.

    Official v0.6: use next collected state and terminated-only recursion. The
    profile switch is explicit because this numerical difference is substantive.
    """
    advantages = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        bootstrap = 1.-terminated[t]
        continuation = bootstrap if official else bootstrap*(1.-truncated[t])
        delta = rewards[t]+gamma*next_values[t]*bootstrap-values[t]
        advantages[t] = last = delta+gamma*lam*continuation*last
    return advantages, advantages+values


class OnlineRunner:
    """Collects actual native chains before any updates; owns persistent env state."""
    def __init__(self, agent, envs):
        self.agent, self.envs = agent, envs
        if len(envs) != agent.config.n_envs:
            raise ValueError("n_envs differs from environment list")
        self.observations = np.stack([e.reset(seed=agent.config.seed+i)[0] for i, e in enumerate(envs)])
        self.first = np.ones(len(envs))
        self.episode_returns = np.zeros(len(envs))
        self.episode_lengths = np.zeros(len(envs), dtype=int)
        self.scaler = RunningRewardScaler(len(envs), gamma=agent.config.gamma)

    def collect(self):
        a, c = self.agent, self.agent.config
        if a.phase != "online":
            raise RuntimeError("Call begin_online after BC initialization")
        observations, chains, logprobs, rewards, values, nextvalues, terminated, truncated, firsts = ([] for _ in range(9))
        finished = []
        for _ in range(c.n_steps):
            with torch.no_grad():
                cond = a.cond(self.observations)
                sample = a.model(cond, deterministic=False, return_chain=True)
                value = a.model.critic(cond).flatten().cpu().numpy()
                logprob = a.model.get_logprobs(cond, sample.chains).reshape(len(self.envs), c.ft_denoising_steps, c.horizon_steps, a.action_dim).cpu().numpy()
            observations.append(self.observations.copy())
            chains.append(sample.chains.cpu().numpy())
            values.append(value)
            logprobs.append(logprob)
            firsts.append(self.first.copy())
            step_rewards, term, trunc, bootstrap_obs = [], [], [], []
            next_obs = []
            for i, (env, action_chunk) in enumerate(zip(self.envs, sample.trajectories.cpu().numpy())):
                reward_sum = 0.
                ended, timedout = False, False
                for action in action_chunk[:c.act_steps]:
                    observation, reward, ended, timedout, info = env.step(action)
                    a.counters["online_env_steps"] += 1
                    reward_sum += float(reward)
                    self.episode_returns[i] += float(reward)
                    self.episode_lengths[i] += 1
                    if ended or timedout:
                        break
                bootstrap_obs.append(observation)
                if ended or timedout:
                    finished.append({"return": float(self.episode_returns[i]), "length": int(self.episode_lengths[i]),
                                     "success": bool(info.get("success", ended))})
                    self.episode_returns[i], self.episode_lengths[i] = 0, 0
                    observation, _ = env.reset()
                next_obs.append(observation)
                step_rewards.append(reward_sum)
                term.append(ended)
                trunc.append(timedout)
            self.observations = np.asarray(next_obs)
            self.first = np.asarray(term) | np.asarray(trunc)
            with torch.no_grad():
                # Port uses the state before auto-reset for timeout bootstrap.
                nv_obs = self.observations if c.profile == "official_code_v0.6" else bootstrap_obs
                nextvalues.append(a.model.critic(a.cond(np.asarray(nv_obs))).flatten().cpu().numpy())
            rewards.append(step_rewards)
            terminated.append(term)
            truncated.append(trunc)
        reward_array = np.asarray(rewards, dtype=np.float64)
        if c.reward_scale_running:
            reward_array = self.scaler(reward_array.T, np.asarray(firsts).T).T
        advantages, returns = generalized_advantages(reward_array*c.reward_scale_const,
            np.asarray(values), np.asarray(nextvalues), np.asarray(terminated), np.asarray(truncated),
            c.gamma, c.gae_lambda, official=c.profile == "official_code_v0.6")
        flatten = lambda x: np.asarray(x).reshape((-1,)+np.asarray(x).shape[2:])
        return {"source": "real_on_policy_rollout", "observations": flatten(observations),
            "chains": flatten(chains), "logprobs": flatten(logprobs), "values": flatten(values),
            "advantages": flatten(advantages), "returns": flatten(returns),
            "completed_episodes": finished, "raw_reward_mean": float(np.mean(rewards))}

    def state_dict(self):
        if not all(hasattr(e, "state_dict") for e in self.envs):
            raise NotImplementedError("Exact resume requires environment state_dict/load_state_dict; adapter must include simulator and RNG state")
        return dict(observations=self.observations, first=self.first,
            episode_returns=self.episode_returns, episode_lengths=self.episode_lengths,
            scaler=copy.deepcopy(self.scaler.__dict__), envs=[e.state_dict() for e in self.envs])

    def load_state_dict(self, state):
        for key in ("observations", "first", "episode_returns", "episode_lengths"):
            setattr(self, key, state[key])
        self.scaler.__dict__ = state["scaler"]
        for env, env_state in zip(self.envs, state["envs"]):
            env.load_state_dict(env_state)


def train_online(agent, envs, iterations=None):
    agent.begin_online()
    runner = OnlineRunner(agent, envs)
    for _ in range(iterations or agent.config.online_iterations):
        rollout = runner.collect()
        yield {**agent.ppo_update(rollout), **agent.counters, "raw_reward_mean": rollout["raw_reward_mean"]}
