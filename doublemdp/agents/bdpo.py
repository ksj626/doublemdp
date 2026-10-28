"""Minimally extracted FlowRL BDPO components.
Source: typoverflow/flow-rl, bae22f4cad3f9ce0a220aceb877b0e979de89808.
Copyright and MIT license: third_party/flow-rl/LICENSE. See docs/bdpo_audit.md.
"""
from __future__ import annotations
from .bdpo_nn import *
from .bdpo_diffusion import DDPM, DDPMBackbone, jit_update_ddpm

EPS = 1e-6

@partial(jax.jit, static_argnames=("maxQ", "q_target", "rho"))
def get_target(rng: PRNGKey, vs: jnp.ndarray, maxQ: bool, q_target: str, rho: float) -> Tuple[PRNGKey, jnp.ndarray]:
    # vs is of shape (E, B, N, 1)
    if maxQ:
        vs = vs.max(axis=-2)
    else:
        vs = vs.mean(axis=-2)
    if q_target == "mean":
        # Explicit numerical profile: rho=0 LCB objective without undefined
        # derivatives of std at a zero-spread ensemble. Official uses lcb.
        vs = vs.mean(axis=0)
    elif q_target == "min":
        vs = vs.min(axis=0)
    elif q_target == "convex":
        vs = rho * vs.min(axis=0) + (1-rho) * vs.max(axis=0)
    elif q_target == "lcb":
        vs = vs.mean(axis=0) - rho * vs.std(axis=0)
    elif q_target == "rand_convex":
        rng, alpha_key = jax.random.split(rng)
        alphas = jax.random.uniform(alpha_key, vs.shape)
        alphas /= (alphas.sum(axis=0, keepdims=True) + EPS)
        vs = (vs * alphas).sum(axis=0)
    else:
        raise NotImplementedError(f"Unrecognized Q-target type: {q_target}. ")
    return rng, vs

@partial(jax.jit, static_argnames=("T"))
def get_penalty(
    actor_eps,
    behavior_eps,
    t: jnp.ndarray,
    T: int,
    alphas: jnp.ndarray,
    alpha_hats: jnp.ndarray,
    betas: jnp.ndarray
) -> jnp.ndarray:
    return 0.5 * betas[t] * ((actor_eps - behavior_eps)**2) / (1 - betas[t]) / (1 - alpha_hats[t])

@partial(jax.jit, static_argnames=("training", "num_samples", "solver", "temperature"))
def jit_sample_and_select(
    rng: PRNGKey,
    model: DDPM,
    q0: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    solver: str,
    temperature: float,
) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], model.x_dim))
    rng, actions, _ = model.sample(rng, xT, obs_repeat, training, solver)
    if temperature is None:
        return rng, actions[:, 0]
    else:
        qs = q0(
            obs_repeat,
            actions,
        )
        qs = qs.mean(axis=0).reshape(B, num_samples)
        if temperature <= 0.0:
            idx = qs.argmax(axis=-1)
        else:
            rng, select_rng = jax.random.split(rng)
            idx = jax.random.categorical(select_rng, logits=qs/(1e-6+temperature), axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), idx]
        return rng, actions

@partial(jax.jit, static_argnames=("ema", "do_ema_update"))
def jit_update_behavior(
    rng: PRNGKey,
    behavior: DDPM,
    behavior_target: DDPM,
    batch: Batch,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    rng, new_behavior, metrics = jit_update_ddpm(rng, behavior, batch.action, batch.obs)
    if do_ema_update:
        new_behavior_target = ema_update(new_behavior, behavior_target, ema)
    else:
        new_behavior_target = behavior_target
    return rng, new_behavior, new_behavior_target, metrics

@partial(jax.jit, static_argnames=("T", "discount", "eta", "rho", "num_q_samples", "q_target", "maxQ", "solver", "ema", "do_ema_update"))
def jit_update_critic(
    rng: PRNGKey,
    q0: Model,
    q0_target: Model,
    vt: Model,
    vt_target: Model,
    actor_target: DDPM,
    behavior_target: DDPM,
    batch: Batch,
    T: int,
    discount: float,
    eta: float,
    rho: float,
    num_q_samples: int,
    q_target: str,
    maxQ: bool,
    solver: str,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]
    alphas = actor_target.alphas
    alpha_hats = actor_target.alpha_hats
    betas = actor_target.betas

    rng, xT_rng = jax.random.split(rng)

    # q0 target
    next_obs_repeat = batch.next_obs[..., jnp.newaxis, :].repeat(num_q_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*next_obs_repeat.shape[:-1], A))
    rng, next_action, history = actor_target.sample(
        rng,
        xT,
        next_obs_repeat,
        training=False,
        solver=solver,
    )
    q0_target_value = q0_target(
        next_obs_repeat,
        next_action
    )
    rng, q0_target_value = get_target(rng, q0_target_value, maxQ, q_target, rho)

    q0_xt, q0_actor_eps = history
    q0_t = jnp.arange(T, 0, -1).repeat(B*num_q_samples, axis=0).reshape(T, B, num_q_samples, 1)
    q0_behavior_eps = behavior_target(
        q0_xt,
        q0_t,
        next_obs_repeat[jnp.newaxis, ...].repeat(T, axis=0),
    )
    q0_penalty = get_penalty(q0_actor_eps, q0_behavior_eps, q0_t, T, alphas, alpha_hats, betas)
    q0_penalty = q0_penalty.sum(axis=0).mean(axis=-2).sum(axis=-1, keepdims=True)

    q0_target_value = q0_target_value - eta * q0_penalty
    q0_target_value = batch.reward + discount * (1-batch.terminal) * q0_target_value

    # vt target
    obs_repeat = batch.obs[..., jnp.newaxis, :].repeat(num_q_samples, axis=-2)
    rng, rep_xt_1, xt, rep_t_1, t, history = actor_target.onestep_sample(
        rng,
        batch.action,
        batch.obs,
        training=False,
        num_samples=num_q_samples,
        solver=solver,
        sample_xt=True,
        t=None,
    )
    vt_target_value1 = vt_target(
        obs_repeat,
        rep_xt_1,
        rep_t_1
    ) # (E, B, N, 1)
    rng, vt_target_value1 = get_target(rng, vt_target_value1, False, q_target, rho)
    vt_target_value2 = q0_target(
        obs_repeat,
        rep_xt_1
    )
    rng, vt_target_value2 = get_target(rng, vt_target_value2, False, q_target, rho)
    vt_target_value = (t != 1) * vt_target_value1 + (t == 1) * vt_target_value2

    vt_actor_eps = history
    vt_behavior_eps = behavior_target(xt, t, batch.obs)
    vt_penalty = get_penalty(vt_actor_eps, vt_behavior_eps, t, T, alphas, alpha_hats, betas)

    vt_target_value = vt_target_value - eta * vt_penalty.sum(axis=-1, keepdims=True)

    def q0_loss_fn(q0_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        pred = q0.apply(
            {"params": q0_params},
            batch.obs,
            batch.action
        )
        loss = ((pred - q0_target_value[jnp.newaxis, :])**2).mean()
        return loss, {
            "loss/q0_loss": loss,
            "misc/q0_mean": pred.mean(),
            "misc/q0_penalty": q0_penalty.mean(),
            "misc/reward": batch.reward.mean(),
        }
    def vt_loss_fn(vt_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        pred = vt.apply(
            {"params": vt_params},
            batch.obs,
            xt,
            t
        )
        loss = ((pred - vt_target_value[jnp.newaxis, :])**2).mean()
        return loss, {
            "loss/vt_loss": loss,
            "misc/vt_mean": pred.mean(),
            "misc/vt_std": pred.std(axis=0).mean(),
            "misc/vt_penalty": vt_penalty.mean(),
        }
    new_q0, q0_metrics = q0.apply_gradient(q0_loss_fn)
    new_vt, vt_metrics = vt.apply_gradient(vt_loss_fn)
    if do_ema_update:
        new_q0_target = ema_update(new_q0, q0_target, ema)
        new_vt_target = ema_update(new_vt, vt_target, ema)
    else:
        new_q0_target = q0_target
        new_vt_target = vt_target
    return rng, new_q0, new_q0_target, new_vt, new_vt_target, {
        **q0_metrics,
        **vt_metrics
    }

@partial(jax.jit, static_argnames=("T", "eta", "rho", "num_q_samples", "q_target", "maxQ", "solver", "ema", "do_ema_update"))
def jit_update_actor(
    rng: PRNGKey,
    actor: DDPM,
    actor_target: DDPM,
    behavior_target: DDPM,
    q0_target: Model,
    vt_target: Model,
    batch: Batch,
    T: int,
    eta,
    rho: float,
    num_q_samples: int,
    q_target: str,
    maxQ: bool,
    solver: str,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]
    alphas = actor.alphas
    alpha_hats = actor.alpha_hats
    betas = actor.betas

    rng, sample_rng = jax.random.split(rng)

    def actor_loss_fn(actor_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        sample_rng_, rep_xt_1, xt, rep_t_1, t, history = actor.onestep_sample(
            sample_rng,
            batch.action,
            batch.obs,
            training=True,
            num_samples=num_q_samples,
            solver=solver,
            sample_xt=True,
            t=None,
            params=actor_params
        )
        obs_repeat = batch.obs[..., jnp.newaxis, :].repeat(num_q_samples, axis=-2)
        target_1 = vt_target(
            obs_repeat,
            rep_xt_1,
            rep_t_1
        )
        sample_rng_, target_1 = get_target(sample_rng_, target_1, False, q_target, rho=rho)
        target_2 = q0_target(
            obs_repeat,
            rep_xt_1
        )
        sample_rng_, target_2 = get_target(sample_rng_, target_2, False, q_target, rho=rho)
        target = (t != 1) * target_1 + (t == 1) * target_2

        actor_eps = history
        behavior_eps = behavior_target(xt, t, batch.obs)
        penalty = get_penalty(actor_eps, behavior_eps, t, T, alphas, alpha_hats, betas)
        target = target - eta * penalty.sum(axis=-1, keepdims=True)
        actor_loss = - target.mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss
        }

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    if do_ema_update:
        new_actor_target = ema_update(new_actor, actor_target, ema)
    else:
        new_actor_target = actor_target
    metrics["misc/eta"] = eta
    return rng, new_actor, new_actor_target, metrics


class BDPOAgent:
    """
    Behavior-Regularized Diffusion Policy Optimization (BDPO)
    https://arxiv.org/abs/2502.04778
    """
    name = "BDPOAgent"
    model_names = ["behavior", "behavior_target", "actor", "actor_target", "q0", "q0_target", "vt", "vt_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: BDPOConfig, seed: int):
        self.obs_dim, self.act_dim, self.cfg = obs_dim, act_dim, cfg
        self.rng = jax.random.PRNGKey(seed)
        self.cfg = cfg
        self.rng, behavior_rng, actor_rng, q0_rng, vt_rng = jax.random.split(self.rng, 5)

        # define behavior and actor
        time_embedding = PositionalEmbedding(output_dim=cfg.diffusion.time_dim)
        if cfg.diffusion.resnet:
            noise_predictor = ResidualMLP(
                hidden_dims=cfg.diffusion.resnet_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=True,
                dropout=cfg.diffusion.dropout,
                multiplier=1,
            )
        else:
            noise_predictor = MLP(
                hidden_dims=cfg.diffusion.mlp_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=cfg.diffusion.layer_norm,
                dropout=cfg.diffusion.dropout
            )
        backbone_def = DDPMBackbone(
            noise_predictor=noise_predictor,
            time_embedding=time_embedding
        )

        def create_ddpm(network_def: nn.Module, rng: PRNGKey, cfg: BDPODiffusionConfig, train_cfg: BDPODiffusionTrainConfig):
            if train_cfg.lr_decay_steps is not None:
                lr = optax.cosine_decay_schedule(train_cfg.lr, train_cfg.lr_decay_steps)
            else:
                lr = train_cfg.lr
            ddpm = DDPM.create(
                network=network_def,
                rng=rng,
                inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
                x_dim=self.act_dim,
                steps=cfg.steps,
                noise_schedule=cfg.noise_schedule,
                noise_schedule_params=None,
                approx_postvar=True,
                clip_sampler=cfg.clip_sampler,
                x_min=cfg.x_min,
                x_max=cfg.x_max,
                optimizer=optax.adam(learning_rate=lr),
                clip_grad_norm=train_cfg.clip_grad_norm,
            )
            ddpm_target = DDPM.create(
                network=network_def,
                rng=rng,
                x_dim=self.act_dim,
                inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
                steps=cfg.steps,
                noise_schedule=cfg.noise_schedule,
                noise_schedule_params=None,
                approx_postvar=True,
                clip_sampler=cfg.clip_sampler,
                x_min=cfg.x_min,
                x_max=cfg.x_max,
            )
            return ddpm, ddpm_target

        self.behavior, self.behavior_target = create_ddpm(backbone_def, behavior_rng, cfg.diffusion, cfg.diffusion.behavior)
        self.actor, self.actor_target = create_ddpm(backbone_def, actor_rng, cfg.diffusion, cfg.diffusion.actor)

        # define critic networks
        q0_lr = vt_lr = cfg.critic.lr
        if cfg.critic.lr_decay_steps is not None:
            q0_lr = optax.cosine_decay_schedule(cfg.critic.lr, cfg.critic.lr_decay_steps)
            vt_lr = optax.cosine_decay_schedule(cfg.critic.lr, cfg.critic.lr_decay_steps)
        q0_def = Ensemblize(
            base=ScalarCritic(
                backbone=MLP(
                    hidden_dims=cfg.critic.hidden_dims,
                    activation=mish,
                    layer_norm=cfg.critic.layer_norm,
                )
            ),
            ensemble_size=cfg.critic.ensemble_size,
        )
        vt_def = Ensemblize(
            base=ScalarCriticWithDiscreteTime(
                backbone=MLP(
                    hidden_dims=cfg.critic.hidden_dims,
                    activation=mish,
                    layer_norm=True,
                ),
                time_embedding=time_embedding,
            ),
            ensemble_size=cfg.critic.ensemble_size,
        )
        self.q0 = Model.create(
            q0_def,
            q0_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=q0_lr),
            clip_grad_norm=cfg.critic.clip_grad_norm,
        )
        self.q0_target = Model.create(
            q0_def,
            q0_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )
        self.vt = Model.create(
            vt_def,
            vt_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.zeros((1, 1))),
            optimizer=optax.adam(learning_rate=vt_lr),
            clip_grad_norm = cfg.critic.clip_grad_norm,
        )
        self.vt_target = Model.create(
            vt_def,
            vt_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.zeros((1, 1))),
        )

        self.warmup_steps = cfg.warmup_steps
        self._is_pretraining = True # will switch to False after prepared for training
        self._n_pretraining_steps = 0
        self._n_training_steps = 0

    @property
    def saved_model_names(self) -> List[str]:
        if self._is_pretraining:
            return ["behavior_target"]
        else:
            return self.model_names

    def prepare_training(self):
        self.actor = ema_update(self.behavior_target, self.actor, 1.0)
        self.actor_target = ema_update(self.behavior_target, self.actor_target, 1.0)
        self._is_pretraining = False

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}
        self.rng, self.q0, self.q0_target, self.vt, self.vt_target, critic_metrics = jit_update_critic(
            self.rng,
            self.q0,
            self.q0_target,
            self.vt,
            self.vt_target,
            self.actor_target,
            self.behavior_target,
            batch,
            self.cfg.diffusion.steps,
            self.cfg.critic.discount,
            self.cfg.critic.eta,
            self.cfg.critic.rho,
            self.cfg.critic.num_samples,
            self.cfg.critic.q_target,
            self.cfg.critic.maxQ,
            self.cfg.critic.solver,
            self.cfg.critic.ema,
            self._n_training_steps % self.cfg.critic.ema_every == 0
        )
        metrics.update(critic_metrics)

        if self._n_training_steps >= self.warmup_steps\
            and self._n_training_steps % self.cfg.critic.update_ratio == 0:
            self.rng, self.actor, self.actor_target, actor_metrics = jit_update_actor(
                self.rng,
                self.actor,
                self.actor_target,
                self.behavior_target,
                self.q0_target,
                self.vt_target,
                batch,
                self.cfg.diffusion.steps,
                self.cfg.critic.eta,
                self.cfg.critic.rho,
                self.cfg.critic.num_samples,
                self.cfg.critic.q_target,
                self.cfg.critic.maxQ,
                self.cfg.critic.solver,
                self.cfg.diffusion.actor.ema,
                self._n_training_steps % self.cfg.diffusion.actor.ema_every == 0
            )
            self.rng, self.behavior, self.behavior_target, behavior_metrics = jit_update_behavior(
                self.rng,
                self.behavior,
                self.behavior_target,
                batch,
                self.cfg.diffusion.behavior.ema,
                self._n_pretraining_steps % self.cfg.diffusion.behavior.ema_every == 0
            )
            metrics.update(actor_metrics)
            metrics.update(behavior_metrics)
        self._n_training_steps += 1
        return metrics

    def pretrain_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.behavior, self.behavior_target, metrics = jit_update_behavior(
            self.rng,
            self.behavior,
            self.behavior_target,
            batch,
            self.cfg.diffusion.behavior.ema,
            self._n_pretraining_steps % self.cfg.diffusion.behavior.ema_every == 0
        )
        self._n_pretraining_steps += 1
        return metrics

    def _upstream_sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        if self._is_pretraining:
            use_model = self.behavior_target
            temperature = None
            num_samples = 1
        else:
            use_model = self.actor_target
            temperature = self.cfg.temperature
            num_samples = num_samples
        self.rng, action = jit_sample_and_select(
            self.rng,
            use_model,
            self.q0_target,
            obs,
            training=False,
            num_samples=num_samples,
            solver=self.cfg.diffusion.solver,
            temperature=temperature,
        )
        return action, {}

    @classmethod
    def create(cls, seed, observations, actions, config=None):
        """Create from example arrays and an explicit BDPO configuration dictionary."""
        from types import SimpleNamespace
        import copy
        config = copy.deepcopy(config or default_config())
        def namespace(value):
            if isinstance(value, dict):
                return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
            if isinstance(value, list):
                return tuple(value)
            return value
        obs_dim, act_dim = observations.shape[-1], actions.shape[-1]
        self = cls(obs_dim, act_dim, namespace(config), seed)
        self.config_dict = config
        self.pretrain_steps = int(config.get('pretrain_steps', 100000))
        self.eval_num_samples = int(config.get('eval_num_samples', 1))
        if self.pretrain_steps == 0:
            self.prepare_training()
        return self

    def update(self, batch):
        """One offline update; returns self and metrics, without any oracle access."""
        if not isinstance(batch, Batch):
            batch = Batch(
                obs=jnp.asarray(batch['observations']),
                action=jnp.asarray(batch['actions']),
                reward=jnp.asarray(batch['rewards']).reshape(-1, 1),
                terminal=1 - jnp.asarray(batch['masks']).reshape(-1, 1),
                next_obs=jnp.asarray(batch['next_observations']),
                next_action=None,
            )
        if self._is_pretraining:
            metrics = self.pretrain_step(batch, self._n_pretraining_steps)
            if self._n_pretraining_steps >= self.pretrain_steps:
                self.prepare_training()
        else:
            metrics = self.train_step(batch, self._n_training_steps)
        return self, metrics

    def sample_actions(self, observations, seed=None, num_samples=None, **kwargs):
        """Native stochastic sampler with optional reproducible, non-mutating RNG."""
        obs = jnp.asarray(observations)
        unbatched = obs.ndim == 1
        obs = obs[None] if unbatched else obs
        key = self.rng if seed is None else (jax.random.PRNGKey(seed) if isinstance(seed, int) else seed)
        model = self.behavior_target if self._is_pretraining else self.actor_target
        count = 1 if self._is_pretraining else (self.eval_num_samples if num_samples is None else num_samples)
        temperature = None if self._is_pretraining else self.cfg.temperature
        rng, actions = jit_sample_and_select(key, model, self.q0_target, obs, False,
                                             count, self.cfg.diffusion.solver, temperature)
        if seed is None:
            self.rng = rng
        return actions[0] if unbatched else actions

    def sample_trace(self, observations, seed=0, num_samples=1):
        """Native latent states, including all candidate chains and selected index.

        This deliberately keeps candidates separate. A selected best-of-N chain
        alone is not a Markov state for continuation-value identities.
        """
        obs = jnp.atleast_2d(jnp.asarray(observations))
        key = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
        model = self.behavior_target if self._is_pretraining else self.actor_target
        temperature = None if self._is_pretraining else self.cfg.temperature
        count = 1 if self._is_pretraining else num_samples
        rng, x_rng = jax.random.split(key)
        repeat_obs = obs[..., None, :].repeat(count, axis=-2)
        xT = jax.random.normal(x_rng, (*repeat_obs.shape[:-1], self.act_dim))
        rng, candidates, (xt, eps) = model.sample(rng, xT, repeat_obs, False, self.cfg.diffusion.solver)
        qs = self.q0_target(repeat_obs, candidates)
        means = qs.mean(axis=0)[..., 0]
        if temperature is None:
            selected = jnp.zeros(obs.shape[0], dtype=jnp.int32)
        elif temperature <= 0:
            selected = means.argmax(-1)
        else:
            rng, select_rng = jax.random.split(rng)
            selected = jax.random.categorical(select_rng, means / (1e-6 + temperature))
        latents = jnp.concatenate([xt, candidates[None]], axis=0)
        times = jnp.arange(model.steps, -1, -1)
        repeated_time = jnp.broadcast_to(times[:-1, None, None, None], (*xt.shape[:-1], 1))
        native_inner = self.vt_target(jnp.broadcast_to(repeat_obs, (model.steps, *repeat_obs.shape)), xt, repeated_time)
        native_values = jnp.concatenate([native_inner, qs[:, None]], axis=1)
        behavior_eps = self.behavior_target(xt, repeated_time, jnp.broadcast_to(repeat_obs, (model.steps, *repeat_obs.shape)))
        penalty = get_penalty(eps, behavior_eps, repeated_time, model.steps,
                              model.alphas, model.alpha_hats, model.betas).sum(-1)
        return {'latents': latents, 'native_times': times,
                'progress': jnp.linspace(0, 1, model.steps+1),
                'actions': candidates[jnp.arange(obs.shape[0]), selected],
                'candidate_actions': candidates, 'selected_indices': selected,
                'native_values': native_values, 'step_path_kl': penalty,
                'value_semantics': 'Native regularized Vt/Q0 ensemble; includes remaining and future path costs, not unregularized Q^pi.',
                'policy': ('single_candidate' if count == 1 else ('first_candidate_of_N' if temperature is None else ('best_of_N_all_candidates_retained' if temperature <= 0 else 'softmax_selected_all_candidates_retained')))}

    def continue_from_latent(self, observations, latent, native_time, seed=0, return_info=False):
        """Continue a single candidate; remaining KL follows the native penalty.

        A fresh continuation key supplies independent remaining noise. Native
        times count down from T to zero, unlike generation progress.
        """
        obs = jnp.atleast_2d(jnp.asarray(observations))
        latent = jnp.atleast_2d(jnp.asarray(latent))
        key = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
        model = self.behavior_target if self._is_pretraining else self.actor_target
        if not 0 <= int(native_time) <= model.steps:
            raise ValueError('native_time must be an integer between zero and diffusion steps')
        actions, kl = continue_sampler(key, model, self.behavior_target, obs, latent,
                                       int(native_time), self.cfg.diffusion.solver)
        if return_info:
            return {'actions': actions, 'remaining_path_kl': kl}
        return actions

    def state_dict(self):
        """Full checkpoint, including both phases, dropout RNGs and optimizers."""
        from flax.serialization import to_state_dict
        return {'format': 1, 'config': self.config_dict, 'obs_dim': self.obs_dim,
                'act_dim': self.act_dim, 'rng': jax.device_get(self.rng),
                'pretraining': self._is_pretraining,
                'pretrain_counter': self._n_pretraining_steps,
                'train_counter': self._n_training_steps,
                'models': {name: jax.device_get(to_state_dict(getattr(self, name)))
                           for name in self.model_names}}

    @classmethod
    def from_state_dict(cls, state):
        from flax.serialization import from_state_dict
        self = cls.create(0, jnp.zeros((1, state['obs_dim'])),
                          jnp.zeros((1, state['act_dim'])), state['config'])
        for name, model_state in state['models'].items():
            setattr(self, name, from_state_dict(getattr(self, name), model_state))
        self.rng = jnp.asarray(state['rng'])
        self._is_pretraining = state['pretraining']
        self._n_pretraining_steps = state['pretrain_counter']
        self._n_training_steps = state['train_counter']
        return self

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.__dict__.update(type(self).from_state_dict(state).__dict__)


def default_config(profile='official_code', task='halfcheetah-medium-v2'):
    """Pinned author defaults plus launch-script overrides; no hidden tuning."""
    import copy
    train_steps = 2000000
    cfg = {
        'profile': profile, 'name': 'bdpo', 'pretrain_steps': 2000000,
        'train_steps': train_steps, 'batch_size': 256, 'eval_num_samples': 10,
        'warmup_steps': 50000, 'temperature': 0.0,
        'diffusion': {
            'actor': {'lr': 1e-5, 'lr_decay_steps': train_steps, 'clip_grad_norm': 1., 'ema': .005, 'ema_every': 5},
            'behavior': {'lr': 3e-4, 'lr_decay_steps': None, 'clip_grad_norm': 1., 'ema': .005, 'ema_every': 1},
            'noise_schedule': 'vp', 'resnet': False, 'dropout': 0., 'layer_norm': False,
            'time_dim': 64, 'mlp_hidden_dims': [256]*4, 'resnet_hidden_dims': [256]*3,
            'solver': 'ddpm', 'steps': 5, 'clip_sampler': True, 'x_min': -1., 'x_max': 1.},
        'critic': {'discount': .99, 'q_target': 'lcb', 'maxQ': False,
            'ensemble_size': 10, 'rho': .5, 'eta': .05, 'hidden_dims': [256]*3,
            'lr': 3e-4, 'lr_decay_steps': train_steps, 'clip_grad_norm': 1.,
            'layer_norm': False, 'ema': .005, 'ema_every': 1, 'steps': 5,
            'num_samples': 10, 'solver': 'ddpm', 'update_ratio': 5}}
    if task.startswith('hopper'):
        cfg['critic'].update(rho=2., eta=.2, ensemble_size=20 if task == 'hopper-medium-v2' else 10)
    elif task.startswith('walker2d'):
        cfg['critic'].update(rho=1., eta=.15)
    elif task.startswith('antmaze'):
        cfg['critic'].update(rho=.8, eta=.5 if 'umaze' in task else (.2 if 'medium' in task else 1.),
                             layer_norm=True, maxQ=True, discount=.995)
        cfg['diffusion']['mlp_hidden_dims'] = [512]*4
        cfg['diffusion']['behavior']['lr'] = 1e-4
        if task == 'antmaze-umaze-diverse-v0':
            cfg['temperature'], cfg['eval_num_samples'] = None, 1
    if profile in ('toy_official_code', 'toy_paper_eta', 'toy_diagnostic', 'smoke'):
        cfg.update(pretrain_steps=100000, train_steps=100000, batch_size=512,
                   eval_num_samples=1, warmup_steps=5000, temperature=None)
        cfg['diffusion'].update(steps=50, clip_sampler=False, x_min=-4., x_max=4.)
        cfg['critic'].update(steps=50, rho=0., eta=.06 if profile == 'toy_paper_eta' else .05)
        cfg['diffusion']['actor']['lr_decay_steps'] = cfg['critic']['lr_decay_steps'] = 100000
    if profile == 'smoke':
        cfg.update(pretrain_steps=10, train_steps=20, batch_size=32, warmup_steps=0)
        cfg['diffusion'].update(steps=5, time_dim=8, mlp_hidden_dims=[32,32])
        cfg['critic'].update(steps=5, ensemble_size=2, hidden_dims=[32,32], num_samples=2)
    return copy.deepcopy(cfg)


@partial(jax.jit, static_argnames=('native_time', 'solver'))
def continue_sampler(rng, model, behavior, obs, latent, native_time, solver):
    def step(carry, time):
        key, xt, cost = carry
        times = jnp.full((*xt.shape[:-1], 1), time, jnp.int32)
        key, next_samples, _, _, _, actor_eps = model.onestep_sample(
            key, xt, obs, False, 1, solver, False, times)
        behavior_eps = behavior(xt, times, obs)
        penalty = get_penalty(actor_eps, behavior_eps, times, model.steps,
                              model.alphas, model.alpha_hats, model.betas).sum(-1)
        return (key, next_samples[:, 0], cost + penalty), None
    (_, actions, cost), _ = jax.lax.scan(step,
        (rng, latent, jnp.zeros(latent.shape[:-1])), jnp.arange(native_time, 0, -1))
    return actions, cost
