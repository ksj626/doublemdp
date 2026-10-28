# Adapted from aoberai/rql at 229c956efb4494c2b9bb0bbddbd67b761c93f1cc.
# Copyright (c) 2025 TRL Authors. MIT license: third_party/licenses/rql-LICENSE.md.
import copy
from typing import Any

from functools import partial
import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange, repeat

from .rql_flax import ModuleDict, TrainState, nonpytree_field
from .rql_networks import Actor, Value


class RQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()


    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        batch_size, action_dim = self.config["batch_size"], self.config["action_dim"]

        rng, n_rng, u_rng, r_rng = jax.random.split(rng, 4)

        next_state = jnp.concatenate([batch["observations"][-1], jax.random.normal(n_rng, (batch_size, action_dim)), jnp.zeros((batch_size, 1,))], axis=-1) # s', x_0', 0 
        next_qs = self.network.select("target_value")(next_state)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        d = jnp.concatenate(
            [
                jax.random.uniform(u_rng, (batch_size // 2,)),
                jax.random.randint(r_rng, (batch_size // 2,), 0, self.config["flow_steps"] + 1)
                / self.config["flow_steps"],
            ],
            0,
        )
        d_b = d / self.config["flow_steps"]

        actions = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")

        # Reversing flow
        x_f = jnp.copy(actions)
        f = jnp.ones((batch_size, 1))
        for i in range(self.config["flow_steps"]):
            fm_actor = jnp.concatenate([batch["observations"][0], x_f, f], -1)
            out = self.network.select("actor")(fm_actor).mode()
            x_f = x_f - out * d_b[..., None]
            f = f - d_b[..., None]

        state = jnp.concatenate([batch["observations"][0], jax.lax.stop_gradient(x_f), f], axis=-1) # s, x_f, f

        q = self.network.select("value")(state, params=grad_params)

        rs_terminals = jnp.concatenate([jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]], axis=0) # right shift terminals
        n_rews = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1 - rs_terminals)
        ).sum(0)
        tqt_q  = (
            n_rews
            + (self.config["discount"] ** (self.config["h"]))
            * next_q
            * batch["masks"][-2]
        )
        
        s = rs_terminals.sum(0)
        valids = (
            (s <= 1).astype(s.dtype) # 1 for term and shift removes second (h=1)
        )
        critic_loss = (self.expectile_loss(tqt_q - q, tqt_q - q, self.config["expectile"]) * valids).mean()

        # BC flow loss
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")
        t = jax.random.uniform(t_rng, (batch_size, 1))

        x_t = (1 - t) * x_0 + t * x_1
        tgt = x_1 - x_0
        fm_actor = jnp.concatenate([batch["observations"][0], x_t, t], axis=-1)
        pred = self.network.select("actor")(
            fm_actor, params=grad_params
        ).mode()
        q_pe = self.network.select("value")(
                jnp.concatenate([batch["observations"][0], 
                    x_t + pred
                    * jnp.minimum(1 / self.config["flow_steps"], 1 - t), 
		    jnp.clip(t + 1 / self.config["flow_steps"], max=1)], axis=-1)
                )
        q_pe = q_pe.mean(axis=0)

        ac_mask = repeat(1 - rs_terminals[:-1], 'h b -> b (h r)', r=self.config["action_dim"] // self.config["h"]) # mask repeated actions at end of ep
        bc_loss = (jnp.square(pred - tgt) * ac_mask).mean()
        actor_loss = -(q_pe * valids).mean()

        total_loss = (
            actor_loss + bc_loss * self.config["alpha"] + critic_loss
        )

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "q": q.mean(),
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),

            "q_pe_mean": q_pe.mean(),
            "q_pe_max": q_pe.max(),
            "q_pe_min": q_pe.min(),
        }

    def target_update(self, network, module_name, d):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, "value", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1-self.config["ema"])

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(
        self,
        observations,
        noise,
        seed=None,
        temperature=0.0,
    ):
        actions = noise
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            fm_actor = jnp.concatenate([observations, actions, t], axis=-1)
            out = self.network.select("actor" if temperature > 0 else "target_actor")(fm_actor).mode()
            actions = actions + (out / self.config["flow_steps"])
        actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(
        self,
        obs,
        seed=None,
        temperature=0.0,
    ):
        action_rng, n_rng = jax.random.split(seed)

        obs = jnp.atleast_2d(obs)[-1:]
        noise = jax.random.normal(
            n_rng,
            (
                1,
                self.config["action_dim"],
            ),
        )
        actions = self.compute_flow_actions(obs, seed=action_rng, noise=noise, temperature=temperature)[0]
        actions = rearrange(actions, "(h d) -> h d", h=self.config["h"])
        return actions

    def sample_trace(self, obs, seed, num_samples=1, temperature=0.0, noise=None):
        """Record real Euler states without clipping latents or changing the sampler.

        ``latents`` is [F+1, B, H*A], ``native_values`` is [F+1, E, B],
        and ``final_actions`` is [B, H, A]. Times are native normalized flow
        times. Native values are expectile/pessimistic heads, not MC returns.
        With B=1, ``final_actions[0]`` matches ``sample_actions`` for this key.
        """
        observations = jnp.atleast_2d(obs)
        if observations.shape[0] == 1:
            observations = jnp.repeat(observations, num_samples, axis=0)
        elif observations.shape[0] != num_samples:
            raise ValueError("Provide one observation or num_samples observations")
        _, n_rng = jax.random.split(seed)
        if noise is None:
            noise = jax.random.normal(n_rng, (num_samples, self.config["action_dim"]))
        noise = jnp.asarray(noise)
        if noise.shape != (num_samples, self.config["action_dim"]):
            raise ValueError("Noise must have shape (num_samples, h * action_dim)")
        latents, values = self._trace_from_noise(observations, noise, temperature=temperature)
        final_actions = jnp.clip(latents[-1], -1, 1).reshape(num_samples, self.config["h"], -1)
        times = jnp.arange(self.config["flow_steps"] + 1) / self.config["flow_steps"]
        return dict(latents=latents, native_times=times, progress=times,
                    final_actions=final_actions, native_values=values,
                    initial_noise=noise, observations=observations)

    @partial(jax.jit, static_argnames=("temperature",))
    def _trace_from_noise(self, observations, noise, temperature=0.0):
        actions = noise
        latents, values = [actions], []
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            inputs = jnp.concatenate([observations, actions, t], axis=-1)
            values.append(self.network.select("value")(inputs))
            out = self.network.select("actor" if temperature > 0 else "target_actor")(inputs).mode()
            actions = actions + (out / self.config["flow_steps"])
            latents.append(actions)
        values.append(self.network.select("value")(
            jnp.concatenate([observations, actions, jnp.ones_like(t)], axis=-1)))
        return jnp.stack(latents), jnp.stack(values)

    @partial(jax.jit, static_argnames=("start_step", "temperature"))
    def continue_flow(self, observations, latent, start_step, temperature=0.0):
        """Finish deterministic native generation for a single-candidate probe."""
        if not 0 <= start_step <= self.config["flow_steps"]:
            raise ValueError("start_step must be a native step in [0, flow_steps]")
        observations, actions = jnp.atleast_2d(observations), jnp.atleast_2d(latent)
        observations = jnp.broadcast_to(observations, (actions.shape[0], observations.shape[-1]))
        for i in range(start_step, self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            out = self.network.select("actor" if temperature > 0 else "target_actor")(
                jnp.concatenate([observations, actions, t], axis=-1)).mode()
            actions = actions + (out / self.config["flow_steps"])
        return jnp.clip(actions, -1, 1).reshape(actions.shape[0], self.config["h"], -1)

    @jax.jit
    def reverse_cycle(self, observations, endpoint_actions, duration=1.0):
        """Measure finite-Euler reconstruction error using the training actor.

        Reconstruct for F steps of duration/F, then integrate forward with the
        same step size. Neither direction clips; exact invertibility is not
        assumed. This is an evaluation-only diagnostic, outside the loss.
        """
        observations = jnp.atleast_2d(observations)
        endpoint_actions = jnp.asarray(endpoint_actions).reshape(observations.shape[0], -1)
        x = endpoint_actions
        dt = jnp.asarray(duration) / self.config["flow_steps"]
        dt = jnp.broadcast_to(dt, (observations.shape[0],)).reshape(-1, 1)
        t = jnp.ones((observations.shape[0], 1))
        for _ in range(self.config["flow_steps"]):
            out = self.network.select("actor")(jnp.concatenate([observations, x, t], -1)).mode()
            x, t = x - out * dt, t - dt
        reconstructed = x
        for _ in range(self.config["flow_steps"]):
            out = self.network.select("actor")(jnp.concatenate([observations, x, t], -1)).mode()
            x, t = x + out * dt, t + dt
        residual = x - endpoint_actions
        return dict(reconstructed=reconstructed, cycled=x, residual=residual,
                    rmse=jnp.sqrt(jnp.mean(residual ** 2)),
                    max_abs=jnp.max(jnp.abs(residual)))

    def state_dict(self):
        """Optimizer, targets, online networks, update counter and RNG state."""
        return flax.serialization.to_state_dict(self)

    def restore_state_dict(self, state):
        return flax.serialization.from_state_dict(self, state)

    def to_bytes(self):
        return flax.serialization.to_bytes(self)

    def restore_bytes(self, payload):
        return flax.serialization.from_bytes(self, payload)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):

        # Infrastructure validation only; the loss and update below are the
        # pinned official implementation. Avoid mutating the caller's config.
        config = {**get_config().to_dict(), **dict(config)}
        for key in ("actor_hidden_dims", "value_hidden_dims"):
            config[key] = tuple(config[key])
        if config["batch_size"] < 2 or config["batch_size"] % 2:
            raise ValueError("RQL's half-continuous/half-discrete duration batch must be positive and even")
        if config["ensemble_ct"] < 2:
            raise ValueError("The official reduction requires ensemble_ct >= 2")
        if config["h"] < 1 or config["flow_steps"] < 1 or not 0 < config["discount"] < 1:
            raise ValueError("Require h >= 1, flow_steps >= 1, and 0 < discount < 1")
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        ex_times = ex_actions[..., :1]
        ex_in = jnp.concatenate([ex_observations, ex_actions, ex_times], -1)
        action_dim = ex_actions.shape[-1]

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )

        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=1,
        )

        network_info = dict(
            value=(value_def, (ex_in,)),
            target_value=(copy.deepcopy(value_def), (ex_in)),
            actor=(actor_def, (ex_in,)),
            target_actor=(
                copy.deepcopy(actor_def),
                (ex_in,)
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_actor"] = params["modules_actor"]
        config["action_dim"] = action_dim

        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf])
        )

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="rql",
            h=3,
            alpha=1.0,
            expectile=0.5,
            ensemble_ct=10,
            rho=0.0, 
            lr=3e-4,
            discount=0.99,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            tau=0.005,
            ema=0.999,
            flow_steps=10,
            q_agg="mean",
        )
    )
    return config
