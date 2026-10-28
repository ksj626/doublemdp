"""Minimally extracted FlowRL BDPO components.
Source: typoverflow/flow-rl, bae22f4cad3f9ce0a220aceb877b0e979de89808.
Copyright and MIT license: third_party/flow-rl/LICENSE. See docs/bdpo_audit.md.
"""
from __future__ import annotations
from abc import ABC
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, List
from dataclasses import field as dc_field
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, dataclass, field
from flax.training.train_state import TrainState
from flax.linen.initializers import Initializer
PRNGKey = Any
Param = Any
Metric = dict

def default_kernel_init(scale=None):
    return nn.initializers.orthogonal(jnp.sqrt(2) if scale is None else scale)
def default_bias_init():
    return nn.initializers.zeros_init()
def mish(x):
    return x * jnp.tanh(nn.softplus(x))
@dataclass
class Batch:
    obs: Any
    action: Any
    reward: Any
    terminal: Any
    next_obs: Any
    next_action: Any = None

def empty_optimizer() -> optax.GradientTransformation:
    """Returns an empty optimizer, which does not update any parameters."""
    def init_fn(params):
        return True # bool is a valid opt_state
    def update_fn(grads, state, params=None):
        raise ValueError("Empty optimizer does not update any parameters.")
    return optax.GradientTransformation(init_fn, update_fn)

@dataclass
class Model(PyTreeNode, ABC):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    """

    Model is initialized with Model.create() method
    """

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,  # PRNGKey
        inputs: Sequence[jnp.ndarray],  # sample of inputs
        optimizer: Optional[optax.GradientTransformation] = None,
        clip_grad_norm: float = None
    ) -> 'Model':
        param_rng, dropout_rng = jax.random.split(rng)
        params = network.init(param_rng, *inputs)  # params = {"params": ...}

        if optimizer is not None:
            if clip_grad_norm:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(max_norm=clip_grad_norm),
                    optimizer
                )
        else:
            optimizer = empty_optimizer()

        state = TrainState.create(
            apply_fn=network.apply,
            params=params.get("params", {}),
            tx=optimizer
        )
        return cls(
            state=state,
            dropout_rng=dropout_rng,
        )

    @property
    def params(self):
        return self.state.params

    def __call__(self, *args, **kwargs):
        # the network defined by the jax nn.Model should be used by apply function with {'params': P} and other ..
        out = self.state.apply_fn(
            {'params': self.state.params},
            *args,
            **kwargs,
        )
        return out

    def apply(self, *args, **kwargs):
        return self.state.apply_fn(*args,**kwargs)

    def apply_gradient(self, loss_fn) -> Tuple['Model', Metric]:
        grad_fn = jax.grad(loss_fn, has_aux=True)  # here auxiliary data is just the info dict
        dropout_rng, next_dropout_rng = jax.random.split(self.dropout_rng)
        grads, info = grad_fn(self.state.params, dropout_rng)
        state = self.state.apply_gradients(grads=grads)
        return self.replace(
            state=state,
            dropout_rng=next_dropout_rng
        ), info

    def reset_optim(self, optimizer: optax.GradientTransformation) -> 'Model':
        """Reset the optimizer to a new one."""
        new_state = TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            tx=optimizer
        )
        return self.replace(state=new_state)

class MLP(nn.Module):
    hidden_dims: Sequence[int] = dc_field(default_factory=lambda: [])
    output_dim: int = 0
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = default_bias_init

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init(), bias_init=self.bias_init())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation(x)
            if self.dropout and self.dropout > 0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        if self.output_dim > 0:
            x = nn.Dense(self.output_dim, kernel_init=self.kernel_init(), bias_init=self.bias_init())(x)
        return x

class ResidualLinear(nn.Module):
    dim: int
    multiplier: int = 4
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = default_bias_init

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        residual = x
        if self.dropout and self.dropout > 0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not training)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim * self.multiplier, kernel_init=self.kernel_init(), bias_init=self.bias_init())(x)
        x = self.activation(x)
        x = nn.Dense(self.dim, kernel_init=self.kernel_init(), bias_init=self.bias_init())(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.dim, kernel_init=self.kernel_init(), bias_init=self.bias_init())(residual)

        return residual + x

class ResidualMLP(nn.Module):
    hidden_dims: Sequence[int] = dc_field(default_factory=lambda: [])
    output_dim: int = 0
    multiplier: int = 4
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = default_bias_init

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        if len(self.hidden_dims) > 0:
            x = nn.Dense(self.hidden_dims[0], kernel_init=self.kernel_init(), bias_init=self.bias_init())(x)
        for i, size in enumerate(self.hidden_dims):
            x = ResidualLinear(
                size,
                self.multiplier,
                self.activation,
                self.layer_norm,
                self.dropout,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x, training)
        if self.output_dim > 0:
            x = self.activation(x)
            x = nn.Dense(self.output_dim, kernel_init=self.kernel_init(), bias_init=self.bias_init())(x)
        return x

class PositionalEmbedding(nn.Module):
    output_dim: int
    max_positions: int = 10000

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        half_dim = self.output_dim // 2
        f = jnp.log(self.max_positions) / (half_dim - 1)
        f = jnp.exp(jnp.arange(half_dim) * -f)
        f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class Ensemblize(nn.Module):
    base: nn.Module
    ensemble_size: int

    @nn.compact
    def __call__(self, *args, **kwargs) -> jnp.ndarray:
        def call_base(base_instance, *args, **kwargs):
            return base_instance(*args, **kwargs)

        vmap_fn = nn.vmap(
            call_base,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        return vmap_fn(self.base, *args, **kwargs)

class ScalarCritic(nn.Module):
    backbone: nn.Module
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = default_bias_init

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        if action is None:
            x = obs
        else:
            x = jnp.concatenate([obs, action], axis=-1)
        x = self.backbone(x)
        x = nn.Dense(
            1, kernel_init=self.kernel_init(), bias_init=self.bias_init()
        )(x)
        return x

class ScalarCriticWithDiscreteTime(nn.Module):
    backbone: nn.Module
    time_embedding: nn.Module
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = default_bias_init

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        t: jnp.ndarray = None,
        training: bool = False,
    ) -> jnp.ndarray:
        t_ff = self.time_embedding(t)
        t_ff = MLP(
            hidden_dims=[t_ff.shape[-1]*2, t_ff.shape[-1]],
            activation=mish,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(t_ff)
        x = jnp.concatenate([item for item in [obs, action, t_ff] if item is not None], axis=-1)
        x = self.backbone(x)
        x = nn.Dense(
            1, kernel_init=self.kernel_init(), bias_init=self.bias_init()
        )(x)
        return x

@partial(jax.jit, static_argnames=("ema_rate"))
def ema_update(
    src: Model,
    tgt: Model,
    ema_rate: float,
) -> Model:
    new_params = jax.tree_util.tree_map(
        lambda p, tp: p * ema_rate + tp * (1 - ema_rate),
        src.state.params,
        tgt.state.params,
    )
    return tgt.replace(state=tgt.state.replace(params=new_params))
