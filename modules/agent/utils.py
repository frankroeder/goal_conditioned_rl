from typing import List

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(
        self,
        loc: jax.Array,
        scale_diag: jax.Array,
    ):
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        bijector = distrax.Block(distrax.Tanh(), ndims=1)
        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jax.Array:
        return self.bijector.forward(self.distribution.mode())


def get_action_dist(actor_state: TrainState, params, inputs: List[jax.Array]) -> TanhMultivariateNormalDiag:
    mu, log_sigma = actor_state.apply_fn(params, *inputs)
    return TanhMultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_sigma))


class QTrainState(TrainState):
    target_params: flax.core.FrozenDict = None

    def soft_update(self, tau: float):
        new_target_params = optax.incremental_update(self.params, self.target_params, tau)
        return self.replace(target_params=new_target_params)


class Temperature(nn.Module):
    initial_temp: float = 1.0

    @nn.compact
    def __call__(self) -> jax.Array:
        log_temp = self.param("log_temp", init_fn=lambda _: jnp.full((), jnp.log(self.initial_temp)))
        return jnp.exp(log_temp)


class ConstantTemperature(nn.Module):
    initial_temp: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        self.param("dummy_param", init_fn=lambda _: jnp.full((), self.initial_temp))
        return self.initial_temp
