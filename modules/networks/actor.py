from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from modules.networks.base import FeatureExtractor
from modules.networks.utils import default_init


LOG_STD_MIN: float = -20
LOG_STD_MAX: float = 2


class GaussianActor(nn.Module):
    action_dim: int
    cfg: DictConfig
    env_params: DictConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = FeatureExtractor(self.cfg, self.env_params, activation=self.cfg.agent.actor.activation)(x)
        for size in self.cfg.agent.actor.hidden_size:
            x = nn.Dense(size, kernel_init=default_init())(x)
            x = getattr(nn, self.cfg.agent.actor.activation)(x)

        mu = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        log_sigma = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        log_sigma = jnp.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_sigma


class DeterministicActor(nn.Module):
    action_dim: int
    cfg: DictConfig
    env_params: DictConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x = FeatureExtractor(self.cfg, self.env_params, activation=self.cfg.agent.actor.activation)(x)
        for size in self.cfg.agent.actor.hidden_size:
            x = nn.Dense(size, kernel_init=default_init())(x)
            x = getattr(nn, self.cfg.agent.actor.activation)(x)

        mu = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        return jax.nn.tanh(mu)
