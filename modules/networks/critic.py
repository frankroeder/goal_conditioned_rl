import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from modules.networks.base import FeatureExtractor
from modules.networks.utils import default_init


class Critic(nn.Module):
    cfg: DictConfig
    env_params: DictConfig

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs, action], -1)

        for size in self.cfg.agent.critic.hidden_size:
            x = nn.Dense(size, kernel_init=default_init())(x)
            if self.cfg.agent.critic.dropout > 0:
                x = nn.Dropout(rate=self.cfg.agent.critic.dropout)(x, deterministic=False)
            if self.cfg.agent.critic.layer_norm:
                x = nn.LayerNorm()(x)
            x = getattr(nn, self.cfg.agent.critic.activation)(x)

        return nn.Dense(1, kernel_init=default_init())(x)


class VectorCritic(nn.Module):
    cfg: DictConfig
    env_params: DictConfig

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.cfg.agent.critic.ensemble_size,
        )
        obs_emb = FeatureExtractor(
            self.cfg,
            self.env_params,
            activation=self.cfg.agent.critic.activation,
        )(obs)
        action_emb = FeatureExtractor(
            self.cfg,
            self.env_params,
            activation=self.cfg.agent.critic.activation,
        )(action)
        return vmap_critic(self.cfg, self.env_params)(obs_emb, action_emb)
