import flax.linen as nn
import jax
from omegaconf import DictConfig

from modules.networks.utils import default_init


class MLP(nn.Module):
    hidden_size: int
    output_size: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_size, kernel_init=default_init())(x)
        x = getattr(nn, self.activation)(x)
        x = nn.Dense(self.hidden_size, kernel_init=default_init())(x)
        x = getattr(nn, self.activation)(x)
        return nn.Dense(self.output_size, kernel_init=default_init())(x)


class FeatureExtractor(nn.Module):
    cfg: DictConfig
    env_params: DictConfig
    activation: str = "relu"

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        obs_embedding = nn.Dense(self.cfg.feature_embedding_size, kernel_init=default_init())(observation)
        return getattr(nn, self.activation)(obs_embedding)
