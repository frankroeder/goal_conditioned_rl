from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper
from omegaconf import DictConfig

from modules.utils import get_env_params


def setup_environments(cfg: DictConfig, rank_seed: int, **kwargs) -> List[gym.Env]:
    envs = []
    for i in range(cfg.episode_batch_size):
        env = gym.make(cfg.env_name, **kwargs)
        env.action_space.seed(rank_seed + i)
        env.observation_space.seed(rank_seed + i)
        env_params = get_env_params(env, rank_seed + i)
        envs.append(env)
    return envs, env_params


def setup_wrappers(envs: List[gym.Env], cfg: DictConfig, env_params: Dict) -> Tuple[List[gym.Env], Dict]:
    wrapped_envs = []
    for env in envs:
        if cfg.obs_noise:
            env = ObservationNoise(env)
        wrapped_envs.append(env)
    return wrapped_envs, env_params


class ObservationNoise(ObservationWrapper):
    def __init__(self, env, noise_level=0.1):
        self._env = env
        self.noise_level = noise_level
        self.obs_space = env.observation_space["observation"]

    def observation(self, obs):
        noise = np.random.normal(loc=0, scale=self.noise_level, size=obs["observation"].shape)
        noisy_obs = np.clip(obs["observation"] + noise, self.obs_space.low, self.obs_space.high)
        obs["observation"] = noisy_obs.copy()
        return obs

    def __getattr__(self, name):
        return getattr(self._env, name)
