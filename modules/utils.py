import os
from typing import Dict, List, Optional, Tuple

import gymnasium
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict


def check_hydra_config(cfg: DictConfig, comm) -> None:
    with open_dict(cfg):
        cfg["num_workers"] = comm.Get_size()
        if cfg["seed"] is None:
            cfg["seed"] = np.random.randint(2**14)
    assert isinstance(cfg.env_name, str)
    assert isinstance(cfg.clip_range, int)
    assert cfg.agent.name in ["sac", "ddpg"], f"{cfg.agent.name} is not a valid agent"
    assert isinstance(cfg.n_epochs, int)
    assert isinstance(cfg.n_cycles, int)
    assert isinstance(cfg.utd_ratio, int)
    assert isinstance(cfg.n_test_rollouts, int)
    assert isinstance(cfg.buffer_size, int)
    assert isinstance(cfg.batch_size, int)
    assert isinstance(cfg.gamma, float)
    assert isinstance(cfg.feature_embedding_size, int)
    assert isinstance(cfg.done_signal, bool)
    assert isinstance(cfg.normalize_goal, bool)


def init_storage(cfg: DictConfig) -> Tuple[str, str]:
    logdir = os.getcwd()
    model_path = os.path.join(logdir, "models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    video_path = os.path.join(logdir, "videos")
    if cfg.log_video and not os.path.exists(video_path):
        os.mkdir(video_path)
    with open(os.path.join(logdir, "omega_config.yaml"), "w") as file:
        OmegaConf.save(config=cfg, f=file)
    return logdir, model_path


def get_env_params(env, seed) -> DictConfig:
    obs, _ = env.reset(seed=seed)
    params = {
        "obs": obs["observation"].shape[0],
        "obs_shape": obs["observation"].shape,
        "action": env.action_space.shape[0],
        "max_episode_steps": env._max_episode_steps,
    }
    # goal-conditioned environment
    if "desired_goal" in obs.keys():
        params["goal"] = obs["desired_goal"].shape[0]
    return DictConfig(params)


def get_env_samples(env: gymnasium.Env) -> Dict:
    """Environment samples with batch size of 1 for parameter initialization"""
    obs, _ = env.reset()
    env_dict = {
        "obs": obs["observation"][None, :].copy(),
        "action": env.action_space.sample()[None, :].copy(),
        "action_space_low": env.action_space.low,
        "action_space_high": env.action_space.high,
    }
    if "desired_goal" in obs.keys():
        env_dict["flatten_obs"] = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])[
            None, :
        ].copy()
        env_dict["actor"] = env_dict["flatten_obs"]
        env_dict["critic"] = [env_dict["flatten_obs"].copy(), env_dict["action"].copy()]
    return env_dict


class BatchEnv:
    def __init__(self, envs: List[gymnasium.Env]):
        assert len(envs) > 0
        self._envs = envs
        first_obs_space = self._envs[0].observation_space
        first_action_space = self._envs[0].action_space
        for env in self._envs[1:]:
            assert env.observation_space == first_obs_space, "Mismatch in observation spaces"
            assert env.action_space == first_action_space, "Mismatch in action spaces"

    @property
    def observation_space(self):
        return self._envs[0].observation_space

    @property
    def action_space(self):
        return self._envs[0].action_space

    def __len__(self):
        return len(self._envs)

    def __getitem__(self, index):
        return self._envs[index]

    def step(self, action: np.ndarray) -> Tuple[Dict, List[float], List[bool], List[bool], List[Dict]]:
        next_obs = []
        rewards = []
        terminals = []
        truncations = []
        infos = []
        assert len(action) == len(
            self._envs
        ), f"Mistmatch number of actions {action.shape=} and envs {len(self._envs)=}"
        for i, _env in enumerate(self._envs):
            _next_obs, _reward, _terminated, _truncated, _info = _env.step(action[i])
            next_obs.append(_next_obs)
            rewards.append(_reward)
            terminals.append(_terminated)
            truncations.append(_truncated)
            infos.append(_info)
        next_obs = {k: jnp.stack([ele[k] for ele in next_obs]) for k in _next_obs.keys()}
        infos = {k: [ele[k] for ele in infos if k in ele.keys()] for k in _info.keys()}
        return (next_obs, rewards, terminals, truncations, infos)

    def render(self):
        return self._envs[0].render()

    def close(self) -> None:
        [env.close() for env in self._envs]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        obs = []
        infos = []
        for _env in self._envs:
            _obs, _info = _env.reset(seed=seed, options=options)
            obs.append(_obs)
            infos.append(_info)
        obs = {k: jnp.stack([ele[k] for ele in obs]) for k in _obs.keys()}
        infos = {k: [ele[k] for ele in infos if k in ele.keys()] for k in _info.keys()}
        return (obs, infos)
