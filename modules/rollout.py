from typing import Dict, Tuple

import numpy as np
from omegaconf import DictConfig

from modules.utils import BatchEnv


class RolloutWorker:
    _timestep_counter: int = 0

    def __init__(
        self,
        env: BatchEnv,
        policy,
        cfg: DictConfig,
        env_params: DictConfig,
    ):
        self.env = env
        self.policy = policy
        self.cfg = cfg
        self.env_params = env_params

    def get_current_timesteps(self) -> int:
        return self._timestep_counter

    def generate_rollout(self, train_mode: bool = False, animated: bool = False) -> Dict:
        ep_obs, ep_actions, ep_success, ep_rewards, ep_dones = [], [], [], [], []
        dict_obs, info = self.env.reset()
        obs = dict_obs["observation"]
        ag = dict_obs["achieved_goal"]
        g = dict_obs["desired_goal"]
        ep_ag, ep_g = [], []

        for _ in range(self.env_params["max_episode_steps"]):
            action = self.policy.act(obs.copy(), ag.copy(), g.copy(), train_mode)
            if animated:
                self.env.render()

            observation_new, reward, terminated, truncated, info = self.env.step(action)
            done = np.logical_or(terminated, truncated)

            if train_mode:
                self._timestep_counter += int(len(action))

            obs_new = observation_new["observation"]
            ag_new = observation_new["achieved_goal"]

            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            ep_rewards.append(np.vstack(reward))
            ep_dones.append(np.vstack(done))
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())

            obs = obs_new
            ag = ag_new

            if "is_success" in info.keys():
                ep_success.append(np.vstack(info["is_success"]))
            else:
                ep_success.append(np.vstack(reward > 0))

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())

        episode_data = dict(
            obs=np.stack(ep_obs, axis=1).copy(),
            action=np.stack(ep_actions, axis=1).copy(),
            reward=np.stack(ep_rewards, axis=1).copy(),
            done=np.stack(ep_dones, axis=1).copy(),
            success=np.stack(ep_success, axis=1).copy(),
        )

        episode_data["g"] = np.stack(ep_g, axis=1).copy()
        episode_data["ag"] = np.stack(ep_ag, axis=1).copy()
        return episode_data

    def generate_test_rollout(self, animated: bool = False) -> Tuple[list, list]:
        rollout_data = {}
        for _ in range(self.cfg.n_test_rollouts):
            rollout = self.generate_rollout(train_mode=False, animated=animated)
            for key, val in rollout.items():
                if key in rollout_data.keys():
                    rollout_data[key] = np.concatenate([rollout_data[key], val])
                else:
                    rollout_data[key] = val
        # only consider the last transitions to calculate the episodic success
        successes = rollout_data["success"][:, -1].flatten()
        rewards = rollout_data["reward"].sum(axis=1).flatten()
        return successes, rewards
