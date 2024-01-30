import threading
from collections.abc import Callable
from typing import Dict

import jax
import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        env_params: Dict,
        buffer_size: int,
        sample_func: Callable,
    ):
        self.env_params = env_params
        self.T = env_params["max_episode_steps"]
        self.size = buffer_size // self.T

        self.sample_func = sample_func
        self.current_size = 0
        self.lock = threading.Lock()

        obs_storage = np.empty(
            [self.size, self.T + 1, *self.env_params["obs_shape"]],
            dtype=np.float32,
        )
        self.buffer = {
            "obs": obs_storage,
            "action": np.empty([self.size, self.T, self.env_params["action"]]),
        }
        self.buffer["ag"] = np.empty([self.size, self.T + 1, self.env_params["goal"]])
        self.buffer["g"] = np.empty([self.size, self.T, self.env_params["goal"]])
        self.buffer["done"] = np.empty([self.size, self.T, 1])

    def store_episode(self, episode_batch: Dict[str, np.ndarray]) -> None:
        batch_size = len(episode_batch["reward"])
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            for key, val in episode_batch.items():
                if key in self.buffer.keys():
                    self.buffer[key][idxs] = val

    def sample(self, batch_size: int) -> Dict:
        with self.lock:
            temp_buffers = jax.tree_map(lambda arr: arr[: self.current_size], self.buffer)
        temp_buffers["next_obs"] = temp_buffers["obs"][:, 1:, :]
        temp_buffers["next_ag"] = temp_buffers["ag"][:, 1:, :]
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None) -> np.ndarray:
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            if self.current_size == 0:
                raise ValueError(f"Unable to sample overflow indices with {self.current_size=}")
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx
