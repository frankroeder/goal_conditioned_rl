from typing import Callable, Dict, Optional

import jax
import numpy as np
from omegaconf import DictConfig


class HerSampler:
    replay_strategy = None
    replay_k = None
    method_name = None

    def __init__(self, cfg: DictConfig, reward_func: Optional[Callable] = None):
        if "hindsight" in cfg:
            self.replay_strategy = cfg.hindsight.replay_strategy
            self.replay_k = cfg.hindsight.replay_k
            self.method_name = cfg.hindsight.name
        if self.replay_strategy == "future":
            self.future_p = 1 - (1.0 / (1 + self.replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch: Dict, batch_size_in_transitions: int) -> Dict:
        assert self.method_name == "her"
        t = episode_batch["action"].shape[1]
        rollout_batch_size = episode_batch["action"].shape[0]
        batch_size = batch_size_in_transitions

        # select episodes and transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(t, size=batch_size)
        transitions = jax.tree_map(lambda x: x[episode_idxs, t_samples], episode_batch)

        # select her indices for the batch of transitions
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (t - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace goal with achieved goal
        future_ag = episode_batch["ag"][episode_idxs[her_indexes], future_t]
        transitions["g"][her_indexes] = future_ag
        # re-compute reward
        transitions["reward"] = np.expand_dims(
            np.array(
                [self.reward_func(next_ag, g, None) for next_ag, g in zip(transitions["next_ag"], transitions["g"])]
            ),
            1,
        )
        return transitions
