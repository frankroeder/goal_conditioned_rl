import os
import pickle
from typing import Dict

import flax
import flax.linen as nn
import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from mpi4py import MPI
from omegaconf import DictConfig

from modules.agent.utils import QTrainState
from modules.buffer import ReplayBuffer
from modules.mpi_utils import logger
from modules.mpi_utils.mpi_utils import tree_bcast


class Agent:
    target_entropy: jax.Array = None
    buffer: ReplayBuffer
    actor_network: nn.Module
    critic_network: nn.Module
    actor_state: TrainState
    critic_state: QTrainState
    temp_coef: nn.Module
    temp_state: TrainState
    grad_steps: int = 0

    def __init__(self, env_samples: Dict, cfg: DictConfig, env_params: Dict):
        self.cfg = cfg
        self.env_params = env_params
        self.env_samples = env_samples
        self.comm = MPI.COMM_WORLD

    def setup(self, actor_key, critic_key, dropout_key) -> None:
        if self.comm.Get_rank() == 0:
            logger.info(
                self.actor_network.tabulate(
                    actor_key,
                    *self.env_samples["actor"],
                    depth=1 if not self.cfg.debug else None,
                )
            )
            logger.info(
                self.critic_network.tabulate(
                    {"params": critic_key, "dropout": dropout_key},
                    *self.env_samples["critic"],
                    depth=1 if not self.cfg.debug else None,
                )
            )

        actor_tx_chain = []
        if actor_decay := self.cfg.agent.actor.weight_decay:
            actor_tx_chain.append(optax.additive_weight_decay(actor_decay))
        if max_norm := self.cfg.agent.actor.max_norm:
            actor_tx_chain.append(optax.clip_by_global_norm(max_norm))
        if self.cfg.agent.actor.nan_to_zero:
            actor_tx_chain.append(optax.zero_nans())
        actor_tx_chain.append(optax.adam(learning_rate=self.cfg.agent.actor.lr))
        self.actor_network.apply = jax.jit(self.actor_network.apply)
        self.actor_state = TrainState.create(
            apply_fn=self.actor_network.apply,
            params=self.actor_network.init(actor_key, *self.env_samples["actor"]),
            tx=optax.chain(*actor_tx_chain),
        )
        global_params = tree_bcast(self.actor_state.params)
        self.actor_state = self.actor_state.replace(params=global_params)
        global_opt_state = tree_bcast(self.actor_state.opt_state)
        self.actor_state = self.actor_state.replace(opt_state=global_opt_state)

        self.critic_network.apply = jax.jit(self.critic_network.apply)
        critic_tx_chain = []
        if critic_decay := self.cfg.agent.actor.weight_decay:
            critic_tx_chain.append(optax.additive_weight_decay(critic_decay))
        if max_norm := self.cfg.agent.critic.max_norm:
            critic_tx_chain.append(optax.clip_by_global_norm(max_norm))
        if self.cfg.agent.critic.nan_to_zero:
            critic_tx_chain.append(optax.zero_nans())
        critic_tx_chain.append(optax.adam(learning_rate=self.cfg.agent.critic.lr))
        critic_variables = self.critic_network.init(
            {"params": critic_key, "dropout": dropout_key}, *self.env_samples["critic"]
        )
        self.critic_state = QTrainState.create(
            apply_fn=self.critic_network.apply,
            params=critic_variables,
            target_params=critic_variables.copy(),
            tx=optax.chain(*critic_tx_chain),
        )
        global_params = tree_bcast(self.critic_state.params)
        self.critic_state = self.critic_state.replace(params=global_params)
        global_opt_state = tree_bcast(self.critic_state.opt_state)
        self.critic_state = self.critic_state.replace(opt_state=global_opt_state)

    def store(self, episodes: Dict[str, np.ndarray]) -> None:
        self.buffer.store_episode(episode_batch=episodes)

    def get_normalizer_stats(self):
        raise NotImplementedError

    def _update_networks(self, transitions: Dict[str, jax.Array]):
        raise NotImplementedError

    def save(self, model_path: str, epoch="final"):
        ckpt = {
            "actor": flax.serialization.to_bytes(self.actor_state),
            "critic": flax.serialization.to_bytes(self.critic_state),
            "normalizer": self.get_normalizer_stats(),
        }
        with open(os.path.join(model_path, f"model_{epoch}.pkl"), "wb") as f:
            pickle.dump(ckpt, f)

    def load(self, model_path: str):
        with open(model_path, "rb") as f:
            state_restored = pickle.load(f)
        self.actor_state = flax.serialization.from_bytes(self.actor_state, state_restored["actor"])
        self.critic_state = flax.serialization.from_bytes(self.critic_state, state_restored["critic"])
        for key, val in state_restored["normalizer"].items():
            for k, v in val.items():
                setattr(getattr(self, key), k, v)

    def train(self) -> Dict:
        metric_list = []
        utd_ratio = self.cfg.utd_ratio * self.cfg.episode_batch_size
        transitions = self.buffer.sample(self.cfg.batch_size * utd_ratio)

        # Taken from Smith et al. 2022
        # "A Walk in the Park: Learning to Walk in 20 Minutes With Model-Free Reinforcement Learning"
        for i in range(utd_ratio):

            def _slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]  # noqa: B023

            mini_batch = jax.tree_util.tree_map(_slice, transitions)
            metric_list.append(self._update_networks(mini_batch))
            self.grad_steps += 1
        metric_dict = {}
        for dict_item in metric_list:
            for key, value in dict_item.items():
                if key not in metric_dict:
                    metric_dict[key] = [value]
                else:
                    metric_dict[key] = np.concatenate([metric_dict[key], [value]])
        return metric_dict

    def get_current_grad_steps(self) -> int:
        return self.grad_steps
