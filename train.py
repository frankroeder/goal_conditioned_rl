import os
import random
import time
import uuid
from datetime import datetime

import hydra
import jax
import numpy as np
import wandb
from gymnasium.wrappers.record_video import RecordVideo
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf

from modules import RolloutWorker
from modules.agent import DDPG, SAC
from modules.gym_wrapper import setup_environments, setup_wrappers
from modules.mpi_utils import logger
from modules.mpi_utils.mpi_utils import get_metric_stats
from modules.utils import BatchEnv, check_hydra_config, get_env_samples, init_storage


def launch(cfg: DictConfig, comm):
    rank = comm.Get_rank()
    if rank == 0:
        logger.info(OmegaConf.to_yaml(cfg))
    t_total_init = time.time()
    rank_seed = cfg.seed + rank
    envs, env_params = setup_environments(cfg, rank_seed)
    envs, env_params = setup_wrappers(envs, cfg, env_params)
    envs = BatchEnv(envs)

    env_samples = get_env_samples(envs[0])
    os.environ["PYTHONHASHSEED"] = str(rank_seed)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    rng_key = jax.random.PRNGKey(rank_seed)

    if rank == 0:
        logger.info(f"Jax Default Backend: {jax.default_backend()}")
        logger.info(f"Jax Devices: {jax.devices()}")
        logger.info(f"Jax Local Devices: {jax.local_devices()}")

    if cfg.agent.name == "sac":
        policy = SAC(
            rng_key,
            env_samples,
            cfg,
            env_params,
            envs[0].unwrapped.compute_reward,
        )
    elif cfg.agent.name == "ddpg":
        policy = DDPG(
            rng_key,
            env_samples,
            cfg,
            env_params,
            envs[0].unwrapped.compute_reward,
        )
    else:
        raise NotImplementedError

    if rank == 0:
        logdir, model_path = init_storage(cfg)
        logger.configure(dir=logdir, format_strs=cfg.logging_formats)
        start_time = time.time()
        if cfg.use_wandb:
            wandb_args = {
                "name": f"trial_{str(uuid.uuid4())[:5]}",
                "config": OmegaConf.to_container(cfg, resolve=True),
                "reinit": False,
                **cfg.wandb,
            }
            if "tensorboard" in cfg.logging_formats:
                wandb_args["sync_tensorboard"] = True
                wandb_args["monitor_gym"] = True
            run = wandb.init(**wandb_args)
            wandb.save(os.path.join(logdir, "omega_config.yaml"), policy="now")

    rollout_worker = RolloutWorker(envs, policy, cfg, env_params)
    episode_ctr = 0

    for epoch in range(cfg.n_epochs):
        t_init = time.time()
        time_dict = dict(
            train_eps=0.0,
            eval_eps=0.0,
            store=0.0,
            norm_update=0.0,
            train=0.0,
            epoch=0.0,
        )
        train_metrics = {}

        for _ in range(cfg.n_cycles):
            # Environment interactions
            t_i = time.time()
            train_episodes = rollout_worker.generate_rollout(train_mode=True)
            time_dict["train_eps"] += time.time() - t_i

            # log the last step
            train_metrics.setdefault("success_rate", []).extend(train_episodes["success"][:, -1].flatten())
            train_metrics.setdefault("rewards", []).extend(np.sum(train_episodes["reward"], axis=1).flatten())
            assert cfg.episode_batch_size == len(train_episodes["reward"])
            episode_ctr += len(train_episodes["reward"])

            # Storing episodes
            t_i = time.time()
            policy.store(train_episodes)
            time_dict["store"] += time.time() - t_i

            # Updating observation normalization
            t_i = time.time()
            policy._update_normalizer(train_episodes)
            time_dict["norm_update"] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            policy_metrics = policy.train()
            for _key, _val in policy_metrics.items():
                train_metrics.setdefault(_key, []).extend(_val)
            time_dict["train"] += time.time() - t_i

        time_dict["epoch"] += time.time() - t_init
        time_dict["total"] = time.time() - t_total_init

        # evaluate
        t_i = time.time()
        global_train_metrics = {}

        # start video recording
        if rank == 0 and cfg.log_video and epoch > 0 and cfg.video_freq % epoch == 0:
            video_env, video_env_params = setup_environments(cfg, rank_seed, render_mode="rgb_array")
            video_env, _ = setup_wrappers(video_env, cfg, video_env_params)
            video_env[0] = RecordVideo(
                video_env[0],
                video_folder=os.path.join(logdir, "videos"),
                episode_trigger=lambda x: x == cfg.n_test_rollouts,
                video_length=0,
                name_prefix=f"vid_{epoch}",
            )
            video_env = BatchEnv(video_env)
            rollout_worker.env = video_env

        eval_successes, eval_rewards = rollout_worker.generate_test_rollout()

        # close video recording
        if rank == 0 and cfg.log_video and epoch > 0 and cfg.video_freq % epoch == 0:
            rollout_worker.env._envs[0].close_video_recorder()
            rollout_worker.env = envs
            video_env.close()
            # wandb should log the videos by itself when tensorboard is not enabled
            if cfg.use_wandb and "tensorboard" not in cfg.logging_formats:
                wandb.log(
                    {
                        "video":
                        # only log the last test rollout of the episode
                        wandb.Video(
                            os.path.join(logdir, "videos", f"vid_{epoch}-episode-{0}.mp4"),
                            fps=4,
                            format="gif",
                        )
                    }
                )

        time_dict["eval_eps"] += time.time() - t_i
        timesteps = rollout_worker.get_current_timesteps()
        grad_steps = policy.get_current_grad_steps()

        for _key, _val in train_metrics.items():
            if "loss" in _key or "grad" in _key:
                global_train_metrics[_key] = comm.allreduce(np.mean(_val), op=MPI.SUM)
                continue
            global_train_metrics = get_metric_stats(comm, _key, _val, global_train_metrics)

        global_time_dict = {}
        for _key, _val in time_dict.items():
            global_time_dict[_key] = comm.allreduce(_val, op=MPI.SUM)

        eval_metrics = {"success_rate": eval_successes, "reward": eval_rewards}
        global_eval_metrics = {}

        for _key, _val in eval_metrics.items():
            global_eval_metrics = get_metric_stats(comm, _key, _val, global_eval_metrics)

        global_timesteps = comm.allreduce(timesteps, op=MPI.SUM)
        global_episode_ctr = comm.allreduce(episode_ctr, op=MPI.SUM)
        global_grads_steps = comm.allreduce(grad_steps, op=MPI.SUM)

        if rank == 0:
            for _key in global_train_metrics:
                global_train_metrics[_key] /= comm.Get_size()

            for _key in global_eval_metrics:
                global_eval_metrics[_key] /= comm.Get_size()

            for _key in global_time_dict:
                global_time_dict[_key] /= comm.Get_size()

            time_elapsed = time.time() - start_time
            current_sps = int(global_timesteps / (time_elapsed + 1e-8))
            log_data = {
                "epoch": epoch,
                # NOTE: Logged twice because it is easier to have it in the main panel #
                "success_rate": global_eval_metrics["success_rate_mean"],
                "reward": global_eval_metrics["reward_mean"],
                ##################
                "timesteps": int(global_timesteps),
                "episodes": int(global_episode_ctr),
                "grad_steps": int(global_grads_steps),
                "SPS": current_sps,
                **{"time/" + key: val for key, val in global_time_dict.items()},
                **{"train/" + key: val for key, val in global_train_metrics.items()},
                **{"eval/" + key: val for key, val in global_eval_metrics.items()},
            }
            if cfg.use_wandb:
                wandb.log(log_data)
            {logger.logkv(_k, _v) for _k, _v in log_data.items()}
            logger.dumpkvs()
            data_str = " ".join(
                [
                    f"{key}: {val}"
                    for key, val in log_data.items()
                    if key
                    in [
                        "epoch",
                        "timesteps",
                        "SPS",
                    ]
                ]
            )
            data_str += f" success_rate: {log_data['success_rate']:1.2f} ± {log_data['eval/success_rate_std']:1.2f}"
            data_str += f" reward: {log_data['reward']:1.2f} ± {log_data['eval/reward_std']:1.2f}"

            logger.info(f"[{datetime.now()}] " + data_str)
            # Saving model states
            if cfg.save_freq and epoch % cfg.save_freq == 0:
                policy.save(model_path, str(epoch))
                if cfg.use_wandb:
                    wandb.save(
                        os.path.join(model_path, f"model_{epoch}.pkl"),
                        base_path=os.path.split(model_path)[0],
                    )

    if rank == 0:
        policy.save(model_path)
        if cfg.use_wandb:
            wandb.save(
                os.path.join(model_path, "model_final.pkl"),
                base_path=os.path.split(model_path)[0],
            )
            wandb.save(os.path.join(logdir, "progress.csv"))
            run.finish()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    check_hydra_config(cfg, comm)
    try:
        import tensorflow as tf

        tf.config.experimental.set_visible_devices([], "GPU")
    except:
        pass
    os.environ.update(
        XLA_FLAGS=(
            # Limit ourselves to single-threaded jax/xla operations to avoid
            # thrashing. See https://github.com/google/jax/issues/743.
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
        ),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    launch(cfg, comm)


if __name__ == "__main__":
    main()
