"""
This demo.py script allows loading a trained agent for demonstration.
The location to restore the agent can either be a local folder (--demo_path) or
a remotely accessible WANDB trial in a project (--wandb_url).
"""
import argparse
import os
import random

import jax
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from modules import RolloutWorker
from modules.agent import DDPG, SAC
from modules.gym_wrapper import setup_environments, setup_wrappers
from modules.mpi_utils import logger
from modules.utils import BatchEnv, get_env_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_epochs", type=int, default=10, help="the demo epochs")
    parser.add_argument("--demo_path", type=str, default="", help="Load experiment from local path")
    parser.add_argument("--model_file", type=str, default="model_final.pkl")
    parser.add_argument(
        "--wandb_url",
        type=str,
        default="",
        help="""Download and run the trained agent locally, using the path
        WANDB schema <entity>/<project name>/<run url id>""",
    )

    demo_args = parser.parse_args()
    MODEL_LOAD_PATH = f"models/{demo_args.model_file}"
    if demo_args.wandb_url:
        CONFIG_LOAD_PATH = "config.yaml"
    else:
        CONFIG_LOAD_PATH = "omega_config.yaml"
    if demo_args.wandb_url:
        TMP_DEMO_DIR = "/tmp/gcrl_demo"
        wandb_url = demo_args.wandb_url.replace("/runs", "")
        # create a temporary folder called lcjax_demo to download the experiment files
        if os.path.exists(TMP_DEMO_DIR):
            os.system(f"rm -rf {TMP_DEMO_DIR}")
        os.makedirs(TMP_DEMO_DIR)
        wandb.restore(MODEL_LOAD_PATH, run_path=wandb_url, root=TMP_DEMO_DIR)
        wandb.restore(CONFIG_LOAD_PATH, run_path=wandb_url, root=TMP_DEMO_DIR)
        path = TMP_DEMO_DIR
        model_path = os.path.join(path, MODEL_LOAD_PATH)
        logger.warn(f"Using temporary folder {TMP_DEMO_DIR} to download and store experiment files.")
    else:
        path = demo_args.demo_path
        model_path = os.path.join(path, MODEL_LOAD_PATH)

    with open(os.path.join(path, CONFIG_LOAD_PATH)) as f:
        cfg = OmegaConf.load(f.name)

    if demo_args.wandb_url:
        with open_dict(cfg):
            for rng_key, val in cfg.items():
                cfg[rng_key] = val["value"] if isinstance(val, DictConfig) else val

    logger.info(OmegaConf.to_yaml(cfg))

    with open_dict(cfg):
        cfg.episode_batch_size = 1
        cfg.n_test_rollouts = 10

    seed = np.random.randint(2**14)
    envs, env_params = setup_environments(cfg, seed, render_mode="human")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)

    envs, env_params = setup_wrappers(envs, cfg, env_params)
    env_samples = get_env_samples(envs[0])
    envs = BatchEnv(envs)

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

    policy.load(model_path)
    rollout_worker = RolloutWorker(envs, policy, cfg, env_params)
    demo_successes = []
    demo_rewards = []
    for _ in range(demo_args.demo_epochs):
        eval_successes, eval_rewards = rollout_worker.generate_test_rollout(animated=True)
        demo_successes.append(np.mean(eval_successes))
        demo_rewards.append(np.mean(eval_rewards))

    logger.info(f"Avg Success Rate: {np.mean(demo_successes)} Avg Reward {np.mean(demo_rewards)}")
