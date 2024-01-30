# Goal-Conditioned Reinforcement Learning (Jax/Flax/Optax)
This repository contains a collection of goal-conditioned reinforcement learning algorithms.
It is compatible with the latest [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API and uses very recent version of jax, flax and optax.
We support multiprocessing via [mpi4jax](https://github.com/mpi4jax/mpi4jax) like the deprecated OpenAI [baselines](https://github.com/openai/baselines).

## Supported Algorithms

- [x] Deep Deterministic Policy Gradient (DDPG [paper link](https://arxiv.org/abs/1509.02971))
- [x] Soft Actor-Critic (SAC [paper link](https://arxiv.org/abs/1801.01290))
- [x] DroQ ([paper link](https://arxiv.org/abs/2110.02034))

## Installation
- `git clone https://github.com/frankroeder/goal_conditioned_rl.git`
- pip users: `pip install -r requirements.txt`
- conda users: `conda create --file= conda_env.yaml`
- libraries: `apt install libopenmpi-dev`

### Jax CUDA Support
> https://github.com/google/jax#installation
To install on a machine with an Nvidia GPU, run
```bash
# install packages
pip install -r requirements.txt
# remove jaxlib and install cuda version of necessary
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
## Training

### Single process
```bash
# SAC
python train.py n_epochs=10 agent=sac env_name=FetchPush-v2 hindsight=her agent.critic.dropout=0.0
# DDPG
python train.py n_epochs=10 agent=ddpg env_name=FetchPush-v2 hindsight=her
# DroQ
python train.py n_epochs=10 agent=sac env_name=FetchPush-v2 hindsight=her agent.critic.dropout=0.01
```

### Multiple processes
```bash
mpirun -np 4 python -u train.py n_epochs=10 agent=sac env_name=FetchPush-v2 hindsight=her
```

## Enjoy your trained agent
```bash
python demo.py --demo_path <path to the trial folder>
# or
python demo.py --wandb_url <wandb trial url>
```
