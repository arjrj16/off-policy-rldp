#!/bin/bash

#SBATCH -J idql-train-og-bench                  # job name
#SBATCH -p mit_normal_gpu              # partition
#SBATCH --gres=gpu:l40:1               # 1x H200 GPU
#SBATCH -N 1                           # 1 node
# Pretrain needs little CPU (DataLoader num_workers=0 on cuda datasets) but
# 1G was marginal (torch + CUDA context + 132MB dataset). Finetune runs 50
# async MuJoCo workers -> bump to ~16 CPUs / 32G before switching the command.
#SBATCH --cpus-per-task=16            # CPUs for parallel envs
#SBATCH --mem=32G                      # memory
#SBATCH -t 06:00:00                    # walltime limit
#SBATCH -o logs/%x-%j.out              # log file

set -e -o pipefail

source ~/.bashrc

# Go to DPPO project directory
cd $W/off-policy-rldp-ogbench

# Set required environment variables
export DPPO_DATA_DIR=$W/off-policy-rldp-ogbench/data      # where datasets/normalization are stored
export DPPO_LOG_DIR=$W/off-policy-rldp-ogbench/log        # where checkpoints/logs go
export DPPO_WANDB_ENTITY=arjunmelwani-massachusetts-institute-of-technology                       # your WandB username (or set wandb=null below)
# Suppress d4rl import warnings
export D4RL_SUPPRESS_IMPORT_ERROR=1

# set up conda env for mujoco to work:
module load miniforge/24.3.0-0
conda activate mjgl

export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Headless rendering:
export MUJOCO_GL=egl


# ============================================
# Choose ONE of these training commands:
# ============================================

# Option 1: Gym - Hopper (fine-tune pre-trained policy)
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/gym/finetune/hopper-v2 
    
# Option 2: Gym - Walker2D
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/gym/finetune/walker2d-v2

# Option 3: Gym - HalfCheetah
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/gym/finetune/halfcheetah-v2

# Option 4: Robomimic - Can
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/robomimic/finetune/can 


# Option 5: Train from scratch (no pre-training)
# uv run python script/run.py --config-name=idql_diffusion_mlp \
#     --config-dir=cfg/gym/scratch/hopper-v2

# # Option 6: Robomimic - transport
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/robomimic/finetune/transport 

# # Option 7: push-t - IDQL
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/pusht/finetune

# # Option 8: push-t - DPPO
# uv run python script/run.py --config-name=ft_ppo_diffusion_mlp \
#     --config-dir=cfg/pusht/finetune

# transport pretrain
# uv run python script/run.py --config-name=pre_diffusion_mlp \
#     --config-dir=cfg/robomimic/pretrain/transport

# OGBench - cube-triple pretrain
# uv run python script/run.py --config-name=pre_diffusion_mlp --config-dir=cfg/ogbench/pretrain/cube-triple

# OGBench - cube-triple IDQL finetune
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp --config-dir=cfg/ogbench/finetune/cube-triple


#OG BENCH - singel cube pretrain:
# uv run python script/run.py --config-name=pre_diffusion_mlp --config-dir=cfg/ogbench/pretrain/cube-single

# OGBench - cube-single IDQL finetune
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp --config-dir=cfg/ogbench/finetune/cube-single

# OGBench - cube-single IDQL critic pretrain (offline Q/V; independent of BC pretrain)
# uv run python script/run.py --config-name=pre_idql_critic_mlp --config-dir=cfg/ogbench/pretrain/cube-single

# ---- cube-double (data processed via:
#   python script/dataset/process_ogbench_dataset.py \
#     --env_name cube-double-play-singletask-task2-v0 --save_dir data/ogbench/cube-double) ----

# OGBench - cube-double BC actor pretrain
# uv run python script/run.py --config-name=pre_diffusion_mlp --config-dir=cfg/ogbench/pretrain/cube-double

# OGBench - cube-double IDQL critic pretrain
# uv run python script/run.py --config-name=pre_idql_critic_mlp --config-dir=cfg/ogbench/pretrain/cube-double

# OGBench - cube-double IDQL finetune (set base_policy_path timestamp first;
# add model.critic_path=.../checkpoint/state_500.pt for the critic-pretrained arm,
# train.use_bc_warmup=true train.bc_warmup_iters=K for the BC-warmup arm,
# env.save_video=true train.render.num=1 to upload rollout videos to wandb)
uv run python script/run.py --config-name=ft_idql_diffusion_mlp --config-dir=cfg/ogbench/finetune/cube-double
