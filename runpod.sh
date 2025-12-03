set -e -o pipefail

source ~/.bashrc

# Go to DPPO project directory
cd $W/off-policy-rldp/

# Set required environment variables
export DPPO_DATA_DIR=$W/off-policy-rldp/data      # where datasets/normalization are stored
export DPPO_LOG_DIR=$W/off-policy-rldp/log        # where checkpoints/logs go
export DPPO_WANDB_ENTITY=arjunmelwani-massachusetts-institute-of-technology                       # your WandB username (or set wandb=null below)
# Suppress d4rl import warnings
export D4RL_SUPPRESS_IMPORT_ERROR=1
export HYDRA_FULL_ERROR=1
# set up conda env for mujoco to work:
# module load miniforge/24.3.0-0
# conda activate mjgl

export CUDA_VISIBLE_DEVICES=0

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

# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/robomimic/finetune/square 



# # Option 7: push-t - IDQL
# uv run python script/run.py --config-name=ft_idql_diffusion_mlp \
#     --config-dir=cfg/pusht/finetune

# # Option 8: push-t - DPPO
# uv run python script/run.py --config-name=ft_ppo_diffusion_mlp \
#     --config-dir=cfg/pusht/finetune

# transport pretrain
uv run python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/transport
