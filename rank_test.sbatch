#!/bin/bash
#SBATCH -J idql-ranktest
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:35:00
#SBATCH -o logs/%x-%j.out

set -e -o pipefail
source ~/.bashrc
cd $W/off-policy-rldp-ogbench

export DPPO_DATA_DIR=$W/off-policy-rldp-ogbench/data
export DPPO_LOG_DIR=$W/off-policy-rldp-ogbench/log
export D4RL_SUPPRESS_IMPORT_ERROR=1

module load miniforge/24.3.0-0
conda activate mjgl
export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl

uv run python script/rank_test.py
