"""
Add sparse rewards to an existing NPZ dataset.

For robomimic tasks (transport, can, lift, square), the reward is sparse:
- reward = 0 for all intermediate steps
- reward = 1 for the final step of each trajectory (successful demonstration)

This script simply adds these sparse rewards to an existing NPZ file,
enabling offline Q/V pretraining without needing the environment.

Usage:
    python script/dataset/add_sparse_rewards.py \
        --input_path /path/to/train.npz \
        --output_path /path/to/train_with_rewards.npz \
        --final_reward 1.0
"""

import argparse
import logging
import os
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def add_sparse_rewards(
    input_path: str,
    output_path: str,
    final_reward: float = 1.0,
    intermediate_reward: float = 0.0,
):
    """
    Add sparse rewards to an NPZ dataset.
    
    Args:
        input_path: Path to input NPZ file (must have states, actions, traj_lengths)
        output_path: Path to save output NPZ file with rewards
        final_reward: Reward for final step of each trajectory
        intermediate_reward: Reward for all other steps (default 0)
    """
    log.info(f"Loading dataset from {input_path}")
    
    # Load existing dataset
    data = np.load(input_path, allow_pickle=False)
    
    states = data["states"]
    actions = data["actions"]
    traj_lengths = data["traj_lengths"]
    
    # Check if terminals exist, otherwise create them
    if "terminals" in data:
        terminals = data["terminals"]
    else:
        # Create terminals: True only at last step of each trajectory
        terminals = np.zeros(len(states), dtype=bool)
        cumsum = np.cumsum(traj_lengths)
        for end_idx in cumsum:
            terminals[end_idx - 1] = True
    
    total_steps = len(states)
    num_trajectories = len(traj_lengths)
    
    log.info(f"Dataset has {num_trajectories} trajectories, {total_steps} total steps")
    log.info(f"States shape: {states.shape}")
    log.info(f"Actions shape: {actions.shape}")
    
    # Create sparse rewards
    rewards = np.full(total_steps, intermediate_reward, dtype=np.float32)
    
    # Set final reward for last step of each trajectory
    cumsum = np.cumsum(traj_lengths)
    for end_idx in cumsum:
        rewards[end_idx - 1] = final_reward
    
    log.info(f"Created sparse rewards:")
    log.info(f"  Final reward: {final_reward}")
    log.info(f"  Intermediate reward: {intermediate_reward}")
    log.info(f"  Total reward per trajectory: {final_reward} (sparse)")
    log.info(f"  Sum of all rewards: {np.sum(rewards)}")
    log.info(f"  Non-zero rewards: {np.sum(rewards != 0)}/{total_steps}")
    
    # Save new dataset
    log.info(f"Saving dataset with rewards to {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    np.savez_compressed(
        output_path,
        states=states,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        traj_lengths=traj_lengths,
    )
    
    log.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add sparse rewards to NPZ dataset for offline Q/V pretraining"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input NPZ file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output NPZ file with rewards",
    )
    parser.add_argument(
        "--final_reward",
        type=float,
        default=1.0,
        help="Reward for final step of each trajectory (default: 1.0)",
    )
    parser.add_argument(
        "--intermediate_reward",
        type=float,
        default=0.0,
        help="Reward for intermediate steps (default: 0.0)",
    )
    
    args = parser.parse_args()
    
    add_sparse_rewards(
        input_path=args.input_path,
        output_path=args.output_path,
        final_reward=args.final_reward,
        intermediate_reward=args.intermediate_reward,
    )
