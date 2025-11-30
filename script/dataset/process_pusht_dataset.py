"""
Process Push-T dataset from Zarr format and convert to the format required for diffusion training.

The Push-T dataset uses a Zarr storage format with:
- data/state: (N, 5) array with [agent_x, agent_y, block_x, block_y, block_angle]
- data/action: (N, 2) array with [target_agent_x, target_agent_y]
- meta/episode_ends: 1-D array marking end indices for each episode

This script converts it to our npz format with:
- states: normalized states [-1, 1]
- actions: normalized actions [-1, 1]
- rewards: reward at each step (computed from coverage)
- terminals: whether episode ended due to success
- traj_lengths: length of each trajectory
- normalization.npz: min/max values for obs and action

Usage:
    python process_pusht_dataset.py --dataset_path pusht_cchi_v7_replay.zarr.zip --save_dir ./data/pusht
"""

import os
import logging
import numpy as np
import zarr
from tqdm import tqdm
import random
from copy import deepcopy


def process_pusht_dataset(
    dataset_path,
    save_dir,
    save_name_prefix="",
    val_split=0.0,
    max_episodes=-1,
    logger=None,
):
    """
    Process Push-T dataset from Zarr format to npz format for diffusion training.

    Args:
        dataset_path: Path to the zarr dataset (e.g., pusht_cchi_v7_replay.zarr.zip)
        save_dir: Directory to save processed data
        save_name_prefix: Prefix for saved file names
        val_split: Fraction of data to use for validation
        max_episodes: Maximum number of episodes to use (-1 for all)
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Load zarr dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset_root = zarr.open(dataset_path, "r")

    # Extract data
    states = dataset_root["data"]["state"][:]  # (N, 5)
    actions = dataset_root["data"]["action"][:]  # (N, 2)
    episode_ends = dataset_root["meta"]["episode_ends"][:]  # (num_episodes,)

    logger.info("\n========== Basic Info ===========")
    logger.info(f"States shape: {states.shape}")
    logger.info(f"Actions shape: {actions.shape}")
    logger.info(f"Number of episodes: {len(episode_ends)}")

    # Compute trajectory lengths
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    traj_lengths = episode_ends - episode_starts

    logger.info(f"Total transitions: {np.sum(traj_lengths)}")
    logger.info(
        f"Trajectory length mean/std: {np.mean(traj_lengths):.2f}, {np.std(traj_lengths):.2f}"
    )
    logger.info(
        f"Trajectory length min/max: {np.min(traj_lengths)}, {np.max(traj_lengths)}"
    )

    # Compute normalization statistics
    obs_min = np.min(states, axis=0)
    obs_max = np.max(states, axis=0)
    action_min = np.min(actions, axis=0)
    action_max = np.max(actions, axis=0)

    logger.info(f"obs min: {obs_min}")
    logger.info(f"obs max: {obs_max}")
    logger.info(f"action min: {action_min}")
    logger.info(f"action max: {action_max}")

    # Limit episodes if specified
    if max_episodes > 0:
        episode_ends = episode_ends[:max_episodes]
        episode_starts = episode_starts[:max_episodes]
        traj_lengths = traj_lengths[:max_episodes]

    num_traj = len(traj_lengths)

    # Split into train and validation
    num_train = int(num_traj * (1 - val_split))
    train_indices = set(random.sample(range(num_traj), k=num_train))

    # Prepare data containers
    out_train = {
        "states": [],
        "actions": [],
        "rewards": [],
        "terminals": [],
        "traj_lengths": [],
    }
    out_val = deepcopy(out_train)

    train_episode_reward_all = []
    val_episode_reward_all = []

    # Process each trajectory
    for i in tqdm(range(num_traj), desc="Processing trajectories"):
        start = episode_starts[i]
        end = episode_ends[i]
        traj_length = traj_lengths[i]

        # Extract trajectory data
        traj_states = states[start:end]
        traj_actions = actions[start:end]

        # Compute rewards based on coverage (simplified - using proxy from state)
        # In the actual env, reward is based on T-block overlap with goal
        # Here we use a simple heuristic based on block position relative to goal
        # Goal position: (256, 256, pi/4)
        goal_pos = np.array([256, 256])
        goal_angle = np.pi / 4

        # Compute distance-based reward proxy
        block_pos = traj_states[:, 2:4]
        block_angle = traj_states[:, 4]

        pos_dist = np.linalg.norm(block_pos - goal_pos, axis=1)
        angle_diff = np.abs(np.mod(block_angle - goal_angle + np.pi, 2 * np.pi) - np.pi)

        # Normalize rewards to [0, 1] range
        max_pos_dist = 512 * np.sqrt(2)  # diagonal of 512x512 space
        pos_reward = 1 - (pos_dist / max_pos_dist)
        angle_reward = 1 - (angle_diff / np.pi)
        traj_rewards = 0.5 * pos_reward + 0.5 * angle_reward

        # Terminal: True for last step of successful episodes (high reward)
        traj_terminals = np.zeros(traj_length, dtype=bool)
        if traj_rewards[-1] > 0.9:  # Consider success if final reward > 0.9
            traj_terminals[-1] = True

        # Normalize states and actions
        traj_states_norm = (
            2 * (traj_states - obs_min) / (obs_max - obs_min + 1e-6) - 1
        )
        traj_actions_norm = (
            2 * (traj_actions - action_min) / (action_max - action_min + 1e-6) - 1
        )

        # Add to appropriate split
        if i in train_indices:
            out = out_train
            episode_reward_all = train_episode_reward_all
        else:
            out = out_val
            episode_reward_all = val_episode_reward_all

        out["states"].append(traj_states_norm)
        out["actions"].append(traj_actions_norm)
        out["rewards"].append(traj_rewards)
        out["terminals"].append(traj_terminals)
        out["traj_lengths"].append(traj_length)
        episode_reward_all.append(np.sum(traj_rewards))

    # Concatenate trajectories
    for key in ["states", "actions", "rewards", "terminals"]:
        out_train[key] = np.concatenate(out_train[key], axis=0)
        if val_split > 0 and len(out_val[key]) > 0:
            out_val[key] = np.concatenate(out_val[key], axis=0)

    out_train["traj_lengths"] = np.array(out_train["traj_lengths"])
    out_val["traj_lengths"] = np.array(out_val["traj_lengths"])

    # Save datasets
    os.makedirs(save_dir, exist_ok=True)

    train_save_path = os.path.join(save_dir, f"{save_name_prefix}train.npz")
    np.savez_compressed(
        train_save_path,
        states=out_train["states"].astype(np.float32),
        actions=out_train["actions"].astype(np.float32),
        rewards=out_train["rewards"].astype(np.float32),
        terminals=out_train["terminals"],
        traj_lengths=out_train["traj_lengths"],
    )
    logger.info(f"Saved train dataset to {train_save_path}")

    if val_split > 0:
        val_save_path = os.path.join(save_dir, f"{save_name_prefix}val.npz")
        np.savez_compressed(
            val_save_path,
            states=out_val["states"].astype(np.float32),
            actions=out_val["actions"].astype(np.float32),
            rewards=out_val["rewards"].astype(np.float32),
            terminals=out_val["terminals"],
            traj_lengths=out_val["traj_lengths"],
        )
        logger.info(f"Saved val dataset to {val_save_path}")

    # Save normalization info
    normalization_save_path = os.path.join(
        save_dir, f"{save_name_prefix}normalization.npz"
    )
    np.savez(
        normalization_save_path,
        obs_min=obs_min.astype(np.float32),
        obs_max=obs_max.astype(np.float32),
        action_min=action_min.astype(np.float32),
        action_max=action_max.astype(np.float32),
    )
    logger.info(f"Saved normalization to {normalization_save_path}")

    # Summary statistics
    logger.info("\n========== Final Summary ===========")
    logger.info(
        f"Train - Trajectories: {len(out_train['traj_lengths'])}, "
        f"Transitions: {np.sum(out_train['traj_lengths'])}"
    )
    logger.info(
        f"Train - States shape: {out_train['states'].shape}, "
        f"Actions shape: {out_train['actions'].shape}"
    )
    if val_split > 0:
        logger.info(
            f"Val - Trajectories: {len(out_val['traj_lengths'])}, "
            f"Transitions: {np.sum(out_val['traj_lengths'])}"
        )
    logger.info(
        f"Train - Mean episode reward: {np.mean(train_episode_reward_all):.2f}"
    )


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the zarr dataset (e.g., pusht_cchi_v7_replay.zarr.zip)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/pusht",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--save_name_prefix",
        type=str,
        default="",
        help="Prefix for saved file names",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="Maximum number of episodes to use (-1 for all)",
    )
    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(
        args.save_dir,
        f"process_pusht_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )

    logger = logging.getLogger("process_pusht_dataset")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    process_pusht_dataset(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        save_name_prefix=args.save_name_prefix,
        val_split=args.val_split,
        max_episodes=args.max_episodes,
        logger=logger,
    )

