"""
Relabel robomimic dataset with rewards by replaying trajectories through the environment.

This script:
1. Loads the original HDF5 file to get initial states for each trajectory
2. Loads the existing NPZ file (states, actions, traj_lengths)
3. Creates a robomimic environment
4. For each trajectory: resets env to initial state -> replays recorded actions -> collects rewards
5. Saves a new NPZ file with proper rewards

Usage:
    python script/dataset/relabel_robomimic_rewards.py \
        --hdf5_path /path/to/robomimic/dataset.hdf5 \
        --npz_path /path/to/processed/train.npz \
        --env_meta_path cfg/robomimic/env_meta/transport.json \
        --output_path /path/to/output/train_with_rewards.npz \
        --normalization_path /path/to/normalization.npz

Note: This requires the robomimic environment to be installed and accessible.
"""

import argparse
import json
import logging
import os
import numpy as np
import h5py
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def create_robomimic_env(env_meta_path, normalization_path=None, low_dim_keys=None):
    """Create a robomimic environment from metadata."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    
    # Default low_dim_keys for transport
    if low_dim_keys is None:
        low_dim_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot1_eef_pos",
            "robot1_eef_quat",
            "robot1_gripper_qpos",
            "object",
        ]
    
    # Initialize observation modality
    obs_modality_dict = {"low_dim": low_dim_keys}
    ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
    
    # Load env metadata
    with open(env_meta_path, "r") as f:
        env_meta = json.load(f)
    
    # Create environment
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    
    # Disable hard reset for efficiency
    env.env.hard_reset = False
    
    return env, low_dim_keys


def get_observation(raw_obs, low_dim_keys, obs_min=None, obs_max=None, normalize=False):
    """Extract and optionally normalize observation from raw environment output."""
    obs = np.concatenate([raw_obs[key] for key in low_dim_keys], axis=0)
    if normalize and obs_min is not None and obs_max is not None:
        obs = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-6) - 1
    return obs


def unnormalize_action(action, action_min, action_max):
    """Convert normalized action [-1, 1] back to original scale."""
    action = (action + 1) / 2  # [-1, 1] -> [0, 1]
    return action * (action_max - action_min) + action_min


def relabel_rewards(
    hdf5_path,
    npz_path,
    env_meta_path,
    output_path,
    normalization_path=None,
    low_dim_keys=None,
    max_episodes=None,
    verify_states=True,
    state_mismatch_threshold=0.1,
):
    """
    Relabel the NPZ dataset with rewards by replaying through the environment.
    
    Args:
        hdf5_path: Path to original robomimic HDF5 file (contains initial states)
        npz_path: Path to processed NPZ file (contains states, actions, traj_lengths)
        env_meta_path: Path to environment metadata JSON
        output_path: Path to save the relabeled NPZ file
        normalization_path: Path to normalization stats (if data is normalized)
        low_dim_keys: List of observation keys to use
        max_episodes: Maximum number of episodes to process (None = all)
        verify_states: Whether to verify that replayed states match recorded states
        state_mismatch_threshold: Threshold for state mismatch warning
    """
    log.info(f"Loading HDF5 from {hdf5_path}")
    log.info(f"Loading NPZ from {npz_path}")
    
    # Load NPZ dataset
    npz_data = np.load(npz_path, allow_pickle=False)
    states = npz_data["states"]
    actions = npz_data["actions"]
    traj_lengths = npz_data["traj_lengths"]
    terminals = npz_data.get("terminals", np.zeros(len(states), dtype=bool))
    
    log.info(f"NPZ dataset: {len(traj_lengths)} trajectories, {len(states)} total steps")
    log.info(f"States shape: {states.shape}, Actions shape: {actions.shape}")
    
    # Load normalization stats if provided
    normalize = normalization_path is not None
    if normalize:
        norm_data = np.load(normalization_path)
        obs_min = norm_data["obs_min"]
        obs_max = norm_data["obs_max"]
        action_min = norm_data["action_min"]
        action_max = norm_data["action_max"]
        log.info(f"Loaded normalization stats from {normalization_path}")
    else:
        obs_min = obs_max = action_min = action_max = None
    
    # Create environment
    log.info(f"Creating environment from {env_meta_path}")
    env, low_dim_keys = create_robomimic_env(
        env_meta_path,
        normalization_path=normalization_path,
        low_dim_keys=low_dim_keys,
    )
    
    # Limit episodes if specified
    if max_episodes is not None:
        traj_lengths = traj_lengths[:max_episodes]
        total_steps = np.sum(traj_lengths)
        states = states[:total_steps]
        actions = actions[:total_steps]
        terminals = terminals[:total_steps]
        log.info(f"Limited to {max_episodes} episodes, {total_steps} steps")
    
    # Initialize rewards array
    rewards = np.zeros(len(states), dtype=np.float32)
    
    # Load HDF5 to get initial states
    with h5py.File(hdf5_path, "r") as hdf5_file:
        demos = sorted(list(hdf5_file["data"].keys()))
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        
        if max_episodes is not None:
            demos = demos[:max_episodes]
        
        log.info(f"Processing {len(demos)} demonstrations")
        
        # Track statistics
        total_reward = 0
        successful_episodes = 0
        state_mismatches = 0
        
        # Process each trajectory
        step_idx = 0
        for ep_idx, (ep_name, traj_len) in enumerate(tqdm(zip(demos, traj_lengths), total=len(demos), desc="Relabeling")):
            # Get initial state from HDF5
            initial_state = hdf5_file[f"data/{ep_name}"]["states"][0]
            
            # Reset environment to initial state
            try:
                raw_obs = env.reset_to({"states": initial_state})
            except Exception as e:
                log.warning(f"Failed to reset to initial state for episode {ep_idx}: {e}")
                log.warning("Trying regular reset...")
                raw_obs = env.reset()
            
            episode_reward = 0
            
            # Replay actions and collect rewards
            for t in range(traj_len):
                # Get the action for this step
                action = actions[step_idx]
                
                # Unnormalize action if necessary
                if normalize:
                    action_raw = unnormalize_action(action, action_min, action_max)
                else:
                    action_raw = action
                
                # Step environment
                raw_obs, reward, done, info = env.step(action_raw)
                
                # Store reward
                rewards[step_idx] = reward
                episode_reward += reward
                
                # Optionally verify states match
                if verify_states and t < traj_len - 1:
                    obs_from_env = get_observation(
                        raw_obs, low_dim_keys, obs_min, obs_max, normalize
                    )
                    obs_from_data = states[step_idx + 1]
                    mismatch = np.max(np.abs(obs_from_env - obs_from_data))
                    if mismatch > state_mismatch_threshold:
                        state_mismatches += 1
                        if state_mismatches <= 10:  # Only log first 10
                            log.warning(
                                f"State mismatch at ep {ep_idx}, step {t}: max diff = {mismatch:.4f}"
                            )
                
                step_idx += 1
            
            total_reward += episode_reward
            if episode_reward > 0:
                successful_episodes += 1
        
        log.info(f"Total reward collected: {total_reward:.2f}")
        log.info(f"Successful episodes (reward > 0): {successful_episodes}/{len(demos)}")
        log.info(f"Average episode reward: {total_reward / len(demos):.4f}")
        if verify_states:
            log.info(f"State mismatches (>{state_mismatch_threshold}): {state_mismatches}")
    
    # Save relabeled dataset
    log.info(f"Saving relabeled dataset to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez_compressed(
        output_path,
        states=states,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        traj_lengths=traj_lengths,
    )
    
    log.info("Done!")
    
    # Print reward statistics
    log.info(f"Reward statistics:")
    log.info(f"  Min: {np.min(rewards):.4f}")
    log.info(f"  Max: {np.max(rewards):.4f}")
    log.info(f"  Mean: {np.mean(rewards):.4f}")
    log.info(f"  Std: {np.std(rewards):.4f}")
    log.info(f"  Non-zero: {np.sum(rewards != 0)}/{len(rewards)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relabel robomimic dataset with rewards by replaying through environment"
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to original robomimic HDF5 file",
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to processed NPZ file",
    )
    parser.add_argument(
        "--env_meta_path",
        type=str,
        required=True,
        help="Path to environment metadata JSON",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save relabeled NPZ file",
    )
    parser.add_argument(
        "--normalization_path",
        type=str,
        default=None,
        help="Path to normalization stats NPZ (if data is normalized)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process",
    )
    parser.add_argument(
        "--no_verify_states",
        action="store_true",
        help="Disable state verification (faster but less safe)",
    )
    parser.add_argument(
        "--low_dim_keys",
        nargs="+",
        default=None,
        help="Observation keys to use (default: transport keys)",
    )
    
    args = parser.parse_args()
    
    relabel_rewards(
        hdf5_path=args.hdf5_path,
        npz_path=args.npz_path,
        env_meta_path=args.env_meta_path,
        output_path=args.output_path,
        normalization_path=args.normalization_path,
        low_dim_keys=args.low_dim_keys,
        max_episodes=args.max_episodes,
        verify_states=not args.no_verify_states,
    )
