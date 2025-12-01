"""
Collect rollouts from a trained policy and save states/actions for analysis.

This script loads a trained diffusion policy and collects rollout data 
to compare with the original dataset.

Usage:
    python script/dataset/collect_rollouts.py \
        --checkpoint_path $DPPO_LOG_DIR/robomimic-finetune/transport_ft_diffusion_mlp_ta8_td20_tdf10/.../checkpoint/state_200.pt \
        --env_cfg_path cfg/robomimic/env_meta/transport.json \
        --normalization_path $DPPO_DATA_DIR/robomimic/transport/normalization.npz \
        --output_path ./dppo_rollouts.npz \
        --n_episodes 100

"""

import numpy as np
import torch
import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_env(env_cfg_path, normalization_path, n_envs=1):
    """Create robomimic environment with wrappers."""
    import robomimic.utils.env_utils as EnvUtils
    from env.gym_utils.wrapper import MujocoWrapper, RobomimicLowdimWrapper, MultiStepWrapper
    
    with open(env_cfg_path, 'r') as f:
        env_meta = json.load(f)
    
    def make_env():
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
        )
        env = MujocoWrapper(env)
        return env
    
    # Create single env for simplicity
    env = make_env()
    
    # Apply wrappers
    low_dim_keys = [
        'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
        'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos',
        'object'
    ]
    env = RobomimicLowdimWrapper(
        env, 
        normalization_path=normalization_path,
        low_dim_keys=low_dim_keys
    )
    env = MultiStepWrapper(
        env,
        n_obs_steps=1,
        n_action_steps=8,
        max_episode_steps=800,
        reset_within_step=True
    )
    
    return env


def load_policy(checkpoint_path, device='cuda:0'):
    """Load diffusion policy from checkpoint."""
    from model.diffusion.diffusion_ppo import PPODiffusion
    from model.diffusion.diffusion import DiffusionModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to infer model type from checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check if it's PPO or base diffusion
    has_critic = any('critic' in k for k in state_dict.keys())
    
    # We'll just return the state dict and let the caller handle model creation
    return state_dict, has_critic


def collect_rollouts_simple(env, state_dict, n_episodes, device='cuda:0', 
                            denoising_steps=20, horizon_steps=8, action_dim=14, obs_dim=59):
    """
    Collect rollouts using a simplified policy loading approach.
    
    This function manually constructs the model to handle different checkpoint formats.
    """
    from model.diffusion.mlp_diffusion import DiffusionMLP
    from model.diffusion.diffusion import DiffusionModel
    
    # Create the diffusion network
    network = DiffusionMLP(
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        cond_dim=obs_dim,
        time_dim=32,
        mlp_dims=[1024, 1024, 1024],
        residual_style=True,
    )
    
    # Create diffusion model
    model = DiffusionModel(
        network=network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        device=device,
        predict_epsilon=True,
        denoised_clip_value=1.0,
    )
    
    # Load weights (handle different checkpoint formats)
    if 'model' in state_dict:
        model_state = state_dict['model']
    else:
        model_state = state_dict
    
    # Filter to only network weights
    network_state = {}
    for k, v in model_state.items():
        # Handle different key formats
        if k.startswith('network.'):
            network_state[k.replace('network.', '')] = v
        elif k.startswith('actor.'):
            network_state[k.replace('actor.', '')] = v
        elif not any(prefix in k for prefix in ['critic', 'ema', 'target']):
            network_state[k] = v
    
    try:
        model.network.load_state_dict(network_state, strict=False)
        print("Loaded network weights successfully")
    except Exception as e:
        print(f"Warning: Could not load all weights: {e}")
        print("Attempting partial load...")
    
    model.to(device)
    model.eval()
    
    # Collect rollouts
    all_states = []
    all_actions = []
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_states = []
        ep_actions = []
        
        while not done:
            # Get state observation
            state = torch.from_numpy(obs['state']).float().to(device)
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dim
            
            cond = {'state': state}
            
            with torch.no_grad():
                action = model(cond=cond, deterministic=True).cpu().numpy()
            
            if action.ndim == 3:
                action = action[0]  # Remove batch dim
            
            # Store normalized state (already normalized by wrapper)
            ep_states.append(obs['state'].flatten())
            ep_actions.append(action.flatten())
            
            # Step environment  
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        episode_rewards.append(ep_reward)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}, Reward: {ep_reward:.2f}, "
                  f"Avg: {np.mean(episode_rewards):.2f}")
    
    return {
        'states': np.array(all_states),
        'actions': np.array(all_actions),
        'episode_rewards': np.array(episode_rewards),
    }


def main(args):
    print(f"Loading checkpoint from {args.checkpoint_path}")
    state_dict, has_critic = load_policy(args.checkpoint_path, args.device)
    print(f"Checkpoint has critic: {has_critic}")
    
    print(f"Creating environment...")
    env = create_env(args.env_cfg_path, args.normalization_path)
    
    print(f"Collecting {args.n_episodes} rollouts...")
    rollout_data = collect_rollouts_simple(
        env=env,
        state_dict=state_dict,
        n_episodes=args.n_episodes,
        device=args.device,
        denoising_steps=args.denoising_steps,
        horizon_steps=args.horizon_steps,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
    )
    
    print(f"\nRollout Statistics:")
    print(f"  Total states: {rollout_data['states'].shape}")
    print(f"  Total actions: {rollout_data['actions'].shape}")
    print(f"  Episode rewards - mean: {rollout_data['episode_rewards'].mean():.2f}, "
          f"std: {rollout_data['episode_rewards'].std():.2f}")
    print(f"  Episode rewards - min: {rollout_data['episode_rewards'].min():.2f}, "
          f"max: {rollout_data['episode_rewards'].max():.2f}")
    
    # Save rollouts
    np.savez_compressed(
        args.output_path,
        states=rollout_data['states'],
        actions=rollout_data['actions'],
        episode_rewards=rollout_data['episode_rewards'],
    )
    print(f"\nSaved rollouts to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained policy checkpoint")
    parser.add_argument("--env_cfg_path", type=str, 
                        default="cfg/robomimic/env_meta/transport.json",
                        help="Path to environment config")
    parser.add_argument("--normalization_path", type=str, required=True,
                        help="Path to normalization.npz")
    parser.add_argument("--output_path", type=str, default="./rollouts.npz",
                        help="Path to save rollout data")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--denoising_steps", type=int, default=20)
    parser.add_argument("--horizon_steps", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--obs_dim", type=int, default=59)
    args = parser.parse_args()
    
    main(args)

