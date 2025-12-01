"""
Analyze the Transport dataset to understand state/action coverage.

This script helps investigate why IDQL underperforms on Transport while DPPO succeeds.
Hypothesis: The optimal policy visits states/actions that the initial dataset rarely covers.

Usage:
    python script/dataset/analyze_transport_dataset.py \
        --dataset_path $DPPO_DATA_DIR/robomimic/transport/train.npz \
        --output_dir ./transport_analysis

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json


def load_dataset(dataset_path):
    """Load the preprocessed transport dataset."""
    data = np.load(dataset_path, allow_pickle=False)
    
    # Handle datasets with or without rewards/terminals
    result = {
        'states': data['states'],
        'actions': data['actions'],
        'traj_lengths': data['traj_lengths'],
    }
    
    # Optional fields
    if 'rewards' in data:
        result['rewards'] = data['rewards']
    if 'terminals' in data:
        result['terminals'] = data['terminals']
    
    print(f"Dataset keys: {list(data.keys())}")
    return result


def compute_basic_stats(data):
    """Compute basic statistics about the dataset."""
    states = data['states']
    actions = data['actions']
    traj_lengths = data['traj_lengths']
    
    stats = {
        'num_trajectories': len(traj_lengths),
        'total_transitions': len(states),
        'traj_length_mean': float(np.mean(traj_lengths)),
        'traj_length_std': float(np.std(traj_lengths)),
        'traj_length_min': int(np.min(traj_lengths)),
        'traj_length_max': int(np.max(traj_lengths)),
        'state_dim': states.shape[1],
        'action_dim': actions.shape[1],
        'state_min': states.min(axis=0).tolist(),
        'state_max': states.max(axis=0).tolist(),
        'state_mean': states.mean(axis=0).tolist(),
        'state_std': states.std(axis=0).tolist(),
        'action_min': actions.min(axis=0).tolist(),
        'action_max': actions.max(axis=0).tolist(),
        'action_mean': actions.mean(axis=0).tolist(),
        'action_std': actions.std(axis=0).tolist(),
    }
    
    if 'rewards' in data:
        rewards = data['rewards']
        stats['reward_sum_per_traj_mean'] = float(np.mean([
            rewards[sum(traj_lengths[:i]):sum(traj_lengths[:i+1])].sum() 
            for i in range(len(traj_lengths))
        ]))
        stats['reward_max'] = float(np.max(rewards))
        stats['reward_min'] = float(np.min(rewards))
    
    return stats


def analyze_trajectory_diversity(data):
    """Analyze how diverse the trajectories are."""
    states = data['states']
    traj_lengths = data['traj_lengths']
    
    # Extract start and end states for each trajectory
    start_states = []
    end_states = []
    
    idx = 0
    for length in traj_lengths:
        start_states.append(states[idx])
        end_states.append(states[idx + length - 1])
        idx += length
    
    start_states = np.array(start_states)
    end_states = np.array(end_states)
    
    return {
        'start_states': start_states,
        'end_states': end_states,
        'start_state_std': start_states.std(axis=0),
        'end_state_std': end_states.std(axis=0),
    }


def compute_state_coverage(states, n_bins=50):
    """Estimate state space coverage using histogram-based density."""
    coverage_per_dim = []
    for dim in range(states.shape[1]):
        hist, bin_edges = np.histogram(states[:, dim], bins=n_bins)
        # Coverage = fraction of non-empty bins
        coverage = np.sum(hist > 0) / n_bins
        # Also compute entropy as measure of uniformity
        hist_normalized = hist / hist.sum()
        hist_normalized = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        max_entropy = np.log(n_bins)
        coverage_per_dim.append({
            'coverage': coverage,
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy,
        })
    return coverage_per_dim


def compute_action_coverage(actions, n_bins=50):
    """Estimate action space coverage."""
    coverage_per_dim = []
    for dim in range(actions.shape[1]):
        hist, _ = np.histogram(actions[:, dim], bins=n_bins, range=(-1, 1))
        coverage = np.sum(hist > 0) / n_bins
        hist_normalized = hist / hist.sum()
        hist_normalized = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        max_entropy = np.log(n_bins)
        coverage_per_dim.append({
            'coverage': coverage,
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy,
        })
    return coverage_per_dim


def analyze_temporal_patterns(data):
    """Analyze how states/actions evolve over time within trajectories."""
    states = data['states']
    actions = data['actions']
    traj_lengths = data['traj_lengths']
    
    # Normalize timestep to [0, 1] within each trajectory
    normalized_times = []
    idx = 0
    for length in traj_lengths:
        normalized_times.extend(np.linspace(0, 1, length).tolist())
        idx += length
    normalized_times = np.array(normalized_times)
    
    # Compute state/action statistics at different phases
    early_mask = normalized_times < 0.33
    mid_mask = (normalized_times >= 0.33) & (normalized_times < 0.66)
    late_mask = normalized_times >= 0.66
    
    temporal_stats = {}
    for phase, mask in [('early', early_mask), ('mid', mid_mask), ('late', late_mask)]:
        temporal_stats[phase] = {
            'state_mean': states[mask].mean(axis=0).tolist(),
            'state_std': states[mask].std(axis=0).tolist(),
            'action_mean': actions[mask].mean(axis=0).tolist(),
            'action_std': actions[mask].std(axis=0).tolist(),
        }
    
    return temporal_stats, normalized_times


def identify_rare_regions(states, actions, percentile=5):
    """Identify states/actions that are visited rarely (in the tail of distribution)."""
    state_low = np.percentile(states, percentile, axis=0)
    state_high = np.percentile(states, 100 - percentile, axis=0)
    action_low = np.percentile(actions, percentile, axis=0)
    action_high = np.percentile(actions, 100 - percentile, axis=0)
    
    return {
        'state_low_threshold': state_low.tolist(),
        'state_high_threshold': state_high.tolist(),
        'action_low_threshold': action_low.tolist(),
        'action_high_threshold': action_high.tolist(),
    }


def plot_state_distributions(states, output_dir):
    """Plot distribution of each state dimension."""
    n_dims = states.shape[1]
    n_cols = 6
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    
    # Transport state breakdown (59 dims total):
    # robot0_eef_pos: 3, robot0_eef_quat: 4, robot0_gripper_qpos: 2 = 9
    # robot1_eef_pos: 3, robot1_eef_quat: 4, robot1_gripper_qpos: 2 = 9  
    # object: 41 (various object states)
    
    dim_labels = []
    if states.shape[1] == 59:  # Transport specific
        dim_labels = (
            ['r0_eef_x', 'r0_eef_y', 'r0_eef_z'] +
            ['r0_quat_w', 'r0_quat_x', 'r0_quat_y', 'r0_quat_z'] +
            ['r0_grip_0', 'r0_grip_1'] +
            ['r1_eef_x', 'r1_eef_y', 'r1_eef_z'] +
            ['r1_quat_w', 'r1_quat_x', 'r1_quat_y', 'r1_quat_z'] +
            ['r1_grip_0', 'r1_grip_1'] +
            [f'obj_{i}' for i in range(41)]
        )
    
    for dim in range(n_dims):
        ax = axes[dim]
        ax.hist(states[:, dim], bins=50, density=True, alpha=0.7)
        label = dim_labels[dim] if dim < len(dim_labels) else f'dim_{dim}'
        ax.set_title(f'{label}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'state_distributions.png'), dpi=150)
    plt.close()


def plot_action_distributions(actions, output_dir):
    """Plot distribution of each action dimension."""
    n_dims = actions.shape[1]
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    
    # Transport action breakdown (14 dims total):
    # Each robot has 7 dims (6 for OSC_POSE + 1 gripper)
    dim_labels = (
        ['r0_dx', 'r0_dy', 'r0_dz', 'r0_drx', 'r0_dry', 'r0_drz', 'r0_grip'] +
        ['r1_dx', 'r1_dy', 'r1_dz', 'r1_drx', 'r1_dry', 'r1_drz', 'r1_grip']
    )
    
    for dim in range(n_dims):
        ax = axes[dim]
        ax.hist(actions[:, dim], bins=50, density=True, alpha=0.7, range=(-1.1, 1.1))
        label = dim_labels[dim] if dim < len(dim_labels) else f'dim_{dim}'
        ax.set_title(f'{label}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.axvline(x=-1, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distributions.png'), dpi=150)
    plt.close()


def plot_eef_positions(states, traj_lengths, output_dir):
    """Plot end-effector positions for both robots (Transport has 2 robots)."""
    if states.shape[1] != 59:
        print("Skipping EEF plot - not transport dataset")
        return
    
    # Extract EEF positions
    r0_eef = states[:, :3]  # robot0 x, y, z
    r1_eef = states[:, 9:12]  # robot1 x, y, z
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Robot 0 XY, XZ, YZ
    axes[0, 0].scatter(r0_eef[:, 0], r0_eef[:, 1], alpha=0.01, s=1)
    axes[0, 0].set_xlabel('X'); axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Robot 0 EEF: XY plane')
    
    axes[0, 1].scatter(r0_eef[:, 0], r0_eef[:, 2], alpha=0.01, s=1)
    axes[0, 1].set_xlabel('X'); axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('Robot 0 EEF: XZ plane')
    
    axes[0, 2].scatter(r0_eef[:, 1], r0_eef[:, 2], alpha=0.01, s=1)
    axes[0, 2].set_xlabel('Y'); axes[0, 2].set_ylabel('Z')
    axes[0, 2].set_title('Robot 0 EEF: YZ plane')
    
    # Robot 1 XY, XZ, YZ
    axes[1, 0].scatter(r1_eef[:, 0], r1_eef[:, 1], alpha=0.01, s=1)
    axes[1, 0].set_xlabel('X'); axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('Robot 1 EEF: XY plane')
    
    axes[1, 1].scatter(r1_eef[:, 0], r1_eef[:, 2], alpha=0.01, s=1)
    axes[1, 1].set_xlabel('X'); axes[1, 1].set_ylabel('Z')
    axes[1, 1].set_title('Robot 1 EEF: XZ plane')
    
    axes[1, 2].scatter(r1_eef[:, 1], r1_eef[:, 2], alpha=0.01, s=1)
    axes[1, 2].set_xlabel('Y'); axes[1, 2].set_ylabel('Z')
    axes[1, 2].set_title('Robot 1 EEF: YZ plane')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eef_positions.png'), dpi=150)
    plt.close()


def plot_trajectory_start_end(data, output_dir):
    """Plot start vs end positions for each trajectory to see trajectory diversity."""
    if data['states'].shape[1] != 59:
        return
    
    diversity = analyze_trajectory_diversity(data)
    start_states = diversity['start_states']
    end_states = diversity['end_states']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Robot 0
    axes[0, 0].scatter(start_states[:, 0], start_states[:, 1], alpha=0.5, s=10, c='blue', label='Start')
    axes[0, 0].scatter(end_states[:, 0], end_states[:, 1], alpha=0.5, s=10, c='red', label='End')
    axes[0, 0].set_xlabel('X'); axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Robot 0: Trajectory Start vs End (XY)')
    axes[0, 0].legend()
    
    axes[0, 1].scatter(start_states[:, 0], start_states[:, 2], alpha=0.5, s=10, c='blue')
    axes[0, 1].scatter(end_states[:, 0], end_states[:, 2], alpha=0.5, s=10, c='red')
    axes[0, 1].set_xlabel('X'); axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('Robot 0: Trajectory Start vs End (XZ)')
    
    axes[0, 2].scatter(start_states[:, 1], start_states[:, 2], alpha=0.5, s=10, c='blue')
    axes[0, 2].scatter(end_states[:, 1], end_states[:, 2], alpha=0.5, s=10, c='red')
    axes[0, 2].set_xlabel('Y'); axes[0, 2].set_ylabel('Z')
    axes[0, 2].set_title('Robot 0: Trajectory Start vs End (YZ)')
    
    # Robot 1
    axes[1, 0].scatter(start_states[:, 9], start_states[:, 10], alpha=0.5, s=10, c='blue', label='Start')
    axes[1, 0].scatter(end_states[:, 9], end_states[:, 10], alpha=0.5, s=10, c='red', label='End')
    axes[1, 0].set_xlabel('X'); axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('Robot 1: Trajectory Start vs End (XY)')
    axes[1, 0].legend()
    
    axes[1, 1].scatter(start_states[:, 9], start_states[:, 11], alpha=0.5, s=10, c='blue')
    axes[1, 1].scatter(end_states[:, 9], end_states[:, 11], alpha=0.5, s=10, c='red')
    axes[1, 1].set_xlabel('X'); axes[1, 1].set_ylabel('Z')
    axes[1, 1].set_title('Robot 1: Trajectory Start vs End (XZ)')
    
    axes[1, 2].scatter(start_states[:, 10], start_states[:, 11], alpha=0.5, s=10, c='blue')
    axes[1, 2].scatter(end_states[:, 10], end_states[:, 11], alpha=0.5, s=10, c='red')
    axes[1, 2].set_xlabel('Y'); axes[1, 2].set_ylabel('Z')
    axes[1, 2].set_title('Robot 1: Trajectory Start vs End (YZ)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_start_end.png'), dpi=150)
    plt.close()


def analyze_gripper_usage(data, output_dir):
    """Analyze gripper states and actions."""
    states = data['states']
    actions = data['actions']
    
    if states.shape[1] != 59 or actions.shape[1] != 14:
        return
    
    # Gripper states (qpos)
    r0_grip_state = states[:, 7:9]  # 2 dims
    r1_grip_state = states[:, 16:18]  # 2 dims
    
    # Gripper actions
    r0_grip_action = actions[:, 6]  # 1 dim
    r1_grip_action = actions[:, 13]  # 1 dim
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gripper states
    axes[0, 0].hist(r0_grip_state[:, 0], bins=50, alpha=0.7, label='qpos_0')
    axes[0, 0].hist(r0_grip_state[:, 1], bins=50, alpha=0.7, label='qpos_1')
    axes[0, 0].set_title('Robot 0 Gripper State')
    axes[0, 0].legend()
    
    axes[0, 1].hist(r1_grip_state[:, 0], bins=50, alpha=0.7, label='qpos_0')
    axes[0, 1].hist(r1_grip_state[:, 1], bins=50, alpha=0.7, label='qpos_1')
    axes[0, 1].set_title('Robot 1 Gripper State')
    axes[0, 1].legend()
    
    # Gripper actions
    axes[1, 0].hist(r0_grip_action, bins=50, alpha=0.7)
    axes[1, 0].set_title('Robot 0 Gripper Action')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    axes[1, 1].hist(r1_grip_action, bins=50, alpha=0.7)
    axes[1, 1].set_title('Robot 1 Gripper Action')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gripper_analysis.png'), dpi=150)
    plt.close()


def plot_temporal_evolution(data, output_dir):
    """Plot how EEF positions evolve through the trajectory phases."""
    if data['states'].shape[1] != 59:
        return
    
    states = data['states']
    traj_lengths = data['traj_lengths']
    
    # Collect states at different phases
    early_states = []
    mid_states = []
    late_states = []
    
    idx = 0
    for length in traj_lengths:
        traj_states = states[idx:idx + length]
        n = len(traj_states)
        early_states.append(traj_states[:n//3])
        mid_states.append(traj_states[n//3:2*n//3])
        late_states.append(traj_states[2*n//3:])
        idx += length
    
    early_states = np.vstack(early_states)
    mid_states = np.vstack(mid_states)
    late_states = np.vstack(late_states)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Robot 0 XY at different phases
    axes[0, 0].scatter(early_states[:, 0], early_states[:, 1], alpha=0.01, s=1, c='blue', label='Early')
    axes[0, 0].scatter(mid_states[:, 0], mid_states[:, 1], alpha=0.01, s=1, c='green', label='Mid')
    axes[0, 0].scatter(late_states[:, 0], late_states[:, 1], alpha=0.01, s=1, c='red', label='Late')
    axes[0, 0].set_xlabel('X'); axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Robot 0 EEF: XY by Phase')
    
    axes[0, 1].scatter(early_states[:, 0], early_states[:, 2], alpha=0.01, s=1, c='blue')
    axes[0, 1].scatter(mid_states[:, 0], mid_states[:, 2], alpha=0.01, s=1, c='green')
    axes[0, 1].scatter(late_states[:, 0], late_states[:, 2], alpha=0.01, s=1, c='red')
    axes[0, 1].set_xlabel('X'); axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('Robot 0 EEF: XZ by Phase')
    
    axes[0, 2].scatter(early_states[:, 1], early_states[:, 2], alpha=0.01, s=1, c='blue')
    axes[0, 2].scatter(mid_states[:, 1], mid_states[:, 2], alpha=0.01, s=1, c='green')
    axes[0, 2].scatter(late_states[:, 1], late_states[:, 2], alpha=0.01, s=1, c='red')
    axes[0, 2].set_xlabel('Y'); axes[0, 2].set_ylabel('Z')
    axes[0, 2].set_title('Robot 0 EEF: YZ by Phase')
    
    # Robot 1
    axes[1, 0].scatter(early_states[:, 9], early_states[:, 10], alpha=0.01, s=1, c='blue', label='Early')
    axes[1, 0].scatter(mid_states[:, 9], mid_states[:, 10], alpha=0.01, s=1, c='green', label='Mid')
    axes[1, 0].scatter(late_states[:, 9], late_states[:, 10], alpha=0.01, s=1, c='red', label='Late')
    axes[1, 0].set_xlabel('X'); axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('Robot 1 EEF: XY by Phase')
    
    axes[1, 1].scatter(early_states[:, 9], early_states[:, 11], alpha=0.01, s=1, c='blue')
    axes[1, 1].scatter(mid_states[:, 9], mid_states[:, 11], alpha=0.01, s=1, c='green')
    axes[1, 1].scatter(late_states[:, 9], late_states[:, 11], alpha=0.01, s=1, c='red')
    axes[1, 1].set_xlabel('X'); axes[1, 1].set_ylabel('Z')
    axes[1, 1].set_title('Robot 1 EEF: XZ by Phase')
    
    axes[1, 2].scatter(early_states[:, 10], early_states[:, 11], alpha=0.01, s=1, c='blue')
    axes[1, 2].scatter(mid_states[:, 10], mid_states[:, 11], alpha=0.01, s=1, c='green')
    axes[1, 2].scatter(late_states[:, 10], late_states[:, 11], alpha=0.01, s=1, c='red')
    axes[1, 2].set_xlabel('Y'); axes[1, 2].set_ylabel('Z')
    axes[1, 2].set_title('Robot 1 EEF: YZ by Phase')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Early (0-33%)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Mid (33-66%)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Late (66-100%)', markersize=10),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.dataset_path}")
    data = load_dataset(args.dataset_path)
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    stats = compute_basic_stats(data)
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"{key}: (list of {len(value)} values)")
        else:
            print(f"{key}: {value}")
    
    # Save stats to JSON
    with open(os.path.join(args.output_dir, 'basic_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Trajectory diversity
    print("\n=== Trajectory Diversity ===")
    diversity = analyze_trajectory_diversity(data)
    print(f"Start state std (mean across dims): {diversity['start_state_std'].mean():.4f}")
    print(f"End state std (mean across dims): {diversity['end_state_std'].mean():.4f}")
    print(f"Number of trajectories: {len(diversity['start_states'])}")
    
    # State coverage
    print("\n=== State Space Coverage ===")
    state_coverage = compute_state_coverage(data['states'])
    avg_coverage = np.mean([c['coverage'] for c in state_coverage])
    avg_entropy = np.mean([c['normalized_entropy'] for c in state_coverage])
    print(f"Average state coverage (fraction of bins): {avg_coverage:.4f}")
    print(f"Average normalized entropy: {avg_entropy:.4f}")
    
    # Find dimensions with lowest coverage
    coverages = [c['coverage'] for c in state_coverage]
    worst_dims = np.argsort(coverages)[:5]
    print(f"Dimensions with worst coverage: {worst_dims.tolist()}")
    for dim in worst_dims:
        print(f"  Dim {dim}: coverage={coverages[dim]:.3f}")
    
    # Action coverage
    print("\n=== Action Space Coverage ===")
    action_coverage = compute_action_coverage(data['actions'])
    avg_action_coverage = np.mean([c['coverage'] for c in action_coverage])
    avg_action_entropy = np.mean([c['normalized_entropy'] for c in action_coverage])
    print(f"Average action coverage (fraction of bins): {avg_action_coverage:.4f}")
    print(f"Average normalized entropy: {avg_action_entropy:.4f}")
    
    # Identify rare regions
    print("\n=== Rare Region Analysis ===")
    rare = identify_rare_regions(data['states'], data['actions'])
    with open(os.path.join(args.output_dir, 'rare_regions.json'), 'w') as f:
        json.dump(rare, f, indent=2)
    print("Saved rare region thresholds to rare_regions.json")
    
    # Temporal patterns
    print("\n=== Temporal Patterns ===")
    temporal_stats, _ = analyze_temporal_patterns(data)
    with open(os.path.join(args.output_dir, 'temporal_stats.json'), 'w') as f:
        json.dump(temporal_stats, f, indent=2)
    print("Saved temporal statistics to temporal_stats.json")
    
    # Generate plots
    print("\n=== Generating Plots ===")
    plot_state_distributions(data['states'], args.output_dir)
    print("Generated state_distributions.png")
    
    plot_action_distributions(data['actions'], args.output_dir)
    print("Generated action_distributions.png")
    
    plot_eef_positions(data['states'], data['traj_lengths'], args.output_dir)
    print("Generated eef_positions.png")
    
    plot_trajectory_start_end(data, args.output_dir)
    print("Generated trajectory_start_end.png")
    
    analyze_gripper_usage(data, args.output_dir)
    print("Generated gripper_analysis.png")
    
    plot_temporal_evolution(data, args.output_dir)
    print("Generated temporal_evolution.png")
    
    # Summary for hypothesis testing
    print("\n" + "="*60)
    print("SUMMARY FOR HYPOTHESIS TESTING")
    print("="*60)
    print(f"""
Your hypothesis: Transport's optimal policy visits states/actions that 
the initial dataset rarely covers, making IDQL conservative.

Key findings from this analysis:
1. Dataset has {stats['num_trajectories']} trajectories with {stats['total_transitions']} transitions
2. State coverage: {avg_coverage*100:.1f}% of bins occupied (higher = better coverage)
3. Action coverage: {avg_action_coverage*100:.1f}% of bins occupied
4. State entropy: {avg_entropy:.3f} (1.0 = uniform, lower = concentrated)
5. Action entropy: {avg_action_entropy:.3f}

To test your hypothesis further:
1. Check temporal_evolution.png - does the dataset cover diverse paths?
2. Check trajectory_start_end.png - are trajectories diverse or clustered?
3. Check action_distributions.png - if actions are concentrated in narrow ranges,
   the dataset may lack diversity for optimal behavior

Next step: Collect rollouts from DPPO and compare with this dataset
to see what regions DPPO visits that the dataset doesn't cover.
""")
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the train.npz dataset file")
    parser.add_argument("--output_dir", type=str, default="./transport_analysis",
                        help="Directory to save analysis outputs")
    args = parser.parse_args()
    
    main(args)

