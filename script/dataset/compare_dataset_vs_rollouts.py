"""
Compare dataset distribution with online rollouts from a trained policy.

This script collects rollouts from a trained policy (e.g., DPPO) and compares
the state/action distributions with the original training dataset.

This directly tests the hypothesis: "The optimal policy visits states/actions 
that the initial dataset rarely covers."

Usage:
    python script/dataset/compare_dataset_vs_rollouts.py \
        --dataset_path $DPPO_DATA_DIR/robomimic/transport/train.npz \
        --rollout_path ./dppo_rollouts.npz \
        --output_dir ./dataset_vs_rollouts_analysis

To collect rollouts, you can add code to save states/actions during evaluation
in the train_ppo_diffusion_agent.py or create a separate rollout collection script.

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from scipy import stats as scipy_stats


def load_dataset(path):
    """Load dataset from npz file."""
    data = np.load(path, allow_pickle=False)
    return {
        'states': data['states'],
        'actions': data['actions'],
    }


def compute_kl_divergence_estimate(p_samples, q_samples, n_bins=50, dim=None):
    """
    Estimate KL divergence D(P||Q) using histogram-based density estimation.
    p_samples: samples from P (e.g., successful policy)
    q_samples: samples from Q (e.g., dataset)
    
    Returns KL divergence for each dimension.
    """
    if dim is not None:
        p = p_samples[:, dim]
        q = q_samples[:, dim]
        
        # Compute shared bin edges
        all_data = np.concatenate([p, q])
        bin_edges = np.linspace(all_data.min() - 1e-6, all_data.max() + 1e-6, n_bins + 1)
        
        p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
        q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()
        
        kl = np.sum(p_hist * np.log(p_hist / q_hist))
        return kl
    else:
        kl_per_dim = []
        for d in range(p_samples.shape[1]):
            kl = compute_kl_divergence_estimate(p_samples, q_samples, n_bins, dim=d)
            kl_per_dim.append(kl)
        return np.array(kl_per_dim)


def compute_wasserstein_distance(p_samples, q_samples, dim=None):
    """Compute 1D Wasserstein distance for each dimension."""
    if dim is not None:
        return scipy_stats.wasserstein_distance(p_samples[:, dim], q_samples[:, dim])
    else:
        distances = []
        for d in range(p_samples.shape[1]):
            dist = scipy_stats.wasserstein_distance(p_samples[:, d], q_samples[:, d])
            distances.append(dist)
        return np.array(distances)


def compute_out_of_distribution_fraction(rollout_samples, dataset_samples, percentile=5):
    """
    Compute fraction of rollout samples that fall outside the dataset's distribution.
    Uses percentile-based thresholds.
    """
    low_thresh = np.percentile(dataset_samples, percentile, axis=0)
    high_thresh = np.percentile(dataset_samples, 100 - percentile, axis=0)
    
    # Count samples outside thresholds for each dimension
    below = rollout_samples < low_thresh
    above = rollout_samples > high_thresh
    ood = below | above
    
    # Per-dimension OOD fraction
    ood_fraction_per_dim = ood.mean(axis=0)
    
    # Overall OOD fraction (any dimension)
    ood_any_dim = ood.any(axis=1).mean()
    
    return {
        'per_dim': ood_fraction_per_dim,
        'any_dim': ood_any_dim,
        'low_thresh': low_thresh,
        'high_thresh': high_thresh,
    }


def plot_distribution_comparison(dataset, rollouts, output_dir, name='state', n_dims=None):
    """Plot side-by-side comparison of distributions."""
    if n_dims is None:
        n_dims = dataset.shape[1]
    
    n_cols = 6
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    
    for dim in range(n_dims):
        ax = axes[dim]
        
        # Compute shared bins
        all_data = np.concatenate([dataset[:, dim], rollouts[:, dim]])
        bins = np.linspace(all_data.min(), all_data.max(), 50)
        
        ax.hist(dataset[:, dim], bins=bins, density=True, alpha=0.5, label='Dataset', color='blue')
        ax.hist(rollouts[:, dim], bins=bins, density=True, alpha=0.5, label='Rollouts', color='orange')
        ax.set_title(f'Dim {dim}')
        ax.legend(fontsize=6)
    
    for idx in range(n_dims, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{name.capitalize()} Distribution: Dataset vs Rollouts', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_distribution_comparison.png'), dpi=150)
    plt.close()


def plot_kl_and_wasserstein(kl_divs, wasserstein_dists, output_dir, name='state'):
    """Plot KL divergence and Wasserstein distance per dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    dims = np.arange(len(kl_divs))
    
    axes[0].bar(dims, kl_divs, alpha=0.7)
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('KL Divergence')
    axes[0].set_title(f'{name.capitalize()} KL Divergence (Rollouts || Dataset)')
    axes[0].axhline(y=np.mean(kl_divs), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(kl_divs):.3f}')
    axes[0].legend()
    
    axes[1].bar(dims, wasserstein_dists, alpha=0.7)
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Wasserstein Distance')
    axes[1].set_title(f'{name.capitalize()} Wasserstein Distance')
    axes[1].axhline(y=np.mean(wasserstein_dists), color='r', linestyle='--',
                    label=f'Mean: {np.mean(wasserstein_dists):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_divergence_metrics.png'), dpi=150)
    plt.close()


def plot_ood_analysis(ood_state, ood_action, output_dir):
    """Plot OOD analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # State OOD
    dims = np.arange(len(ood_state['per_dim']))
    axes[0].bar(dims, ood_state['per_dim'] * 100, alpha=0.7)
    axes[0].set_xlabel('State Dimension')
    axes[0].set_ylabel('OOD Fraction (%)')
    axes[0].set_title(f'State: Fraction of Rollouts Outside Dataset Distribution\n'
                      f'(Overall: {ood_state["any_dim"]*100:.1f}% samples have at least one OOD dim)')
    axes[0].axhline(y=5, color='r', linestyle='--', label='5% threshold')
    axes[0].legend()
    
    # Action OOD
    dims = np.arange(len(ood_action['per_dim']))
    axes[1].bar(dims, ood_action['per_dim'] * 100, alpha=0.7)
    axes[1].set_xlabel('Action Dimension')
    axes[1].set_ylabel('OOD Fraction (%)')
    axes[1].set_title(f'Action: Fraction of Rollouts Outside Dataset Distribution\n'
                      f'(Overall: {ood_action["any_dim"]*100:.1f}% samples have at least one OOD dim)')
    axes[1].axhline(y=5, color='r', linestyle='--', label='5% threshold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_analysis.png'), dpi=150)
    plt.close()


def plot_eef_comparison(dataset_states, rollout_states, output_dir):
    """Compare EEF positions for transport task."""
    if dataset_states.shape[1] != 59:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Robot 0
    axes[0, 0].scatter(dataset_states[:, 0], dataset_states[:, 1], 
                       alpha=0.02, s=1, c='blue', label='Dataset')
    axes[0, 0].scatter(rollout_states[:, 0], rollout_states[:, 1], 
                       alpha=0.02, s=1, c='orange', label='Rollouts')
    axes[0, 0].set_xlabel('X'); axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_title('Robot 0 EEF: XY')
    axes[0, 0].legend()
    
    axes[0, 1].scatter(dataset_states[:, 0], dataset_states[:, 2], 
                       alpha=0.02, s=1, c='blue')
    axes[0, 1].scatter(rollout_states[:, 0], rollout_states[:, 2], 
                       alpha=0.02, s=1, c='orange')
    axes[0, 1].set_xlabel('X'); axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title('Robot 0 EEF: XZ')
    
    axes[0, 2].scatter(dataset_states[:, 1], dataset_states[:, 2], 
                       alpha=0.02, s=1, c='blue')
    axes[0, 2].scatter(rollout_states[:, 1], rollout_states[:, 2], 
                       alpha=0.02, s=1, c='orange')
    axes[0, 2].set_xlabel('Y'); axes[0, 2].set_ylabel('Z')
    axes[0, 2].set_title('Robot 0 EEF: YZ')
    
    # Robot 1
    axes[1, 0].scatter(dataset_states[:, 9], dataset_states[:, 10], 
                       alpha=0.02, s=1, c='blue', label='Dataset')
    axes[1, 0].scatter(rollout_states[:, 9], rollout_states[:, 10], 
                       alpha=0.02, s=1, c='orange', label='Rollouts')
    axes[1, 0].set_xlabel('X'); axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_title('Robot 1 EEF: XY')
    
    axes[1, 1].scatter(dataset_states[:, 9], dataset_states[:, 11], 
                       alpha=0.02, s=1, c='blue')
    axes[1, 1].scatter(rollout_states[:, 9], rollout_states[:, 11], 
                       alpha=0.02, s=1, c='orange')
    axes[1, 1].set_xlabel('X'); axes[1, 1].set_ylabel('Z')
    axes[1, 1].set_title('Robot 1 EEF: XZ')
    
    axes[1, 2].scatter(dataset_states[:, 10], dataset_states[:, 11], 
                       alpha=0.02, s=1, c='blue')
    axes[1, 2].scatter(rollout_states[:, 10], rollout_states[:, 11], 
                       alpha=0.02, s=1, c='orange')
    axes[1, 2].set_xlabel('Y'); axes[1, 2].set_ylabel('Z')
    axes[1, 2].set_title('Robot 1 EEF: YZ')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eef_comparison.png'), dpi=150)
    plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_dataset(args.dataset_path)
    
    print(f"Loading rollouts from {args.rollout_path}")
    rollouts = load_dataset(args.rollout_path)
    
    print(f"Dataset: {dataset['states'].shape[0]} samples")
    print(f"Rollouts: {rollouts['states'].shape[0]} samples")
    
    # Compute divergence metrics
    print("\n=== Computing KL Divergence (Rollouts || Dataset) ===")
    state_kl = compute_kl_divergence_estimate(rollouts['states'], dataset['states'])
    action_kl = compute_kl_divergence_estimate(rollouts['actions'], dataset['actions'])
    print(f"State KL (mean): {state_kl.mean():.4f}")
    print(f"Action KL (mean): {action_kl.mean():.4f}")
    
    print("\n=== Computing Wasserstein Distance ===")
    state_wasserstein = compute_wasserstein_distance(rollouts['states'], dataset['states'])
    action_wasserstein = compute_wasserstein_distance(rollouts['actions'], dataset['actions'])
    print(f"State Wasserstein (mean): {state_wasserstein.mean():.4f}")
    print(f"Action Wasserstein (mean): {action_wasserstein.mean():.4f}")
    
    print("\n=== Computing OOD Fraction ===")
    ood_state = compute_out_of_distribution_fraction(rollouts['states'], dataset['states'])
    ood_action = compute_out_of_distribution_fraction(rollouts['actions'], dataset['actions'])
    print(f"State: {ood_state['any_dim']*100:.1f}% of rollout samples have at least one OOD dimension")
    print(f"Action: {ood_action['any_dim']*100:.1f}% of rollout samples have at least one OOD dimension")
    
    # Find dimensions with highest divergence
    print("\n=== Top 5 State Dimensions with Highest KL Divergence ===")
    top_state_dims = np.argsort(state_kl)[::-1][:5]
    for dim in top_state_dims:
        print(f"  Dim {dim}: KL={state_kl[dim]:.4f}, OOD={ood_state['per_dim'][dim]*100:.1f}%")
    
    print("\n=== Top 5 Action Dimensions with Highest KL Divergence ===")
    top_action_dims = np.argsort(action_kl)[::-1][:5]
    for dim in top_action_dims:
        print(f"  Dim {dim}: KL={action_kl[dim]:.4f}, OOD={ood_action['per_dim'][dim]*100:.1f}%")
    
    # Save metrics
    metrics = {
        'state_kl_per_dim': state_kl.tolist(),
        'state_kl_mean': float(state_kl.mean()),
        'action_kl_per_dim': action_kl.tolist(),
        'action_kl_mean': float(action_kl.mean()),
        'state_wasserstein_per_dim': state_wasserstein.tolist(),
        'state_wasserstein_mean': float(state_wasserstein.mean()),
        'action_wasserstein_per_dim': action_wasserstein.tolist(),
        'action_wasserstein_mean': float(action_wasserstein.mean()),
        'state_ood_per_dim': ood_state['per_dim'].tolist(),
        'state_ood_any_dim': float(ood_state['any_dim']),
        'action_ood_per_dim': ood_action['per_dim'].tolist(),
        'action_ood_any_dim': float(ood_action['any_dim']),
    }
    
    with open(os.path.join(args.output_dir, 'divergence_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    print("\n=== Generating Plots ===")
    
    # Limit dimensions for readability
    n_state_dims = min(dataset['states'].shape[1], 24)
    n_action_dims = dataset['actions'].shape[1]
    
    plot_distribution_comparison(dataset['states'], rollouts['states'], 
                                 args.output_dir, name='state', n_dims=n_state_dims)
    print("Generated state_distribution_comparison.png")
    
    plot_distribution_comparison(dataset['actions'], rollouts['actions'], 
                                 args.output_dir, name='action', n_dims=n_action_dims)
    print("Generated action_distribution_comparison.png")
    
    plot_kl_and_wasserstein(state_kl, state_wasserstein, args.output_dir, name='state')
    print("Generated state_divergence_metrics.png")
    
    plot_kl_and_wasserstein(action_kl, action_wasserstein, args.output_dir, name='action')
    print("Generated action_divergence_metrics.png")
    
    plot_ood_analysis(ood_state, ood_action, args.output_dir)
    print("Generated ood_analysis.png")
    
    plot_eef_comparison(dataset['states'], rollouts['states'], args.output_dir)
    print("Generated eef_comparison.png")
    
    # Summary
    print("\n" + "="*60)
    print("HYPOTHESIS TEST SUMMARY")
    print("="*60)
    print(f"""
Your hypothesis: Transport's optimal policy visits states/actions 
that the initial dataset rarely covers.

Key findings:
1. State OOD: {ood_state['any_dim']*100:.1f}% of rollout samples visit states 
   outside the dataset's 5th-95th percentile range
2. Action OOD: {ood_action['any_dim']*100:.1f}% of rollout samples use actions
   outside the dataset's 5th-95th percentile range
3. Mean State KL Divergence: {state_kl.mean():.4f} (higher = more different)
4. Mean Action KL Divergence: {action_kl.mean():.4f}

Interpretation:
- If OOD fractions are HIGH (>10-20%), the optimal policy indeed visits
  regions the dataset doesn't cover well → supports your hypothesis
- If KL divergence is HIGH, the rollout distribution differs significantly
  from the dataset → IDQL would struggle because it learned to stay close
  to the dataset distribution

Check eef_comparison.png to visually see where the robot goes during 
successful rollouts vs. the original dataset coverage.
""")
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the original train.npz dataset")
    parser.add_argument("--rollout_path", type=str, required=True,
                        help="Path to rollout data (npz with 'states' and 'actions' keys)")
    parser.add_argument("--output_dir", type=str, default="./dataset_vs_rollouts",
                        help="Directory to save analysis outputs")
    args = parser.parse_args()
    
    main(args)

