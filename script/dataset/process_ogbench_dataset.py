"""
Download and convert OGBench singletask datasets into the DPPO NPZ format.

Outputs:
    {save_dir}/train.npz   — states, actions, rewards, terminals, masks, traj_lengths
    {save_dir}/normalization.npz — obs_min, obs_max, action_min, action_max

Usage:
    python script/dataset/process_ogbench_dataset.py \
        --env_name cube-triple-play-singletask-task2-v0 \
        --save_dir data/ogbench/cube-triple

    python script/dataset/process_ogbench_dataset.py \
        --env_name scene-play-singletask-task3-v0 \
        --save_dir data/ogbench/scene

The --normalize flag (default True) maps obs/actions to [-1, 1] and writes
normalization.npz with the original ranges.
"""

import argparse
import os
import ssl
import numpy as np


def load_and_relabel(env_name, dataset_dir=None):
    """Use OGBench API to download dataset and relabel with rewards."""
    import ogbench
    from ogbench.relabel_utils import relabel_dataset
    import gymnasium

    splits = env_name.split("-")
    pos = splits.index("singletask")
    # gymnasium env name: remove the dataset type
    gym_env_name = "-".join(splits[:pos - 1] + splits[pos:])
    # dataset name: remove singletask and taskN words
    dataset_name = "-".join(splits[:pos] + splits[-1:])

    env = gymnasium.make(gym_env_name)

    if dataset_dir is None:
        dataset_dir = os.path.expanduser("~/.ogbench/data")

    # Download raw dataset via ogbench (with SSL fallback for HPC clusters)
    try:
        ogbench.download_datasets([dataset_name], dataset_dir)
    except Exception as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e) or "SSL" in str(e):
            print(f"SSL error during download: {e}")
            print("Retrying with SSL verification disabled...")
            _orig_ctx = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                ogbench.download_datasets([dataset_name], dataset_dir)
            finally:
                ssl._create_default_https_context = _orig_ctx
        else:
            raise

    train_path = os.path.join(dataset_dir, f"{dataset_name}.npz")
    val_path = os.path.join(dataset_dir, f"{dataset_name}-val.npz")

    datasets = {}
    for split, path in [("train", train_path), ("val", val_path)]:
        raw = np.load(path)
        ds = {
            "observations": raw["observations"].astype(np.float32),
            "actions": raw["actions"].astype(np.float32),
            "terminals": raw["terminals"].astype(np.float32),
        }
        # Add qpos/qvel/button_states needed by relabel_dataset
        for k in ["qpos", "qvel", "button_states"]:
            if k in raw:
                ds[k] = raw[k]

        # Build next_observations from the sequential layout
        ob_mask = (1.0 - ds["terminals"]).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        ds["next_observations"] = ds["observations"][next_ob_mask]
        ds["observations"] = ds["observations"][ob_mask]
        ds["actions"] = ds["actions"][ob_mask]
        new_terminals = np.concatenate([ds["terminals"][1:], [1.0]])
        ds["terminals"] = new_terminals[ob_mask].astype(np.float32)
        for k in ["qpos", "qvel", "button_states"]:
            if k in ds:
                ds[k] = ds[k][ob_mask]

        relabel_dataset(gym_env_name, env, ds)
        datasets[split] = ds

    env.close()
    return datasets


def convert_to_dppo_format(dataset):
    """
    Convert an OGBench dataset (with next_observations, rewards, terminals,
    masks) into the stitched DPPO format (states, actions, rewards, terminals,
    masks, traj_lengths).
    """
    obs = dataset["observations"]
    acts = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]
    # masks come from OGBench relabeling: 0 where the task is fully solved.
    # They are the correct TD bootstrap signal for critic pretraining;
    # `terminals` only mark the ends of the 1000-step play trajectories
    # (data-collection truncations, NOT task termination).
    masks = dataset["masks"]

    # Reconstruct trajectory boundaries from terminal flags
    terminal_indices = np.where(terminals > 0.5)[0]

    all_states = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_masks = []
    traj_lengths = []

    prev_end = 0
    for term_idx in terminal_indices:
        # Include the terminal step in the trajectory
        traj_len = term_idx - prev_end + 1
        traj_lengths.append(traj_len)
        all_states.append(obs[prev_end : term_idx + 1])
        all_actions.append(acts[prev_end : term_idx + 1])
        all_rewards.append(rewards[prev_end : term_idx + 1])
        all_terminals.append(terminals[prev_end : term_idx + 1])
        all_masks.append(masks[prev_end : term_idx + 1])
        prev_end = term_idx + 1

    # Handle trailing data without a terminal flag
    if prev_end < len(obs):
        traj_len = len(obs) - prev_end
        traj_lengths.append(traj_len)
        all_states.append(obs[prev_end:])
        all_actions.append(acts[prev_end:])
        all_rewards.append(rewards[prev_end:])
        all_terminals.append(terminals[prev_end:])
        all_masks.append(masks[prev_end:])

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards_cat = np.concatenate(all_rewards, axis=0)
    terminals_cat = np.concatenate(all_terminals, axis=0)
    masks_cat = np.concatenate(all_masks, axis=0)
    traj_lengths = np.array(traj_lengths, dtype=np.int64)

    return states, actions, rewards_cat, terminals_cat, masks_cat, traj_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True,
                        help="Full singletask env name, e.g. cube-triple-play-singletask-task2-v0")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory, e.g. data/ogbench/cube-triple")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize obs/actions to [-1, 1]")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Directory to cache raw OGBench datasets (default: ~/.ogbench/data)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading and relabeling dataset for {args.env_name} ...")
    datasets = load_and_relabel(args.env_name, dataset_dir=args.dataset_dir)

    for split in ["train", "val"]:
        if split not in datasets:
            continue
        ds = datasets[split]
        states, actions, rewards, terminals, masks, traj_lengths = convert_to_dppo_format(ds)

        print(f"\n--- {split} split ---")
        print(f"  Total transitions: {len(states)}")
        print(f"  Total trajectories: {len(traj_lengths)}")
        print(f"  Traj length mean: {traj_lengths.mean():.1f}")
        print(f"  Traj length std: {traj_lengths.std():.1f}")
        print(f"  Traj length min: {traj_lengths.min()}")
        print(f"  Traj length max: {traj_lengths.max()}")
        print(f"  States shape: {states.shape}")
        print(f"  Actions shape: {actions.shape}")
        print(f"  Obs range: [{states.min():.4f}, {states.max():.4f}]")
        print(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
        print(f"  Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        print(f"  Non-zero rewards: {(rewards != 0).sum()}")
        print(f"  Success steps (mask==0): {(masks < 0.5).sum()}")

        if split == "train":
            # Compute normalization stats from training data only
            obs_min = states.min(axis=0).astype(np.float32)
            obs_max = states.max(axis=0).astype(np.float32)
            action_min = actions.min(axis=0).astype(np.float32)
            action_max = actions.max(axis=0).astype(np.float32)

        if args.normalize:
            states = 2 * (
                (states - obs_min) / (obs_max - obs_min + 1e-6) - 0.5
            )
            actions = 2 * (
                (actions - action_min) / (action_max - action_min + 1e-6) - 0.5
            )

        out_path = os.path.join(args.save_dir, f"{split}.npz")
        np.savez(
            out_path,
            states=states.astype(np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            terminals=terminals.astype(np.float32),
            masks=masks.astype(np.float32),
            traj_lengths=traj_lengths,
        )
        print(f"  Saved to {out_path}")

    # Save normalization stats
    norm_path = os.path.join(args.save_dir, "normalization.npz")
    np.savez(
        norm_path,
        obs_min=obs_min,
        obs_max=obs_max,
        action_min=action_min,
        action_max=action_max,
    )
    print(f"\nSaved normalization to {norm_path}")


if __name__ == "__main__":
    main()
