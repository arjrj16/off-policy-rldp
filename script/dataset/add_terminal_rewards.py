"""
Add terminal-only rewards to a preprocessed dataset.

Creates rewards and terminals arrays based on traj_lengths:
    rewards[t] = reward_value at the final step of each trajectory, else 0
    terminals[t] = True at the final step of each trajectory, else False

Existing keys are preserved, while rewards / terminals are overwritten.

Note: The Q-learning dataset should read rewards/dones from the end of each
action chunk (reward_mode="end") to consume these terminal-only signals.
"""

import argparse
import numpy as np


def build_terminal_signals(traj_lengths, reward_value):
    total_steps = int(np.sum(traj_lengths))
    rewards = np.zeros(total_steps, dtype=np.float32)
    terminals = np.zeros(total_steps, dtype=np.float32)

    offset = 0
    for length in traj_lengths:
        if length <= 0:
            raise ValueError(f"Invalid trajectory length: {length}")
        last = offset + length - 1
        rewards[last] = reward_value
        terminals[last] = 1.0
        offset += length
    return rewards, terminals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--reward_value", type=float, default=1.0)
    args = parser.parse_args()

    data = np.load(args.input_path, allow_pickle=False)
    if "traj_lengths" not in data:
        raise KeyError("Expected key 'traj_lengths' in dataset")
    traj_lengths = data["traj_lengths"]

    total_steps = int(np.sum(traj_lengths))
    for key in ("states", "actions"):
        if key in data and data[key].shape[0] != total_steps:
            raise ValueError(
                f"Key '{key}' has length {data[key].shape[0]}, expected {total_steps}"
            )

    rewards, terminals = build_terminal_signals(traj_lengths, args.reward_value)

    output = {
        key: data[key]
        for key in data.files
        if key not in ("rewards", "terminals")
    }
    output["rewards"] = rewards
    output["terminals"] = terminals

    np.savez_compressed(args.output_path, **output)
    print(f"Wrote dataset with terminal rewards to {args.output_path}")


if __name__ == "__main__":
    main()
