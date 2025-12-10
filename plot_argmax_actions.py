#!/usr/bin/env python

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_npz(path: str):
    data = np.load(path)
    if "actions" not in data.files:
        raise ValueError(
            f"'actions' not found in {path}. "
            "Make sure rollout_argmax_traj saved actions."
        )
    actions = data["actions"]        # [T, n_env, H, A]
    q_entropy = data["q_entropy"] if "q_entropy" in data.files else None
    return actions, q_entropy


def plot_translation_3d(
    actions: np.ndarray,
    q_entropy=None,
    max_env: int = 4,
    title=None,
    out_path=None,
):
    """
    actions:   [T, n_env, H, A] with A = 14 (two 7-d arms).
               We use only translation dims:
                   arm 1: a[0:3]
                   arm 2: a[7:10]  (currently unused)
    q_entropy: [T, n_env] or None (unused here but left for future coloring).

    We interpret the translation actions as delta positions and
    compose them over time via cumulative sum to get trajectories.
    """

    T, n_env, H, A = actions.shape
    if H != 1:
        # assume first executed action is at index 0
        actions = actions[:, :, 0, :]  # [T, n_env, A]
    else:
        actions = actions[:, :, 0, :]

    if A < 10:
        raise ValueError(f"Expected at least 10 action dims (got {A}).")

    n_plot = min(n_env, max_env)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for env_idx in range(n_plot):
        traj = actions[:, env_idx, :]      # [T, A]

        # Drop NaN rows (e.g. post-reset garbage)
        mask = ~np.any(np.isnan(traj), axis=-1)
        traj = traj[mask]
        if traj.shape[0] < 2:
            continue

        # First gripper translation deltas (dims 0:3)
        arm1_delta = traj[:, 0:3]    # [T, 3]

        # Compose into positions by cumulative sum.
        # Start at origin (0,0,0); if you have an actual initial pose,
        # just add it to arm1_pos afterwards.
        arm1_pos = np.cumsum(arm1_delta, axis=0)  # [T, 3]

        # Plot as a trajectory line
        ax.plot(
            arm1_pos[:, 0],
            arm1_pos[:, 1],
            arm1_pos[:, 2],
            alpha=0.8,
            linewidth=1.5,
        )

        # If you also want arm 2, uncomment this:
        arm2_delta = traj[:, 7:10]
        arm2_pos = np.cumsum(arm2_delta, axis=0)
        ax.plot(
            arm2_pos[:, 0],
            arm2_pos[:, 1],
            arm2_pos[:, 2],
            alpha=0.8,
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_zlabel("z position")
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="3D plot of composed argmax-Q translation trajectories (two grippers)."
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to argmax_traj_itrXXXXXX.npz",
    )
    parser.add_argument(
        "--max-env",
        type=int,
        default=4,
        help="Max number of env trajectories to overlay.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output image path (.png, .pdf, ...).",
    )
    args = parser.parse_args()

    actions, q_entropy = load_npz(args.npz)
    title = os.path.basename(args.npz).replace(".npz", "")

    plot_translation_3d(
        actions=actions,
        q_entropy=q_entropy,
        max_env=args.max_env,
        title=title,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
