#!/usr/bin/env python3
"""
Compare weights between two PyTorch checkpoints and produce visualizations.

Checkpoints are expected to be .pt files containing at least a "model" key
(state_dict). Diffusion checkpoints may also have an "ema" key.

Usage:
  python script/compare_checkpoints.py path1.pt path2.pt [--out-dir OUT] [--ema2]
  # Second path can use env vars, e.g. $DPPO_LOG_DIR/.../state_8000.pt
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import torch

# Use non-interactive backend so script works in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_state_dict_from_checkpoint(path: str, state_dict_key: str = "model"):
    """Load a state dict from a checkpoint file. Resolves env vars in path."""
    path = os.path.expandvars(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if state_dict_key not in data:
        raise KeyError(
            f"Key '{state_dict_key}' not in checkpoint. Keys: {list(data.keys())}"
        )
    return data[state_dict_key], path


def normalize_key_for_match(key: str, prefixes: tuple = ("actor.", "network.")):
    """Strip common prefixes so we can match actor.* and network.* to same param."""
    for p in prefixes:
        if key.startswith(p):
            return key[len(p) :]
    return key


def find_matching_params(sd1, sd2, match_by_normalized_name: bool = False):
    """
    Find params that exist in both state dicts with the same shape.
    Returns list of (name1, name2) where name1 in sd1, name2 in sd2, same shape.
    """
    if match_by_normalized_name:
        # Build suffix -> [(name1, t1), ...]; then for each n2 find unique n1 with same suffix and shape
        by_suffix = defaultdict(list)
        for n1, t1 in sd1.items():
            suf = normalize_key_for_match(n1)
            by_suffix[suf].append((n1, t1))
        used_n1 = set()
        for n2, t2 in sd2.items():
            suf = normalize_key_for_match(n2)
            for (n1, t1) in by_suffix.get(suf, []):
                if n1 not in used_n1 and t1.shape == t2.shape:
                    used_n1.add(n1)
                    yield (n1, n2)
                    break
        return
    # Exact name match
    for n1 in sd1:
        if n1 in sd2 and sd1[n1].shape == sd2[n1].shape:
            yield (n1, n1)


def compute_diff_stats(t1: torch.Tensor, t2: torch.Tensor):
    """Compute L2 diff, cosine sim, mean/max abs diff (flattened)."""
    a, b = t1.double().flatten(), t2.double().flatten()
    l2_a = a.norm(2).item()
    l2_b = b.norm(2).item()
    l2_diff = (a - b).norm(2).item()
    mean_abs_diff = (a - b).abs().mean().item()
    max_abs_diff = (a - b).abs().max().item()
    cos_sim = (
        (a @ b) / (l2_a * l2_b + 1e-12)
        if (l2_a > 1e-12 and l2_b > 1e-12)
        else float("nan")
    )
    return {
        "l2_diff": l2_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "cosine_sim": cos_sim,
        "numel": a.numel(),
    }


def run_comparison(sd1, sd2, match_by_normalized_name: bool = False):
    """Compare two state dicts; return matching pairs and per-tensor stats."""
    pairs = list(find_matching_params(sd1, sd2, match_by_normalized_name))
    results = []
    for name1, name2 in pairs:
        stats = compute_diff_stats(sd1[name1], sd2[name2])
        stats["name"] = name1 if name1 == name2 else f"{name1} <-> {name2}"
        stats["name1"] = name1
        stats["name2"] = name2
        stats["shape"] = tuple(sd1[name1].shape)
        results.append(stats)
    return results


def plot_layer_metrics(results, out_path: str):
    """Bar chart of per-layer mean abs diff and cosine similarity."""
    names = [r["name"] for r in results]
    # Shorten for x-axis
    short = [re.sub(r"^(actor|network)\.", "", n) for n in names]
    mean_diffs = [r["mean_abs_diff"] for r in results]
    cos_sims = [r["cosine_sim"] if not np.isnan(r["cosine_sim"]) else 0 for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(short) * 0.35), 6), sharex=True)
    x = np.arange(len(short))
    w = 0.35
    ax1.bar(x - w / 2, mean_diffs, width=w, color="steelblue", label="Mean |Δ|")
    ax1.set_ylabel("Mean absolute difference")
    ax1.set_title("Per-layer weight difference")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x + w / 2, cos_sims, width=w, color="coral", label="Cosine sim")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_xlabel("Parameter")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=45, ha="right")
    ax2.set_ylim(-1.05, 1.05)
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer metrics to {out_path}")


def plot_weight_histograms(sd1, sd2, pairs, out_path: str, max_layers: int = 6):
    """Overlaid histograms of weight values for first few matching layers."""
    n_plot = min(max_layers, len(pairs))
    fig, axes = plt.subplots(2, (n_plot + 1) // 2, figsize=(4 * ((n_plot + 1) // 2), 6))
    axes = np.atleast_2d(axes)
    for i, (name1, name2) in enumerate(pairs[:n_plot]):
        ax = axes.flat[i]
        w1 = sd1[name1].flatten().numpy()
        w2 = sd2[name2].flatten().numpy()
        ax.hist(w1, bins=50, alpha=0.6, label="Checkpoint 1", color="steelblue", density=True)
        ax.hist(w2, bins=50, alpha=0.6, label="Checkpoint 2", color="coral", density=True)
        short = re.sub(r"^(actor|network)\.", "", name1)
        ax.set_title(short[:40] + ("..." if len(short) > 40 else ""))
        ax.set_xlabel("Weight value")
        ax.legend(fontsize=8)
    for j in range(i + 1, axes.size):
        axes.flat[j].set_visible(False)
    plt.suptitle("Weight distribution comparison (first layers)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved weight histograms to {out_path}")


def plot_diff_heatmaps(sd1, sd2, pairs, out_path: str, max_mats: int = 4):
    """Heatmaps of absolute difference for 2D weight matrices."""
    two_d = [(n1, n2) for n1, n2 in pairs if sd1[n1].dim() >= 2]
    n_plot = min(max_mats, len(two_d))
    if n_plot == 0:
        return
    fig, axes = plt.subplots(1, n_plot, figsize=(4 * n_plot, 4))
    if n_plot == 1:
        axes = [axes]
    for i, (name1, name2) in enumerate(two_d[:n_plot]):
        diff = (sd1[name1] - sd2[name2]).abs()
        if diff.dim() > 2:
            diff = diff.flatten(0, -3).mean(0)
        im = axes[i].imshow(diff.numpy(), aspect="auto", cmap="hot")
        short = re.sub(r"^(actor|network)\.", "", name1)
        axes[i].set_title(short[:36] + ("..." if len(short) > 36 else ""))
        plt.colorbar(im, ax=axes[i], label="|Δ|")
    plt.suptitle("Absolute weight difference (2D view)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved diff heatmaps to {out_path}")


def plot_global_summary(results, path1: str, path2: str, out_path: str):
    """Summary text and small overview plot."""
    if not results:
        return
    mean_diffs = [r["mean_abs_diff"] for r in results]
    cos_sims = [r["cosine_sim"] for r in results]
    cos_sims = [c for c in cos_sims if not np.isnan(c)]
    total_params = sum(r["numel"] for r in results)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(mean_diffs, cos_sims, alpha=0.7, s=20)
    ax.set_xlabel("Mean absolute difference")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Layers (n={len(results)}, params={total_params:,})")
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare weights between two PyTorch checkpoints and visualize."
    )
    parser.add_argument(
        "checkpoint1",
        nargs="?",
        default="/home/melwani/67920/code2/off-policy-rldp/log/robomimic-pretrain/transport/transport_pre_diffusion_mlp_ta8_td20/2024-07-08_11-18-59/checkpoint/state_8000.pt",
        help="Path to first checkpoint (.pt)",
    )
    parser.add_argument(
        "checkpoint2",
        nargs="?",
        default="/home/melwani/67920/code2/off-policy-rldp/log/robomimic-pretrain/transport_idql_critic_pretrain_mlp_ta8/2026-01-29_12-13-05_42/checkpoint/state_500.pt",
        help="Path to second checkpoint (.pt); may use $VAR/...",
    )
    parser.add_argument(
        "--state-dict-key1",
        default="model",
        help="Key for state dict in first checkpoint (default: model)",
    )
    parser.add_argument(
        "--state-dict-key2",
        default="model",
        help="Key for state dict in second checkpoint (default: model). Use 'ema' for diffusion EMA.",
    )
    parser.add_argument(
        "--match-normalized",
        action="store_true",
        help="Match parameters by normalized name (actor.* <-> network.*) for different archs",
    )
    parser.add_argument(
        "--out-dir",
        default="./compare_out",
        help="Directory for output plots (default: current dir)",
    )
    parser.add_argument(
        "--prefix",
        default="compare",
        help="Filename prefix for outputs (default: compare)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    sd1, resolved1 = load_state_dict_from_checkpoint(
        args.checkpoint1, args.state_dict_key1
    )
    sd2, resolved2 = load_state_dict_from_checkpoint(
        args.checkpoint2, args.state_dict_key2
    )

    print(f"Checkpoint 1: {resolved1}")
    print(f"  Keys: {len(sd1)}")
    print(f"Checkpoint 2: {resolved2}")
    print(f"  Keys: {len(sd2)}")

    results = run_comparison(sd1, sd2, match_by_normalized_name=args.match_normalized)
    pairs = [(r["name1"], r["name2"]) for r in results]

    if not results:
        print("No matching parameters (same name and shape). Try --match-normalized if comparing actor vs network.")
        return

    print(f"\nMatching parameters: {len(results)}")
    total_diff = sum(r["l2_diff"] ** 2 for r in results) ** 0.5
    mean_cos = np.nanmean([r["cosine_sim"] for r in results])
    print(f"  Total L2 difference: {total_diff:.6f}")
    print(f"  Mean cosine similarity: {mean_cos:.6f}")
    print("\nPer-layer summary (first 10):")
    for r in results[:10]:
        print(f"  {r['name'][:50]:50} shape={r['shape']} mean|Δ|={r['mean_abs_diff']:.6f} cos={r['cosine_sim']:.4f}")

    prefix = args.prefix
    plot_layer_metrics(results, os.path.join(out_dir, f"{prefix}_layer_metrics.png"))
    plot_weight_histograms(sd1, sd2, pairs, os.path.join(out_dir, f"{prefix}_histograms.png"))
    plot_diff_heatmaps(sd1, sd2, pairs, os.path.join(out_dir, f"{prefix}_heatmaps.png"))
    plot_global_summary(
        results, resolved1, resolved2, os.path.join(out_dir, f"{prefix}_summary.png")
    )


if __name__ == "__main__":
    main()
