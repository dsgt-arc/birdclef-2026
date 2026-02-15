#!/usr/bin/env python3
"""Plot pretraining metrics from TensorBoard event files.

Usage:
    python scripts/plot_pretrain.py [--log-dirs DIR1 DIR2 ...] [--output artifacts/plots/pretrain_curves.png]

Reads TensorBoard event files and generates matplotlib plots of:
  - train/loss (total loss per epoch)
  - train/loss_mpg (reconstruction loss per epoch)
  - train/monitor_mse and train/monitor_l1 (cross-metrics per epoch)
  - train/grad_norm (per step)
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_tfevents(log_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Read scalar summaries from TensorBoard event files.

    Returns dict mapping tag -> list of (step, value) tuples.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def epoch_metrics(scalars: dict, tag: str, use_index: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Extract epoch-level metrics.

    If use_index=True, x-axis is sequential epoch index (0, 1, 2, ...)
    rather than the global step number that TensorBoard records.
    """
    if tag not in scalars:
        return np.array([]), np.array([])
    data = scalars[tag]
    values = np.array([v for s, v in data])
    if use_index:
        steps = np.arange(len(values))
    else:
        steps = np.array([s for s, v in data])
    return steps, values


def plot_pretrain_curves(
    runs: dict[str, dict],
    output_path: str,
):
    """Generate a multi-panel plot of pretraining metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SSAMBA Pretraining Curves (MPG)", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    # Panel 1: Total loss (epoch)
    ax = axes[0, 0]
    for i, (name, scalars) in enumerate(runs.items()):
        steps, vals = epoch_metrics(scalars, "train/loss_epoch")
        if len(steps) == 0:
            steps, vals = epoch_metrics(scalars, "train/loss")
        if len(steps) > 0:
            ax.plot(steps, vals, color=colors[i % len(colors)], label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: MPG reconstruction loss (epoch)
    ax = axes[0, 1]
    for i, (name, scalars) in enumerate(runs.items()):
        steps, vals = epoch_metrics(scalars, "train/loss_mpg_epoch")
        if len(steps) == 0:
            steps, vals = epoch_metrics(scalars, "train/loss_mpg")
        if len(steps) > 0:
            ax.plot(steps, vals, color=colors[i % len(colors)], label=name, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("MPG Reconstruction Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Cross-metrics (epoch) — MSE and L1 on same axis
    ax = axes[1, 0]
    for i, (name, scalars) in enumerate(runs.items()):
        steps_mse, vals_mse = epoch_metrics(scalars, "train/monitor_mse")
        steps_l1, vals_l1 = epoch_metrics(scalars, "train/monitor_l1")
        if len(steps_mse) > 0:
            ax.plot(steps_mse, vals_mse, color=colors[i % len(colors)],
                    linestyle="-", label=f"{name} (MSE)", linewidth=1.5)
        if len(steps_l1) > 0:
            ax.plot(steps_l1, vals_l1, color=colors[i % len(colors)],
                    linestyle="--", label=f"{name} (L1)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Cross-Metrics (MSE & L1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Gradient norm (per step, convert to epochs)
    ax = axes[1, 1]
    for i, (name, scalars) in enumerate(runs.items()):
        if "train/grad_norm" in scalars:
            data = scalars["train/grad_norm"]
            steps = np.array([s for s, v in data])
            vals = np.array([v for s, v in data])
            # Estimate steps_per_epoch from epoch tag
            if "epoch" in scalars:
                epoch_data = scalars["epoch"]
                # Find step where epoch transitions
                spe_estimates = []
                for j in range(1, len(epoch_data)):
                    s0, e0 = epoch_data[j-1]
                    s1, e1 = epoch_data[j]
                    if e1 > e0 and (s1 - s0) > 0:
                        spe_estimates.append(int((s1 - s0) / (e1 - e0)))
                steps_per_epoch = int(np.median(spe_estimates)) if spe_estimates else 911
            else:
                steps_per_epoch = 911
            epochs = steps / steps_per_epoch
            # Subsample if too many points
            if len(epochs) > 2000:
                idx = np.linspace(0, len(epochs) - 1, 2000, dtype=int)
                epochs, vals = epochs[idx], vals[idx]
            ax.plot(epochs, vals, color=colors[i % len(colors)], label=name,
                    linewidth=0.5, alpha=0.7)
            # Rolling average
            if len(vals) > 50:
                window = min(100, len(vals) // 5)
                rolling = np.convolve(vals, np.ones(window)/window, mode="valid")
                ax.plot(epochs[window-1:], rolling, color=colors[i % len(colors)],
                        linewidth=2, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm (with rolling avg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)

    # Print summary stats
    for name, scalars in runs.items():
        print(f"\n=== {name} ===")
        for tag in ["train/loss_epoch", "train/loss_mpg_epoch",
                     "train/monitor_mse", "train/monitor_l1"]:
            steps, vals = epoch_metrics(scalars, tag)
            if len(vals) > 0:
                print(f"  {tag}: {len(vals)} epochs, "
                      f"first={vals[0]:.4f}, last={vals[-1]:.4f}, "
                      f"min={vals.min():.4f} @ epoch {int(steps[vals.argmin()])}")


def main():
    parser = argparse.ArgumentParser(description="Plot pretraining curves from TensorBoard logs")
    parser.add_argument(
        "--log-dirs", nargs="+",
        default=["logs/pretrain-full/lightning_logs/run"],
        help="TensorBoard log directories to plot",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels for each run (defaults to directory names)",
    )
    parser.add_argument(
        "--output", default="artifacts/plots/pretrain_curves.png",
        help="Output image path",
    )
    args = parser.parse_args()

    labels = args.labels or [Path(d).parent.parent.name for d in args.log_dirs]
    if len(labels) != len(args.log_dirs):
        labels = [f"run_{i}" for i in range(len(args.log_dirs))]

    runs = {}
    for label, log_dir in zip(labels, args.log_dirs):
        print(f"Reading {log_dir} ...")
        scalars = read_tfevents(log_dir)
        print(f"  Found {len(scalars)} scalar tags: {sorted(scalars.keys())}")
        runs[label] = scalars

    plot_pretrain_curves(runs, args.output)


if __name__ == "__main__":
    main()
