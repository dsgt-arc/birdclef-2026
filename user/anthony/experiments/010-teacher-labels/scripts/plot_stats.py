#!/usr/bin/env python3
"""Plot statistics from teacher label outputs.

Generates:
- BamBird cluster quality (signal vs noise)
- Species frequency bar chart (top N species per model)
- Confidence distribution (BirdNET vs Perch)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bambird_cluster_quality(stats: dict, ax: plt.Axes):
    """Plot BamBird cluster quality stats."""
    bambird = stats.get("bambird", {})

    cluster = bambird.get("cluster", {})
    if not cluster:
        ax.set_title("BamBird: No cluster data")
        return

    signal = cluster.get("total_signal_rois", 0)
    noise = cluster.get("total_noise_rois", 0)
    total = signal + noise

    labels = ["Signal", "Noise"]
    values = [signal, noise]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    ax.set_ylabel("Number of ROIs")
    ax.set_title(f"BamBird: Cluster Quality ({total} ROIs)")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=10)

    # Add signal ratio annotation
    ratio = cluster.get("signal_ratio", 0)
    ax.annotate(
        f"Signal ratio: {ratio:.1%}\n"
        f"Files all-noise: {cluster.get('files_all_noise', 0)}\n"
        f"Species: {cluster.get('num_species', 0)}",
        xy=(0.95, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )


def plot_top_species(stats: dict, model: str, ax: plt.Axes, n: int = 15):
    """Plot top N species by mean confidence for a model."""
    # Combine focal + soundscape
    all_species = {}
    for src_stats in stats.get(model, {}).values():
        for entry in src_stats.get("top_20_species", []):
            sp = entry["species"]
            conf = entry["mean_conf"]
            all_species[sp] = max(all_species.get(sp, 0), conf)

    top = sorted(all_species.items(), key=lambda x: -x[1])[:n]
    if not top:
        ax.set_title(f"{model}: No data")
        return

    species = [s for s, _ in top]
    confs = [c for _, c in top]

    ax.barh(range(len(species)), confs, alpha=0.8)
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species, fontsize=7)
    ax.set_xlabel("Mean confidence")
    ax.set_title(f"{model}: Top {n} Species")
    ax.invert_yaxis()


def plot_confidence_comparison(stats: dict, ax: plt.Axes):
    """Compare confidence distributions between models."""
    models = []
    means = []
    medians = []
    p95s = []
    for model in ["birdnet", "perch"]:
        for src, s in stats.get(model, {}).items():
            label = f"{model}/{src.replace('train_', '')}"
            models.append(label)
            means.append(s.get("max_conf_mean", 0))
            medians.append(s.get("max_conf_median", 0))
            p95s.append(s.get("max_conf_p95", 0))

    x = np.arange(len(models))
    w = 0.25
    ax.bar(x - w, means, w, label="Mean", alpha=0.8)
    ax.bar(x, medians, w, label="Median", alpha=0.8)
    ax.bar(x + w, p95s, w, label="P95", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Max confidence score")
    ax.set_title("Model Confidence Distribution")
    ax.legend()


def main():
    stats_path = sys.argv[1] if len(sys.argv) > 1 else "stats.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "teacher_stats.png"

    with open(stats_path) as f:
        stats = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Teacher Label Statistics", fontsize=14)

    plot_bambird_cluster_quality(stats, axes[0, 0])
    plot_top_species(stats, "birdnet", axes[0, 1])
    plot_top_species(stats, "perch", axes[1, 0])
    plot_confidence_comparison(stats, axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
