#!/usr/bin/env python3
"""Plot statistics from teacher label outputs.

Generates:
- Event density histogram (BamBird ROIs per file)
- Species frequency bar chart (top N species per model)
- Confidence distribution (BirdNET vs Perch)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bambird_density(stats: dict, ax: plt.Axes):
    """Plot BamBird ROI density across sources."""
    sources = []
    means = []
    medians = []
    for src, s in stats.get("bambird", {}).items():
        sources.append(src.replace("train_", ""))
        means.append(s["rois_per_file_mean"])
        medians.append(s["rois_per_file_median"])

    x = np.arange(len(sources))
    ax.bar(x - 0.15, means, 0.3, label="Mean", alpha=0.8)
    ax.bar(x + 0.15, medians, 0.3, label="Median", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel("ROIs per file")
    ax.set_title("BamBird: ROI Density")
    ax.legend()


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

    plot_bambird_density(stats, axes[0, 0])
    plot_top_species(stats, "birdnet", axes[0, 1])
    plot_top_species(stats, "perch", axes[1, 0])
    plot_confidence_comparison(stats, axes[1, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
