"""Plot functions for SED model attribution analysis.

Pure functions extracted from model_explorer.ipynb. Each takes AttributionResult(s)
and a Matplotlib axis/figure, writes to disk, and closes the figure.

Used by both the notebook (interactive) and export_figures.py (bulk export).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from attribution import AttributionResult


def plot_four_panel(
    result: AttributionResult,
    species_list: list[str],
    sample_meta: dict,
    out_path: str | Path,
    top_k_species: int = 10,
) -> None:
    """Save the 4-panel figure: mel, saliency, top-10 species, per-frame analysis.

    Args:
        result: AttributionResult for the GT species.
        species_list: Full ordered species list (len 234).
        sample_meta: Dict with keys idx, primary_label, source, loss, gt_rank.
        out_path: PNG destination.
        top_k_species: Bars to show in top-k panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: Mel spectrogram
    ax = axes[0, 0]
    ax.imshow(result.mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Mel Spectrogram")
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time frame")

    # Panel 2: Gradient saliency overlay
    ax = axes[0, 1]
    ax.imshow(result.mel, aspect="auto", origin="lower", cmap="gray", alpha=0.4)
    im = ax.imshow(
        result.gradient_saliency,
        aspect="auto",
        origin="lower",
        cmap="hot",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"Gradient Saliency: {result.species_name}")
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time frame")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Top-k species bar chart
    ax = axes[1, 0]
    top_indices = np.argsort(result.clipwise_probs)[::-1][:top_k_species]
    top_names = [species_list[i] for i in top_indices]
    top_probs = [result.clipwise_probs[i] for i in top_indices]
    colors = [
        "#d32f2f" if species_list[i] == result.species_name else "#1976d2" for i in top_indices
    ]
    bars = ax.barh(range(top_k_species), top_probs, color=colors)
    ax.set_yticks(range(top_k_species))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel("Probability")
    ax.set_title(f"Top-{top_k_species} Predicted Species")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    for bar, prob in zip(bars, top_probs, strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}",
            va="center",
            fontsize=7,
        )
    gt_patch = mpatches.Patch(color="#d32f2f", label="Ground truth")
    pred_patch = mpatches.Patch(color="#1976d2", label="Other predicted")
    ax.legend(handles=[gt_patch, pred_patch], fontsize=8, loc="lower right")

    # Panel 4: Per-frame analysis
    ax = axes[1, 1]
    n_frames = result.frame_probs.shape[1]
    frame_x = np.arange(n_frames)
    target_frame_probs = result.frame_probs[result.species_idx]
    ax.bar(
        frame_x - 0.15,
        target_frame_probs,
        width=0.3,
        color="#d32f2f",
        alpha=0.8,
        label=f"Frame prob: {result.species_name}",
    )
    ax.bar(
        frame_x + 0.15,
        result.effective_attention,
        width=0.3,
        color="#1976d2",
        alpha=0.8,
        label="LSE attention",
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.set_title(f"Per-Frame Analysis: {result.species_name}")
    ax.legend(fontsize=8)
    ax.set_xticks(frame_x)
    ax.set_xticklabels([f"F{i}" for i in frame_x])

    plt.suptitle(
        f"Sample idx={sample_meta['idx']} | {sample_meta['primary_label']} "
        f"({sample_meta['source']}) | loss={sample_meta['loss']:.4f} | "
        f"gt_rank={sample_meta['gt_rank']}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_competitor_comparison(
    gt_result: AttributionResult,
    competitor_result: AttributionResult,
    sample_meta: dict,
    out_path: str | Path,
    competitor_role: str,
) -> None:
    """Save side-by-side saliency comparison of GT species vs a competitor.

    Args:
        gt_result: Attribution for the ground truth species.
        competitor_result: Attribution for the competitor (top-1 or top-2).
        sample_meta: Dict with keys idx, loss, primary_label, source.
        out_path: PNG destination.
        competitor_role: "top1" or "top2" — affects title only.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    ax.imshow(gt_result.mel, aspect="auto", origin="lower", cmap="gray", alpha=0.4)
    ax.imshow(
        gt_result.gradient_saliency,
        aspect="auto",
        origin="lower",
        cmap="hot",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    gt_prob = gt_result.clipwise_probs[gt_result.species_idx]
    ax.set_title(f"GT: {gt_result.species_name} (prob={gt_prob:.3f})")
    ax.set_ylabel("Mel bin")
    ax.set_xlabel("Time frame")

    ax = axes[1]
    ax.imshow(competitor_result.mel, aspect="auto", origin="lower", cmap="gray", alpha=0.4)
    ax.imshow(
        competitor_result.gradient_saliency,
        aspect="auto",
        origin="lower",
        cmap="hot",
        alpha=0.7,
        vmin=0,
        vmax=1,
    )
    comp_prob = competitor_result.clipwise_probs[competitor_result.species_idx]
    role_label = "Top-1" if competitor_role == "top1" else "Top-2"
    ax.set_title(f"{role_label}: {competitor_result.species_name} (prob={comp_prob:.3f})")
    ax.set_xlabel("Time frame")

    plt.suptitle(
        f"GT vs {role_label} Saliency | idx={sample_meta['idx']} | "
        f"loss={sample_meta['loss']:.4f} | source={sample_meta['source']}",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_batch_grid(
    results: list[AttributionResult],
    sample_metas: list[dict],
    title: str,
    out_path: str | Path,
) -> None:
    """Save a 2×N grid of mel + saliency for a batch of samples.

    Args:
        results: AttributionResult per sample (len N, N typically 5).
        sample_metas: Sample metadata dicts matching results.
        title: Suptitle.
        out_path: PNG destination.
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (res, meta) in enumerate(zip(results, sample_metas, strict=False)):
        axes[0, col].imshow(res.mel, aspect="auto", origin="lower", cmap="magma")
        axes[0, col].set_title(f"{meta['primary_label']}\nloss={meta['loss']:.3f}", fontsize=9)
        if col == 0:
            axes[0, col].set_ylabel("Mel")

        axes[1, col].imshow(res.mel, aspect="auto", origin="lower", cmap="gray", alpha=0.4)
        axes[1, col].imshow(
            res.gradient_saliency,
            aspect="auto",
            origin="lower",
            cmap="hot",
            alpha=0.7,
            vmin=0,
            vmax=1,
        )
        axes[1, col].set_title(
            f"gt_rank={meta['gt_rank']}, top1={meta['top1_species']}", fontsize=8
        )
        if col == 0:
            axes[1, col].set_ylabel("Saliency")

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
