#!/usr/bin/env python3
"""Per-frame analysis of v5j (LSEHead) val predictions.

Operates on per-frame logits saved by `precompute_losses.py` (extended on
2026-04-25 to dump them). Produces structural diagnostics about how the
LSEHead-pooled model fails:

- LSE-implicit attention entropy: how peaked vs diffuse is the per-frame
  weighting on the GT class? (Compares across triage categories.)
- Per-frame top-1 wandering: how often does the model's top-1 prediction
  change across frames? (Stable wrong vs unstable wandering.)
- Peak-frame agreement: where in the 5s window does the GT-class logit
  peak? (Random spread = no localization; clustered at a frame = the
  model has learned a positional bias.)
- Bimodal-species deep-dive: same species across `correct_confident` vs
  `mixed_uncertainty` — does the head behave differently?

Goal: surface model-side structural patterns analogous to what Salman
described as "investigating attention values on highest/lowest loss
examples." LSEHead doesn't learn attention weights, but the per-frame
logit distribution reveals what the model "thinks" at each frame, which
is the LSEHead analogue.

Usage:
    cd user/murilo/analysis
    python per_frame_analysis.py \\
        --triage data/triage_v5j.parquet \\
        --perframe data/losses_v5j_perframe_logits.npy \\
        --output-dir data/perframe/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def shannon_entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy in nats."""
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def compute_per_sample_metrics(
    triage: pd.DataFrame,
    perframe: np.ndarray,
    species_list: list[str],
    lse_temperature: float = 1.0,
) -> pd.DataFrame:
    """Compute per-sample LSE-attention diagnostics.

    Args:
        triage: DataFrame with one row per val sample (must have `idx`,
            `primary_label`, `top1_species`, `triage_category`).
        perframe: (N, n_classes, T) per-frame logits.
        species_list: list mapping class index → species string.
        lse_temperature: α used in LSE pooling. v5j uses 1.0.

    Returns:
        triage augmented with per-frame columns:
            - `gt_attn_entropy`: entropy of LSE-implicit attention on GT class (nats)
            - `gt_attn_max`: max attention weight on any one frame (peakedness)
            - `gt_peak_frame`: index of the frame with highest GT-class logit
            - `gt_logit_std`: std of GT-class logits across frames (variability)
            - `top1_wander`: number of distinct frame-level top-1 species
            - `top1_majority_frac`: fraction of frames where top-1 == sample's
              clipwise-top1 (high = stable, low = wandering)
    """
    N, C, T = perframe.shape
    species_to_idx = {s: i for i, s in enumerate(species_list)}

    # Allocate output columns
    cols = {
        "gt_attn_entropy": np.full(N, np.nan, dtype=np.float32),
        "gt_attn_max": np.full(N, np.nan, dtype=np.float32),
        "gt_peak_frame": np.full(N, -1, dtype=np.int32),
        "gt_logit_std": np.full(N, np.nan, dtype=np.float32),
        "top1_wander": np.full(N, -1, dtype=np.int32),
        "top1_majority_frac": np.full(N, np.nan, dtype=np.float32),
    }

    # Cast perframe to float32 for math (was stored as float16)
    perframe_f32 = perframe.astype(np.float32)

    # Per-sample loop (vectorizable but readability matters more here for now)
    for row in triage.itertuples(index=False):
        i = int(row.idx)
        primary = row.primary_label
        if primary not in species_to_idx:
            continue
        g = species_to_idx[primary]

        z_g_t = perframe_f32[i, g, :]  # (T,)
        attn_g = softmax(z_g_t * lse_temperature)
        cols["gt_attn_entropy"][i] = float(shannon_entropy(attn_g))
        cols["gt_attn_max"][i] = float(attn_g.max())
        cols["gt_peak_frame"][i] = int(z_g_t.argmax())
        cols["gt_logit_std"][i] = float(z_g_t.std())

        # Top-1 per frame (across all classes)
        per_frame_top1 = perframe_f32[i, :, :].argmax(axis=0)  # (T,)
        unique_top1, counts = np.unique(per_frame_top1, return_counts=True)
        cols["top1_wander"][i] = int(len(unique_top1))

        # Fraction of frames where top-1 matches the sample's clipwise top-1
        clipwise_top1 = row.top1_species
        if clipwise_top1 in species_to_idx:
            c_top = species_to_idx[clipwise_top1]
            cols["top1_majority_frac"][i] = float((per_frame_top1 == c_top).mean())

    out = triage.copy()
    for k, v in cols.items():
        out[k] = v
    return out


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-frame stats by triage_category."""
    agg = df.groupby("triage_category").agg(
        n=("idx", "size"),
        mean_loss=("loss", "mean"),
        mean_top1_prob=("top1_prob", "mean"),
        mean_gt_prob=("gt_prob", "mean"),
        mean_gt_attn_entropy=("gt_attn_entropy", "mean"),
        median_gt_attn_entropy=("gt_attn_entropy", "median"),
        mean_gt_attn_max=("gt_attn_max", "mean"),
        mean_gt_logit_std=("gt_logit_std", "mean"),
        mean_top1_wander=("top1_wander", "mean"),
        mean_top1_majority_frac=("top1_majority_frac", "mean"),
    )
    return agg.round(3)


def bimodal_species_deep_dive(
    df: pd.DataFrame,
    target_species: list[str],
) -> pd.DataFrame:
    """For each target species, compare per-frame stats between
    correct_confident and mixed_uncertainty crops.

    These should be the same species' calls under different SNR conditions
    (per the 04-25 audio finding). If the head behaves differently across
    them, the failure is head-level. If identically, the failure is upstream.
    """
    rows = []
    for sp in target_species:
        sub = df[df["primary_label"] == sp]
        if sub.empty:
            continue
        for cat in ("correct_confident", "correct_uncertain", "mixed_uncertainty"):
            cell = sub[sub["triage_category"] == cat]
            if cell.empty:
                continue
            rows.append(
                {
                    "species": sp,
                    "category": cat,
                    "n": len(cell),
                    "mean_top1_prob": round(cell["top1_prob"].mean(), 3),
                    "mean_gt_prob": round(cell["gt_prob"].mean(), 3),
                    "mean_attn_entropy": round(cell["gt_attn_entropy"].mean(), 3),
                    "mean_attn_max": round(cell["gt_attn_max"].mean(), 3),
                    "mean_logit_std": round(cell["gt_logit_std"].mean(), 3),
                    "mean_top1_wander": round(cell["top1_wander"].mean(), 2),
                    "mean_top1_majority_frac": round(cell["top1_majority_frac"].mean(), 3),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Per-frame analysis of v5j")
    parser.add_argument(
        "--triage",
        type=str,
        default="data/triage_v5j.parquet",
        help="Path to triage parquet (with certainty columns).",
    )
    parser.add_argument(
        "--perframe",
        type=str,
        default="data/losses_v5j_perframe_logits.npy",
        help="Path to per-frame logits (N, C, T) saved by precompute_losses.py.",
    )
    parser.add_argument(
        "--species-list-source",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv",
        help="Path to taxonomy.csv to build species list.",
    )
    parser.add_argument(
        "--lse-temperature",
        type=float,
        default=1.0,
        help="LSE alpha used by the model; matches config lse_temperature.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/perframe/",
        help="Directory for outputs (parquet + summary text).",
    )
    args = parser.parse_args()

    triage_path = Path(args.triage)
    perframe_path = Path(args.perframe)
    output_dir = Path(args.output_dir)

    if not triage_path.exists():
        raise SystemExit(f"Triage parquet not found: {triage_path}")
    if not perframe_path.exists():
        raise SystemExit(
            f"Per-frame logits not found: {perframe_path}.\n"
            f"Run precompute_losses.py first (which now also dumps "
            f"_perframe_logits.npy)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading triage from {triage_path}")
    triage = pd.read_parquet(triage_path)

    logger.info(f"Loading per-frame logits from {perframe_path}")
    perframe = np.load(perframe_path)
    logger.info(f"Per-frame logits shape: {perframe.shape}")

    # Build species list from triage parquet (ensures consistent ordering)
    # We re-derive from taxonomy.csv since that's how training was set up.
    taxonomy = pd.read_csv(args.species_list_source)
    species_list = sorted(taxonomy["primary_label"].astype(str).unique().tolist())
    logger.info(f"Species list: {len(species_list)} species")

    # Cross-check: classes axis of perframe should match
    if perframe.shape[1] != len(species_list):
        logger.warning(
            f"Per-frame n_classes ({perframe.shape[1]}) != taxonomy species "
            f"({len(species_list)}). Continuing anyway, but check alignment."
        )

    logger.info("Computing per-sample metrics...")
    out = compute_per_sample_metrics(
        triage, perframe, species_list, lse_temperature=args.lse_temperature
    )

    # Save enriched parquet
    enriched_path = output_dir / "triage_v5j_perframe.parquet"
    out.to_parquet(enriched_path, index=False)
    logger.info(f"Saved enriched parquet to {enriched_path}")

    # Category-level summary
    summary = summarize_by_category(out)
    print("\n=== Per-frame metrics by triage category ===")
    print(summary.to_string())
    summary_path = output_dir / "perframe_summary_by_category.csv"
    summary.to_csv(summary_path)
    logger.info(f"Saved category summary to {summary_path}")

    # Bimodal-species deep dive (the 4 species we already inspected via audio)
    bimodal_targets = ["roahaw", "coffal1", "socfly1", "compot1", "whtdov", "barant1"]
    deep = bimodal_species_deep_dive(out, bimodal_targets)
    if not deep.empty:
        print("\n=== Bimodal-species deep dive ===")
        print(deep.to_string(index=False))
        deep_path = output_dir / "perframe_bimodal_species_deepdive.csv"
        deep.to_csv(deep_path, index=False)
        logger.info(f"Saved bimodal deep-dive to {deep_path}")

    # Reference: theoretical max entropy for uniform distribution over T frames
    T = perframe.shape[2]
    uniform_entropy = float(np.log(T))
    print(
        f"\nReference: T={T} frames; uniform-attention entropy = "
        f"log({T}) = {uniform_entropy:.3f} nats. "
        f"A perfectly peaked attention has entropy 0."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
