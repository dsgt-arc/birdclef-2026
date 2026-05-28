#!/usr/bin/env python3
"""Generate pseudo-labels for iterative pseudo-labeling (exp 025).

Config-driven pipeline that produces versioned pseudo-label parquets.
Reads a YAML config (e.g., configs/v1.yaml) and generates:
  - SED pseudo-labels (Tiers A+B): max-across-folds → temperature → power transform
  - Perch pseudo-labels (Tier C): per-species z-normalization → sigmoid → power transform
  - Merged pseudo-labels: one parquet with all tiers combined

Usage (from experiment directory):
    cd user/murilo/experiments/025-iter-pseudo-labeling
    python generate_pseudo_labels.py --config configs/v1.yaml
    python generate_pseudo_labels.py --config configs/v1.yaml --stats-only

Output directory: shared/pseudo-labels/birdclef-2026/v{N}/
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ============================================================================
# Paths
# ============================================================================

PROJECT = Path("/storage/project/ps-clef2026_birdclef-0")
SHARED = PROJECT / "shared"
DATA = SHARED / "data" / "birdclef-2026"
REPO = PROJECT / "mgustineli3" / "birdclef-2026"

META_COLS = ["filename", "source", "chunk_idx", "start_sec", "end_sec"]


# ============================================================================
# Math utilities
# ============================================================================


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def to_logit(p, eps=1e-7):
    """Convert probability to logit (inverse sigmoid)."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)


def temperature_scale(probs, temperature):
    """Apply temperature scaling: prob → logit → logit/T → sigmoid."""
    logits = to_logit(probs)
    return sigmoid(logits / temperature)


def power_transform(probs, exponent):
    """Apply power transform: p_new = p ^ exponent."""
    return np.power(np.clip(probs, 0, 1), exponent)


# ============================================================================
# Tier assignment
# ============================================================================


def assign_tiers(species_cols, sed_max_probs, perch_raw_probs, unlabeled_mask, cfg):
    """Assign each species to a tier based on model coverage.

    Returns dict: {species_label: tier} where tier is 'A', 'B', 'C', or 'D'.
    """
    # Load competition metadata
    sc_labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")

    # Seen species (in labeled soundscapes)
    seen = set()
    for x in sc_labels["primary_label"]:
        if pd.notna(x):
            seen.update(t.strip() for t in str(x).split(";") if t.strip())

    # Tier D from config
    tier_d = {str(s) for s in cfg["tier_d_species"]}

    # SED coverage on unlabeled segments
    sed_unlabeled = sed_max_probs[unlabeled_mask]
    sed_max_per_species = sed_unlabeled.max(axis=0)

    # Tier assignment
    tiers = {}
    for i, sp in enumerate(species_cols):
        if sp in seen:
            tiers[sp] = "A"
        elif sp in tier_d:
            tiers[sp] = "D"
        elif sed_max_per_species[i] > 0.5:
            tiers[sp] = "B"
        else:
            tiers[sp] = "C"

    return tiers


# ============================================================================
# SED pseudo-labels (Tiers A+B)
# ============================================================================


def generate_sed_labels(meta_df, species_cols, unlabeled_mask, tiers, cfg):
    """Generate SED pseudo-labels for Tier A+B species."""
    sed_cfg = cfg["sed"]
    source_dir = SHARED / sed_cfg["source_dir"].replace("shared/", "")
    n_folds = sed_cfg["n_folds"]
    temperature = sed_cfg["temperature"]
    power = sed_cfg["power_transform"]

    print("\n--- SED Pseudo-Labels (Tiers A+B) ---")
    print(f"  Source: {source_dir}")
    print(f"  Strategy: max-across-{n_folds}-folds → T={temperature} → P^{power}")

    # Load folds and compute max-across-folds
    fold_probs = []
    for fold in range(n_folds):
        logits_df = pd.read_parquet(source_dir / f"fold{fold}" / "logits.parquet")
        logits_df = logits_df.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)
        sp_cols = [c for c in logits_df.columns if c not in META_COLS]
        fold_probs.append(sigmoid(logits_df[sp_cols].values.astype(np.float32)))
        print(f"  Loaded fold {fold}: {fold_probs[-1].shape}")

    sed_max = np.maximum.reduce(fold_probs)
    del fold_probs
    print(f"  Max-across-folds: {sed_max.shape}")

    # Apply temperature scaling
    sed_scaled = temperature_scale(sed_max, temperature)

    # Apply power transform
    sed_final = power_transform(sed_scaled, power)

    # Zero out non-Tier-A/B species
    tier_ab_mask = np.array([tiers.get(sp) in ("A", "B") for sp in species_cols])
    sed_final[:, ~tier_ab_mask] = 0.0

    # Stats
    n_ab = tier_ab_mask.sum()
    sed_unlabeled = sed_final[unlabeled_mask]
    above_05 = (sed_unlabeled > 0.5).sum()
    species_covered = (sed_unlabeled.max(axis=0) > 0.5).sum()
    print(f"  Tier A+B species: {n_ab}")
    print(f"  Pseudo-labels >0.5 (unlabeled): {above_05:,}")
    print(f"  Species with coverage >0.5: {species_covered}/{n_ab}")

    return sed_final, sed_max


# ============================================================================
# Perch pseudo-labels (Tier C)
# ============================================================================


def generate_perch_labels(meta_df, species_cols, unlabeled_mask, tiers, cfg):
    """Generate Perch pseudo-labels for Tier C species using z-normalization."""
    perch_cfg = cfg["perch"]
    source_path = SHARED / perch_cfg["source_path"].replace("shared/", "")
    z_threshold = perch_cfg["z_threshold"]
    min_std = perch_cfg["min_std"]
    power = perch_cfg["power_transform"]

    print("\n--- Perch Pseudo-Labels (Tier C) ---")
    print(f"  Source: {source_path}")
    print(f"  Strategy: per-species z-normalization → z>{z_threshold} → sigmoid → P^{power}")

    # Load Perch raw probs
    perch_df = pd.read_parquet(source_path)
    perch_df = perch_df.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)
    perch_meta_cols = ["filename", "chunk_idx", "site", "hour_utc"]
    perch_sp_cols = [c for c in perch_df.columns if c not in perch_meta_cols]
    perch_raw = perch_df[perch_sp_cols].values.astype(np.float32)

    # Verify alignment
    assert len(perch_raw) == len(
        meta_df
    ), f"Row mismatch: Perch={len(perch_raw)}, SED={len(meta_df)}"
    assert (perch_df["filename"].values == meta_df["filename"].values).all(), "Filename mismatch"

    # Identify Tier C species
    tier_c_species = [sp for sp in species_cols if tiers.get(sp) == "C"]
    tier_c_idx = [species_cols.index(sp) for sp in tier_c_species]
    print(f"  Tier C species: {len(tier_c_species)}")

    # Z-normalize per species on ALL segments (including labeled, for stable stats)
    perch_labels = np.zeros_like(perch_raw)
    skipped = []
    per_species_stats = []

    for sp, sp_idx in zip(tier_c_species, tier_c_idx, strict=False):
        col = perch_raw[:, sp_idx]
        mean = col.mean()
        std = col.std()

        if std < min_std:
            skipped.append((sp, mean, std))
            continue

        z = (col - mean) / std
        # Keep only segments above z_threshold
        mask = z > z_threshold
        n_kept = mask.sum()

        # Convert z-scores to soft labels via sigmoid, then power transform
        soft = sigmoid(z)
        soft[~mask] = 0.0  # zero out below-threshold segments
        soft = power_transform(soft, power)

        perch_labels[:, sp_idx] = soft

        per_species_stats.append(
            {
                "species": sp,
                "mean": float(mean),
                "std": float(std),
                "z_threshold": z_threshold,
                "n_above_threshold": int(n_kept),
                "n_unlabeled_above": int(mask[unlabeled_mask].sum()),
                "max_soft_label": float(soft.max()),
            }
        )

    if skipped:
        print(f"  Skipped {len(skipped)} species (std < {min_std}):")
        for sp, m, s in skipped:
            print(f"    {sp}: mean={m:.4f}, std={s:.6f}")

    # Stats
    perch_unlabeled = perch_labels[unlabeled_mask]
    above_05 = (perch_unlabeled > 0.5).sum()
    species_covered = (perch_unlabeled.max(axis=0) > 0.5).sum()
    active = len(per_species_stats)
    print(f"  Active Tier C species (not skipped): {active}")
    print(f"  Pseudo-labels >0.5 (unlabeled): {above_05:,}")
    print(f"  Species with coverage >0.5: {species_covered}")

    if per_species_stats:
        segs = [s["n_unlabeled_above"] for s in per_species_stats]
        print(
            f"  Segments per species: min={min(segs)}, median={int(np.median(segs))}, max={max(segs)}"
        )

    return perch_labels, per_species_stats, skipped


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from config")
    parser.add_argument(
        "--config", required=True, help="Path to config YAML (e.g., configs/v1.yaml)"
    )
    parser.add_argument("--stats-only", action="store_true", help="Print stats without saving")
    args = parser.parse_args()

    t0 = time.time()

    # Load config — resolve relative to script directory (experiment dir)
    SCRIPT_DIR = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to cwd first, then relative to script directory
        if not config_path.exists():
            config_path = SCRIPT_DIR / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    version = cfg["version"]
    output_dir = SHARED / cfg["output_dir"].replace("shared/", "")
    # Pure-SED configs (e.g., exp 033 pl_gen_v2.yaml) omit `perch:` to disable
    # Tier C fallback. Tier C species end up with 0 labels (effectively Tier D).
    use_perch = "perch" in cfg
    print(f"=== Generating pseudo-labels: {version} ===")
    print(f"  Config: {config_path}")
    print(f"  Output: {output_dir}")
    print(f"  Perch tier C: {'enabled' if use_perch else 'DISABLED (pure-SED)'}")

    # Load competition metadata
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    species_cols = sample_sub.columns[1:].tolist()

    # Load SED fold0 metadata for segment alignment (all folds have same order)
    sed_source = SHARED / cfg["sed"]["source_dir"].replace("shared/", "")
    meta_df = pd.read_parquet(
        sed_source / "fold0" / "logits.parquet",
        columns=["filename", "chunk_idx"],
    )
    meta_df = meta_df.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)

    # Labeled file mask
    sc_labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    labeled_files = set(sc_labels["filename"].unique())
    unlabeled_mask = ~meta_df["filename"].isin(labeled_files).values

    print(f"  Total segments: {len(meta_df):,}")
    print(f"  Unlabeled: {unlabeled_mask.sum():,}")
    print(f"  Labeled: {(~unlabeled_mask).sum():,}")
    print(f"  Species: {len(species_cols)}")

    # --- Generate SED labels ---
    sed_labels, sed_max_raw = generate_sed_labels(meta_df, species_cols, unlabeled_mask, {}, cfg)

    # We need sed_max_raw (before temp/power) for tier assignment
    # Recompute tiers now that we have sed_max_raw
    if use_perch:
        # Load Perch for tier assignment
        perch_source = SHARED / cfg["perch"]["source_path"].replace("shared/", "")
        perch_df = pd.read_parquet(perch_source)
        perch_df = perch_df.sort_values(["filename", "chunk_idx"]).reset_index(drop=True)
        perch_meta_cols = ["filename", "chunk_idx", "site", "hour_utc"]
        perch_sp = [c for c in perch_df.columns if c not in perch_meta_cols]
        perch_raw = perch_df[perch_sp].values.astype(np.float32)
        tiers = assign_tiers(species_cols, sed_max_raw, perch_raw, unlabeled_mask, cfg)
    else:
        # Pure-SED: assign_tiers doesn't actually read perch_raw_probs (only uses
        # sed_max_probs + sc_labels + tier_d), so passing None is safe.
        tiers = assign_tiers(species_cols, sed_max_raw, None, unlabeled_mask, cfg)

    # Print tier summary
    tier_counts = {}
    for t in tiers.values():
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print("\n--- Tier Assignment ---")
    for t in ["A", "B", "C", "D"]:
        print(f"  Tier {t}: {tier_counts.get(t, 0)} species")

    # Re-generate SED labels with correct tier mask
    sed_labels, _ = generate_sed_labels(meta_df, species_cols, unlabeled_mask, tiers, cfg)

    # --- Generate Perch labels (optional — pure-SED configs skip this) ---
    if use_perch:
        del perch_df, perch_raw
        perch_labels, perch_stats, perch_skipped = generate_perch_labels(
            meta_df, species_cols, unlabeled_mask, tiers, cfg
        )
    else:
        # Tier C species get 0 (effectively Tier D). Merging logic below
        # writes perch_labels[:, i] for tier C — all zeros is a no-op since
        # `merged` is pre-zeroed.
        perch_labels = np.zeros((len(meta_df), len(species_cols)), dtype=np.float32)
        perch_stats = []
        perch_skipped = []
        print("\n--- Perch Pseudo-Labels: SKIPPED (pure-SED mode) ---")

    # --- Merge ---
    print("\n--- Merging ---")
    merged = np.zeros((len(meta_df), len(species_cols)), dtype=np.float32)

    for i, sp in enumerate(species_cols):
        tier = tiers[sp]
        if tier in ("A", "B"):
            merged[:, i] = sed_labels[:, i]
        elif tier == "C":
            merged[:, i] = perch_labels[:, i]
        # Tier D stays 0

    # Final stats (pre-filter)
    merged_unlabeled = merged[unlabeled_mask]
    print(f"  Merged shape: {merged.shape}")
    print(f"  Total pseudo-labels >0.5 (unlabeled): {(merged_unlabeled > 0.5).sum():,}")
    print(f"  Total pseudo-labels >0.3 (unlabeled): {(merged_unlabeled > 0.3).sum():,}")
    print(f"  Species with any >0.5 (unlabeled): {(merged_unlabeled.max(axis=0) > 0.5).sum()}/234")
    print(f"  Mean labels per segment >0.5: {(merged_unlabeled > 0.5).sum(axis=1).mean():.2f}")

    # --- Sydorskyi-style filter (optional; backward-compat: no-op when absent) ---
    # See user/murilo/analysis/notes/2026-05-27-exp033-sydorskyi-divergence-audit.md
    #
    # Sydorskyi 2025 2nd place applies two filters AFTER ensemble + soft-label
    # generation:
    #   1. Chunk discard: drop rows where max(prob) < primary_label_min_prob.
    #      ~83% discard rate in their iteration 1 (117K → 19.4K chunks).
    #   2. Per-class trim: zero per-class probs < trim_min_prob.
    #      Eliminates teacher uncertainty noise that BCE would otherwise
    #      train the student to match.
    #
    # Order follows Sydorskyi (discard first, then trim) — both yield the same
    # surviving set since max > 0.5 > 0.1, but the intent is clearer this way.
    #
    # NOTE: We filter BOTH labeled and unlabeled chunks here. The training-time
    # PL filter (DistilledSEDDataModule.setup) still excludes val-fold SS and
    # labeled-SS train files from the PL pool, so labeled rows in the parquet
    # are harmless (just smaller storage win).
    filter_cfg = cfg.get("filter", {}) or {}
    min_max_prob = filter_cfg.get("min_max_prob")
    trim_min_prob = filter_cfg.get("trim_min_prob")
    filter_stats = {}

    if min_max_prob is not None:
        print(f"\n--- Filter: chunk discard (min_max_prob={min_max_prob}) ---")
        chunk_max = merged.max(axis=1)
        keep_mask = chunk_max >= min_max_prob
        n_before = len(merged)
        n_after = int(keep_mask.sum())
        n_dropped = n_before - n_after
        print(f"  Chunks before: {n_before:,}")
        print(f"  Chunks after:  {n_after:,}")
        print(f"  Dropped:       {n_dropped:,} ({n_dropped / n_before:.1%})")
        # Keep all arrays/dataframes aligned
        meta_df = meta_df.iloc[keep_mask].reset_index(drop=True)
        merged = merged[keep_mask]
        sed_labels = sed_labels[keep_mask]
        if use_perch:
            perch_labels = perch_labels[keep_mask]
        unlabeled_mask = unlabeled_mask[keep_mask]
        filter_stats["min_max_prob"] = float(min_max_prob)
        filter_stats["chunks_before_discard"] = int(n_before)
        filter_stats["chunks_after_discard"] = int(n_after)

    if trim_min_prob is not None:
        print(f"\n--- Filter: per-class trim (trim_min_prob={trim_min_prob}) ---")
        n_probs_before = int((merged > 0).sum())
        merged[merged < trim_min_prob] = 0.0
        # Keep sed_labels and perch_labels in sync with the merged matrix so
        # the per-source parquets reflect the same filter. (Sydorskyi-faithful
        # — they trim the saved soft labels, not just the merge.)
        sed_labels[sed_labels < trim_min_prob] = 0.0
        if use_perch:
            perch_labels[perch_labels < trim_min_prob] = 0.0
        n_probs_after = int((merged > 0).sum())
        n_zeroed = n_probs_before - n_probs_after
        print(f"  Non-zero probs before: {n_probs_before:,}")
        print(f"  Non-zero probs after:  {n_probs_after:,}")
        print(f"  Zeroed:                {n_zeroed:,} ({n_zeroed / max(n_probs_before, 1):.1%})")
        filter_stats["trim_min_prob"] = float(trim_min_prob)
        filter_stats["nonzero_probs_before_trim"] = n_probs_before
        filter_stats["nonzero_probs_after_trim"] = n_probs_after

    # Re-print post-filter stats so logs show both before/after
    if min_max_prob is not None or trim_min_prob is not None:
        merged_unlabeled = merged[unlabeled_mask]
        print("\n--- Post-filter stats ---")
        print(f"  Merged shape: {merged.shape}")
        print(f"  Total pseudo-labels >0.5 (unlabeled): {(merged_unlabeled > 0.5).sum():,}")
        print(f"  Total pseudo-labels >0.3 (unlabeled): {(merged_unlabeled > 0.3).sum():,}")
        print(
            f"  Species with any >0.5 (unlabeled): {(merged_unlabeled.max(axis=0) > 0.5).sum()}/234"
        )
        if len(merged_unlabeled) > 0:
            mean_labels = (merged_unlabeled > 0.5).sum(axis=1).mean()
        else:
            mean_labels = 0.0
        print(f"  Mean labels per segment >0.5: {mean_labels:.2f}")

    if args.stats_only:
        print("\n  --stats-only mode, not saving.")
        return

    # --- Save ---
    print(f"\n--- Saving to {output_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    shutil.copy2(config_path, output_dir / "config.yaml")

    # Tier assignment
    tier_df = pd.DataFrame([{"species": sp, "tier": tiers[sp]} for sp in species_cols])
    tier_df.to_csv(output_dir / "tier_assignment.csv", index=False)

    # SED pseudo-labels
    sed_df = pd.concat(
        [
            meta_df[["filename", "chunk_idx"]],
            pd.DataFrame(sed_labels, columns=species_cols),
        ],
        axis=1,
    )
    sed_df.to_parquet(output_dir / "sed_pseudo_labels.parquet", index=False)
    print(f"  Saved sed_pseudo_labels.parquet ({sed_df.shape})")

    # Perch pseudo-labels (skip when pure-SED — file would just be all zeros)
    if use_perch:
        perch_out_df = pd.concat(
            [
                meta_df[["filename", "chunk_idx"]],
                pd.DataFrame(perch_labels, columns=species_cols),
            ],
            axis=1,
        )
        perch_out_df.to_parquet(output_dir / "perch_pseudo_labels.parquet", index=False)
        print(f"  Saved perch_pseudo_labels.parquet ({perch_out_df.shape})")

    # Merged pseudo-labels
    merged_df = pd.concat(
        [
            meta_df[["filename", "chunk_idx"]],
            pd.DataFrame(merged, columns=species_cols),
        ],
        axis=1,
    )
    merged_df.to_parquet(output_dir / "pseudo_labels.parquet", index=False)
    print(f"  Saved pseudo_labels.parquet ({merged_df.shape})")

    # Stats JSON
    stats = {
        "version": version,
        "config_path": str(config_path),
        "total_segments": int(len(meta_df)),
        "unlabeled_segments": int(unlabeled_mask.sum()),
        "species": len(species_cols),
        "tiers": {t: int(c) for t, c in sorted(tier_counts.items())},
        "sed": {
            "temperature": cfg["sed"]["temperature"],
            "power_transform": cfg["sed"]["power_transform"],
            "fold_strategy": cfg["sed"]["fold_strategy"],
            "pseudo_labels_above_0.5": int((sed_labels[unlabeled_mask] > 0.5).sum()),
            "species_covered_above_0.5": int((sed_labels[unlabeled_mask].max(axis=0) > 0.5).sum()),
        },
        "merged": {
            "pseudo_labels_above_0.5": int((merged_unlabeled > 0.5).sum()),
            "pseudo_labels_above_0.3": int((merged_unlabeled > 0.3).sum()),
            "species_covered_above_0.5": int((merged_unlabeled.max(axis=0) > 0.5).sum()),
            "mean_labels_per_segment_above_0.5": float((merged_unlabeled > 0.5).sum(axis=1).mean()),
        },
    }
    if use_perch:
        stats["perch"] = {
            "z_threshold": cfg["perch"]["z_threshold"],
            "min_std": cfg["perch"]["min_std"],
            "power_transform": cfg["perch"]["power_transform"],
            "active_species": len(perch_stats),
            "skipped_species": len(perch_skipped),
            "pseudo_labels_above_0.5": int((perch_labels[unlabeled_mask] > 0.5).sum()),
            "per_species": perch_stats,
        }
    if filter_stats:
        stats["filter"] = filter_stats
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  Saved stats.json")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
