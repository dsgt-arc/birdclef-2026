#!/usr/bin/env python3
"""Diagnose pseudo-label regression: A/B comparison of baseline vs student model.

Compares inference logits from two models (e.g. v5j baseline vs v5j-pl-v1 student)
broken down by species tier (A/B/C/D) to identify WHERE the regression occurred.

Usage:
    python diagnose_regression.py \
        --baseline-model hgnet-b0-sed-v5j \
        --student-model hgnet-b0-sed-v5j-pl-v1 \
        --pseudo-label-path shared/pseudo-labels/birdclef-2026/v1/pseudo_labels.parquet \
        --tier-path shared/pseudo-labels/birdclef-2026/v1/tier_assignment.csv

Requires: pandas, numpy, scikit-learn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data" / "birdclef-2026"
INFERENCE_BASE = SHARED / "inference" / "birdclef-2026"

META_COLS = ["filename", "source", "chunk_idx", "start_sec", "end_sec"]


def time_to_sec(t: str) -> float:
    """Convert HH:MM:SS string to seconds."""
    parts = t.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def load_data(
    baseline_model: str,
    student_model: str,
    pseudo_label_path: str | None = None,
    tier_path: str | None = None,
):
    """Load inference logits from both models + metadata."""
    # Baseline model
    base_dir = INFERENCE_BASE / baseline_model
    bl_ss = pd.read_parquet(base_dir / "train_soundscapes" / "logits.parquet")
    bl_ta = pd.read_parquet(base_dir / "train_audio" / "logits.parquet")
    species_cols = [c for c in bl_ss.columns if c not in META_COLS]

    # Student model
    st_dir = INFERENCE_BASE / student_model
    st_ss = pd.read_parquet(st_dir / "train_soundscapes" / "logits.parquet")
    st_ta = pd.read_parquet(st_dir / "train_audio" / "logits.parquet")

    # Ground truth
    labels = pd.read_csv(DATA_DIR / "train_soundscapes_labels.csv")
    labels["start_sec"] = labels["start"].apply(time_to_sec).astype(float)
    train_csv = pd.read_csv(DATA_DIR / "train.csv")
    sp_to_class = dict(zip(train_csv["primary_label"], train_csv["class_name"], strict=False))

    # Labeled / unlabeled split
    labeled_files = set(labels["filename"].unique())
    bl_ss["is_labeled"] = bl_ss["filename"].isin(labeled_files)
    st_ss["is_labeled"] = st_ss["filename"].isin(labeled_files)

    # Species in labeled soundscapes
    ss_species = set()
    for slist in labels["primary_label"].str.split(";"):
        ss_species.update(s.strip() for s in slist)

    # Tier assignment
    tiers = None
    if tier_path:
        tiers = pd.read_csv(tier_path)
        # Ensure species column is string
        tiers.columns = ["species", "tier"]
        tiers["species"] = tiers["species"].astype(str)

    # Pseudo-labels
    pl_df = None
    if pseudo_label_path:
        pl_df = pd.read_parquet(pseudo_label_path)

    return {
        "bl_ss": bl_ss,
        "bl_ta": bl_ta,
        "st_ss": st_ss,
        "st_ta": st_ta,
        "labels": labels,
        "train_csv": train_csv,
        "species_cols": species_cols,
        "sp_to_class": sp_to_class,
        "labeled_files": labeled_files,
        "ss_species": ss_species,
        "tiers": tiers,
        "pl_df": pl_df,
        "baseline_model": baseline_model,
        "student_model": student_model,
    }


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _get_tier(species: str, tiers: pd.DataFrame | None, ss_species: set) -> str:
    """Get tier for a species. Falls back to A/unseen heuristic if no tier file."""
    if tiers is not None:
        match = tiers[tiers["species"] == str(species)]
        if len(match) > 0:
            return match.iloc[0]["tier"]
    # Fallback
    return "A" if species in ss_species else "unknown"


# --------------------------------------------------------------------------- #
# Section A: Labeled soundscape per-species AUC delta (75 seen species)
# --------------------------------------------------------------------------- #
def section_a_labeled_auc_delta(data: dict):
    print("\n" + "=" * 70)
    print("SECTION A: LABELED SOUNDSCAPE AUC DELTA (75 seen species)")
    print("=" * 70)

    labels = data["labels"]
    species_cols = data["species_cols"]
    sp_to_class = data["sp_to_class"]

    results = []
    for _model_key, ss_df, name in [
        ("bl_ss", data["bl_ss"], data["baseline_model"]),
        ("st_ss", data["st_ss"], data["student_model"]),
    ]:
        pred = ss_df[ss_df["is_labeled"]].copy()
        pred["key"] = pred["filename"] + "_" + pred["start_sec"].astype(str)
        labels_copy = labels.copy()
        labels_copy["key"] = labels_copy["filename"] + "_" + labels_copy["start_sec"].astype(str)
        matched_keys = set(pred["key"].values) & set(labels_copy["key"].values)
        pred_matched = (
            pred[pred["key"].isin(matched_keys)].sort_values("key").reset_index(drop=True)
        )

        # Build GT matrix
        label_dict = {}
        for _, row in labels_copy.iterrows():
            label_dict[row["key"]] = [s.strip() for s in row["primary_label"].split(";")]

        sp_to_idx = {sp: i for i, sp in enumerate(species_cols)}
        gt_matrix = np.zeros((len(pred_matched), len(species_cols)), dtype=np.float32)
        for i, key in enumerate(pred_matched["key"].values):
            if key in label_dict:
                for sp in label_dict[key]:
                    if sp in sp_to_idx:
                        gt_matrix[i, sp_to_idx[sp]] = 1.0

        probs = _sigmoid(pred_matched[species_cols].values.astype(np.float32))

        for j, sp in enumerate(species_cols):
            n_pos = int(gt_matrix[:, j].sum())
            if n_pos > 0 and n_pos < len(gt_matrix):
                auc = roc_auc_score(gt_matrix[:, j], probs[:, j])
            else:
                auc = np.nan
            results.append({"model": name, "species": sp, "auc": auc, "n_pos": n_pos})

    df = pd.DataFrame(results)
    bl_name = data["baseline_model"]
    st_name = data["student_model"]

    bl_df = df[df["model"] == bl_name].set_index("species")
    st_df = df[df["model"] == st_name].set_index("species")

    # Only species with valid AUC in both
    valid_sp = bl_df.dropna(subset=["auc"]).index & st_df.dropna(subset=["auc"]).index
    comparison = pd.DataFrame(
        {
            "bl_auc": bl_df.loc[valid_sp, "auc"],
            "st_auc": st_df.loc[valid_sp, "auc"],
            "n_pos": bl_df.loc[valid_sp, "n_pos"],
        }
    )
    comparison["delta"] = comparison["st_auc"] - comparison["bl_auc"]
    comparison["taxon"] = comparison.index.map(sp_to_class).fillna("soundscape_only")
    comparison = comparison.sort_values("delta")

    print(f"\nSpecies with valid AUC in both models: {len(comparison)}")
    print(f"Baseline macro AUC:  {comparison['bl_auc'].mean():.4f}")
    print(f"Student macro AUC:   {comparison['st_auc'].mean():.4f}")
    print(f"Delta (student-bl):  {comparison['delta'].mean():+.4f}")

    # Count improved / regressed / unchanged
    improved = (comparison["delta"] > 0.001).sum()
    regressed = (comparison["delta"] < -0.001).sum()
    unchanged = len(comparison) - improved - regressed
    print(f"\nImproved: {improved}  Regressed: {regressed}  Unchanged: {unchanged}")

    print("\n--- WORST 15 regressions ---")
    for sp, row in comparison.head(15).iterrows():
        print(
            f"  {sp:20s}  bl={row['bl_auc']:.3f}  st={row['st_auc']:.3f}  "
            f"delta={row['delta']:+.3f}  n_pos={row['n_pos']:3.0f}  {row['taxon']}"
        )

    print("\n--- BEST 10 improvements ---")
    for sp, row in comparison.tail(10).iloc[::-1].iterrows():
        print(
            f"  {sp:20s}  bl={row['bl_auc']:.3f}  st={row['st_auc']:.3f}  "
            f"delta={row['delta']:+.3f}  n_pos={row['n_pos']:3.0f}  {row['taxon']}"
        )

    return comparison


# --------------------------------------------------------------------------- #
# Section B: Focal recording accuracy delta (all 234 species)
# --------------------------------------------------------------------------- #
def section_b_focal_accuracy_delta(data: dict):
    print("\n" + "=" * 70)
    print("SECTION B: FOCAL RECORDING ACCURACY DELTA (234 species)")
    print("=" * 70)

    train_csv = data["train_csv"]
    species_cols = data["species_cols"]
    sp_to_class = data["sp_to_class"]

    results = []
    for ta_df, name in [
        (data["bl_ta"], data["baseline_model"]),
        (data["st_ta"], data["student_model"]),
    ]:
        probs = _sigmoid(ta_df[species_cols].values.astype(np.float32))
        pred_species = [species_cols[i] for i in probs.argmax(axis=1)]
        true_species = (
            ta_df["filename"]
            .map(dict(zip(train_csv["filename"], train_csv["primary_label"], strict=False)))
            .values
        )

        for sp in species_cols:
            sp_mask = true_species == sp
            n_total = sp_mask.sum()
            if n_total == 0:
                results.append(
                    {
                        "model": name,
                        "species": sp,
                        "top1_acc": np.nan,
                        "mean_conf": np.nan,
                        "n_total": 0,
                    }
                )
                continue
            sp_preds = np.array(pred_species)[sp_mask]
            top1_acc = (sp_preds == sp).mean()
            sp_idx = species_cols.index(sp)
            mean_conf = probs[sp_mask, sp_idx].mean()
            results.append(
                {
                    "model": name,
                    "species": sp,
                    "top1_acc": top1_acc,
                    "mean_conf": mean_conf,
                    "n_total": n_total,
                }
            )

    df = pd.DataFrame(results)
    bl_name = data["baseline_model"]
    st_name = data["student_model"]

    bl_df = df[df["model"] == bl_name].set_index("species")
    st_df = df[df["model"] == st_name].set_index("species")

    valid_sp = bl_df[bl_df["n_total"] > 0].index & st_df[st_df["n_total"] > 0].index
    comparison = pd.DataFrame(
        {
            "bl_acc": bl_df.loc[valid_sp, "top1_acc"],
            "st_acc": st_df.loc[valid_sp, "top1_acc"],
            "bl_conf": bl_df.loc[valid_sp, "mean_conf"],
            "st_conf": st_df.loc[valid_sp, "mean_conf"],
            "n_total": bl_df.loc[valid_sp, "n_total"],
        }
    )
    comparison["acc_delta"] = comparison["st_acc"] - comparison["bl_acc"]
    comparison["conf_delta"] = comparison["st_conf"] - comparison["bl_conf"]
    comparison["taxon"] = comparison.index.map(sp_to_class).fillna("unknown")

    print(f"\nSpecies with focal recordings: {len(comparison)}")
    print(f"Baseline mean top-1 acc: {comparison['bl_acc'].mean():.4f}")
    print(f"Student mean top-1 acc:  {comparison['st_acc'].mean():.4f}")
    print(f"Acc delta:               {comparison['acc_delta'].mean():+.4f}")
    print(f"Baseline mean conf:      {comparison['bl_conf'].mean():.4f}")
    print(f"Student mean conf:       {comparison['st_conf'].mean():.4f}")
    print(f"Conf delta:              {comparison['conf_delta'].mean():+.4f}")

    improved = (comparison["acc_delta"] > 0.01).sum()
    regressed = (comparison["acc_delta"] < -0.01).sum()
    print(f"\nImproved (>1%): {improved}  Regressed (<-1%): {regressed}")

    print("\n--- WORST 15 accuracy regressions ---")
    for sp, row in comparison.sort_values("acc_delta").head(15).iterrows():
        print(
            f"  {sp:20s}  bl={row['bl_acc']:.3f}  st={row['st_acc']:.3f}  "
            f"delta={row['acc_delta']:+.3f}  n={row['n_total']:5.0f}  {row['taxon']}"
        )

    return comparison


# --------------------------------------------------------------------------- #
# Section C: Confidence shift by tier on unlabeled soundscapes
# --------------------------------------------------------------------------- #
def section_c_confidence_by_tier(data: dict):
    print("\n" + "=" * 70)
    print("SECTION C: CONFIDENCE SHIFT BY TIER (unlabeled soundscapes)")
    print("=" * 70)

    species_cols = data["species_cols"]
    tiers = data["tiers"]
    ss_species = data["ss_species"]

    # Get unlabeled segments
    bl_unlab = data["bl_ss"][~data["bl_ss"]["is_labeled"]]
    st_unlab = data["st_ss"][~data["st_ss"]["is_labeled"]]

    bl_probs = _sigmoid(bl_unlab[species_cols].values.astype(np.float32))
    st_probs = _sigmoid(st_unlab[species_cols].values.astype(np.float32))

    # Assign tiers to species
    tier_map = {}
    for sp in species_cols:
        tier_map[sp] = _get_tier(sp, tiers, ss_species)

    for tier in ["A", "B", "C", "D"]:
        tier_sp = [sp for sp in species_cols if tier_map.get(sp) == tier]
        if not tier_sp:
            continue
        tier_idx = [species_cols.index(sp) for sp in tier_sp]

        bl_tier = bl_probs[:, tier_idx]
        st_tier = st_probs[:, tier_idx]

        # Per-species max prob
        bl_max = bl_tier.max(axis=0)
        st_max = st_tier.max(axis=0)

        # Coverage: species with any segment > 0.5
        bl_cov = (bl_tier > 0.5).any(axis=0).sum()
        st_cov = (st_tier > 0.5).any(axis=0).sum()

        # Mean prediction per species (across all segments)
        bl_mean = bl_tier.mean(axis=0)
        st_mean = st_tier.mean(axis=0)
        mean_delta = st_mean - bl_mean

        # Pseudo-labels > 0.5 count
        bl_pl_count = (bl_tier > 0.5).sum()
        st_pl_count = (st_tier > 0.5).sum()

        print(f"\n--- Tier {tier} ({len(tier_sp)} species) ---")
        print(f"  Coverage @>0.5:    bl={bl_cov}  st={st_cov}  delta={st_cov - bl_cov:+d}")
        print(
            f"  Predictions @>0.5: bl={bl_pl_count}  st={st_pl_count}  "
            f"delta={st_pl_count - bl_pl_count:+d}"
        )
        print(
            f"  Mean max-prob:     bl={bl_max.mean():.4f}  st={st_max.mean():.4f}  "
            f"delta={st_max.mean() - bl_max.mean():+.4f}"
        )
        print(
            f"  Mean prediction:   bl={bl_mean.mean():.4f}  st={st_mean.mean():.4f}  "
            f"delta={mean_delta.mean():+.4f}"
        )

        # Top movers within this tier
        species_delta = pd.DataFrame(
            {
                "species": tier_sp,
                "bl_max_prob": bl_max,
                "st_max_prob": st_max,
                "delta_max": st_max - bl_max,
                "bl_mean": bl_mean,
                "st_mean": st_mean,
                "delta_mean": mean_delta,
            }
        ).sort_values("delta_mean")

        if len(species_delta) > 5:
            print("\n  Biggest drops (top 5):")
            for _, row in species_delta.head(5).iterrows():
                print(
                    f"    {row['species']:20s}  bl_mean={row['bl_mean']:.4f}  "
                    f"st_mean={row['st_mean']:.4f}  delta={row['delta_mean']:+.4f}"
                )
            print("  Biggest gains (top 5):")
            for _, row in species_delta.tail(5).iloc[::-1].iterrows():
                print(
                    f"    {row['species']:20s}  bl_mean={row['bl_mean']:.4f}  "
                    f"st_mean={row['st_mean']:.4f}  delta={row['delta_mean']:+.4f}"
                )


# --------------------------------------------------------------------------- #
# Section D: Per-species prediction shift (all 234)
# --------------------------------------------------------------------------- #
def section_d_species_prediction_shift(data: dict):
    print("\n" + "=" * 70)
    print("SECTION D: PER-SPECIES PREDICTION SHIFT (unlabeled soundscapes)")
    print("=" * 70)

    species_cols = data["species_cols"]
    tiers = data["tiers"]
    ss_species = data["ss_species"]

    bl_unlab = data["bl_ss"][~data["bl_ss"]["is_labeled"]]
    st_unlab = data["st_ss"][~data["st_ss"]["is_labeled"]]

    bl_probs = _sigmoid(bl_unlab[species_cols].values.astype(np.float32))
    st_probs = _sigmoid(st_unlab[species_cols].values.astype(np.float32))

    results = []
    for i, sp in enumerate(species_cols):
        bl_mean = bl_probs[:, i].mean()
        st_mean = st_probs[:, i].mean()
        bl_cov = (bl_probs[:, i] > 0.5).sum()
        st_cov = (st_probs[:, i] > 0.5).sum()
        tier = _get_tier(sp, tiers, ss_species)
        results.append(
            {
                "species": sp,
                "tier": tier,
                "bl_mean_prob": bl_mean,
                "st_mean_prob": st_mean,
                "delta_mean": st_mean - bl_mean,
                "bl_n_gt05": bl_cov,
                "st_n_gt05": st_cov,
                "delta_n_gt05": st_cov - bl_cov,
            }
        )

    df = pd.DataFrame(results).sort_values("delta_mean")

    print("\n--- Overall shift ---")
    print(f"  Mean prob delta: {df['delta_mean'].mean():+.6f}")
    print(f"  Species with increased predictions:  {(df['delta_mean'] > 0).sum()}")
    print(f"  Species with decreased predictions:  {(df['delta_mean'] < 0).sum()}")

    # By tier
    for tier in ["A", "B", "C", "D"]:
        t = df[df["tier"] == tier]
        if len(t) == 0:
            continue
        print(
            f"\n  Tier {tier} ({len(t)} spp): mean delta={t['delta_mean'].mean():+.6f}  "
            f"coverage delta={t['delta_n_gt05'].sum():+d}"
        )

    print("\n--- TOP 15 species that LOST the most prediction strength ---")
    for _, row in df.head(15).iterrows():
        print(
            f"  {row['species']:20s}  tier={row['tier']}  bl={row['bl_mean_prob']:.4f}  "
            f"st={row['st_mean_prob']:.4f}  delta={row['delta_mean']:+.4f}  "
            f"cov: {row['bl_n_gt05']} → {row['st_n_gt05']}"
        )

    print("\n--- TOP 15 species that GAINED the most prediction strength ---")
    for _, row in df.tail(15).iloc[::-1].iterrows():
        print(
            f"  {row['species']:20s}  tier={row['tier']}  bl={row['bl_mean_prob']:.4f}  "
            f"st={row['st_mean_prob']:.4f}  delta={row['delta_mean']:+.4f}  "
            f"cov: {row['bl_n_gt05']} → {row['st_n_gt05']}"
        )

    return df


# --------------------------------------------------------------------------- #
# Section E: Pseudo-label agreement analysis
# --------------------------------------------------------------------------- #
def section_e_pl_agreement(data: dict):
    print("\n" + "=" * 70)
    print("SECTION E: PSEUDO-LABEL AGREEMENT (student vs teacher PL)")
    print("=" * 70)

    pl_df = data["pl_df"]
    if pl_df is None:
        print("  No pseudo-label file provided — skipping")
        return

    species_cols = data["species_cols"]
    tiers = data["tiers"]
    ss_species = data["ss_species"]

    # Align student predictions with pseudo-labels by (filename, chunk_idx)
    st_unlab = data["st_ss"][~data["st_ss"]["is_labeled"]].copy()

    # Match on filename + chunk_idx
    st_unlab["key"] = st_unlab["filename"] + "_" + st_unlab["chunk_idx"].astype(str)
    pl_df["key"] = pl_df["filename"] + "_" + pl_df["chunk_idx"].astype(str)

    common_keys = set(st_unlab["key"].values) & set(pl_df["key"].values)
    print(f"\nMatched segments: {len(common_keys)} / {len(st_unlab)} student, {len(pl_df)} PL")

    st_matched = (
        st_unlab[st_unlab["key"].isin(common_keys)].sort_values("key").reset_index(drop=True)
    )
    pl_matched = pl_df[pl_df["key"].isin(common_keys)].sort_values("key").reset_index(drop=True)

    # Only use species columns present in both
    common_sp = [sp for sp in species_cols if sp in pl_matched.columns]

    st_probs = _sigmoid(st_matched[common_sp].values.astype(np.float32))
    pl_vals = pl_matched[common_sp].values.astype(np.float32)

    # For each tier, check agreement on PL > 0.5 segments
    for tier in ["A", "B", "C"]:
        tier_sp = [sp for sp in common_sp if _get_tier(sp, tiers, ss_species) == tier]
        if not tier_sp:
            continue
        tier_idx = [common_sp.index(sp) for sp in tier_sp]

        pl_tier = pl_vals[:, tier_idx]
        st_tier = st_probs[:, tier_idx]

        # Where PL was confident (>0.5)
        pl_pos_mask = pl_tier > 0.5
        n_pl_pos = pl_pos_mask.sum()
        if n_pl_pos == 0:
            print(f"\n--- Tier {tier}: 0 positive pseudo-labels — skipping ---")
            continue

        # Student agreement on PL-positive segments
        st_on_pl_pos = st_tier[pl_pos_mask]
        st_agrees = (st_on_pl_pos > 0.5).sum()
        st_mean_on_pos = st_on_pl_pos.mean()

        # Where PL was negative (<0.1)
        pl_neg_mask = pl_tier < 0.1
        n_pl_neg = pl_neg_mask.sum()
        st_on_pl_neg = st_tier[pl_neg_mask]
        st_false_pos = (st_on_pl_neg > 0.5).sum() if n_pl_neg > 0 else 0

        print(f"\n--- Tier {tier} ({len(tier_sp)} species) ---")
        print(f"  PL positives (>0.5): {n_pl_pos:,}")
        print(
            f"  Student agrees (>0.5 where PL>0.5): {st_agrees:,} ({st_agrees / n_pl_pos * 100:.1f}%)"
        )
        print(f"  Student mean prob on PL-positive: {st_mean_on_pos:.3f}")
        print(f"  PL negatives (<0.1): {n_pl_neg:,}")
        if n_pl_neg > 0:
            print(
                f"  Student FP on PL-negative (>0.5 where PL<0.1): {st_false_pos:,} "
                f"({st_false_pos / n_pl_neg * 100:.4f}%)"
            )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Diagnose pseudo-label regression")
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="hgnet-b0-sed-v5j",
        help="Baseline model name (under shared/inference/)",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="hgnet-b0-sed-v5j-pl-v1",
        help="Student model name (under shared/inference/)",
    )
    parser.add_argument(
        "--pseudo-label-path", type=str, default=None, help="Path to pseudo_labels.parquet"
    )
    parser.add_argument("--tier-path", type=str, default=None, help="Path to tier_assignment.csv")
    args = parser.parse_args()

    print("=" * 70)
    print("PSEUDO-LABEL REGRESSION DIAGNOSIS")
    print(f"  Baseline: {args.baseline_model}")
    print(f"  Student:  {args.student_model}")
    print(f"  PL path:  {args.pseudo_label_path}")
    print(f"  Tier path: {args.tier_path}")
    print("=" * 70)

    data = load_data(
        baseline_model=args.baseline_model,
        student_model=args.student_model,
        pseudo_label_path=args.pseudo_label_path,
        tier_path=args.tier_path,
    )

    section_a_labeled_auc_delta(data)
    section_b_focal_accuracy_delta(data)
    section_c_confidence_by_tier(data)
    section_d_species_prediction_shift(data)
    section_e_pl_agreement(data)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
