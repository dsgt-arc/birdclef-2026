#!/usr/bin/env python3
"""Inference EDA: Analyze v5j model predictions on train_audio and train_soundscapes.

Reproduces all findings from user/murilo/notes/2026-04-09-inference-eda-deep-dive.md.

Usage:
    python user/murilo/scripts/inference_eda.py

Requires: pandas, numpy, scikit-learn
Data: shared/inference/birdclef-2026/hgnet-b0-sed-v5j/
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
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # birdclef-2026/
# shared/ is at the project-level parent, not the repo parent
SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data" / "birdclef-2026"
INFERENCE_BASE = SHARED / "inference" / "birdclef-2026"

META_COLS = ["filename", "source", "chunk_idx", "start_sec", "end_sec"]


def time_to_sec(t: str) -> float:
    """Convert HH:MM:SS string to seconds."""
    parts = t.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def load_data(model_name: str = "hgnet-b0-sed-v5j"):
    """Load inference logits and competition metadata."""
    base = INFERENCE_BASE / model_name

    # Soundscape logits (labeled + unlabeled)
    ss_logits = pd.read_parquet(base / "train_soundscapes" / "logits.parquet")
    species_cols = [c for c in ss_logits.columns if c not in META_COLS]

    # Focal logits
    ta_logits = pd.read_parquet(base / "train_audio" / "logits.parquet")

    # Ground truth
    labels = pd.read_csv(DATA_DIR / "train_soundscapes_labels.csv")
    labels["start_sec"] = labels["start"].apply(time_to_sec).astype(float)
    train_csv = pd.read_csv(DATA_DIR / "train.csv")

    # Taxonomy mapping
    sp_to_class = dict(zip(train_csv["primary_label"], train_csv["class_name"], strict=False))

    # Labeled / unlabeled split
    labeled_files = set(labels["filename"].unique())
    ss_logits["is_labeled"] = ss_logits["filename"].isin(labeled_files)
    ss_logits["site"] = ss_logits["filename"].str.extract(r"_(S\d+)_")[0]
    ss_logits["hour"] = ss_logits["filename"].str.extract(r"_(\d{2})\d{4}\.ogg$")[0].astype(int)

    # Species sets
    ss_species = set()
    for slist in labels["primary_label"].str.split(";"):
        ss_species.update(s.strip() for s in slist)
    focal_species = set(train_csv["primary_label"].unique())

    return {
        "ss_logits": ss_logits,
        "ta_logits": ta_logits,
        "labels": labels,
        "train_csv": train_csv,
        "species_cols": species_cols,
        "sp_to_class": sp_to_class,
        "labeled_files": labeled_files,
        "ss_species": ss_species,
        "focal_species": focal_species,
    }


# --------------------------------------------------------------------------- #
# Section 1: Labeled soundscape per-species AUC
# --------------------------------------------------------------------------- #
def section1_labeled_auc(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 1: LABELED SOUNDSCAPE PER-SPECIES AUC")
    print("=" * 70)

    df = data["ss_logits"]
    labels = data["labels"]
    species_cols = data["species_cols"]
    sp_to_class = data["sp_to_class"]

    # Match predictions to labels
    pred = df[df["is_labeled"]].copy()
    pred["key"] = pred["filename"] + "_" + pred["start_sec"].astype(str)
    labels["key"] = labels["filename"] + "_" + labels["start_sec"].astype(str)
    matched_keys = set(pred["key"].values) & set(labels["key"].values)
    pred_matched = pred[pred["key"].isin(matched_keys)].sort_values("key").reset_index(drop=True)

    # Build ground truth matrix
    label_dict = {}
    for _, row in labels.iterrows():
        label_dict[row["key"]] = [s.strip() for s in row["primary_label"].split(";")]

    sp_to_idx = {sp: i for i, sp in enumerate(species_cols)}
    gt_matrix = np.zeros((len(pred_matched), len(species_cols)), dtype=np.float32)
    for i, key in enumerate(pred_matched["key"].values):
        if key in label_dict:
            for sp in label_dict[key]:
                if sp in sp_to_idx:
                    gt_matrix[i, sp_to_idx[sp]] = 1.0

    probs = 1 / (1 + np.exp(-pred_matched[species_cols].values.astype(np.float32)))

    # Per-species AUC
    results = []
    for i, sp in enumerate(species_cols):
        gt_col = gt_matrix[:, i]
        n_pos = int(gt_col.sum())
        if n_pos > 0 and n_pos < len(gt_col):
            auc = roc_auc_score(gt_col, probs[:, i])
            results.append({"species": sp, "auc": auc, "n_pos": n_pos})
        else:
            results.append({"species": sp, "auc": np.nan, "n_pos": n_pos})

    results_df = pd.DataFrame(results)
    results_df["taxon"] = results_df["species"].map(sp_to_class).fillna("soundscape_only")
    valid = results_df.dropna(subset=["auc"]).sort_values("auc")

    print(f"\nMatched segments: {len(pred_matched)}")
    print(f"Species with AUC: {len(valid)} / {len(species_cols)}")
    print(f"Macro AUC: {valid['auc'].mean():.4f}")

    print("\n--- AUC by taxonomy ---")
    for cls in sorted(valid["taxon"].unique()):
        grp = valid[valid["taxon"] == cls]
        print(
            f"  {cls:20s}: mean={grp['auc'].mean():.3f}  n={len(grp):3d}  "
            f"min={grp['auc'].min():.3f}  max={grp['auc'].max():.3f}"
        )

    print("\n--- WORST 25 species ---")
    for _, row in valid.head(25).iterrows():
        print(
            f"  {row['species']:20s}  AUC={row['auc']:.3f}  n_pos={row['n_pos']:4d}  "
            f"taxon={row['taxon']}"
        )

    # Species with 0 positives
    zero = results_df[results_df["n_pos"] == 0]
    print(f"\n--- {len(zero)} species have 0 positives in labeled soundscapes ---")
    for cls, grp in zero.groupby("taxon"):
        print(f"  {cls}: {len(grp)} species")

    return results_df


# --------------------------------------------------------------------------- #
# Section 2: Unseen species predictions on unlabeled soundscapes
# --------------------------------------------------------------------------- #
def section2_unseen_species(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 2: UNSEEN SPECIES ON UNLABELED SOUNDSCAPES")
    print("=" * 70)

    df = data["ss_logits"]
    species_cols = data["species_cols"]
    ss_species = data["ss_species"]
    sp_to_class = data["sp_to_class"]

    logits = df[species_cols].values.astype(np.float32)
    probs = 1 / (1 + np.exp(-logits))
    unlabeled_mask = ~df["is_labeled"].values
    unlab_probs = probs[unlabeled_mask]

    unseen_species = [sp for sp in species_cols if sp not in ss_species]
    seen_species = [sp for sp in species_cols if sp in ss_species]
    unseen_idx = [species_cols.index(sp) for sp in unseen_species]
    seen_idx = [species_cols.index(sp) for sp in seen_species]

    print("\n--- Prediction confidence (unlabeled) ---")
    print(f"  159 unseen: max_prob mean={unlab_probs[:, unseen_idx].max(axis=1).mean():.3f}")
    print(f"  75 seen:    max_prob mean={unlab_probs[:, seen_idx].max(axis=1).mean():.3f}")

    # Pseudo-label coverage
    print("\n--- Pseudo-label coverage for unseen species ---")
    for t in [0.3, 0.5, 0.7, 0.9]:
        covered = sum((unlab_probs[:, idx] > t).sum() > 0 for idx in unseen_idx)
        print(f"  >{t}: {covered} / {len(unseen_species)} species covered")

    # Top and zero-coverage species
    coverage = []
    for sp, idx in zip(unseen_species, unseen_idx, strict=False):
        n = (unlab_probs[:, idx] > 0.5).sum()
        coverage.append({"species": sp, "n_gt05": n, "taxon": sp_to_class.get(sp, "unknown")})
    cov_df = pd.DataFrame(coverage).sort_values("n_gt05", ascending=False)

    print("\n--- Top 20 pseudo-label candidates (unseen, >0.5) ---")
    for _, row in cov_df.head(20).iterrows():
        print(f"  {row['species']:20s}  n={row['n_gt05']:6d}  taxon={row['taxon']}")

    zero = cov_df[cov_df["n_gt05"] == 0]
    print(f"\n--- Zero-coverage species: {len(zero)} ---")
    for cls, grp in zero.groupby("taxon"):
        print(f"  {cls}: {len(grp)}")


# --------------------------------------------------------------------------- #
# Section 3: Focal recording analysis
# --------------------------------------------------------------------------- #
def section3_focal_analysis(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 3: FOCAL RECORDING ANALYSIS")
    print("=" * 70)

    ta = data["ta_logits"]
    train_csv = data["train_csv"]
    species_cols = data["species_cols"]
    sp_to_class = data["sp_to_class"]

    logits = ta[species_cols].values.astype(np.float32)
    probs = 1 / (1 + np.exp(-logits))

    # Merge primary labels
    merged = ta.merge(train_csv[["filename", "primary_label"]], on="filename", how="left")
    merged["taxon"] = merged["primary_label"].map(sp_to_class).fillna("unknown")

    # Top-1 / Top-5
    top_pred = np.array(species_cols)[probs.argmax(axis=1)]
    merged["correct_top1"] = top_pred == merged["primary_label"].values

    top5_idx = np.argsort(-probs, axis=1)[:, :5]
    top5_sp = np.array(species_cols)[top5_idx]
    merged["correct_top5"] = [
        pl in t5 for pl, t5 in zip(merged["primary_label"], top5_sp, strict=False)
    ]

    # Primary species confidence
    primary_probs = np.array(
        [
            probs[i, species_cols.index(pl)] if pl in species_cols else np.nan
            for i, pl in enumerate(merged["primary_label"])
        ]
    )
    merged["primary_prob"] = primary_probs

    print(f"\nTotal segments: {len(merged)}, Unique files: {merged['filename'].nunique()}")
    print("\n--- Accuracy ---")
    print(f"  Top-1: {merged['correct_top1'].mean():.3f}")
    print(f"  Top-5: {merged['correct_top5'].mean():.3f}")

    print("\n--- Primary label confidence ---")
    pp = merged["primary_prob"]
    print(f"  Mean: {pp.mean():.3f}  Median: {pp.median():.3f}")
    print(
        f"  <0.1: {(pp < 0.1).mean():.3f}  <0.5: {(pp < 0.5).mean():.3f}  >0.9: {(pp > 0.9).mean():.3f}"
    )

    print("\n--- Accuracy by taxonomy ---")
    for cls in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        g = merged[merged["taxon"] == cls]
        if len(g) > 0:
            print(
                f"  {cls:15s}: top1={g['correct_top1'].mean():.3f}  "
                f"top5={g['correct_top5'].mean():.3f}  "
                f"prob={g['primary_prob'].mean():.3f}  n={len(g)}"
            )

    print("\n--- 20 worst species (by confidence on own focal recordings) ---")
    sp_stats = (
        merged.groupby("primary_label")
        .agg(
            mean_prob=("primary_prob", "mean"),
            top1=("correct_top1", "mean"),
            n=("primary_prob", "count"),
            taxon=("taxon", "first"),
        )
        .sort_values("mean_prob")
    )
    for sp, row in sp_stats.head(20).iterrows():
        print(
            f"  {sp:20s}  prob={row['mean_prob']:.3f}  top1={row['top1']:.3f}  "
            f"n={row['n']:5d}  {row['taxon']}"
        )

    print("\n--- Top 15 confusion pairs ---")
    wrong = merged[~merged["correct_top1"]].copy()
    wrong["top_pred"] = top_pred[~merged["correct_top1"].values]
    conf = wrong.groupby(["primary_label", "top_pred"]).size().reset_index(name="count")
    for _, row in conf.sort_values("count", ascending=False).head(15).iterrows():
        print(f"  {row['primary_label']:20s} → {row['top_pred']:20s}  n={row['count']}")


# --------------------------------------------------------------------------- #
# Section 4: Unlabeled soundscape landscape
# --------------------------------------------------------------------------- #
def section4_unlabeled_landscape(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 4: UNLABELED SOUNDSCAPE LANDSCAPE")
    print("=" * 70)

    df = data["ss_logits"]
    species_cols = data["species_cols"]

    probs = 1 / (1 + np.exp(-df[species_cols].values.astype(np.float32)))
    df = df.copy()
    df["max_prob"] = probs.max(axis=1)

    unlabeled = df[~df["is_labeled"]]
    labeled_sites = set(df[df["is_labeled"]]["site"].unique())

    print(f"\nTotal unlabeled: {len(unlabeled)} segments, {unlabeled['filename'].nunique()} files")

    print("\n--- Per-site summary ---")
    for site in sorted(unlabeled["site"].unique()):
        g = unlabeled[unlabeled["site"] == site]
        tag = "LABELED" if site in labeled_sites else "NEW"
        print(
            f"  {site}: files={g['filename'].nunique():5d}  "
            f"mean_max_prob={g['max_prob'].mean():.3f}  [{tag}]"
        )

    print("\n--- Site overlap ---")
    unlab_sites = set(unlabeled["site"].unique())
    print(f"  Labeled only: {sorted(labeled_sites - unlab_sites)}")
    print(f"  Both: {sorted(labeled_sites & unlab_sites)}")
    print(f"  Unlabeled only: {sorted(unlab_sites - labeled_sites)}")


# --------------------------------------------------------------------------- #
# Section 5: Pseudo-label quality estimation
# --------------------------------------------------------------------------- #
def section5_pseudolabel_quality(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 5: PSEUDO-LABEL QUALITY ESTIMATION")
    print("=" * 70)

    df = data["ss_logits"]
    species_cols = data["species_cols"]
    probs = 1 / (1 + np.exp(-df[species_cols].values.astype(np.float32)))
    unlab_probs = probs[~df["is_labeled"].values]

    print("\n--- Confidence distribution (unlabeled) ---")
    max_per_seg = unlab_probs.max(axis=1)
    for t in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        n = (max_per_seg >= t).sum()
        print(f"  >={t:.2f}: {n:6d} segments ({n / len(max_per_seg) * 100:.1f}%)")

    print("\n--- Species coverage at thresholds ---")
    for t in [0.3, 0.5, 0.7, 0.9]:
        n_sp = ((unlab_probs > t).sum(axis=0) > 0).sum()
        n_seg = (unlab_probs > t).any(axis=1).sum()
        print(f"  >{t}: {n_sp} species, {n_seg} segments")

    print("\n--- Power transform P^1.5 effect ---")
    powered = unlab_probs**1.5
    for t in [0.3, 0.5, 0.7]:
        before = (unlab_probs > t).sum()
        after = (powered > t).sum()
        print(f"  >{t}: {before:8d} → {after:8d} ({after / before * 100:.1f}% retained)")


# --------------------------------------------------------------------------- #
# Section 6: Logit statistics
# --------------------------------------------------------------------------- #
def section6_logit_stats(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 6: LOGIT STATISTICS (hengck23 diagnostics)")
    print("=" * 70)

    df = data["ss_logits"]
    species_cols = data["species_cols"]
    sp_to_class = data["sp_to_class"]
    ss_species = data["ss_species"]

    logits = df[species_cols].values.astype(np.float32)
    logit_std = np.std(logits, axis=0)
    logit_mean = np.mean(logits, axis=0)

    ldf = pd.DataFrame(
        {
            "species": species_cols,
            "logit_mean": logit_mean,
            "logit_std": logit_std,
            "taxon": [sp_to_class.get(sp, "soundscape_only") for sp in species_cols],
            "seen": [sp in ss_species for sp in species_cols],
        }
    )

    print("\n--- Logit std by taxonomy ---")
    for cls in sorted(ldf["taxon"].unique()):
        g = ldf[ldf["taxon"] == cls]
        print(
            f"  {cls:20s}: mean_std={g['logit_std'].mean():.3f}  "
            f"mean_logit={g['logit_mean'].mean():.3f}  n={len(g)}"
        )

    print("\n--- Seen vs unseen ---")
    for label, group in [("75 seen", ldf[ldf["seen"]]), ("159 unseen", ldf[~ldf["seen"]])]:
        print(
            f"  {label}: mean_std={group['logit_std'].mean():.3f}  "
            f"mean_logit={group['logit_mean'].mean():.3f}"
        )

    print("\n--- Highest logit std (top 15) ---")
    for _, r in ldf.sort_values("logit_std", ascending=False).head(15).iterrows():
        tag = "SEEN" if r["seen"] else "unseen"
        print(
            f"  {r['species']:20s}  std={r['logit_std']:.3f}  "
            f"mean={r['logit_mean']:.3f}  {r['taxon']:15s} [{tag}]"
        )

    print("\n--- Lowest logit std (bottom 15) ---")
    for _, r in ldf.sort_values("logit_std", ascending=True).head(15).iterrows():
        tag = "SEEN" if r["seen"] else "unseen"
        print(
            f"  {r['species']:20s}  std={r['logit_std']:.3f}  "
            f"mean={r['logit_mean']:.3f}  {r['taxon']:15s} [{tag}]"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Inference EDA deep dive")
    parser.add_argument(
        "--model", default="hgnet-b0-sed-v5j", help="Model name under shared/inference/"
    )
    args = parser.parse_args()

    print(f"Loading data for {args.model}...")
    data = load_data(args.model)
    print(f"  Soundscape segments: {len(data['ss_logits'])}")
    print(f"  Focal segments: {len(data['ta_logits'])}")
    print(f"  Species: {len(data['species_cols'])}")

    section1_labeled_auc(data)
    section2_unseen_species(data)
    section3_focal_analysis(data)
    section4_unlabeled_landscape(data)
    section5_pseudolabel_quality(data)
    section6_logit_stats(data)

    print("\n" + "=" * 70)
    print("DONE — see user/murilo/notes/2026-04-09-inference-eda-deep-dive.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
