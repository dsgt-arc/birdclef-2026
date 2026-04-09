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
# Section 7: Single vs Avg-4fold vs Max-4fold pseudo-label coverage
# --------------------------------------------------------------------------- #
def section7_fourfold_coverage(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 7: SINGLE vs AVG-4FOLD vs MAX-4FOLD COVERAGE")
    print("=" * 70)

    species_cols = data["species_cols"]
    ss_species = data["ss_species"]
    sp_to_class = data["sp_to_class"]
    unseen_species = [sp for sp in species_cols if sp not in ss_species]
    unseen_idx = [species_cols.index(sp) for sp in unseen_species]

    df = data["ss_logits"]
    single_logits = df[species_cols].values.astype(np.float32)
    unlabeled_mask = ~df["is_labeled"].values

    # Load 4-fold logits
    base_4fold = INFERENCE_BASE / "hgnet-b0-sed-v5j-4fold" / "train_soundscapes"
    fold_logits = []
    for fold in range(4):
        fdf = pd.read_parquet(base_4fold / f"fold{fold}" / "logits.parquet")
        fold_logits.append(fdf[species_cols].values.astype(np.float32))

    avg_logits = np.mean(fold_logits, axis=0)
    max_logits = np.maximum.reduce(fold_logits)

    single_probs = 1 / (1 + np.exp(-single_logits))
    avg_probs = 1 / (1 + np.exp(-avg_logits))
    max_probs = 1 / (1 + np.exp(-max_logits))

    single_unlab = single_probs[unlabeled_mask]
    avg_unlab = avg_probs[unlabeled_mask]
    max_unlab = max_probs[unlabeled_mask]

    # Coverage comparison
    print("\n--- Species coverage (159 unseen, unlabeled soundscapes) ---")
    print(
        f"{'Thresh':>7s} | {'Single':>10s} | {'Avg-4f':>10s} | {'Max-4f':>10s} | {'Max vs Single':>14s}"
    )
    print("-" * 62)
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = sum((single_unlab[:, i] > t).sum() > 0 for i in unseen_idx)
        a = sum((avg_unlab[:, i] > t).sum() > 0 for i in unseen_idx)
        m = sum((max_unlab[:, i] > t).sum() > 0 for i in unseen_idx)
        print(f"  >{t:.1f}  | {s:>6d}/159 | {a:>6d}/159 | {m:>6d}/159 | {m - s:>+12d}")

    # Total pseudo-label predictions
    print("\n--- Total predictions (all 234 species, unlabeled) ---")
    for t in [0.3, 0.5, 0.7]:
        s = (single_unlab > t).sum()
        a = (avg_unlab > t).sum()
        m = (max_unlab > t).sum()
        print(f"  >{t}: single={s:,d}  avg={a:,d}  max={m:,d}")

    # Species recovered by max-folds at 0.5
    recovered = []
    for sp, idx in zip(unseen_species, unseen_idx, strict=False):
        s_n = int((single_unlab[:, idx] > 0.5).sum())
        m_n = int((max_unlab[:, idx] > 0.5).sum())
        if m_n > 0 and s_n == 0:
            recovered.append(
                {"species": sp, "max_folds": m_n, "taxon": sp_to_class.get(sp, "unknown")}
            )
    rec_df = (
        pd.DataFrame(recovered).sort_values("max_folds", ascending=False)
        if recovered
        else pd.DataFrame()
    )
    print(f"\n--- Recovered by max-folds at >0.5 (single=0, max>0): {len(rec_df)} species ---")
    for _, row in rec_df.head(15).iterrows():
        print(f"    {row['species']:20s}  max={row['max_folds']:4d}  {row['taxon']}")

    # Zero coverage counts
    s_zero = {
        sp
        for sp, i in zip(unseen_species, unseen_idx, strict=False)
        if (single_unlab[:, i] > 0.5).sum() == 0
    }
    a_zero = {
        sp
        for sp, i in zip(unseen_species, unseen_idx, strict=False)
        if (avg_unlab[:, i] > 0.5).sum() == 0
    }
    m_zero = {
        sp
        for sp, i in zip(unseen_species, unseen_idx, strict=False)
        if (max_unlab[:, i] > 0.5).sum() == 0
    }
    print("\n--- Zero coverage at >0.5 ---")
    print(
        f"  Single: {len(s_zero)}  Avg: {len(a_zero)}  Max: {len(m_zero)}  All three: {len(s_zero & a_zero & m_zero)}"
    )

    # Truly hard at 0.3
    m_zero_03 = {
        sp
        for sp, i in zip(unseen_species, unseen_idx, strict=False)
        if (max_unlab[:, i] > 0.3).sum() == 0
    }
    print(f"  Zero at >0.3 (max-folds): {len(m_zero_03)} truly hard species")

    # Per-fold unique contributions
    fold_probs_unlab = [(1 / (1 + np.exp(-fl[unlabeled_mask]))) for fl in fold_logits]
    print("\n--- Per-fold unique contributions (unseen, >0.5) ---")
    for f in range(4):
        total = sum((fold_probs_unlab[f][:, i] > 0.5).sum() > 0 for i in unseen_idx)
        unique = 0
        for idx in unseen_idx:
            if (fold_probs_unlab[f][:, idx] > 0.5).sum() > 0:
                if not any(
                    (fold_probs_unlab[o][:, idx] > 0.5).sum() > 0 for o in range(4) if o != f
                ):
                    unique += 1
        print(f"  Fold {f}: {total} species, {unique} unique")


# --------------------------------------------------------------------------- #
# Section 8: Secondary labels & confusion analysis
# --------------------------------------------------------------------------- #
def section8_secondary_labels(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 8: SECONDARY LABELS & CONFUSION ANALYSIS")
    print("=" * 70)

    ta = data["ta_logits"]
    train_csv = data["train_csv"]
    species_cols = data["species_cols"]

    # Parse secondary labels
    train_csv = train_csv.copy()
    train_csv["sec_parsed"] = train_csv["secondary_labels"].apply(
        lambda x: {
            s.strip().strip("'\"") for s in x.strip("[]").split(",") if s.strip().strip("'\"") != ""
        }
        if pd.notna(x) and x.strip() not in ["", "[]"]
        else set()
    )
    file_info = dict(
        zip(
            train_csv["filename"],
            zip(train_csv["primary_label"], train_csv["sec_parsed"], strict=False),
            strict=False,
        )
    )

    logits = ta[species_cols].values.astype(np.float32)
    probs = 1 / (1 + np.exp(-logits))
    top_pred = np.array(species_cols)[probs.argmax(axis=1)]

    primaries = ta["filename"].map(lambda f: file_info.get(f, ("", set()))[0]).values
    secondaries = ta["filename"].map(lambda f: file_info.get(f, ("", set()))[1]).values

    wrong_mask = top_pred != primaries
    wrong_is_sec = np.array(
        [
            tp in sec if isinstance(sec, set) else False
            for tp, sec in zip(top_pred[wrong_mask], secondaries[wrong_mask], strict=False)
        ]
    )

    total_wrong = wrong_mask.sum()
    n_sec = wrong_is_sec.sum()
    print(f"\nTotal wrong top-1: {total_wrong}")
    print(f"Prediction matches secondary label: {n_sec} ({n_sec / total_wrong * 100:.1f}%)")

    # Top confusion pairs with secondary breakdown
    wrong_df = pd.DataFrame(
        {
            "primary": primaries[wrong_mask],
            "top_pred": top_pred[wrong_mask],
            "is_secondary": wrong_is_sec,
        }
    )
    conf = (
        wrong_df.groupby(["primary", "top_pred"])
        .agg(
            count=("is_secondary", "size"),
            n_sec=("is_secondary", "sum"),
        )
        .reset_index()
    )
    conf["pct_sec"] = (conf["n_sec"] / conf["count"] * 100).round(1)

    print("\n--- Top 20 confusion pairs ---")
    print(f"{'True':>20s} → {'Predicted':>20s}  {'N':>5s}  {'2nd':>5s}  {'%':>5s}")
    print("-" * 62)
    for _, row in conf.sort_values("count", ascending=False).head(20).iterrows():
        print(
            f"  {row['primary']:>18s} → {row['top_pred']:>18s}  "
            f"{row['count']:5d}  {row['n_sec']:5d}  {row['pct_sec']:5.1f}%"
        )

    # Confusion rate by secondary label presence
    has_sec = np.array([len(s) > 0 if isinstance(s, set) else False for s in secondaries])
    print("\n--- Confusion rate: with/without secondary labels ---")
    print(f"  With secondary:    {wrong_mask[has_sec].mean():.3f} (n={has_sec.sum()})")
    print(f"  Without secondary: {wrong_mask[~has_sec].mean():.3f} (n={(~has_sec).sum()})")


# --------------------------------------------------------------------------- #
# Section 9: Focal sample count vs pseudo-label coverage
# --------------------------------------------------------------------------- #
def section9_sample_vs_coverage(data: dict):
    print("\n" + "=" * 70)
    print("SECTION 9: FOCAL SAMPLE COUNT vs PSEUDO-LABEL COVERAGE")
    print("=" * 70)

    df = data["ss_logits"]
    species_cols = data["species_cols"]
    ss_species = data["ss_species"]
    sp_to_class = data["sp_to_class"]
    train_csv = data["train_csv"]

    probs = 1 / (1 + np.exp(-df[species_cols].values.astype(np.float32)))
    unlab_probs = probs[~df["is_labeled"].values]

    unseen_species = [sp for sp in species_cols if sp not in ss_species]
    focal_counts = train_csv["primary_label"].value_counts().to_dict()

    rows = []
    for sp in unseen_species:
        idx = species_cols.index(sp)
        rows.append(
            {
                "species": sp,
                "focal_count": focal_counts.get(sp, 0),
                "n_pl_05": int((unlab_probs[:, idx] > 0.5).sum()),
                "n_pl_03": int((unlab_probs[:, idx] > 0.3).sum()),
                "max_prob": float(unlab_probs[:, idx].max()),
                "taxon": sp_to_class.get(sp, "unknown"),
            }
        )
    rdf = pd.DataFrame(rows)

    from scipy import stats

    corr, pval = stats.spearmanr(rdf["focal_count"], rdf["n_pl_05"])
    print(f"\nSpearman (focal_count vs pseudo-labels@0.5): r={corr:.3f}, p={pval:.4f}")

    corr_max, pval_max = stats.spearmanr(rdf["focal_count"], rdf["max_prob"])
    print(f"Spearman (focal_count vs max_prob): r={corr_max:.3f}, p={pval_max:.4f}")

    bins = [(0, 10), (10, 50), (50, 100), (100, 200), (200, 500)]
    print("\n--- Coverage by focal count bucket ---")
    print(f"{'Range':>12s} | {'N':>4s} | {'Cov@0.5':>8s} | {'Cov@0.3':>8s} | {'MaxProb':>8s}")
    print("-" * 52)
    for lo, hi in bins:
        b = rdf[(rdf["focal_count"] >= lo) & (rdf["focal_count"] < hi)]
        n = len(b)
        if n == 0:
            continue
        c5 = (b["n_pl_05"] > 0).sum()
        c3 = (b["n_pl_03"] > 0).sum()
        print(
            f"  {lo:>3d}-{hi:<4d}  | {n:>4d} | {c5:>4d}/{n:<3d} | {c3:>4d}/{n:<3d} | {b['max_prob'].mean():>8.3f}"
        )

    # Data quality suspects
    suspects = rdf[(rdf["focal_count"] >= 50) & (rdf["n_pl_05"] == 0)].sort_values(
        "focal_count", ascending=False
    )
    print(f"\n--- Data quality suspects (≥50 focal, 0 pseudo-labels): {len(suspects)} ---")
    for _, row in suspects.head(10).iterrows():
        print(
            f"  {row['species']:20s}  focal={row['focal_count']:4d}  "
            f"max_prob={row['max_prob']:.4f}  {row['taxon']}"
        )

    # Bright spots
    bright = rdf[(rdf["focal_count"] < 50) & (rdf["n_pl_05"] > 10)].sort_values(
        "n_pl_05", ascending=False
    )
    if len(bright) > 0:
        print("\n--- Bright spots (<50 focal, >10 pseudo-labels) ---")
        for _, row in bright.iterrows():
            print(
                f"  {row['species']:20s}  focal={row['focal_count']:4d}  "
                f"n_pl@0.5={row['n_pl_05']:4d}  {row['taxon']}"
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
    section7_fourfold_coverage(data)
    section8_secondary_labels(data)
    section9_sample_vs_coverage(data)

    print("\n" + "=" * 70)
    print("DONE — see user/murilo/notes/2026-04-09-inference-eda-deep-dive.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
