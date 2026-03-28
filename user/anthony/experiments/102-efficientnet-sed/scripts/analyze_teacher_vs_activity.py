#!/usr/bin/env python3
"""Compare teacher predictions with activity scores.

Checks correlation between model-based teacher_prob and acoustic index-based
activity_score to see if they measure the same thing or complement each other.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    teacher = pd.read_parquet("artifacts/teacher_predictions.parquet")
    activity = pd.read_parquet("artifacts/focal_activity.parquet")

    logger.info(f"Teacher: {len(teacher)} rows, Activity: {len(activity)} rows")

    # Join on (filename, chunk_start)
    merged = activity.merge(teacher, on=["filename", "chunk_start"], how="inner")
    logger.info(f"Merged: {len(merged)} rows")

    print("\n=== Correlation: teacher_prob vs activity scores ===")
    for col in ["activity_score", "activity_score_global", "spectral_contrast_mean", "bioacoustic_index_mean"]:
        if col in merged.columns:
            corr = merged["teacher_prob"].corr(merged[col])
            print(f"  teacher_prob vs {col:30s}: r = {corr:.3f}")

    print(f"\n  teacher_prob vs active_frames/total:         r = {merged['teacher_prob'].corr(merged['active_frames'] / merged['total_frames']):.3f}")

    # Binned analysis: what does activity_score look like for high vs low teacher_prob?
    print("\n=== Activity score by teacher confidence ===")
    bins = [0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.01]
    labels = ["<0.1", "0.1-0.3", "0.3-0.5", "0.5-0.8", "0.8-0.95", ">0.95"]
    merged["teacher_bin"] = pd.cut(merged["teacher_prob"], bins=bins, labels=labels)
    for label in labels:
        group = merged[merged["teacher_bin"] == label]
        if len(group) > 0:
            print(f"  teacher_prob {label:>8s}: n={len(group):>6d}  "
                  f"activity_score mean={group['activity_score'].mean():.3f}  "
                  f"spectral_contrast mean={group['spectral_contrast_mean'].mean():.2f}")

    # Reverse: what does teacher_prob look like for high vs low activity?
    print("\n=== Teacher prob by activity score quartile ===")
    merged["activity_q"] = pd.qcut(merged["activity_score"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
        group = merged[merged["activity_q"] == q]
        print(f"  {q:12s}: n={len(group):>6d}  "
              f"teacher_prob mean={group['teacher_prob'].mean():.3f}  "
              f"median={group['teacher_prob'].median():.3f}  "
              f"frac<0.3={100*(group['teacher_prob'] < 0.3).mean():.1f}%")

    # Can activity score predict "teacher disagrees" (teacher_prob < 0.3)?
    print("\n=== Predicting teacher disagreement (teacher_prob < 0.3) ===")
    y_true = (merged["teacher_prob"] < 0.3).astype(int).values
    for col in ["activity_score", "spectral_contrast_mean", "bioacoustic_index_mean"]:
        if col in merged.columns:
            vals = merged[col].values
            mask = ~np.isnan(vals)
            # Low activity = more likely to disagree, so negate for AUROC
            auc = roc_auc_score(y_true[mask], -vals[mask])
            print(f"  {col:30s}: AUROC = {auc:.3f}")

    # How much do the two signals overlap vs complement?
    print("\n=== Overlap analysis ===")
    low_teacher = merged["teacher_prob"] < 0.3
    low_activity = merged["activity_score"] < 0.2
    both = low_teacher & low_activity
    teacher_only = low_teacher & ~low_activity
    activity_only = ~low_teacher & low_activity
    print(f"  Low teacher (<0.3) AND low activity (<0.2): {both.sum():>6d}")
    print(f"  Low teacher only:                            {teacher_only.sum():>6d}")
    print(f"  Low activity only:                           {activity_only.sum():>6d}")
    print(f"  Neither:                                     {(~low_teacher & ~low_activity).sum():>6d}")

    # Mislabel analysis
    print("\n=== Mislabel candidates (max_other_prob > teacher_prob) ===")
    mislabel = merged["max_other_prob"] > merged["teacher_prob"]
    print(f"  Total: {mislabel.sum()} ({100*mislabel.mean():.1f}%)")
    print(f"  Activity score of mislabel candidates: mean={merged.loc[mislabel, 'activity_score'].mean():.3f}")
    print(f"  Activity score of correct labels:      mean={merged.loc[~mislabel, 'activity_score'].mean():.3f}")


if __name__ == "__main__":
    main()
