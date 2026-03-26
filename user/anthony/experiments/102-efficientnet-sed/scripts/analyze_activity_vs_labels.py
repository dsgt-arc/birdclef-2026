#!/usr/bin/env python3
"""Analyze how well acoustic indices predict call presence in labeled soundscapes.

Uses pre-computed acoustic indices from exp010-2026 and compares against
soundscape labels to measure predictive power.

Outputs:
  - Summary stats (spectral contrast by call/no-call)
  - AUROC for each acoustic index
  - Logistic regression fit and coefficients
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

INDICES_ROOT = Path("~/scratch/birdclef-2026/outputs/exp010-2026/labels/indices").expanduser()
LABELS_CSV = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes_labels.csv"
INDEX_FEATURES = ["ndsi", "spectral_contrast", "spectral_flatness", "bioacoustic_index"]


def time_to_sec(t):
    parts = t.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-duration", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load pre-computed acoustic indices for soundscapes
    parts_dir = INDICES_ROOT / "train_soundscapes" / "parts"
    parts = sorted(parts_dir.glob("*.parquet"))
    logger.info(f"Loading {len(parts)} index parquets from {parts_dir}")
    indices = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    indices["filename"] = indices["file"].apply(lambda f: Path(f).name)
    logger.info(f"Index rows: {len(indices)}, files: {indices['filename'].nunique()}")

    # Aggregate to 5s chunks
    indices["chunk_start"] = (indices["window_start"] // args.chunk_duration) * args.chunk_duration
    agg_dict = {}
    for feat in INDEX_FEATURES:
        if feat in indices.columns:
            agg_dict[feat] = (feat, "mean")
    chunks = indices.groupby(["filename", "chunk_start"]).agg(**agg_dict).reset_index()
    logger.info(f"5s chunks: {len(chunks)}")

    # Load labels
    labels = pd.read_csv(LABELS_CSV)
    labels["chunk_start"] = labels["start"].apply(time_to_sec).astype(float)
    labels = labels.drop_duplicates(subset=["filename", "chunk_start"])
    logger.info(f"Labeled windows: {len(labels)}")

    # Filter to only chunks in labeled files
    labeled_files = set(labels["filename"].unique())
    chunks_labeled = chunks[chunks["filename"].isin(labeled_files)].copy()

    # Join: mark which chunks have labels (= call present)
    chunks_labeled = chunks_labeled.merge(
        labels[["filename", "chunk_start"]].assign(has_call=True),
        on=["filename", "chunk_start"],
        how="left",
    )
    chunks_labeled["has_call"] = chunks_labeled["has_call"].fillna(False)

    n_call = chunks_labeled["has_call"].sum()
    n_nocall = (~chunks_labeled["has_call"]).sum()
    logger.info(f"Chunks in labeled files: {len(chunks_labeled)} (call={n_call}, no-call={n_nocall})")

    # Summary stats per feature
    print("\n=== Summary Stats by Call Status ===")
    for feat in INDEX_FEATURES:
        if feat not in chunks_labeled.columns:
            continue
        print(f"\n{feat}:")
        for has_call, group in chunks_labeled.groupby("has_call"):
            status = "CALL" if has_call else "NO-CALL"
            vals = group[feat].dropna()
            print(f"  {status:8s}: mean={vals.mean():.4f}, median={vals.median():.4f}, "
                  f"std={vals.std():.4f}, n={len(vals)}")

    # Per-feature AUROC
    print("\n=== Per-Feature AUROC (predicting call presence) ===")
    y_true = chunks_labeled["has_call"].astype(int).values
    available_features = []
    for feat in INDEX_FEATURES:
        if feat not in chunks_labeled.columns:
            continue
        y_score = chunks_labeled[feat].values
        mask = ~np.isnan(y_score)
        if mask.sum() == 0:
            continue
        auc = roc_auc_score(y_true[mask], y_score[mask])
        print(f"  {feat:25s}: AUROC = {auc:.3f}")
        available_features.append(feat)

    # Logistic regression with all features
    print("\n=== Logistic Regression (all features) ===")
    X = chunks_labeled[available_features].values
    y = y_true

    # Drop rows with NaN
    mask = ~np.any(np.isnan(X), axis=1)
    X_clean, y_clean = X[mask], y[mask]
    logger.info(f"Fitting on {len(X_clean)} samples ({mask.sum()} after dropping NaN)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_scaled, y_clean)

    y_pred_proba = lr.predict_proba(X_scaled)[:, 1]
    y_pred = lr.predict(X_scaled)

    auc = roc_auc_score(y_clean, y_pred_proba)
    print(f"  AUROC: {auc:.3f}")
    print(f"\n  Coefficients (standardized):")
    for feat, coef in zip(available_features, lr.coef_[0]):
        print(f"    {feat:25s}: {coef:+.4f}")
    print(f"    {'intercept':25s}: {lr.intercept_[0]:+.4f}")

    print(f"\n  Classification Report (balanced class weights):")
    print(classification_report(y_clean, y_pred, target_names=["NO-CALL", "CALL"]))

    # How many no-call windows would be correctly identified at various thresholds?
    print("=== No-call Detection at Various Thresholds ===")
    nocall_probs = y_pred_proba[y_clean == 0]
    call_probs = y_pred_proba[y_clean == 1]
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        nocall_correct = (nocall_probs < threshold).sum()
        call_missed = (call_probs < threshold).sum()
        print(f"  threshold={threshold:.1f}: "
              f"no-call correct={nocall_correct}/{len(nocall_probs)} "
              f"({100*nocall_correct/max(1,len(nocall_probs)):.0f}%), "
              f"calls missed={call_missed}/{len(call_probs)} "
              f"({100*call_missed/max(1,len(call_probs)):.0f}%)")


if __name__ == "__main__":
    main()
