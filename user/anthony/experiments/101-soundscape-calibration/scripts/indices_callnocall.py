"""Evaluate acoustic indices as call/no-call detectors.

Loads indices from exp010-2026, aggregates to 5s windows, and evaluates
each index + a combined logistic regression model against the labeled windows.

Usage:
    python indices_callnocall.py --output artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data/birdclef-2026"
INDICES_ROOT = Path("~/scratch/birdclef-2026/outputs/exp010-2026/labels/indices").expanduser()
EDA_ARTIFACTS = Path(__file__).resolve().parents[2] / "100-eda/artifacts"

INDICES_FEATURES = ["ndsi", "spectral_contrast", "spectral_flatness", "bioacoustic_index"]
TEST_SITE = "S22"


def load_ground_truth(eda_artifacts):
    labels = pd.read_parquet(eda_artifacts / "soundscape_labels.parquet")
    windows = labels[["filename", "start_sec", "end_sec"]].drop_duplicates().copy()
    windows["site"] = windows["filename"].str.extract(r"_(S\d+)_")

    # Count species per window
    species_counts = (
        labels.groupby(["filename", "start_sec"])["primary_label"]
        .nunique()
        .reset_index()
        .rename(columns={"primary_label": "n_species"})
    )
    windows = windows.merge(species_counts, on=["filename", "start_sec"], how="left")
    windows["n_species"] = windows["n_species"].fillna(0).astype(int)
    windows["is_call"] = (windows["n_species"] > 0).astype(int)

    return windows


def load_soundscape_indices():
    """Load acoustic indices for soundscapes, aggregate to 5s windows."""
    parts_dir = INDICES_ROOT / "train_soundscapes" / "parts"
    parts = sorted(parts_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquets in {parts_dir}")

    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    # Filename cleanup
    df["filename"] = df["file"].apply(lambda f: Path(f).name)

    # Aggregate to 5s windows
    df["window_start"] = (df["window_start"] // 5.0 * 5.0).astype(float)

    # For each index, compute both mean and max within the 5s window
    agg_dict = {}
    for feat in INDICES_FEATURES:
        agg_dict[f"{feat}_mean"] = (feat, "mean")
        agg_dict[f"{feat}_max"] = (feat, "max")
        agg_dict[f"{feat}_std"] = (feat, "std")

    agg = df.groupby(["filename", "window_start"]).agg(**agg_dict).reset_index()
    agg = agg.rename(columns={"window_start": "start_sec"})
    agg = agg.fillna(0)

    return agg


def confusion(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--eda-artifacts", type=Path, default=EDA_ARTIFACTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")
    gt = load_ground_truth(args.eda_artifacts)
    print(f"  {len(gt)} windows: {int(gt['is_call'].sum())} call, {int((1 - gt['is_call']).sum())} no-call")

    # Include unlabeled gaps — windows in labeled files with no annotations
    # These are windows where start_sec exists in predictions but not in labels
    # We'll get them from the indices data
    print("\nLoading acoustic indices...")
    indices = load_soundscape_indices()
    print(f"  {len(indices)} 5s windows from {indices['filename'].nunique()} files")

    # Get the 66 labeled filenames
    labeled_files = set(gt["filename"].unique())
    indices_labeled = indices[indices["filename"].isin(labeled_files)].copy()
    print(f"  {len(indices_labeled)} windows in labeled files")

    # Merge: windows in indices but not in gt are unlabeled gaps (no-call candidates)
    merged = indices_labeled.merge(
        gt[["filename", "start_sec", "is_call", "n_species", "site"]],
        on=["filename", "start_sec"],
        how="left",
    )
    # Unlabeled gaps: is_call is NaN → treat as no-call
    merged["is_call"] = merged["is_call"].fillna(0).astype(int)
    merged["n_species"] = merged["n_species"].fillna(0).astype(int)
    # Fill site from filename for gaps
    merged["site"] = merged["site"].fillna(merged["filename"].str.extract(r"_(S\d+)_")[0])

    n_call = int(merged["is_call"].sum())
    n_nocall = len(merged) - n_call
    print(f"  After including gaps: {n_call} call, {n_nocall} no-call ({len(merged)} total)")

    # --- Step 1: Raw index evaluation ---
    print("\n=== Step 1: Individual index AUROC for call/no-call ===")
    y_true = merged["is_call"].values

    feature_cols = [f"{feat}_{agg}" for feat in INDICES_FEATURES for agg in ["mean", "max", "std"]]
    results = {}

    for col in feature_cols:
        scores = merged[col].values
        # For spectral_flatness, lower = more tonal = call, so negate
        if "flatness" in col:
            scores = -scores

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            auroc = float("nan")
            ap = float("nan")
        else:
            auroc = roc_auc_score(y_true, scores)
            ap = average_precision_score(y_true, scores)

        results[col] = {"auroc": auroc, "ap": ap}
        print(f"  {col:<30} AUROC={auroc:.4f}  AP={ap:.4f}")

    # --- Step 2: Combined logistic regression (LOSO CV) ---
    print("\n=== Step 2: Logistic regression on combined indices ===")

    # Use mean features for the combined model
    X_cols = [f"{feat}_mean" for feat in INDICES_FEATURES]
    X = merged[X_cols].values.astype(np.float32)
    y = merged["is_call"].values
    sites = merged["site"].values

    # Negate flatness so all features are positively correlated with call
    flatness_idx = X_cols.index("spectral_flatness_mean")
    X[:, flatness_idx] = -X[:, flatness_idx]

    # LOSO CV on non-S22 sites — test multiple classifiers
    test_mask = sites == TEST_SITE
    train_sites = sorted(set(sites[~test_mask]))

    classifiers = {
        "logistic": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0),
        "ridge": lambda: RidgeClassifier(class_weight="balanced", alpha=1.0),
        "tree_d3": lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced"),
        "tree_d5": lambda: DecisionTreeClassifier(max_depth=5, class_weight="balanced"),
        "gbdt": lambda: GradientBoostingClassifier(n_estimators=50, max_depth=2, learning_rate=0.1),
    }

    clf_results = {}

    for clf_name, clf_factory in classifiers.items():
        print(f"\n  --- {clf_name} ---")

        cv_aurocs = []
        cv_f1s = []
        for held_out in train_sites:
            val_mask = sites == held_out
            train_mask = ~test_mask & ~val_mask

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            X_val = scaler.transform(X[val_mask])

            clf = clf_factory()
            clf.fit(X_train, y[train_mask])

            if hasattr(clf, "predict_proba"):
                y_val_prob = clf.predict_proba(X_val)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_val_prob = clf.decision_function(X_val)
            else:
                y_val_prob = clf.predict(X_val).astype(float)

            y_val = y[val_mask]

            if y_val.sum() > 0 and y_val.sum() < len(y_val):
                auroc = roc_auc_score(y_val, y_val_prob)
                f1 = f1_score(y_val, (y_val_prob >= 0.5).astype(int)) if hasattr(clf, "predict_proba") else f1_score(y_val, clf.predict(X_val))
                cv_aurocs.append(auroc)
                cv_f1s.append(f1)
                print(f"    Fold {held_out}: AUROC={auroc:.4f} F1={f1:.4f} "
                      f"(n={len(y_val)}, pos={int(y_val.sum())})")
            else:
                print(f"    Fold {held_out}: skipped (all same class)")

        if cv_aurocs:
            print(f"    Mean CV AUROC: {np.mean(cv_aurocs):.4f}  F1: {np.mean(cv_f1s):.4f}")

        # Train on all non-S22, evaluate on S22
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[~test_mask])
        X_test = scaler.transform(X[test_mask])

        clf = clf_factory()
        clf.fit(X_train, y[~test_mask])

        if hasattr(clf, "predict_proba"):
            y_test_prob = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_test_prob = clf.decision_function(X_test)
        else:
            y_test_prob = clf.predict(X_test).astype(float)

        y_test = y[test_mask]

        if y_test.sum() > 0 and y_test.sum() < len(y_test):
            test_auroc = roc_auc_score(y_test, y_test_prob)
            test_ap = average_precision_score(y_test, y_test_prob)
            print(f"    S22 test: AUROC={test_auroc:.4f} AP={test_ap:.4f}")

            # Confusion at 0.5 (or decision boundary for ridge)
            if hasattr(clf, "predict_proba"):
                tp, fp, fn, tn, prec, rec, f1 = confusion(y_test, y_test_prob, 0.5)
            else:
                y_pred = clf.predict(X_test)
                tp = int(((y_pred == 1) & (y_test == 1)).sum())
                fp = int(((y_pred == 1) & (y_test == 0)).sum())
                fn = int(((y_pred == 0) & (y_test == 1)).sum())
                tn = int(((y_pred == 0) & (y_test == 0)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"    S22 @0.5: TP={tp} FP={fp} FN={fn} TN={tn} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        else:
            test_auroc = float("nan")
            print(f"    S22 test: all same class")

        clf_results[clf_name] = {
            "cv_aurocs": cv_aurocs,
            "test_auroc": float(test_auroc) if not np.isnan(test_auroc) else None,
        }

        # Feature importance
        if hasattr(clf, "coef_"):
            coefs = clf.coef_[0] if clf.coef_.ndim > 1 else clf.coef_
            print(f"    Coefficients:")
            for col, coef in zip(X_cols, coefs):
                print(f"      {col:<30} {coef:>8.4f}")
        elif hasattr(clf, "feature_importances_"):
            print(f"    Feature importances:")
            for col, imp in zip(X_cols, clf.feature_importances_):
                print(f"      {col:<30} {imp:>8.4f}")

    # --- Step 3: Isotonic calibration of spectral contrast ---
    print("\n=== Step 3: Isotonic calibration (spectral_contrast_mean) ===")

    sc = merged["spectral_contrast_mean"].values
    y_all = merged["is_call"].values
    sites_all = merged["site"].values

    # LOSO CV: fit isotonic on train folds, predict on held-out + S22
    test_mask_iso = sites_all == TEST_SITE
    train_sites_iso = sorted(set(sites_all[~test_mask_iso]))

    # Collect OOF predictions for CV sites
    oof_probs = np.full(len(sc), np.nan)
    for held_out in train_sites_iso:
        val_mask = sites_all == held_out
        train_mask = ~test_mask_iso & ~val_mask

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(sc[train_mask], y_all[train_mask])
        oof_probs[val_mask] = iso.predict(sc[val_mask])

    # Final model: train on all non-S22, evaluate on S22
    iso_final = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_final.fit(sc[~test_mask_iso], y_all[~test_mask_iso])
    s22_probs = iso_final.predict(sc[test_mask_iso])
    y_s22 = y_all[test_mask_iso]

    if y_s22.sum() > 0 and y_s22.sum() < len(y_s22):
        iso_auroc = roc_auc_score(y_s22, s22_probs)
        iso_ap = average_precision_score(y_s22, s22_probs)
        iso_brier = brier_score_loss(y_s22, s22_probs)
        print(f"  S22 test: AUROC={iso_auroc:.4f} AP={iso_ap:.4f} Brier={iso_brier:.4f}")

        print(f"\n  S22 confusion at thresholds (calibrated):")
        for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            tp, fp, fn, tn, prec, rec, f1 = confusion(y_s22, s22_probs, t)
            print(f"    @{t:.1f}: TP={tp:>4} FP={fp:>4} FN={fn:>4} TN={tn:>4} "
                  f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_s22, s22_probs, n_bins=5, strategy="quantile")
            print(f"\n  Calibration curve (S22, 5 bins):")
            print(f"    {'Predicted':>10} {'Observed':>10} {'Gap':>8}")
            for pt, pp in zip(prob_true, prob_pred):
                print(f"    {pp:>10.3f} {pt:>10.3f} {abs(pt-pp):>8.3f}")
        except Exception as e:
            print(f"  Calibration curve failed: {e}")
    else:
        iso_auroc = float("nan")
        print(f"  S22: all same class")

    # Save the isotonic model mapping for later use
    # The model is defined by (X_, y_) pairs — the learned step function
    iso_mapping = {
        "X_thresholds": iso_final.X_thresholds_.tolist() if hasattr(iso_final, "X_thresholds_") else [],
        "y_thresholds": iso_final.y_thresholds_.tolist() if hasattr(iso_final, "y_thresholds_") else [],
    }

    # Raw vs calibrated comparison
    raw_auroc = results["spectral_contrast_mean"]["auroc"]
    print(f"\n  Raw spectral_contrast AUROC:        {raw_auroc:.4f}")
    print(f"  Isotonic-calibrated AUROC:           {iso_auroc:.4f}")
    print(f"  (AUROC unchanged — isotonic preserves ranking, only calibrates probabilities)")

    # --- Step 4: Compare with model-based call scores ---
    print("\n=== Step 4: Comparison ===")
    print(f"  {'Method':<45} {'S22 AUROC':>10}")
    print(f"  {'-'*57}")

    best_idx = max(results.items(), key=lambda x: x[1]["auroc"])
    print(f"  {'Best index (' + best_idx[0] + ')':<45} {best_idx[1]['auroc']:>10.4f}")
    print(f"  {'Isotonic(spectral_contrast)':<45} {iso_auroc:>10.4f}")
    for clf_name, cr in clf_results.items():
        auroc_str = f"{cr['test_auroc']:.4f}" if cr['test_auroc'] else "N/A"
        print(f"  {('Combined ' + clf_name):<45} {auroc_str:>10}")
    print(f"  {'Ensemble top-5 sum (from prev)':<45} {'0.9005':>10}")
    print(f"  {'Ensemble max (from prev)':<45} {'0.8701':>10}")

    # Save results
    all_results = {
        "individual_indices": results,
        "classifiers": clf_results,
        "isotonic_calibration": {
            "test_auroc": float(iso_auroc) if not np.isnan(iso_auroc) else None,
            "mapping": iso_mapping,
        },
        "data": {
            "n_call": n_call,
            "n_nocall": n_nocall,
            "n_total": len(merged),
            "n_labeled_gaps": n_nocall - int((1 - gt["is_call"]).sum()),
        }
    }
    with open(args.output / "indices_callnocall.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {args.output / 'indices_callnocall.json'}")


if __name__ == "__main__":
    main()
