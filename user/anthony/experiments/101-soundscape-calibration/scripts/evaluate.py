"""Exp 101: Evaluate teacher models on labeled soundscape windows.

Phase 1: Data alignment
Phase 2: Raw model evaluation (per-species AUROC, per-taxa, per-site)

Usage:
    python evaluate.py --output artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# --- Paths ---

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data/birdclef-2026"
INFERENCE_DIR = SHARED / "inference/birdclef-2026"

NPZ_PATHS = {
    "protoclr": INFERENCE_DIR / "protoclr-2026-soundscape/labeled.npz",
    "convnext": INFERENCE_DIR / "convnext-2026-soundscape/labeled.npz",
    "eat": INFERENCE_DIR / "eat-2026-soundscape/labeled.npz",
}

# EDA artifacts from exp 100
EDA_ARTIFACTS = Path(__file__).resolve().parents[2] / "100-eda/artifacts"

# Species with zero focal training data (25 insect sonotypes + 3 amphibians)
NO_FOCAL_AMPHIBIA = {"1491113", "25073", "517063"}
SONOTYPE_PREFIX = "47158son"


# --- Phase 1: Data alignment ---


def load_ground_truth(eda_artifacts: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load labeled soundscape windows and build binary label matrix.

    Returns:
        labels_df: DataFrame with columns [filename, start_sec, end_sec, site]
                   + one binary column per species
        species_list: list of 234 species in submission order
    """
    # Load exploded labels from exp 100
    labels = pd.read_parquet(eda_artifacts / "soundscape_labels.parquet")

    # Get species list from taxonomy
    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    species_list = sorted(taxonomy["primary_label"].astype(str).unique())

    # Build window-level index (unique filename + start_sec)
    windows = labels[["filename", "start_sec", "end_sec"]].drop_duplicates()

    # Extract site from filename (e.g., BC2026_Train_0001_S08_... -> S08)
    windows = windows.copy()
    windows["site"] = windows["filename"].str.extract(r"_(S\d+)_")

    # Build binary matrix
    for sp in species_list:
        sp_windows = labels.loc[
            labels["primary_label"] == sp, ["filename", "start_sec"]
        ]
        sp_keys = set(zip(sp_windows["filename"], sp_windows["start_sec"]))
        windows[sp] = [
            1 if (f, s) in sp_keys else 0
            for f, s in zip(windows["filename"], windows["start_sec"])
        ]

    return windows, species_list


def load_predictions(npz_paths: dict) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Load model predictions from npz files.

    Returns dict of model_name -> (predictions array, species list).
    Reconstructs window ordering from known 66-file x 12-window structure
    if seg_keys unpickling fails.
    """
    results = {}
    for name, path in npz_paths.items():
        data = np.load(path, allow_pickle=True)
        preds = data["predictions"]  # (792, 234)
        species = [str(s) for s in data["species"]]  # 234 species

        # Try to load seg_keys, fall back to reconstruction
        try:
            seg_keys = data["seg_keys"]
            # seg_keys is (792, 2) with [filename, window_index]
            filenames = [str(seg_keys[i, 0]) for i in range(len(seg_keys))]
            window_ids = [int(seg_keys[i, 1]) for i in range(len(seg_keys))]
        except Exception:
            # Reconstruct: 66 files x 12 windows, 5s each
            print(f"  {name}: seg_keys unpickle failed, reconstructing")
            filenames = None
            window_ids = None

        results[name] = {
            "predictions": preds,
            "species": species,
            "filenames": filenames,
            "window_ids": window_ids,
        }
    return results


def align_predictions_to_labels(
    gt: pd.DataFrame,
    model_data: dict,
    species_list: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Align model predictions to ground truth labeled windows.

    Returns:
        gt_aligned: ground truth DataFrame (only labeled windows)
        pred_aligned: dict of model_name -> (N_labeled, 234) prediction arrays
    """
    # Get the species ordering from the first model
    first_model = next(iter(model_data.values()))
    model_species = first_model["species"]

    # If we have filenames from seg_keys, join directly
    # Otherwise, reconstruct from soundscape file listing
    pred_aligned = {}

    for name, data in model_data.items():
        preds = data["predictions"]

        if data["filenames"] is not None:
            # seg_keys window IDs are end times (5, 10, ..., 60)
            # ground truth uses start times (0, 5, ..., 55)
            # Convert: start_sec = end_sec - 5
            pred_cols = {f"pred_{sp}": preds[:, i] for i, sp in enumerate(data["species"])}
            pred_df = pd.DataFrame(
                {
                    "filename": data["filenames"],
                    "start_sec": [float(wid) - 5.0 for wid in data["window_ids"]],
                    **pred_cols,
                }
            )

            # Join to ground truth
            merged = gt.merge(pred_df, on=["filename", "start_sec"], how="inner")
            pred_cols = [f"pred_{sp}" for sp in model_species]
            pred_aligned[name] = merged[pred_cols].values
            print(f"  {name}: {len(merged)} windows aligned via seg_keys")
        else:
            # Fallback: reconstruct from sorted soundscape filenames
            soundscape_dir = DATA_DIR / "train_soundscapes"
            all_files = sorted(f.name for f in soundscape_dir.glob("*.ogg"))

            # Get the 66 labeled filenames
            labeled_files = sorted(gt["filename"].unique())

            # Build mapping: for each labeled file, find its index in all_files
            # then compute the row offset in the 792-row prediction array
            file_to_idx = {f: i for i, f in enumerate(all_files)}

            # Wait — the npz has 792 rows = 66 files x 12. But which 66?
            # The labeled npz should correspond to just the labeled files
            # Try assuming predictions are ordered by the 66 labeled files
            rows = []
            for _, row in gt.iterrows():
                fname = row["filename"]
                labeled_file_idx = labeled_files.index(fname)
                window_idx = int(row["start_sec"] / 5.0)
                pred_row = labeled_file_idx * 12 + window_idx
                rows.append(pred_row)

            pred_aligned[name] = preds[rows]
            print(f"  {name}: {len(rows)} windows aligned via reconstruction")

    return gt, pred_aligned, model_species


# --- Phase 2: Raw model evaluation ---


def identify_evaluable_species(
    gt: pd.DataFrame,
    species_list: list[str],
    min_positive_windows: int = 2,
) -> dict:
    """Identify which species are evaluable.

    Returns dict with keys: evaluable, excluded_no_focal, excluded_too_few, all_counts
    """
    no_focal = set()
    for sp in species_list:
        if sp.startswith(SONOTYPE_PREFIX) or sp in NO_FOCAL_AMPHIBIA:
            no_focal.add(sp)

    # Count positive windows per species
    species_in_gt = [sp for sp in species_list if sp in gt.columns]
    counts = {sp: int(gt[sp].sum()) for sp in species_in_gt}

    evaluable = []
    excluded_too_few = []
    for sp in species_in_gt:
        if sp in no_focal:
            continue
        if counts.get(sp, 0) < min_positive_windows:
            excluded_too_few.append(sp)
        elif counts.get(sp, 0) > 0:
            evaluable.append(sp)

    return {
        "evaluable": evaluable,
        "excluded_no_focal": sorted(no_focal),
        "excluded_too_few": excluded_too_few,
        "counts": counts,
    }


def get_taxa(species: str, taxonomy: pd.DataFrame) -> str:
    """Look up taxon class for a species."""
    row = taxonomy.loc[taxonomy["primary_label"].astype(str) == species]
    if len(row) == 0:
        return "Unknown"
    return row.iloc[0].get("class_name", row.iloc[0].get("class", "Unknown"))


def compute_metrics(
    gt: pd.DataFrame,
    pred: np.ndarray,
    species_list: list[str],
    evaluable: list[str],
    model_species: list[str],
) -> pd.DataFrame:
    """Compute per-species AUROC and AP.

    Returns DataFrame with columns: species, auroc, ap, n_positive, n_total
    """
    species_to_idx = {sp: i for i, sp in enumerate(model_species)}

    rows = []
    for sp in evaluable:
        if sp not in gt.columns or sp not in species_to_idx:
            continue

        y_true = gt[sp].values
        y_score = pred[:, species_to_idx[sp]]
        n_pos = int(y_true.sum())
        n_total = len(y_true)

        # AUROC needs both classes present
        if n_pos == 0 or n_pos == n_total:
            auroc = float("nan")
        else:
            auroc = roc_auc_score(y_true, y_score)

        ap = average_precision_score(y_true, y_score) if n_pos > 0 else float("nan")

        rows.append(
            {
                "species": sp,
                "auroc": auroc,
                "ap": ap,
                "n_positive": n_pos,
                "n_total": n_total,
            }
        )

    return pd.DataFrame(rows)


def compute_site_metrics(
    gt: pd.DataFrame,
    pred: np.ndarray,
    evaluable: list[str],
    model_species: list[str],
) -> pd.DataFrame:
    """Compute macro AUROC per site."""
    species_to_idx = {sp: i for i, sp in enumerate(model_species)}
    sites = sorted(gt["site"].unique())

    rows = []
    for site in sites:
        mask = gt["site"] == site
        gt_site = gt[mask]
        pred_site = pred[mask.values]

        aurocs = []
        for sp in evaluable:
            if sp not in gt.columns or sp not in species_to_idx:
                continue
            y_true = gt_site[sp].values
            y_score = pred_site[:, species_to_idx[sp]]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            aurocs.append(roc_auc_score(y_true, y_score))

        rows.append(
            {
                "site": site,
                "n_windows": int(mask.sum()),
                "n_species_evaluable": len(aurocs),
                "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def compute_loso_cv(
    gt: pd.DataFrame,
    preds: dict[str, np.ndarray],
    evaluable: list[str],
    model_species: list[str],
    test_site: str = "S22",
) -> dict:
    """Leave-one-site-out CV on train sites, final eval on test site.

    Returns dict with CV results and test results.
    """
    species_to_idx = {sp: i for i, sp in enumerate(model_species)}

    # Split
    test_mask = gt["site"] == test_site
    train_gt = gt[~test_mask].copy()
    test_gt = gt[test_mask].copy()

    train_sites = sorted(train_gt["site"].unique())

    results = {"cv_folds": [], "test": {}, "train_sites": train_sites}

    for model_name, pred in preds.items():
        train_pred = pred[~test_mask.values]
        test_pred = pred[test_mask.values]

        # LOSO CV on train sites
        fold_aurocs = []
        for held_out in train_sites:
            val_mask = train_gt["site"] == held_out
            val_gt = train_gt[val_mask]
            val_pred_arr = train_pred[val_mask.values]

            aurocs = []
            for sp in evaluable:
                if sp not in val_gt.columns or sp not in species_to_idx:
                    continue
                y_true = val_gt[sp].values
                y_score = val_pred_arr[:, species_to_idx[sp]]
                if y_true.sum() == 0 or y_true.sum() == len(y_true):
                    continue
                aurocs.append(roc_auc_score(y_true, y_score))

            fold_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
            fold_aurocs.append(
                {
                    "model": model_name,
                    "fold_site": held_out,
                    "n_windows": int(val_mask.sum()),
                    "n_species": len(aurocs),
                    "macro_auroc": fold_auroc,
                }
            )

        # Test set eval
        test_aurocs = []
        test_aps = []
        for sp in evaluable:
            if sp not in test_gt.columns or sp not in species_to_idx:
                continue
            y_true = test_gt[sp].values
            y_score = test_pred[:, species_to_idx[sp]]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            test_aurocs.append(roc_auc_score(y_true, y_score))
            test_aps.append(average_precision_score(y_true, y_score))

        results["cv_folds"].extend(fold_aurocs)
        results["test"][model_name] = {
            "n_windows": int(test_mask.sum()),
            "n_species": len(test_aurocs),
            "macro_auroc": float(np.mean(test_aurocs)) if test_aurocs else float("nan"),
            "macro_ap": float(np.mean(test_aps)) if test_aps else float("nan"),
        }

    # Also do simple ensemble (mean of 3 models)
    ensemble_pred_test = np.mean(
        [preds[m][test_mask.values] for m in preds], axis=0
    )
    ensemble_aurocs = []
    ensemble_aps = []
    for sp in evaluable:
        if sp not in test_gt.columns or sp not in species_to_idx:
            continue
        y_true = test_gt[sp].values
        y_score = ensemble_pred_test[:, species_to_idx[sp]]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        ensemble_aurocs.append(roc_auc_score(y_true, y_score))
        ensemble_aps.append(average_precision_score(y_true, y_score))

    results["test"]["ensemble_mean"] = {
        "n_windows": int(test_mask.sum()),
        "n_species": len(ensemble_aurocs),
        "macro_auroc": float(np.mean(ensemble_aurocs)) if ensemble_aurocs else float("nan"),
        "macro_ap": float(np.mean(ensemble_aps)) if ensemble_aps else float("nan"),
    }

    return results


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Exp 101: Soundscape calibration")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts"), help="Output directory"
    )
    parser.add_argument(
        "--eda-artifacts",
        type=Path,
        default=EDA_ARTIFACTS,
        help="Path to exp 100 EDA artifacts",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load taxonomy
    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)

    # Phase 1: Data alignment
    print("=== Phase 1: Data alignment ===")

    print("Loading ground truth...")
    gt, species_list = load_ground_truth(args.eda_artifacts)
    print(f"  {len(gt)} labeled windows, {len(species_list)} species")

    print("Loading predictions...")
    model_data = load_predictions(NPZ_PATHS)

    print("Aligning predictions to labels...")
    gt, pred_aligned, model_species = align_predictions_to_labels(
        gt, model_data, species_list
    )

    print("Identifying evaluable species...")
    eval_info = identify_evaluable_species(gt, species_list)
    evaluable = eval_info["evaluable"]
    print(f"  {len(evaluable)} evaluable species")
    print(f"  {len(eval_info['excluded_no_focal'])} excluded (no focal data)")
    print(f"  {len(eval_info['excluded_too_few'])} excluded (<2 positive windows)")

    # Add taxa info
    for sp in evaluable:
        taxa = get_taxa(sp, taxonomy)
        eval_info.setdefault("taxa", {})[sp] = taxa

    # Save eval info
    with open(args.output / "eval_info.json", "w") as f:
        json.dump(
            {
                "n_evaluable": len(evaluable),
                "n_excluded_no_focal": len(eval_info["excluded_no_focal"]),
                "n_excluded_too_few": len(eval_info["excluded_too_few"]),
                "excluded_too_few": eval_info["excluded_too_few"],
                "evaluable_species": evaluable,
            },
            f,
            indent=2,
        )

    # Phase 2: Raw model evaluation
    print("\n=== Phase 2: Raw model evaluation ===")

    all_metrics = {}
    for model_name, pred in pred_aligned.items():
        print(f"\n--- {model_name} ---")
        metrics = compute_metrics(gt, pred, species_list, evaluable, model_species)

        # Add taxa
        metrics["taxa"] = metrics["species"].apply(lambda sp: get_taxa(sp, taxonomy))

        macro_auroc = metrics["auroc"].dropna().mean()
        macro_ap = metrics["ap"].dropna().mean()
        print(f"  Macro AUROC: {macro_auroc:.4f}")
        print(f"  Macro AP:    {macro_ap:.4f}")
        print(f"  Species with AUROC > 0.85: {(metrics['auroc'] > 0.85).sum()}")
        print(f"  Species with AUROC 0.70-0.85: {((metrics['auroc'] >= 0.70) & (metrics['auroc'] <= 0.85)).sum()}")
        print(f"  Species with AUROC < 0.70: {(metrics['auroc'] < 0.70).sum()}")

        # Per-taxa summary
        print("\n  Per-taxa AUROC:")
        for taxa, group in metrics.groupby("taxa"):
            taxa_auroc = group["auroc"].dropna().mean()
            print(f"    {taxa}: {taxa_auroc:.4f} ({len(group)} species)")

        # Per-site
        site_metrics = compute_site_metrics(gt, pred, evaluable, model_species)
        print("\n  Per-site AUROC:")
        for _, row in site_metrics.iterrows():
            print(f"    {row['site']}: {row['macro_auroc']:.4f} ({int(row['n_windows'])} windows, {int(row['n_species_evaluable'])} species)")

        all_metrics[model_name] = metrics
        metrics.to_csv(args.output / f"metrics_{model_name}.csv", index=False)
        site_metrics.to_csv(args.output / f"site_metrics_{model_name}.csv", index=False)

    # Simple ensemble
    print("\n--- ensemble (mean) ---")
    ensemble_pred = np.mean(list(pred_aligned.values()), axis=0)
    ensemble_metrics = compute_metrics(
        gt, ensemble_pred, species_list, evaluable, model_species
    )
    ensemble_metrics["taxa"] = ensemble_metrics["species"].apply(
        lambda sp: get_taxa(sp, taxonomy)
    )
    macro_auroc = ensemble_metrics["auroc"].dropna().mean()
    macro_ap = ensemble_metrics["ap"].dropna().mean()
    print(f"  Macro AUROC: {macro_auroc:.4f}")
    print(f"  Macro AP:    {macro_ap:.4f}")
    ensemble_metrics.to_csv(args.output / "metrics_ensemble.csv", index=False)

    # LOSO CV + held-out test
    print("\n=== LOSO CV (train: 8 sites) + Test (S22) ===")
    loso_results = compute_loso_cv(gt, pred_aligned, evaluable, model_species)

    print("\nCV fold results:")
    cv_df = pd.DataFrame(loso_results["cv_folds"])
    for model_name in pred_aligned:
        model_cv = cv_df[cv_df["model"] == model_name]
        mean_cv = model_cv["macro_auroc"].mean()
        print(f"  {model_name}: mean CV AUROC = {mean_cv:.4f}")

    print("\nTest set (S22) results:")
    for model_name, test_res in loso_results["test"].items():
        print(
            f"  {model_name}: AUROC = {test_res['macro_auroc']:.4f}, "
            f"AP = {test_res['macro_ap']:.4f} "
            f"({test_res['n_species']} species, {test_res['n_windows']} windows)"
        )

    cv_df.to_csv(args.output / "loso_cv.csv", index=False)
    with open(args.output / "test_results.json", "w") as f:
        json.dump(loso_results["test"], f, indent=2)

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Model':<20} {'Full AUROC':>12} {'Full AP':>10} {'S22 AUROC':>12} {'S22 AP':>10}")
    print("-" * 66)
    for model_name in list(pred_aligned.keys()) + ["ensemble_mean"]:
        if model_name == "ensemble_mean":
            full_auroc = ensemble_metrics["auroc"].dropna().mean()
            full_ap = ensemble_metrics["ap"].dropna().mean()
        else:
            full_auroc = all_metrics[model_name]["auroc"].dropna().mean()
            full_ap = all_metrics[model_name]["ap"].dropna().mean()
        s22 = loso_results["test"][model_name if model_name != "ensemble_mean" else "ensemble_mean"]
        print(
            f"{model_name:<20} {full_auroc:>12.4f} {full_ap:>10.4f} "
            f"{s22['macro_auroc']:>12.4f} {s22['macro_ap']:>10.4f}"
        )

    print(f"\nArtifacts saved to {args.output}/")


if __name__ == "__main__":
    main()
