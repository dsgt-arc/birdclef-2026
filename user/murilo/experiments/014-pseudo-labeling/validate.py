#!/usr/bin/env python3
"""Validate pseudo-labels against ground truth for labeled soundscapes.

Compares ensemble and per-model pseudo-label quality against ground truth
annotations. Reports AUROC and cmAP at overall, per-taxon, and per-species levels.

Usage:
    python validate.py --pseudo-dir shared/pseudo-labels/.../labeled/
    python validate.py --pseudo-dir shared/pseudo-labels/.../labeled/ --per-species
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_ground_truth(data_dir, species_list):
    """Load ground truth labels as binary matrix aligned with pseudo-label segments.

    Returns:
        gt_df: DataFrame with columns [filename, end_sec, *species_list] (binary 0/1)
    """
    labels_csv = Path(data_dir) / "train_soundscapes_labels.csv"
    gt = pd.read_csv(labels_csv)

    # Convert end time "HH:MM:SS" to seconds
    def time_to_sec(t):
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    gt["end_sec"] = gt["end"].apply(time_to_sec)

    # Build binary matrix: (n_segments, n_species)
    species_set = set(species_list)
    rows = []
    for (fn, end_sec), group in gt.groupby(["filename", "end_sec"]):
        labels = set()
        for label_str in group["primary_label"]:
            for sp in str(label_str).split(";"):
                sp = sp.strip()
                if sp in species_set:
                    labels.add(sp)
        row = {"filename": fn, "end_sec": end_sec}
        for sp in species_list:
            row[sp] = 1 if sp in labels else 0
        rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(y_true, y_prob, species_list, label=""):
    """Compute AUROC and cmAP for a prediction matrix.

    Args:
        y_true: (n_segments, n_species) binary ground truth
        y_prob: (n_segments, n_species) predicted probabilities
        species_list: list of species names (for per-species reporting)
        label: label for logging

    Returns:
        dict with overall and per-species metrics
    """
    n_species = y_true.shape[1]

    # Per-species AUROC (only for species with both positives and negatives)
    per_species = {}
    valid_aurocs = []
    valid_aps = []
    for i, sp in enumerate(species_list):
        col_true = y_true[:, i]
        col_prob = y_prob[:, i]
        n_pos = col_true.sum()
        n_neg = len(col_true) - n_pos
        if n_pos > 0 and n_neg > 0:
            auroc = roc_auc_score(col_true, col_prob)
            ap = average_precision_score(col_true, col_prob)
            valid_aurocs.append(auroc)
            valid_aps.append(ap)
            per_species[sp] = {"auroc": auroc, "ap": ap, "n_pos": int(n_pos)}
        elif n_pos > 0:
            per_species[sp] = {
                "auroc": None,
                "ap": None,
                "n_pos": int(n_pos),
                "note": "all-positive (no negatives)",
            }

    macro_auroc = np.mean(valid_aurocs) if valid_aurocs else 0.0
    cmap = np.mean(valid_aps) if valid_aps else 0.0

    return {
        "macro_auroc": macro_auroc,
        "cmap": cmap,
        "n_evaluated_species": len(valid_aurocs),
        "n_total_species": n_species,
        "per_species": per_species,
    }


def compute_taxon_metrics(y_true, y_prob, species_list, taxonomy):
    """Compute metrics grouped by taxon class (Aves, Insecta, etc.)."""
    sp_to_class = dict(
        zip(
            taxonomy["primary_label"].astype(str),
            taxonomy["class_name"],
            strict=False,
        )
    )

    # Group species indices by taxon
    taxon_indices = {}
    for i, sp in enumerate(species_list):
        cls = sp_to_class.get(sp, "Unknown")
        taxon_indices.setdefault(cls, []).append(i)

    taxon_metrics = {}
    for taxon, indices in sorted(taxon_indices.items()):
        t_true = y_true[:, indices]
        t_prob = y_prob[:, indices]
        t_species = [species_list[i] for i in indices]

        # Count species with positives in GT
        n_with_gt = sum(1 for i in indices if y_true[:, i].sum() > 0)

        metrics = compute_metrics(t_true, t_prob, t_species)
        taxon_metrics[taxon] = {
            "macro_auroc": metrics["macro_auroc"],
            "cmap": metrics["cmap"],
            "n_species_total": len(indices),
            "n_species_in_gt": n_with_gt,
            "n_species_evaluated": metrics["n_evaluated_species"],
        }

    return taxon_metrics


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pseudo-dir",
        required=True,
        help="Path to pseudo-label output dir (contains pseudo_labels.parquet + logits/)",
    )
    parser.add_argument(
        "--data-dir",
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026",
        help="Path to BirdCLEF 2026 data directory",
    )
    parser.add_argument(
        "--per-species", action="store_true", help="Print per-species AUROC breakdown"
    )
    args = parser.parse_args()

    pseudo_dir = Path(args.pseudo_dir)
    data_dir = Path(args.data_dir)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    # Pseudo-labels (ensemble) — supports single file or multiple shards
    parquet_files = sorted(pseudo_dir.glob("pseudo_labels*.parquet"))
    if not parquet_files:
        logger.error(f"No pseudo_labels*.parquet found in {pseudo_dir}")
        return
    pl = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    pl = pl.sort_values(["filename", "end_sec"]).reset_index(drop=True)
    species_list = list(pl.columns[2:])  # skip filename, end_sec
    logger.info(
        f"Pseudo-labels: {pl.shape[0]} segments from {len(parquet_files)} file(s), {len(species_list)} species"
    )

    # Ground truth
    gt = load_ground_truth(data_dir, species_list)
    logger.info(f"Ground truth: {gt.shape[0]} labeled segments")

    # Taxonomy
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")

    # Align: inner join on (filename, end_sec)
    merged = pl.merge(gt, on=["filename", "end_sec"], how="left", suffixes=("_pred", "_gt"))

    # For segments with no GT annotation, treat as all-negative
    n_segments = len(merged)
    y_prob = np.zeros((n_segments, len(species_list)), dtype=np.float32)
    y_true = np.zeros((n_segments, len(species_list)), dtype=np.int32)

    for i, sp in enumerate(species_list):
        y_prob[:, i] = merged[f"{sp}_pred"].values
        gt_col = f"{sp}_gt"
        if gt_col in merged.columns:
            y_true[:, i] = merged[gt_col].fillna(0).astype(int).values

    n_gt_segments = gt.shape[0]
    n_unlabeled = n_segments - n_gt_segments
    logger.info(
        f"Aligned: {n_segments} total ({n_gt_segments} labeled, {n_unlabeled} unannotated → all-negative)"
    )

    # Count GT species
    gt_species_mask = y_true.sum(axis=0) > 0
    n_gt_species = gt_species_mask.sum()
    logger.info(f"Species with GT positives: {n_gt_species}/{len(species_list)}")

    # -----------------------------------------------------------------------
    # Ensemble metrics
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ENSEMBLE (mean of teacher sigmoid predictions)")
    print("=" * 70)

    ensemble_metrics = compute_metrics(y_true, y_prob, species_list, "ensemble")
    print(
        f"  Macro AUROC:  {ensemble_metrics['macro_auroc']:.4f}  ({ensemble_metrics['n_evaluated_species']} species)"
    )
    print(f"  cmAP:         {ensemble_metrics['cmap']:.4f}")

    # Per-taxon
    taxon_metrics = compute_taxon_metrics(y_true, y_prob, species_list, taxonomy)
    print(f"\n  {'Taxon':<12} {'AUROC':>8} {'cmAP':>8} {'Species (eval/GT/total)':>25}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*25}")
    for taxon, tm in sorted(taxon_metrics.items(), key=lambda x: -x[1]["macro_auroc"]):
        print(
            f"  {taxon:<12} {tm['macro_auroc']:>8.4f} {tm['cmap']:>8.4f} "
            f"{tm['n_species_evaluated']:>8}/{tm['n_species_in_gt']}/{tm['n_species_total']}"
        )

    # -----------------------------------------------------------------------
    # Per-model metrics (from saved logits)
    # -----------------------------------------------------------------------

    logits_dir = pseudo_dir / "logits"
    if logits_dir.exists():
        print("\n" + "=" * 70)
        print("PER-MODEL COMPARISON")
        print("=" * 70)

        # Load config to get teacher names
        config_path = pseudo_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            teacher_names = config["run_metadata"]["teachers"]
        else:
            teacher_names = [p.stem for p in logits_dir.glob("*.npz")]

        model_results = {}
        for tname in teacher_names:
            npz_path = logits_dir / f"{tname}.npz"
            if not npz_path.exists():
                logger.warning(f"Logits not found: {npz_path}")
                continue

            data = np.load(npz_path, allow_pickle=True)
            logits = data["logits"]  # (n_segments, 234)
            probs = sigmoid(logits)

            model_metrics = compute_metrics(y_true, probs, species_list, tname)
            model_results[tname] = model_metrics

            print(f"\n  {tname}:")
            print(f"    Macro AUROC:  {model_metrics['macro_auroc']:.4f}")
            print(f"    cmAP:         {model_metrics['cmap']:.4f}")

            # Per-taxon for this model
            model_taxon = compute_taxon_metrics(y_true, probs, species_list, taxonomy)
            for taxon, tm in sorted(model_taxon.items(), key=lambda x: -x[1]["macro_auroc"]):
                print(f"    {taxon:<12} AUROC={tm['macro_auroc']:.4f}  cmAP={tm['cmap']:.4f}")

        # Summary comparison
        print(f"\n  {'Model':<30} {'AUROC':>8} {'cmAP':>8}")
        print(f"  {'-'*30} {'-'*8} {'-'*8}")
        print(
            f"  {'Ensemble (mean)':<30} {ensemble_metrics['macro_auroc']:>8.4f} {ensemble_metrics['cmap']:>8.4f}"
        )
        for tname, m in model_results.items():
            print(f"  {tname:<30} {m['macro_auroc']:>8.4f} {m['cmap']:>8.4f}")

    # -----------------------------------------------------------------------
    # Per-species breakdown (optional)
    # -----------------------------------------------------------------------

    if args.per_species:
        sp_to_class = dict(
            zip(
                taxonomy["primary_label"].astype(str),
                taxonomy["class_name"],
                strict=False,
            )
        )
        sp_to_common = dict(
            zip(
                taxonomy["primary_label"].astype(str),
                taxonomy["common_name"],
                strict=False,
            )
        )

        print("\n" + "=" * 70)
        print("PER-SPECIES AUROC (ensemble, species with GT positives)")
        print("=" * 70)

        # Sort by AUROC descending
        species_data = []
        for sp, sm in ensemble_metrics["per_species"].items():
            if sm["auroc"] is not None:
                species_data.append(
                    {
                        "species": sp,
                        "common_name": sp_to_common.get(sp, "?"),
                        "class": sp_to_class.get(sp, "?"),
                        "auroc": sm["auroc"],
                        "ap": sm["ap"],
                        "n_pos": sm["n_pos"],
                    }
                )

        species_data.sort(key=lambda x: -x["auroc"])

        print(
            f"\n  {'Species':<10} {'Common Name':<30} {'Class':<10} {'AUROC':>7} {'AP':>7} {'N+':>5}"
        )
        print(f"  {'-'*10} {'-'*30} {'-'*10} {'-'*7} {'-'*7} {'-'*5}")
        for sd in species_data:
            print(
                f"  {sd['species']:<10} {sd['common_name'][:29]:<30} {sd['class']:<10} "
                f"{sd['auroc']:>7.4f} {sd['ap']:>7.4f} {sd['n_pos']:>5}"
            )

        # Bottom 10 worst
        print("\n  Worst 10 species:")
        for sd in species_data[-10:]:
            print(
                f"    {sd['species']:<10} {sd['common_name'][:29]:<30} {sd['class']:<10} "
                f"AUROC={sd['auroc']:.4f} AP={sd['ap']:.4f} (n={sd['n_pos']})"
            )

    print()


if __name__ == "__main__":
    main()
