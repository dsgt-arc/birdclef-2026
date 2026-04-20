#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from birdclef_107.data import load_labeled_soundscape, loso_folds
from birdclef_107.eval import evaluate_loso
from birdclef_106.data_loading import list_focal_species


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an ensemble of saved LOSO checkpoints via cached fold predictions.")
    p.add_argument("roots", nargs="+", help="Checkpoint roots containing per-fold preds.npy/labels.npy")
    p.add_argument("--features-dir", default="~/scratch/birdclef-2026/features/exp106")
    p.add_argument("--output-json", default=None)
    p.add_argument(
        "--aggregation",
        choices=["mean", "median", "logit_mean"],
        default="mean",
        help="How to combine per-model probabilities.",
    )
    return p.parse_args()


def _aggregate_predictions(preds: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(preds, axis=0)
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "median":
        return np.median(stack, axis=0)
    if mode == "logit_mean":
        eps = 1e-6
        stack = np.clip(stack, eps, 1.0 - eps)
        logits = np.log(stack / (1.0 - stack))
        return 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def main() -> None:
    args = parse_args()
    roots = [Path(r).expanduser() for r in args.roots]
    for root in roots:
        if not root.exists():
            raise SystemExit(f"Missing checkpoint root: {root}")

    fold_sets = []
    for root in roots:
        fold_sets.append({p.parent.name for p in root.glob("*/preds.npy")})
    common_folds = sorted(set.intersection(*fold_sets))
    if not common_folds:
        raise SystemExit("No common fold predictions found across checkpoint roots")

    sc_data = load_labeled_soundscape(args.features_dir)
    folds_by_site = {f.left_out_site: f for f in loso_folds(sc_data.sites)}

    fold_y_true = []
    fold_y_pred = []
    eval_folds = []
    for site in common_folds:
        preds = []
        labels_ref = None
        for root in roots:
            fold_dir = root / site
            preds.append(np.load(fold_dir / "preds.npy"))
            labels = np.load(fold_dir / "labels.npy")
            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                raise SystemExit(f"Label mismatch for fold {site} across roots")
        fold_y_true.append(labels_ref)
        fold_y_pred.append(_aggregate_predictions(preds, args.aggregation))
        eval_folds.append(folds_by_site[site])

    results = evaluate_loso(
        eval_folds,
        fold_y_true,
        fold_y_pred,
        sc_data.species_list,
        focal_species=set(list_focal_species(args.features_dir)),
    )

    if args.output_json:
        out = Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))

    print(f"METRIC pooled_auroc={results['pooled_auroc']}")
    for key in ["without_focal", "with_focal", "bird", "amphibian", "insect"]:
        if key in results.get("subsets", {}):
            print(f"METRIC {key}_auroc={results['subsets'][key]}")
    if "S22" in results.get("per_fold", {}):
        print(f"METRIC s22_fold_auroc={results['per_fold']['S22']['auroc']}")
    print(f"METRIC n_val_windows={results.get('n_windows', 0)}")
    print(f"METRIC n_folds={results.get('n_folds', 0)}")


if __name__ == "__main__":
    main()
