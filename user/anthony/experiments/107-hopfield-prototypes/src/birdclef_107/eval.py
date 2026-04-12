"""Evaluation harness for experiment 107.

Computes pooled AUROC (primary metric), per-fold breakdown, and subset
AUROCs (bird/amphibian/insect/with-focal/without-focal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from birdclef_107.data import LOSOFold
from birdclef_107.model import PrototypeRetrievalModel


@torch.no_grad()
def predict_fold(
    model: PrototypeRetrievalModel,
    embeddings: np.ndarray,
    site_indices: np.ndarray | None,
    site_to_idx: dict[str, int] | None = None,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Run inference on a fold's test set and return probabilities.

    Args:
        model: trained model in eval mode.
        embeddings: (N, 1536) float32.
        site_indices: (N,) int64 site indices, or None for no-site path.
        batch_size: inference batch size.
        device: torch device string.

    Returns:
        (N, n_species) float32 probability array.
    """
    model.eval()
    model.to(device)
    preds = []
    n = len(embeddings)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = torch.from_numpy(embeddings[start:end]).float().to(device)
        si = None
        if site_indices is not None:
            si = torch.from_numpy(site_indices[start:end]).long().to(device)
        out = model(emb, site_idx=si)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_mask: np.ndarray | None = None,
) -> float:
    """Compute macro-averaged AUROC, skipping species with <2 classes.

    Args:
        y_true: (N, S) binary ground truth.
        y_pred: (N, S) predicted probabilities.
        species_mask: (S,) bool mask to restrict evaluation to a subset.

    Returns:
        Macro-averaged AUROC over evaluable species.
    """
    if species_mask is not None:
        y_true = y_true[:, species_mask]
        y_pred = y_pred[:, species_mask]

    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.sum() == 0 or col.sum() == len(col):
            continue  # skip: only one class present
        aucs.append(roc_auc_score(col, y_pred[:, j]))

    return float(np.mean(aucs)) if aucs else 0.0


def pooled_auroc(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    species_mask: np.ndarray | None = None,
) -> float:
    """Compute pooled AUROC by concatenating all fold predictions.

    This is the primary metric from the proposal: concat all held-out
    predictions across LOSO folds, then compute macro AUROC.
    """
    return compute_auroc(all_y_true, all_y_pred, species_mask)


def build_subset_masks(
    species_list: list[str],
    families_csv: str | Path | None = None,
    focal_species: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build boolean masks for evaluation subsets.

    Returns dict with keys: 'bird', 'amphibian', 'insect', 'with_focal',
    'without_focal'. Each value is (n_species,) bool.
    """
    from birdclef_107.family_utils import DEFAULT_FAMILIES_CSV

    import csv

    csv_path = Path(families_csv) if families_csv else DEFAULT_FAMILIES_CSV

    # Read class_name from families.csv
    species_to_class: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row["primary_label"].strip()
            cls = row.get("class_name", "").strip()
            species_to_class[sp] = cls

    n = len(species_list)
    masks: dict[str, np.ndarray] = {}

    bird_mask = np.zeros(n, dtype=bool)
    amphibian_mask = np.zeros(n, dtype=bool)
    insect_mask = np.zeros(n, dtype=bool)

    for i, sp in enumerate(species_list):
        cls = species_to_class.get(sp, "")
        if cls == "Aves":
            bird_mask[i] = True
        elif cls == "Amphibia":
            amphibian_mask[i] = True
        elif cls == "Insecta":
            insect_mask[i] = True

    masks["bird"] = bird_mask
    masks["amphibian"] = amphibian_mask
    masks["insect"] = insect_mask

    if focal_species is not None:
        with_focal = np.array(
            [sp in focal_species for sp in species_list], dtype=bool
        )
        masks["with_focal"] = with_focal
        masks["without_focal"] = ~with_focal
    else:
        masks["with_focal"] = np.ones(n, dtype=bool)
        masks["without_focal"] = np.zeros(n, dtype=bool)

    return masks


def evaluate_loso(
    folds: list[LOSOFold],
    fold_y_true: list[np.ndarray],
    fold_y_pred: list[np.ndarray],
    species_list: list[str],
    focal_species: set[str] | None = None,
) -> dict:
    """Full LOSO evaluation with pooled AUROC and subset breakdowns.

    Args:
        folds: list of LOSOFold objects (for site names).
        fold_y_true: per-fold ground truth arrays.
        fold_y_pred: per-fold prediction arrays.
        species_list: canonical species ordering.
        focal_species: set of species that have focal data.

    Returns:
        Dict with 'pooled_auroc', 'per_fold', and 'subsets' keys.
    """
    # Concat all fold predictions
    all_true = np.concatenate(fold_y_true, axis=0)
    all_pred = np.concatenate(fold_y_pred, axis=0)

    masks = build_subset_masks(species_list, focal_species=focal_species)

    results = {
        "pooled_auroc": pooled_auroc(all_true, all_pred),
        "n_windows": len(all_true),
        "n_folds": len(folds),
        "per_fold": {},
        "subsets": {},
    }

    # Per-fold AUROC
    for fold, yt, yp in zip(folds, fold_y_true, fold_y_pred):
        results["per_fold"][fold.left_out_site] = {
            "auroc": compute_auroc(yt, yp),
            "n_windows": len(yt),
        }

    # Subset AUROCs
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        results["subsets"][name] = pooled_auroc(all_true, all_pred, mask)

    return results


def format_results(results: dict) -> str:
    """Pretty-print LOSO evaluation results."""
    lines = []
    lines.append(f"Pooled AUROC: {results['pooled_auroc']:.4f}")
    lines.append(f"  ({results['n_windows']} windows across {results['n_folds']} folds)")
    lines.append("")

    lines.append("Per-fold breakdown:")
    for site, info in sorted(results["per_fold"].items()):
        lines.append(f"  {site}: {info['auroc']:.4f} ({info['n_windows']} windows)")
    lines.append("")

    lines.append("Subset AUROCs:")
    for name, auroc in sorted(results["subsets"].items()):
        lines.append(f"  {name}: {auroc:.4f}")

    lines.append("")
    lines.append(f"Ridge baseline: 0.749")
    delta = results["pooled_auroc"] - 0.749
    sign = "+" if delta >= 0 else ""
    lines.append(f"Delta: {sign}{delta:.4f}")

    return "\n".join(lines)
