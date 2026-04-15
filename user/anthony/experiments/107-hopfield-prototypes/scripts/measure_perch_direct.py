#!/usr/bin/env python3
"""Measure Perch v2 direct AUROC on labeled soundscape windows.

Reads cached `perch_logits` (14,795-d) from the exp 106 parquet,
projects them to our 234 target species via
`perch_species_mapping.json`, and computes AUROC overall, per taxa
class (Aves / Amphibia / Insecta / Mammalia / Reptilia), by Perch-
coverage (mapped vs unmapped), and per LOSO site.

This is a diagnostic baseline, no training. The numbers answer:

- Does Perch's own classifier (via the eBird-code mapping) compete
  with our v4 prototype retrieval (S22 = 0.7613)?
- Is the "Perch-as-base + exp 107-as-residual" architecture reframe
  justified by the data?

**Design note**: deliberately avoids importing `birdclef_107.*` or
`birdclef_106.*` because those modules import torch, and torch imports
on PACE cpu-small NFS are unusably slow (>10 min). We inline the
minimal logic (parquet load, label parse, site extract, AUROC).

Usage:
    python scripts/measure_perch_direct.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


# ---- Paths ----
CACHE_DIR = Path("~/scratch/birdclef-2026/features/exp106").expanduser()
MAPPING_PATH = Path(
    "~/arc/birdclef-2026/user/anthony/experiments/"
    "106-perch-bayesian/artifacts/perch_species_mapping.json"
).expanduser()
OUT_PATH = Path(
    "~/scratch/birdclef-2026/checkpoints/exp107-v4/perch_direct_auroc.json"
).expanduser()


# ---- Minimal label + site parsing (no birdclef_* imports) ----


SITE_RE = re.compile(r"_S(\d{2})_")


def extract_site(filename: str) -> str:
    m = SITE_RE.search(filename)
    if m is None:
        raise ValueError(f"no S## site in filename: {filename!r}")
    return "S" + m.group(1)


def parse_labels(ground_truth: list[str], species_list: list[str]) -> np.ndarray:
    """Parse semicolon-delimited ground_truth strings → (N, 234) multi-hot."""
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    n = len(ground_truth)
    labels = np.zeros((n, len(species_list)), dtype=np.int8)
    for i, gt in enumerate(ground_truth):
        if not gt:
            continue
        for sp in gt.split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                labels[i, species_to_idx[sp]] = 1
    return labels


# ---- Perch logit → 234 projection ----


def build_projector(
    mapping: dict, species_list: list[str]
) -> tuple[np.ndarray, dict[int, list[int]], np.ndarray]:
    """Build direct+genus-proxy plan.

    Returns:
        direct_idx: (234,) int — Perch column to copy, or -1
        genus_plans: species_idx -> list of Perch columns for max-agg
        is_mapped: (234,) bool
    """
    direct_map = {str(k): int(v) for k, v in mapping["direct_map"].items()}
    genus_proxy_map = {
        str(k): [int(i) for i in v] for k, v in mapping["genus_proxy_map"].items()
    }
    unmapped = set(mapping["unmapped"])

    direct_idx = np.full(len(species_list), -1, dtype=np.int64)
    genus_plans: dict[int, list[int]] = {}
    is_mapped = np.zeros(len(species_list), dtype=bool)

    for s_idx, species in enumerate(species_list):
        if species in direct_map:
            direct_idx[s_idx] = direct_map[species]
            is_mapped[s_idx] = True
        elif species in genus_proxy_map:
            genus_plans[s_idx] = genus_proxy_map[species]
            is_mapped[s_idx] = True
        elif species in unmapped:
            pass

    return direct_idx, genus_plans, is_mapped


def project_logits(
    logits: np.ndarray,
    direct_idx: np.ndarray,
    genus_plans: dict[int, list[int]],
) -> np.ndarray:
    """(N, 14795) → (N, 234). Unmapped species get the global min logit."""
    n, _ = logits.shape
    s = len(direct_idx)
    preds = np.full((n, s), logits.min(), dtype=np.float32)

    direct_mask = direct_idx >= 0
    if direct_mask.any():
        preds[:, direct_mask] = logits[:, direct_idx[direct_mask]]

    for s_idx, perch_indices in genus_plans.items():
        preds[:, s_idx] = logits[:, perch_indices].max(axis=1)

    return preds


# ---- AUROC (inline, mirrors birdclef_107.eval.compute_auroc semantics) ----


def compute_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_mask: np.ndarray | None = None,
) -> float:
    """Macro-averaged AUROC; skip species with <2 classes. Returns 0 if none."""
    if species_mask is not None:
        y_true = y_true[:, species_mask]
        y_pred = y_pred[:, species_mask]
    n_sp = y_true.shape[1]
    scores = []
    for s in range(n_sp):
        if y_true[:, s].min() == y_true[:, s].max():
            continue
        scores.append(roc_auc_score(y_true[:, s], y_pred[:, s]))
    return float(np.mean(scores)) if scores else 0.0


# ---- Main ----


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=== Load mapping ===", flush=True)
    mapping = json.load(open(MAPPING_PATH))
    species_list: list[str] = mapping["birdclef_species"]
    taxa_classes: dict[str, str] = mapping["taxa_classes"]
    print(f"  234 species, 5 classes")

    print("\n=== Load labeled SC parquet ===", flush=True)
    df = pl.read_parquet(CACHE_DIR / "soundscape_labeled.parquet")
    print(f"  {df.height} windows, columns: {df.columns}")

    print("\n=== Extract perch_logits and labels ===", flush=True)
    logits = np.stack(df["perch_logits"].to_list())
    print(f"  logits: {logits.shape} range [{logits.min():.2f}, {logits.max():.2f}]")
    ground_truth = df["ground_truth"].to_list()
    labels = parse_labels(ground_truth, species_list)
    print(
        f"  labels: {labels.shape}, positive rate {labels.mean():.4f}, "
        f"active species {int((labels.sum(axis=0) > 0).sum())}/234"
    )

    print("\n=== Extract sites ===", flush=True)
    filenames = df["filename"].to_list()
    sites = np.array([extract_site(fn) for fn in filenames])
    unique_sites = sorted(np.unique(sites).tolist())
    print(f"  sites: {unique_sites}")

    print("\n=== Build Perch→234 projector ===", flush=True)
    direct_idx, genus_plans, is_mapped = build_projector(mapping, species_list)
    print(
        f"  direct: {(direct_idx >= 0).sum()}, "
        f"genus-proxy: {len(genus_plans)}, "
        f"unmapped: {(~is_mapped).sum()}"
    )

    print("\n=== Project logits ===", flush=True)
    preds = project_logits(logits, direct_idx, genus_plans)
    print(f"  preds: {preds.shape}")

    # ---- Overall ----
    overall = compute_auroc(labels, preds)
    print(f"\n=== Overall AUROC (234 species) ===")
    print(f"  {overall:.4f}")

    # ---- Per-taxa ----
    print("\n=== Per-taxa AUROC ===")
    taxa_aurocs: dict[str, float] = {}
    taxa_n: dict[str, int] = {}
    for cls in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        mask = np.array([taxa_classes[s] == cls for s in species_list], dtype=bool)
        if mask.sum() == 0:
            continue
        auroc = compute_auroc(labels, preds, species_mask=mask)
        taxa_aurocs[cls] = float(auroc)
        taxa_n[cls] = int(mask.sum())
        print(f"  {cls:>10s} ({mask.sum():3d} sp): {auroc:.4f}")

    # ---- Mapped vs unmapped ----
    print("\n=== Mapping coverage ===")
    mapped_auroc = compute_auroc(labels, preds, species_mask=is_mapped)
    unmapped_auroc = compute_auroc(
        labels, preds, species_mask=~is_mapped
    )
    print(
        f"  mapped   ({is_mapped.sum():3d} sp): {mapped_auroc:.4f}  "
        f"(species where Perch has a direct or genus-proxy signal)"
    )
    print(
        f"  unmapped ({(~is_mapped).sum():3d} sp): {unmapped_auroc:.4f}  "
        f"(expected 0 or noisy — 25 insect morphotypes, no Perch signal)"
    )

    # ---- Per-site ----
    print("\n=== Per-site AUROC (LOSO windows) ===")
    per_site: dict[str, float] = {}
    per_site_n: dict[str, int] = {}
    for site in unique_sites:
        mask = sites == site
        if mask.sum() == 0:
            continue
        auroc = compute_auroc(labels[mask], preds[mask])
        per_site[site] = float(auroc)
        per_site_n[site] = int(mask.sum())
        print(f"  {site:>4s} ({mask.sum():3d} windows): {auroc:.4f}")

    # ---- Comparison with known baselines on S22 ----
    print("\n=== Compare vs our models on S22 ===")
    s22 = per_site.get("S22")
    if s22 is not None:
        print(f"  Perch direct S22:  {s22:.4f}")
        print(f"  Exp 106 ridge S22: 0.749   (Δ = {0.749 - s22:+.4f})")
        print(f"  Exp 107 v4 S22:    0.7613  (Δ = {0.7613 - s22:+.4f})")
        if s22 > 0.85:
            print(
                f"  → Perch alone beats v4. "
                f"Perch-as-base + exp107-as-residual reframe is JUSTIFIED."
            )
        elif s22 > 0.7613:
            print(
                f"  → Perch alone beats v4 modestly. "
                f"Reframe is worth considering."
            )
        elif s22 > 0.70:
            print(
                f"  → Perch alone is competitive but not dominant. "
                f"Ensemble Perch+v4 is probably the best move."
            )
        else:
            print(
                f"  → Perch alone is worse than v4. "
                f"Reframe NOT justified; Perch is at best a regularizer/pseudolabeler."
            )

    # ---- Save ----
    out = {
        "overall_auroc": float(overall),
        "taxa_auroc": taxa_aurocs,
        "taxa_n_species": taxa_n,
        "mapped_auroc": float(mapped_auroc),
        "unmapped_auroc": float(unmapped_auroc),
        "mapped_n": int(is_mapped.sum()),
        "unmapped_n": int((~is_mapped).sum()),
        "per_site_auroc": per_site,
        "per_site_n_windows": per_site_n,
        "n_windows": int(labels.shape[0]),
        "n_species": int(labels.shape[1]),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
