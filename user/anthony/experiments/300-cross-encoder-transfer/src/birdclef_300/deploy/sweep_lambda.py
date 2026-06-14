"""Select the ridge-head λ by HELD-OUT focal→soundscape transfer.

The deployed head's λ was carried from the soundscape-only LOSO (~370-window folds);
against the ~266K-window fit it under-regularizes. This sweep streams the focal cache
ONCE into λ-independent moments, then for each λ solves the head and scores it on the
739 labeled soundscapes — which are HELD OUT (focal-only fit), so the number is an
honest transfer estimate, not in-sample. The winning λ is then used to refit the
deployed head on focal+soundscape (`fit-head --lambda <best>`).

Streaming once + re-solving per λ mirrors `ridge_probe.ridge_loso` (Gram once, λ sweep
reuses it).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from ..data_loading import build_binary_label_matrix, embeddings_to_numpy
from ..probe.ridge_probe import _per_column_scores, ridge_solve
from .fit_head import DEFAULT_FEATURES_DIR, accumulate_moments, standardized_system
from .species_map import load_species_mapping, roster_234

logger = logging.getLogger(__name__)

# n is ~720× the soundscape-only LOSO, and XtX diagonals ≈ n under standardization,
# so the optimum sits far above the old 1e4. Bracket up to 1e8.
LAMBDA_GRID = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]


def _load_soundscape_val(features_dir: str, species_list: list[str]):
    df = pl.read_parquet(
        Path(features_dir).expanduser() / "soundscape_labeled.parquet",
        columns=["filename", "ground_truth", "embedding"],
    )
    X = embeddings_to_numpy(df).astype(np.float64)
    Y = build_binary_label_matrix(df, species_list).astype(np.float64)
    return X, Y


def sweep(
    features_dir: str = DEFAULT_FEATURES_DIR,
    mapping_path: str = "artifacts/perch_species_mapping.json",
    grid: list[float] = LAMBDA_GRID,
) -> dict:
    species_list = roster_234(load_species_mapping(mapping_path))
    logger.info("accumulating focal moments (held-out: 739 soundscapes)...")
    m = accumulate_moments(features_dir, ["focal"], species_list)
    XtX_std, XtY_std, mu, std = standardized_system(m)
    Xval, Yval = _load_soundscape_val(features_dir, species_list)
    Xval_s = (Xval - mu) / std

    rows = []
    best = {"macro_auroc": -1.0}
    for lam in grid:
        W = ridge_solve(XtX_std, XtY_std, lam)
        pred = np.clip(Xval_s @ W, 0.0, 1.0)
        aurocs = [a for a, _ in _per_column_scores(Yval, pred).values()]
        macro = float(np.mean(aurocs)) if aurocs else 0.0
        rows.append({"lambda": lam, "macro_auroc": macro, "n_species": len(aurocs)})
        logger.info("  λ=%.0e: held-out macro AUROC=%.4f (%d species)", lam, macro, len(aurocs))
        if macro > best["macro_auroc"]:
            best = {"lambda": lam, "macro_auroc": macro}

    logger.info("BEST λ=%.0e at held-out macro AUROC=%.4f", best["lambda"], best["macro_auroc"])
    return {"best_lambda": best["lambda"], "best_macro_auroc": best["macro_auroc"],
            "n_focal_windows": m["n"], "grid": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep ridge-head λ on held-out soundscape transfer.")
    ap.add_argument("--features-dir", default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--mapping", default="artifacts/perch_species_mapping.json")
    ap.add_argument("--out", default="artifacts/head_lambda_sweep.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = sweep(features_dir=args.features_dir, mapping_path=args.mapping)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
