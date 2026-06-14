"""Fit one ridge head on ALL training embeddings, exported as a portable head.npz.

The exp 300 probe only does LOSO cross-validation. For a deployable Kaggle baseline
we fit a single final head on all available training data — focal recordings (broad
species coverage) + labeled soundscapes (in-domain) — over the full 234-species
roster. The result is a backbone-agnostic artifact the kernel applies as
`(emb - mean) / std @ W`, clipped to [0, 1].

Streaming, single-pass sufficient statistics
---------------------------------------------
Focal is ~266K windows × 1536-d; we never hold all of X in memory. We accumulate,
over one pass of the parquet shards, the raw moments

    n, S = Σx, G = Σx xᵀ, XtY = Σ x yᵀ, ny = Σ y

from which the *standardized* normal equations follow in closed form (μ = S/n,
C = G − S Sᵀ/n, σ² = diag(C)/n):

    XtX' = D⁻¹ C D⁻¹ ,   XtY' = D⁻¹ (XtY − μ ny ᵀ) ,   D = diag(σ)

so one pass suffices for a fit that is identical to standardizing every row first.
The solve reuses `probe.ridge_probe.ridge_solve` — same math as the LOSO probe.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from ..data_loading import build_binary_label_matrix, embeddings_to_numpy
from ..probe.ridge_probe import _per_column_scores, ridge_solve
from .species_map import load_species_mapping, roster_234

logger = logging.getLogger(__name__)

# Default training cache (exp 106 Perch features: focal shards + labeled soundscapes).
DEFAULT_FEATURES_DIR = "~/scratch/birdclef-2026/features/exp106"
# λ=1e5: selected by held-out focal→soundscape sweep on the ~266K-window fit
# (deploy.sweep_lambda, 0.706 vs 0.694 at the old soundscape-LOSO 1e4). Per backbone.
DEFAULT_LAMBDA = 1e5


def _iter_training_frames(features_dir: str, sources: list[str]) -> Iterator[pl.DataFrame]:
    """Yield (filename, ground_truth, embedding) frames for each requested source.

    `focal` streams the 16 shards one at a time; `soundscape` reads the single
    labeled parquet. Only the columns we need are selected to keep memory low.
    """
    base = Path(features_dir).expanduser()
    cols = ["filename", "ground_truth", "embedding"]
    if "focal" in sources:
        shards = sorted((base / "focal").glob("shard-*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no focal shards under {base / 'focal'}")
        for path in shards:
            yield pl.read_parquet(path, columns=cols)
    if "soundscape" in sources:
        path = base / "soundscape_labeled.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        yield pl.read_parquet(path, columns=cols)


def accumulate_moments(features_dir: str, sources: list[str], species_list: list[str]) -> dict:
    """One streaming pass: raw moments n, S=Σx, G=Σxxᵀ, XtY=Σxyᵀ, ny=Σy, pos_counts.

    These are λ-independent and fully determine the standardized ridge system, so a
    λ sweep accumulates once here and re-solves cheaply. See module docstring.
    """
    k = len(species_list)
    n = 0
    embed_dim: int | None = None
    S = G = XtY = ny = None
    pos_counts = np.zeros(k, dtype=np.int64)

    for df in _iter_training_frames(features_dir, sources):
        X = embeddings_to_numpy(df).astype(np.float64)
        Y = build_binary_label_matrix(df, species_list).astype(np.float64)
        if embed_dim is None:
            embed_dim = X.shape[1]
            S = np.zeros(embed_dim)
            G = np.zeros((embed_dim, embed_dim))
            XtY = np.zeros((embed_dim, k))
            ny = np.zeros(k)
        elif X.shape[1] != embed_dim:
            raise ValueError(f"embed_dim mismatch: {X.shape[1]} vs {embed_dim}")
        n += X.shape[0]
        S += X.sum(axis=0)
        G += X.T @ X
        XtY += X.T @ Y
        ny += Y.sum(axis=0)
        pos_counts += (Y > 0).sum(axis=0).astype(np.int64)
        logger.info("  +%d windows (running n=%d)", X.shape[0], n)

    if n == 0 or embed_dim is None:
        raise RuntimeError("no training windows loaded")
    return {"n": n, "S": S, "G": G, "XtY": XtY, "ny": ny,
            "pos_counts": pos_counts, "embed_dim": embed_dim}


def standardized_system(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form standardized normal equations (XtX', XtY', mean, std) from moments."""
    n, S, G, XtY, ny = m["n"], m["S"], m["G"], m["XtY"], m["ny"]
    mu = S / n
    C = G - np.outer(S, S) / n
    std = np.sqrt(np.maximum(np.diag(C) / n, 0.0))
    std[std < 1e-8] = 1.0
    Dinv = 1.0 / std
    XtX_std = (Dinv[:, None] * C) * Dinv[None, :]
    XtY_std = Dinv[:, None] * (XtY - np.outer(mu, ny))
    return XtX_std, XtY_std, mu, std


def fit_head(
    features_dir: str = DEFAULT_FEATURES_DIR,
    mapping_path: str = "artifacts/perch_species_mapping.json",
    sources: tuple[str, ...] = ("focal", "soundscape"),
    lam: float = DEFAULT_LAMBDA,
    encoder: str = "perch",
) -> dict:
    """Accumulate sufficient statistics over `sources` and solve the ridge head.

    Returns a dict ready for `np.savez` (W, mean, std, species_list, covered,
    embed_dim, lambda, encoder, n_windows, sources).
    """
    species_list = roster_234(load_species_mapping(mapping_path))
    m = accumulate_moments(features_dir, list(sources), species_list)
    embed_dim, n = m["embed_dim"], m["n"]
    pos_counts = m["pos_counts"]
    XtX_std, XtY_std, mu, std = standardized_system(m)

    logger.info("solving ridge: D=%d, K=%d, n=%d, λ=%.3g", embed_dim, len(species_list), n, lam)
    W = ridge_solve(XtX_std, XtY_std, lam)  # (D, K)

    covered = pos_counts > 0
    logger.info(
        "head fit: %d/%d species covered by training labels (%d via fallback)",
        int(covered.sum()), len(species_list), int((~covered).sum()),
    )
    return {
        "W": W.astype(np.float32),
        "mean": mu.astype(np.float32),
        "std": std.astype(np.float32),
        "species_list": np.array(species_list, dtype=object),
        "covered": covered,
        "pos_counts": pos_counts,
        "embed_dim": np.int64(embed_dim),
        "lambda": np.float64(lam),
        "encoder": encoder,
        "n_windows": np.int64(n),
        "sources": np.array(list(sources), dtype=object),
    }


def apply_head(head: dict, X: np.ndarray) -> np.ndarray:
    """(emb - mean) / std @ W, clipped to [0, 1] — the exact inference transform."""
    Xs = (X.astype(np.float64) - head["mean"]) / head["std"]
    return np.clip(Xs @ head["W"], 0.0, 1.0)


def validate_on_soundscape(head: dict, features_dir: str = DEFAULT_FEATURES_DIR) -> dict:
    """Pooled macro AUROC of the head on the 739 labeled soundscapes.

    IMPORTANT — interpret by what the head was trained on (`head["sources"]`):
      * soundscape IN sources  → these 739 are IN-SAMPLE. The number is a pipeline
        SMOKE TEST only (a broken standardizer/ordering reads ~0.5), NOT a
        generalization estimate. The deployed head includes soundscape, so this is
        always the case for it; cite exp 300's LOSO 0.620 as the honest figure.
      * soundscape NOT in sources (focal-only) → HELD-OUT: an honest focal→soundscape
        transfer read (expect the exp 300 / 106 ballpark).
    Evaluated over species with both-class support.
    """
    path = Path(features_dir).expanduser() / "soundscape_labeled.parquet"
    df = pl.read_parquet(path, columns=["filename", "ground_truth", "embedding"])
    species_list = list(head["species_list"])
    X = embeddings_to_numpy(df)
    Y = build_binary_label_matrix(df, species_list).astype(np.float64)
    pred = apply_head(head, X)
    scores = _per_column_scores(Y, pred)
    aurocs = [a for a, _ in scores.values()]
    macro = float(np.mean(aurocs)) if aurocs else 0.0
    in_sample = "soundscape" in {str(s) for s in head.get("sources", [])}
    kind = "IN-SAMPLE smoke test" if in_sample else "HELD-OUT transfer"
    logger.info("offline-replay pooled macro AUROC = %.4f over %d species (%d windows) [%s]",
                macro, len(aurocs), X.shape[0], kind)
    return {"macro_auroc": macro, "n_species_evaluated": len(aurocs),
            "n_windows": int(X.shape[0]), "in_sample": in_sample}


def save_head(head: dict, out_path: str) -> None:
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **head)
    logger.info("wrote %s (W %s)", out, head["W"].shape)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a deployable ridge head over all training data.")
    ap.add_argument("--features-dir", default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--mapping", default="artifacts/perch_species_mapping.json")
    ap.add_argument("--sources", default="focal,soundscape",
                    help="comma-separated subset of {focal,soundscape}")
    ap.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--encoder", default="perch")
    ap.add_argument("--out", default="artifacts/head.npz")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the offline-replay AUROC sanity check after fitting")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    head = fit_head(
        features_dir=args.features_dir,
        mapping_path=args.mapping,
        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
        lam=args.lam,
        encoder=args.encoder,
    )
    save_head(head, args.out)
    if not args.no_validate:
        validate_on_soundscape(head, args.features_dir)


if __name__ == "__main__":
    main()
