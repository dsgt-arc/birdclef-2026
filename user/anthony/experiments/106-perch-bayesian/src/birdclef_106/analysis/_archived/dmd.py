"""Dynamic Mode Decomposition on per-site Perch logit time series.

DMD extracts dominant spatiotemporal modes from time-ordered data: which
Perch logit patterns evolve coherently over time at each monitoring site.

Eigenvalues reveal timescales:
    |λ| < 1 → decaying (transient call events)
    |λ| ≈ 1 → persistent (background species)
    Complex λ → oscillatory (diel cycles, period = 2π / arg(λ))

Modes reveal communities: species that load on the same mode appear/disappear
together — data-driven ecological assemblages.

The transition operator A from DMD is used by UKFSmoothing (ukf.py) as the
state transition model.

References:
    Kutz et al. 2015 — Multi-resolution DMD
    PyDMD library
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    extract_site,
    load_unlabeled_shards,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)

PCA_DIM = 100
SVD_RANK = 20
MIN_WINDOWS = 50


def _dmd_numpy(X_seq, svd_rank=SVD_RANK):
    """Standard DMD via truncated SVD. Pure numpy fallback.

    Args:
        X_seq: (d, T) data matrix, columns are time-ordered snapshots.
        svd_rank: truncation rank for SVD.

    Returns:
        eigenvalues: (r,) complex eigenvalues of the reduced DMD operator.
        modes: (d, r) DMD modes (spatial patterns).
        A_tilde: (r, r) reduced DMD operator.
    """
    X1 = X_seq[:, :-1]
    X2 = X_seq[:, 1:]

    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(svd_rank, len(S))
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    A_tilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1.0 / S_r)
    evals, evecs = np.linalg.eig(A_tilde)
    modes = X2 @ Vh_r.T @ np.diag(1.0 / S_r) @ evecs

    return evals, modes, A_tilde


class DMDAnalysis(AnalysisTask):
    """Per-site DMD on Perch logit time series from unlabeled soundscapes."""

    def output(self):
        return self.artifact("dmd_modes/summary.json")

    def run(self):
        self.ensure_parent(self.output())
        out_dir = Path(self.artifacts_dir) / "dmd_modes"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect unlabeled metadata + PCA-reduced logits grouped by site.
        # Only store PCA-reduced vectors (100-d) not full logits (14795-d) to limit memory.
        # Two passes: first pass fits PCA on a subsample, second pass transforms all data.
        logger.info("DMD: pass 1 — fitting PCA on subsample")
        pca_sample = []
        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            logits = logits_to_numpy(shard)
            # Take every 4th row for PCA fitting (~32K total)
            pca_sample.append(logits[::4])
            if i >= 7:
                break
        pca_sample = np.concatenate(pca_sample).astype(np.float64)
        global_pca = PCA(n_components=PCA_DIM)
        global_pca.fit(pca_sample)
        del pca_sample

        logger.info("DMD: pass 2 — projecting and grouping by site")
        site_data: dict[str, list[tuple[str, int, np.ndarray]]] = {}
        for shard in load_unlabeled_shards(self.features_dir):
            logits = logits_to_numpy(shard).astype(np.float64)
            logits_pca = global_pca.transform(logits)  # (N, 100) — much smaller
            filenames = shard["filename"].to_list()
            window_idxs = shard["window_idx"].to_list()

            for i in range(len(shard)):
                site = extract_site(filenames[i])
                if site not in site_data:
                    site_data[site] = []
                site_data[site].append((filenames[i], window_idxs[i], logits_pca[i]))

        logger.info("DMD: %d sites found", len(site_data))

        summary = {}
        for site, rows in sorted(site_data.items()):
            if len(rows) < MIN_WINDOWS:
                logger.info("DMD: skipping %s (%d windows < %d)", site, len(rows), MIN_WINDOWS)
                summary[site] = {"status": "skipped", "n_windows": len(rows)}
                continue

            # Sort chronologically by filename + window_idx
            rows.sort(key=lambda x: (x[0], x[1]))
            X_pca = np.stack([r[2] for r in rows]).astype(np.float64)  # (T, PCA_DIM)

            # DMD: columns are time steps → transpose to (d, T)
            rank = min(SVD_RANK, X_pca.shape[1], X_pca.shape[0] - 2)
            try:
                evals, modes, A_tilde = _dmd_numpy(X_pca.T, svd_rank=rank)
            except Exception as e:
                logger.warning("DMD failed for %s: %s", site, e)
                summary[site] = {"status": "failed", "error": str(e), "n_windows": len(rows)}
                continue

            # Reconstruction error
            X1 = X_pca[:-1].T  # (d, T-1)
            X2 = X_pca[1:].T
            X2_pred = modes @ np.diag(evals) @ np.linalg.pinv(modes) @ X1
            recon_error = float(np.linalg.norm(X2 - X2_pred) / np.linalg.norm(X2))

            # Save per-site (PCA is global, shared across sites)
            np.savez_compressed(
                out_dir / f"{site}.npz",
                eigenvalues=evals,
                modes=modes.astype(np.float32),
                A_tilde=A_tilde.astype(np.float32),
                pca_components=global_pca.components_.astype(np.float32),
                pca_mean=global_pca.mean_.astype(np.float32),
                explained_variance_ratio=global_pca.explained_variance_ratio_.astype(np.float32),
            )

            # Eigenvalue analysis
            magnitudes = np.abs(evals)
            periods = np.where(
                np.abs(np.angle(evals)) > 1e-6,
                2 * np.pi / np.abs(np.angle(evals)),
                np.inf,
            )

            summary[site] = {
                "status": "ok",
                "n_windows": len(rows),
                "pca_dims": int(global_pca.n_components_),
                "svd_rank": rank,
                "reconstruction_error": recon_error,
                "eigenvalue_magnitudes": [float(m) for m in sorted(magnitudes, reverse=True)],
                "eigenvalue_periods": [float(p) for p in periods if np.isfinite(p)],
                "n_persistent_modes": int((magnitudes > 0.95).sum()),
                "n_decaying_modes": int((magnitudes < 0.5).sum()),
            }

            logger.info(
                "DMD %s: %d windows, rank=%d, recon_error=%.3f, %d persistent modes",
                site, len(rows), rank, recon_error, summary[site]["n_persistent_modes"],
            )

        with open(self.output().path, "w") as f:
            json.dump(summary, f, indent=2)
