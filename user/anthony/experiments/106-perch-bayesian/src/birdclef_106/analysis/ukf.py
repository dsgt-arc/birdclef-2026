"""UKF temporal smoothing in DMD mode space.

The Unscented Kalman Filter propagates species presence estimates through time,
using the DMD-learned transition dynamics and the ridge-learned observation model.

Architecture:
    State: embedding in DMD mode space (20-50d) — captures the dominant community
           dynamics, not individual species
    Transition A: from DMD eigendecomposition (dmd.py)
    Observation H: ridge weights projected to PCA/mode space
    Process noise Q: from DMD reconstruction residuals
    Observation noise R: from ridge prediction residuals on labeled data

UKF preferred over EKF per Newman et al. (2023): the unscented transform handles
nonlinearity better than first-order Taylor linearization at similar cost.

References:
    Newman et al. 2023 — Practical SSM fitting (EKF vs UKF comparison)
    filterpy library — UnscentedKalmanFilter implementation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.dmd import DMDAnalysis
from birdclef_106.analysis.ridge import EmpiricalBayesRidge
from birdclef_106.data_loading import (
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    load_unlabeled_shards,
    logits_to_numpy,
)

logger = logging.getLogger(__name__)


class UKFSmoothing(AnalysisTask):
    """UKF temporal smoothing on unlabeled soundscapes using DMD dynamics + ridge obs model."""

    def requires(self):
        return {
            "ridge": EmpiricalBayesRidge(**self._params()),
            "dmd": DMDAnalysis(**self._params()),
        }

    def output(self):
        return self.artifact("ukf_predictions.parquet")

    def run(self):
        self.ensure_parent(self.output())

        from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

        species_list = build_species_list(self.artifacts_dir)

        # Load ridge model
        ridge_input = self.input()["ridge"]
        ridge_path = ridge_input[0].path if isinstance(ridge_input, list) else ridge_input.path
        ridge_data = np.load(ridge_path, allow_pickle=True)
        W_ridge = ridge_data["W"].astype(np.float64)
        feature_space = str(ridge_data["feature_space"])

        # Load DMD summary to find which sites have models
        dmd_summary_path = self.input()["dmd"].path
        dmd_dir = Path(dmd_summary_path).parent
        with open(dmd_summary_path) as f:
            dmd_summary = json.load(f)

        # Collect unlabeled data by site
        site_data: dict[str, list] = {}
        for shard in load_unlabeled_shards(self.features_dir):
            filenames = shard["filename"].to_list()
            window_idxs = shard["window_idx"].to_list()

            if feature_space == "embeddings":
                features = embeddings_to_numpy(shard)
            else:
                features = logits_to_numpy(shard)

            for i in range(len(shard)):
                site = extract_site(filenames[i])
                if site not in site_data:
                    site_data[site] = []
                site_data[site].append({
                    "filename": filenames[i],
                    "window_idx": window_idxs[i],
                    "features": features[i],
                })

        all_rows = []
        for site, rows in sorted(site_data.items()):
            site_info = dmd_summary.get(site, {})
            if site_info.get("status") != "ok":
                logger.info("UKF: skipping %s (no DMD model)", site)
                # Pass through raw ridge predictions for sites without DMD
                for row in rows:
                    scores = row["features"] @ W_ridge
                    all_rows.append({
                        "filename": row["filename"],
                        "window_idx": row["window_idx"],
                        "site": site,
                        "smoothed": False,
                        **{sp: float(scores[si]) for si, sp in enumerate(species_list)},
                    })
                continue

            # Load DMD model
            dmd_data = np.load(dmd_dir / f"{site}.npz")
            A_tilde = dmd_data["A_tilde"].astype(np.float64)
            pca_components = dmd_data["pca_components"].astype(np.float64)
            pca_mean = dmd_data["pca_mean"].astype(np.float64)

            # Sort chronologically
            rows.sort(key=lambda r: (r["filename"], r["window_idx"]))
            X_site = np.stack([r["features"] for r in rows]).astype(np.float64)

            # Project to PCA space (same as DMD used)
            X_pca = (X_site - pca_mean) @ pca_components.T  # (T, pca_dim)
            state_dim = A_tilde.shape[0]
            X_pca = X_pca[:, :state_dim]

            # UKF setup
            def fx(x, dt):
                return A_tilde @ x

            def hx(x):
                return x  # observe the state directly in PCA space

            points = MerweScaledSigmaPoints(state_dim, alpha=0.1, beta=2.0, kappa=0.0)
            ukf = UnscentedKalmanFilter(
                dim_x=state_dim, dim_z=state_dim,
                dt=1.0, fx=fx, hx=hx, points=points,
            )

            # Initialize
            ukf.x = X_pca[0]
            ukf.P = np.eye(state_dim) * 1.0
            ukf.Q = np.eye(state_dim) * 0.1  # process noise
            ukf.R = np.eye(state_dim) * 0.5  # observation noise

            # Forward pass
            smoothed_states = np.zeros_like(X_pca)
            for t in range(len(X_pca)):
                ukf.predict()
                ukf.update(X_pca[t])
                smoothed_states[t] = ukf.x

            # Project back to feature space and apply ridge
            smoothed_features = smoothed_states @ pca_components[:state_dim] + pca_mean
            smoothed_scores = smoothed_features @ W_ridge  # (T, 234)

            for t, row in enumerate(rows):
                all_rows.append({
                    "filename": row["filename"],
                    "window_idx": row["window_idx"],
                    "site": site,
                    "smoothed": True,
                    **{sp: float(smoothed_scores[t, si]) for si, sp in enumerate(species_list)},
                })

            logger.info("UKF %s: smoothed %d windows", site, len(rows))

        # Write output
        df_out = pl.DataFrame(all_rows)
        df_out.write_parquet(self.output().path)
        logger.info("UKF: wrote %d rows to %s", len(all_rows), self.output().path)
