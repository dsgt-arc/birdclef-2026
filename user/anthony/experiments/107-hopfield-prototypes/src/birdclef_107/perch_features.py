"""Utilities for compact Perch sidecar features.

The raw Perch classifier head is 14,795-d per window, which is too large
for the default exp107 cache path. We compress it into two row-aligned
sidecar components:

1. ``mapped_probs`` — dense BirdCLEF-aligned probabilities of shape
   ``(N, n_species)`` using direct + genus-proxy mapping.
2. ``topk_idx`` / ``topk_val`` — sparse residual teacher activations,
   excluding raw classes already consumed by the BirdCLEF mapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from birdclef_106.data_loading import load_species_mapping
from birdclef_107.data import ARTIFACTS_DIR, get_species_list

TOPK_RESIDUAL = 64


class PerchSidecarProjector:
    """Project raw Perch logits into compact mapped + sparse residual form."""

    def __init__(
        self,
        species_list: list[str] | None = None,
        *,
        artifacts_dir: str = ARTIFACTS_DIR,
        topk: int = TOPK_RESIDUAL,
    ) -> None:
        self.species_list = list(species_list or get_species_list(artifacts_dir))
        self.artifacts_dir = str(Path(artifacts_dir))
        self.topk = int(topk)

        mapping = load_species_mapping(self.artifacts_dir)
        direct_map = {str(k): int(v) for k, v in mapping["direct_map"].items()}
        genus_proxy_map = {
            str(k): [int(i) for i in v]
            for k, v in mapping["genus_proxy_map"].items()
        }

        direct_idx = np.full(len(self.species_list), -1, dtype=np.int64)
        genus_species: list[int] = []
        genus_indices: list[np.ndarray] = []
        used_raw: set[int] = set()
        for s_idx, species in enumerate(self.species_list):
            if species in direct_map:
                idx = int(direct_map[species])
                direct_idx[s_idx] = idx
                used_raw.add(idx)
            elif species in genus_proxy_map:
                idxs = np.asarray(genus_proxy_map[species], dtype=np.int64)
                genus_species.append(s_idx)
                genus_indices.append(idxs)
                used_raw.update(int(i) for i in idxs.tolist())

        self.direct_idx = direct_idx
        self.genus_species = genus_species
        self.genus_indices = genus_indices
        self.used_raw_indices = np.asarray(sorted(used_raw), dtype=np.int64)

    def project(self, logits: np.ndarray) -> dict[str, np.ndarray]:
        """Return dense mapped probs + sparse top-k residual activations."""
        if logits.ndim != 2:
            raise ValueError(f"expected 2D logits, got shape {logits.shape}")
        n, raw_dim = logits.shape
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float32, copy=False)))

        mapped = np.zeros((n, len(self.species_list)), dtype=np.float32)
        direct_mask = (self.direct_idx >= 0) & (self.direct_idx < raw_dim)
        if direct_mask.any():
            mapped[:, direct_mask] = probs[:, self.direct_idx[direct_mask]]
        for s_idx, raw_idx in zip(self.genus_species, self.genus_indices):
            raw_idx = raw_idx[raw_idx < raw_dim]
            if len(raw_idx) > 0:
                mapped[:, s_idx] = probs[:, raw_idx].max(axis=1)

        residual_mask = np.ones(raw_dim, dtype=bool)
        valid_used = self.used_raw_indices[self.used_raw_indices < raw_dim]
        residual_mask[valid_used] = False
        residual_cols = np.flatnonzero(residual_mask)

        k = max(1, self.topk)
        topk_idx = np.full((n, k), -1, dtype=np.int32)
        topk_val = np.zeros((n, k), dtype=np.float32)
        if len(residual_cols) == 0:
            return {
                "mapped_probs": mapped,
                "topk_idx": topk_idx,
                "topk_val": topk_val,
            }

        resid = probs[:, residual_cols]
        k_eff = min(k, resid.shape[1])
        if k_eff == resid.shape[1]:
            order = np.argsort(-resid, axis=1)
            local_idx = order[:, :k_eff]
        else:
            local_idx = np.argpartition(-resid, kth=k_eff - 1, axis=1)[:, :k_eff]
            part_vals = np.take_along_axis(resid, local_idx, axis=1)
            order = np.argsort(-part_vals, axis=1)
            local_idx = np.take_along_axis(local_idx, order, axis=1)
        vals = np.take_along_axis(resid, local_idx, axis=1)
        cols = residual_cols[local_idx]
        topk_idx[:, :k_eff] = cols.astype(np.int32, copy=False)
        topk_val[:, :k_eff] = vals.astype(np.float32, copy=False)
        return {
            "mapped_probs": mapped,
            "topk_idx": topk_idx,
            "topk_val": topk_val,
        }


def project_perch_sidecar(
    logits: np.ndarray,
    species_list: list[str] | None = None,
    *,
    artifacts_dir: str = ARTIFACTS_DIR,
    topk: int = TOPK_RESIDUAL,
) -> dict[str, np.ndarray]:
    projector = PerchSidecarProjector(
        species_list=species_list,
        artifacts_dir=artifacts_dir,
        topk=topk,
    )
    return projector.project(logits)


__all__ = ["TOPK_RESIDUAL", "PerchSidecarProjector", "project_perch_sidecar"]
