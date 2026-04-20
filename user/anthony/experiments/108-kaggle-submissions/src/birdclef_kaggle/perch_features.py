"""Compact Perch sidecar projection for BirdCLEF 2026 submissions.

Standalone Kaggle-safe port of
`birdclef_107.perch_features.PerchSidecarProjector`. Reads
`perch_species_mapping.json` directly — no `birdclef_106` / `birdclef_107`
imports — so the Kaggle kernel can ship this without the training dep
chain (polars, sklearn, duckdb, etc.).

The raw Perch classifier head is 14,795-d per window. The
`PrototypeRetrievalModel.perch_adapter` consumes a compressed
representation:

1. ``mapped_probs`` — dense BirdCLEF-aligned probabilities ``(N, 234)``
   using direct + genus-proxy mapping from the species mapping artifact.
2. ``topk_idx`` / ``topk_val`` — sparse top-k residual teacher
   activations over raw classes not already consumed by (1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from birdclef_kaggle.species import DEFAULT_MAPPING_PATH, build_species_list, load_species_mapping

TOPK_RESIDUAL = 64


class PerchSidecarProjector:
    """Project raw Perch logits into compact mapped + sparse residual form."""

    def __init__(
        self,
        species_list: list[str] | None = None,
        *,
        mapping_path: str | Path | None = None,
        topk: int = TOPK_RESIDUAL,
    ) -> None:
        mapping_path = Path(mapping_path) if mapping_path else DEFAULT_MAPPING_PATH
        self.species_list = list(species_list or build_species_list(mapping_path))
        self.topk = int(topk)

        mapping = load_species_mapping(mapping_path)
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


__all__ = ["TOPK_RESIDUAL", "PerchSidecarProjector"]
