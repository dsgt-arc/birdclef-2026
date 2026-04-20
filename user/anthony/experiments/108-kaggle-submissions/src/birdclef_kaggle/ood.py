"""OOD-site fallback for Kaggle inference.

Exp 107's model conditions species retrieval on a site bank. If the hidden
rerun uses sites outside the 23 seen at training time, we cannot retrieve
a matching site prototype. The pre-registered fallback (proposal.md §5.6)
is to set `site_score = 1` so the cross-attention path degenerates to pure
species retrieval — the same behavior focal recordings (which have no
site) already get during training.
"""

from __future__ import annotations

import numpy as np

from .row_id import KNOWN_SITES, site_to_index


def resolve_site_indices(
    sites: list[str],
    known_sites: frozenset[str] | set[str] = KNOWN_SITES,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a list of site strings to (site_idx_array, is_known_mask).

    OOD sites get `site_idx = -1`; callers should consult `is_known` and
    substitute the neutral `site_score = 1` pathway for those rows.
    """
    n = len(sites)
    idx = np.full(n, -1, dtype=np.int64)
    known = np.zeros(n, dtype=bool)
    for i, s in enumerate(sites):
        resolved = site_to_index(s, known_sites=known_sites)
        if resolved is not None:
            idx[i] = resolved
            known[i] = True
    return idx, known


def neutral_site_score(shape: tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """The OOD fallback value `site_score = 1` broadcastable to `shape`."""
    return np.ones(shape, dtype=dtype)
