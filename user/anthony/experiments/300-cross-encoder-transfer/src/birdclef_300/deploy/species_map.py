"""234-species competition roster + Perch native-logit fallback maps.

The competition `submission.csv` needs a probability for all 234 species, but a
ridge head only covers species that have training labels. `perch_species_mapping.json`
(from exp 106 / Murilo 018) carries both the canonical 234-species ordering and the
Perch-specific maps used to fill species the head cannot cover:

    direct_map       {birdclef_label: perch_logit_index}   203 species
    genus_proxy_map  {birdclef_label: [perch_logit_index]} 6 species (max over genus)
    unmapped         [birdclef_label]                       25 insect sonotypes -> 0

Everything except the two Perch maps is backbone-agnostic. A non-Perch backbone
loads the same roster for column ordering and simply skips the fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Perch's `label` output drops index 0 (non-bird background), so a logit index
# stored in the mapping points one slot higher than the row in the 14795-vector.
PERCH_LOGIT_OFFSET = 1


def load_species_mapping(path: str | Path) -> dict:
    """Read perch_species_mapping.json. Keys: direct_map, genus_proxy_map,
    unmapped, birdclef_species (234), birdclef_to_scientific, taxa_classes."""
    with open(Path(path).expanduser()) as f:
        return json.load(f)


def roster_234(mapping: dict) -> list[str]:
    """The canonical 234-species column ordering (sorted, matches submission CSV)."""
    return sorted(mapping["birdclef_species"])


def perch_mapped_scores(logits: np.ndarray, mapping: dict, species_list: list[str]) -> np.ndarray:
    """Map Perch `label` logits -> (N, 234) sigmoid scores, column-aligned to
    `species_list`. Mirrors Murilo 018's `compute_perch_mapped_scores`:
    direct index, genus-proxy max, unmapped -> 0. Used only for the Perch fallback.

    Args:
        logits: (N, P) raw Perch label logits (P ~= 14795).
        mapping: parsed perch_species_mapping.json.
        species_list: the 234-species column order to emit.
    """
    direct_map = mapping["direct_map"]
    genus_proxy_map = mapping.get("genus_proxy_map", {})
    n, p = logits.shape
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    scores = np.zeros((n, len(species_list)), dtype=np.float32)
    for i, label in enumerate(species_list):
        if label in direct_map:
            idx = direct_map[label] - PERCH_LOGIT_OFFSET
            if 0 <= idx < p:
                scores[:, i] = probs[:, idx]
        elif label in genus_proxy_map:
            idxs = [j - PERCH_LOGIT_OFFSET for j in genus_proxy_map[label]]
            idxs = [j for j in idxs if 0 <= j < p]
            if idxs:
                scores[:, i] = probs[:, idxs].max(axis=1)
        # else: unmapped insect sonotype -> stays 0.0
    return scores
