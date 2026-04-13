"""Experiment 107: Hopfield prototype retrieval over frozen Perch v2 embeddings.

See `design.md` for the full architecture and loss derivation.
"""

from birdclef_107.model import (
    HopfieldSiteHead,
    HopfieldSpeciesHead,
    PerchProjection,
    PrototypeRetrievalModel,
    SpeciesSiteCrossAttention,
    affinity_matrix,
)
from birdclef_107.losses import compute_loss

__all__ = [
    "PerchProjection",
    "HopfieldSpeciesHead",
    "HopfieldSiteHead",
    "SpeciesSiteCrossAttention",
    "PrototypeRetrievalModel",
    "affinity_matrix",
    "compute_loss",
]
