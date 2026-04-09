"""Exp 106 analysis tasks — Luigi DAG for Perch v2 feature exploration.

Subpackages:
    eda/        — exploratory analysis, static baselines, covariance, domain shift
    structure/  — star schema: profiles, graphs, prototypes, partitioning
    models/     — learned scoring models: ridge, prototype ridge, temporal pooling
    _archived/  — broken or superseded modules (not imported)
"""

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.eda import (
    ZeroShotBaseline,
    LogitCovariance,
    EmbeddingCovariance,
    DistributionShift,
    FeatureSelection,
)
from birdclef_106.analysis.structure import (
    PartitionFeatures,
    FocalSpeciesProfiles,
    SiteResidualProfiles,
    SpeciesGraph,
    MultiPrototypeProfiles,
    SitePrototypes,
)
from birdclef_106.analysis.models import (
    EmpiricalBayesRidge,
    PrototypeRidge,
    StarSchemaScorer,
    TemporalPooling,
)

__all__ = [
    "AnalysisTask",
    # EDA
    "ZeroShotBaseline",
    "LogitCovariance",
    "EmbeddingCovariance",
    "DistributionShift",
    "FeatureSelection",
    # Structure
    "PartitionFeatures",
    "FocalSpeciesProfiles",
    "SiteResidualProfiles",
    "SpeciesGraph",
    "MultiPrototypeProfiles",
    "SitePrototypes",
    # Models
    "EmpiricalBayesRidge",
    "PrototypeRidge",
    "StarSchemaScorer",
    "TemporalPooling",
]
