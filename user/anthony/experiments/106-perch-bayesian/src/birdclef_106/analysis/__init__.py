"""Exp 106 analysis tasks — Luigi DAG for Perch v2 feature exploration."""

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.zero_shot import ZeroShotBaseline
from birdclef_106.analysis.logit_covariance import LogitCovariance
from birdclef_106.analysis.embedding_covariance import EmbeddingCovariance
from birdclef_106.analysis.distribution_shift import DistributionShift
from birdclef_106.analysis.feature_selection import FeatureSelection
from birdclef_106.analysis.ridge import EmpiricalBayesRidge
from birdclef_106.analysis.temporal_pooling import TemporalPooling
from birdclef_106.analysis.focal_profiles import FocalSpeciesProfiles
from birdclef_106.analysis.site_residuals import SiteResidualProfiles
from birdclef_106.analysis.species_graph import SpeciesGraph
from birdclef_106.analysis.partition import PartitionFeatures
from birdclef_106.analysis.multi_prototype import MultiPrototypeProfiles
from birdclef_106.analysis.site_prototypes import SitePrototypes
from birdclef_106.analysis.prototype_ridge import PrototypeRidge

__all__ = [
    "AnalysisTask",
    "ZeroShotBaseline",
    "LogitCovariance",
    "EmbeddingCovariance",
    "DistributionShift",
    "FeatureSelection",
    "EmpiricalBayesRidge",
    "TemporalPooling",
    "FocalSpeciesProfiles",
    "SiteResidualProfiles",
    "SpeciesGraph",
    "PartitionFeatures",
    "MultiPrototypeProfiles",
    "SitePrototypes",
    "PrototypeRidge",
]
