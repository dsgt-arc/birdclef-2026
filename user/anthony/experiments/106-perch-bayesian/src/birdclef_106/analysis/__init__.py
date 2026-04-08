"""Exp 106 analysis tasks — Luigi DAG for Perch v2 feature exploration."""

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis.zero_shot import ZeroShotBaseline
from birdclef_106.analysis.logit_covariance import LogitCovariance
from birdclef_106.analysis.embedding_covariance import EmbeddingCovariance
from birdclef_106.analysis.distribution_shift import DistributionShift
from birdclef_106.analysis.prototype import PrototypeClassifier
from birdclef_106.analysis.feature_selection import FeatureSelection
from birdclef_106.analysis.ridge import EmpiricalBayesRidge
from birdclef_106.analysis.sparse_logistic import SparseLogistic
from birdclef_106.analysis.temporal_pooling import TemporalPooling
from birdclef_106.analysis.dmd import DMDAnalysis
from birdclef_106.analysis.focal_profiles import FocalSpeciesProfiles
from birdclef_106.analysis.soundscape_profiles import SoundscapeProfiles
from birdclef_106.analysis.site_residuals import SiteResidualProfiles
from birdclef_106.analysis.species_graph import SpeciesGraph
from birdclef_106.analysis.ukf import UKFSmoothing

__all__ = [
    "AnalysisTask",
    "ZeroShotBaseline",
    "LogitCovariance",
    "EmbeddingCovariance",
    "DistributionShift",
    "PrototypeClassifier",
    "FeatureSelection",
    "EmpiricalBayesRidge",
    "SparseLogistic",
    "TemporalPooling",
    "DMDAnalysis",
    "FocalSpeciesProfiles",
    "SoundscapeProfiles",
    "SiteResidualProfiles",
    "SpeciesGraph",
    "UKFSmoothing",
]
