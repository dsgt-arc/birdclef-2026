"""AllAnalysis wrapper task — runs all analysis phases via Luigi."""

from __future__ import annotations

from pathlib import Path

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis import (
    ZeroShotBaseline,
    LogitCovariance,
    EmbeddingCovariance,
    DistributionShift,
    FeatureSelection,
    FocalSpeciesProfiles,
    SiteResidualProfiles,
    SpeciesGraph,
    PartitionFeatures,
    MultiPrototypeProfiles,
    SitePrototypes,
    PrototypeRidge,
    StarSchemaScorer,
    EmpiricalBayesRidge,
    TemporalPooling,
)

import luigi


class AllAnalysis(AnalysisTask):
    """Run all analysis tasks, optionally limited to a specific phase.

    Phase 0 (default): all phases.
    Phase 1: static baselines (zero-shot, covariance, shift, features).
    Phase 2: phase 1 + models (ridge, prototypes, species/site structure).
    """

    phase = luigi.IntParameter(default=0)

    def requires(self):
        p = self._params()
        phase1 = [
            ZeroShotBaseline(**p),
            LogitCovariance(**p),
            EmbeddingCovariance(**p),
            DistributionShift(**p),
            FeatureSelection(**p),
            FocalSpeciesProfiles(**p),
        ]
        phase2 = [
            EmpiricalBayesRidge(**p),
            TemporalPooling(**p),
            SiteResidualProfiles(**p),
            SpeciesGraph(**p),
            PartitionFeatures(**p),
            MultiPrototypeProfiles(**p),
            SitePrototypes(**p),
            PrototypeRidge(**p),
            StarSchemaScorer(**p),
        ]

        if self.phase == 1:
            return phase1
        if self.phase == 2:
            return phase1 + phase2
        return phase1 + phase2

    def output(self):
        return self.artifact(f"_analysis_phase{self.phase}_complete")

    def run(self):
        self.ensure_parent(self.output())
        Path(self.output().path).touch()
