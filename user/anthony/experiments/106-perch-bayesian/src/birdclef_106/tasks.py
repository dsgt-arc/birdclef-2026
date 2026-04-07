"""AllAnalysis wrapper task — runs all analysis phases via Luigi."""

from __future__ import annotations

from pathlib import Path

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.analysis import (
    ZeroShotBaseline,
    LogitCovariance,
    DistributionShift,
    PrototypeClassifier,
    FeatureSelection,
    EmpiricalBayesRidge,
    SparseLogistic,
    TemporalPooling,
    DMDAnalysis,
    UKFSmoothing,
)

import luigi


class AllAnalysis(AnalysisTask):
    """Run all analysis tasks, optionally limited to a specific phase.

    Phase 0 (default): all phases.
    Phase 1: static baselines (zero-shot, covariance, shift, prototypes, features).
    Phase 2: phase 1 + models (ridge, sparse logistic, temporal pooling, DMD).
    Phase 3: phase 1 + 2 + UKF smoothing.
    """

    phase = luigi.IntParameter(default=0)

    def requires(self):
        p = self._params()
        phase1 = [
            ZeroShotBaseline(**p),
            LogitCovariance(**p),
            DistributionShift(**p),
            PrototypeClassifier(**p),
            FeatureSelection(**p),
        ]
        phase2 = [
            EmpiricalBayesRidge(**p),
            SparseLogistic(**p),
            TemporalPooling(**p),
            DMDAnalysis(**p),
        ]
        phase3 = [
            UKFSmoothing(**p),
        ]

        if self.phase == 1:
            return phase1
        if self.phase == 2:
            return phase1 + phase2
        if self.phase == 3:
            return phase1 + phase2 + phase3
        return phase1 + phase2 + phase3

    def output(self):
        return self.artifact(f"_analysis_phase{self.phase}_complete")

    def run(self):
        self.ensure_parent(self.output())
        Path(self.output().path).touch()
