"""Base class for all exp 106 analysis Luigi tasks."""

from __future__ import annotations

from pathlib import Path

import luigi


FEATURES_DIR = "~/scratch/birdclef-2026/features/exp106"
ARTIFACTS_DIR = "artifacts"


class AnalysisTask(luigi.Task):
    """Base task with shared parameters for features and artifacts directories."""

    features_dir = luigi.Parameter(default=FEATURES_DIR)
    artifacts_dir = luigi.Parameter(default=ARTIFACTS_DIR)

    def artifact(self, name: str) -> luigi.LocalTarget:
        path = Path(self.artifacts_dir) / name
        return luigi.LocalTarget(str(path))

    def ensure_parent(self, target: luigi.LocalTarget) -> None:
        Path(target.path).parent.mkdir(parents=True, exist_ok=True)

    def _params(self) -> dict:
        return dict(features_dir=self.features_dir, artifacts_dir=self.artifacts_dir)
