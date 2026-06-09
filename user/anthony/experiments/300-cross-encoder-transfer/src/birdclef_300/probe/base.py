"""Base class for exp 300 probe Luigi tasks (ported from exp 106 analysis/base)."""

from __future__ import annotations

from pathlib import Path

import luigi

FEATURES_DIR = "~/scratch/birdclef-2026/features/exp300"
ARTIFACTS_DIR = "artifacts"


class ProbeTask(luigi.Task):
    """Base task with shared features/artifacts directory parameters."""

    features_dir = luigi.Parameter(default=FEATURES_DIR)
    artifacts_dir = luigi.Parameter(default=ARTIFACTS_DIR)

    def artifact(self, name: str) -> luigi.LocalTarget:
        return luigi.LocalTarget(str(Path(self.artifacts_dir) / name))

    def ensure_parent(self, target: luigi.LocalTarget) -> None:
        Path(target.path).parent.mkdir(parents=True, exist_ok=True)
