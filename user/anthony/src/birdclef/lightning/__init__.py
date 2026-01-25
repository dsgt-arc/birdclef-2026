"""PyTorch Lightning modules for distillation training."""

from birdclef.lightning.distillation_datamodule import DistillationDataModule
from birdclef.lightning.distillation_module import DistillationLitModule

__all__ = ["DistillationLitModule", "DistillationDataModule"]
