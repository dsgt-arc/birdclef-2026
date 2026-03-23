"""PyTorch Lightning modules and data modules.

Imports are lazy to avoid pulling in heavy dependencies (e.g., HuggingFace
`datasets`, `librosa`) when only a subset of modules is needed.
"""


def __getattr__(name):
    if name == "BirdSetDataModule":
        from birdclef.lightning.birdset_datamodule import BirdSetDataModule

        return BirdSetDataModule
    if name == "StreamingBirdSetDataModule":
        from birdclef.lightning.streaming_datamodule import StreamingBirdSetDataModule

        return StreamingBirdSetDataModule
    if name == "SupervisedViTModule":
        from birdclef.lightning.supervised_vit import SupervisedViTModule

        return SupervisedViTModule
    if name == "StudentFinetuneModule":
        from birdclef.lightning.student_finetune import StudentFinetuneModule

        return StudentFinetuneModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BirdSetDataModule",
    "StreamingBirdSetDataModule",
    "StudentFinetuneModule",
    "SupervisedViTModule",
]
