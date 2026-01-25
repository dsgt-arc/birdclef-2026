"""BirdSet dataset loading and preprocessing utilities.

This module re-exports from the split modules for backwards compatibility:
- datasets.py: HuggingFace dataset loading, constants (Phase 2A & 2B)
- torch_data.py: PyTorch Dataset/DataLoader, preprocessing (Phase 2B only)

For new code, prefer importing directly from the specific modules:
    from birdset_benchmark.datasets import load_birdset_dataset, get_num_classes
    from birdset_benchmark.torch_data import BirdSetDataset, create_dataloader

This module maintains backwards compatibility with existing imports:
    from birdset_benchmark.data import load_birdset_dataset, BirdSetDataset
"""

# Re-export from datasets.py (Phase 2A & 2B)
from birdset_benchmark.datasets import (
    BIRDSET_DATASETS,
    BIRDSET_NUM_CLASSES,
    get_num_classes,
    load_birdset_dataset,
)

# Re-export from torch_data.py (Phase 2B)
from birdset_benchmark.torch_data import (
    BirdSetDataset,
    collate_fn,
    create_dataloader,
    pad_or_crop,
    resample_audio,
)

__all__ = [
    # From datasets.py
    "BIRDSET_DATASETS",
    "BIRDSET_NUM_CLASSES",
    "get_num_classes",
    "load_birdset_dataset",
    # From torch_data.py
    "resample_audio",
    "pad_or_crop",
    "BirdSetDataset",
    "collate_fn",
    "create_dataloader",
]
