"""ESC-50 dataset for environmental sound classification."""

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from ..audio import load_audio, pad_or_trim


class ESC50Dataset(Dataset):
    """ESC-50 Environmental Sound Classification dataset.

    The dataset contains 2000 5-second audio clips organized into 50 classes.
    Uses 5-fold cross-validation with pre-defined folds.

    Args:
        root: Path to ESC-50 directory (containing audio/ and meta/)
        folds: List of fold numbers to include (1-5). Use for train/val splits.
        target_sr: Target sample rate (default 16kHz)
        target_length: Target length in samples (default 5 seconds at 16kHz)
        transform: Optional transform to apply to waveforms
    """

    def __init__(
        self,
        root: str | Path,
        folds: list[int] | None = None,
        target_sr: int = 16000,
        target_length: int | None = None,
        transform=None,
    ):
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.target_sr = target_sr
        self.target_length = target_length or (5 * target_sr)  # 5 seconds default
        self.transform = transform

        # Load metadata
        meta_path = self.root / "meta" / "esc50.csv"
        self.metadata = pd.read_csv(meta_path)

        # Filter by folds if specified
        if folds is not None:
            self.metadata = self.metadata[self.metadata["fold"].isin(folds)]

        self.metadata = self.metadata.reset_index(drop=True)

        # Build category mapping
        self.categories = sorted(self.metadata["category"].unique())
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        audio_path = self.audio_dir / row["filename"]

        # Load and preprocess audio
        waveform = load_audio(audio_path, target_sr=self.target_sr)
        waveform = pad_or_trim(waveform, self.target_length)

        if self.transform is not None:
            waveform = self.transform(waveform)

        label = self.cat2idx[row["category"]]
        return waveform, label

    @property
    def num_classes(self) -> int:
        return len(self.categories)

    def get_fold_splits(self, val_fold: int) -> tuple["ESC50Dataset", "ESC50Dataset"]:
        """Create train/val splits for cross-validation.

        Args:
            val_fold: Fold number to use for validation (1-5)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_folds = [f for f in range(1, 6) if f != val_fold]

        train_ds = ESC50Dataset(
            self.root,
            folds=train_folds,
            target_sr=self.target_sr,
            target_length=self.target_length,
            transform=self.transform,
        )
        val_ds = ESC50Dataset(
            self.root,
            folds=[val_fold],
            target_sr=self.target_sr,
            target_length=self.target_length,
            transform=self.transform,
        )
        return train_ds, val_ds
