"""BirdCLEF 2026 focal recording dataset for linear probe evaluation."""

import csv
import hashlib
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .audio import load_audio_chunk

logger = logging.getLogger(__name__)


def _file_hash_fraction(path: str) -> float:
    """Deterministic hash of filename to [0, 1) for reproducible splits."""
    h = hashlib.md5(Path(path).name.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


@torch.no_grad()
def compute_mel_amplitude_to_db(
    mel_spec_layer: torch.nn.Module,
    waveforms: torch.Tensor,
    target_frames: int,
) -> torch.Tensor:
    """Compute AmplitudeToDB-normalized mel spectrogram.

    Replicates SSAMBAPretrainModel.compute_mel() without grid jitter.
    Must be numerically identical to the pretrain pipeline.

    Args:
        mel_spec_layer: nnAudio MelSpectrogram module
        waveforms: (B, num_samples)
        target_frames: Target number of mel frames (pad/truncate)

    Returns:
        (B, 1, n_mels, target_frames) normalized mel spectrogram
    """
    waveforms = waveforms.float()
    mel = mel_spec_layer(waveforms)  # (B, n_mels, T)

    # AmplitudeToDB normalization (matches pretrain.py lines 176-185)
    mel = 10.0 * torch.log10(mel.clamp(min=1e-10))
    mel_max = mel.amax(dim=(-2, -1), keepdim=True)
    mel = torch.clamp(mel, min=mel_max - 80.0)
    mean = mel.mean(dim=(-2, -1), keepdim=True)
    std = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    mel = (mel - mean) / std
    mel_min = mel.amin(dim=(-2, -1), keepdim=True)
    mel_max2 = mel.amax(dim=(-2, -1), keepdim=True)
    mel = (mel - mel_min) / (mel_max2 - mel_min + 1e-6)

    T_mel = mel.shape[2]
    if T_mel > target_frames:
        mel = mel[:, :, :target_frames]
    elif T_mel < target_frames:
        mel = F.pad(mel, (0, target_frames - T_mel), value=0)

    return mel.unsqueeze(1)  # (B, 1, n_mels, target_frames)


def _load_taxonomy(taxonomy_csv: str) -> dict[str, int]:
    """Load taxonomy.csv and return species_id → class_index mapping."""
    label_to_idx = {}
    with open(taxonomy_csv) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            label_to_idx[row["primary_label"]] = i
    return label_to_idx


def _load_train_csv(train_csv: str) -> list[dict]:
    """Load train.csv and parse labels."""
    records = []
    with open(train_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            secondary = row.get("secondary_labels", "[]")
            # Parse secondary_labels: "['sp1', 'sp2']" or "[]"
            if secondary and secondary != "[]":
                secondary = [s.strip().strip("'\"") for s in secondary.strip("[]").split(",") if s.strip()]
            else:
                secondary = []
            records.append({
                "filename": row["filename"],
                "primary_label": row["primary_label"],
                "secondary_labels": secondary,
            })
    return records


def compute_padding_mask(real_frames: int, target_frames: int, patch_size: tuple, n_mels: int) -> torch.Tensor:
    """Compute patch-granularity padding mask for 2D patches.

    Freq-first token order (default Conv2d flatten): token i has
    freq_idx = i // t_dim, time_idx = i % t_dim.
    A token is real if its time position < real_t_patches.

    Args:
        real_frames: Number of real mel frames
        target_frames: Total mel frames (including padding)
        patch_size: (freq_patch, time_patch)
        n_mels: Number of mel bins

    Returns:
        (num_tokens,) boolean tensor, True for real tokens
    """
    f_patch, t_patch = patch_size
    f_dim = n_mels // f_patch
    t_dim = target_frames // t_patch
    num_tokens = f_dim * t_dim

    real_t_patches = min(real_frames // t_patch, t_dim)

    # Freq-first order: token i → time_idx = i % t_dim
    indices = torch.arange(num_tokens)
    mask = (indices % t_dim) < real_t_patches
    return mask


class BirdCLEFFocalDataset(Dataset):
    """BirdCLEF 2026 focal recordings with multi-label species targets.

    Loads raw audio, pads to chunk_duration, returns waveform + labels.
    Mel computation is done in the collate function or model.
    """

    def __init__(
        self,
        train_csv: str,
        taxonomy_csv: str,
        audio_dir: str,
        sample_rate: int = 32000,
        chunk_duration: float = 30.0,
        val_fraction: float = 0.20,
        is_val: bool = False,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)

        self.label_to_idx = _load_taxonomy(taxonomy_csv)
        self.num_classes = len(self.label_to_idx)
        records = _load_train_csv(train_csv)

        # Hash-based train/val split
        self.items = []
        for rec in records:
            frac = _file_hash_fraction(rec["filename"])
            file_is_val = frac < val_fraction
            if file_is_val != is_val:
                continue
            self.items.append(rec)

        split = "val" if is_val else "train"
        logger.info(f"Focal ({split}): {len(self.items)} recordings, {self.num_classes} classes")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        audio_path = self.audio_dir / rec["filename"]

        waveform = load_audio_chunk(str(audio_path), self.sample_rate)
        real_samples = waveform.shape[0]

        # Pad to chunk_duration
        if waveform.shape[0] < self.chunk_samples:
            waveform = F.pad(waveform, (0, self.chunk_samples - waveform.shape[0]))
        elif waveform.shape[0] > self.chunk_samples:
            waveform = waveform[:self.chunk_samples]
            real_samples = self.chunk_samples

        # Multi-hot label vector
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        primary = rec["primary_label"]
        if primary in self.label_to_idx:
            labels[self.label_to_idx[primary]] = 1.0
        for sec in rec["secondary_labels"]:
            if sec in self.label_to_idx:
                labels[self.label_to_idx[sec]] = 1.0

        return {"waveform": waveform, "labels": labels, "real_samples": real_samples}


class FocalDataModule(pl.LightningDataModule):
    """Lightning DataModule for BirdCLEF focal recordings.

    Returns raw waveforms — mel computation happens on GPU in the model.
    """

    def __init__(
        self,
        train_csv: str,
        taxonomy_csv: str,
        audio_dir: str,
        sample_rate: int = 32000,
        chunk_duration: float = 30.0,
        batch_size: int = 16,
        num_workers: int = 6,
        val_fraction: float = 0.20,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_csv = train_csv
        self.taxonomy_csv = taxonomy_csv
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.train_dataset = BirdCLEFFocalDataset(
                self.train_csv, self.taxonomy_csv, self.audio_dir,
                self.sample_rate, self.chunk_duration,
                self.val_fraction, is_val=False,
            )
        if self.val_dataset is None:
            self.val_dataset = BirdCLEFFocalDataset(
                self.train_csv, self.taxonomy_csv, self.audio_dir,
                self.sample_rate, self.chunk_duration,
                self.val_fraction, is_val=True,
            )

    @property
    def num_classes(self):
        if self.train_dataset is not None:
            return self.train_dataset.num_classes
        return len(_load_taxonomy(self.taxonomy_csv))

    @staticmethod
    def _collate(batch):
        """Collate returning raw waveforms + labels + real_samples.

        Mel computation and padding masks are handled on GPU in the model.
        """
        waveforms = torch.stack([item["waveform"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        real_samples = torch.tensor(
            [item["real_samples"] for item in batch], dtype=torch.long,
        )
        return waveforms, labels, real_samples

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
