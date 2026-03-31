"""SSL pretraining data: unlabeled 30s chunks from soundscapes + focal recordings."""

import hashlib
import logging
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import collate_pretrain, load_audio_chunk

logger = logging.getLogger(__name__)


def _file_hash_fraction(path: str) -> float:
    """Deterministic hash of filename to [0, 1) for reproducible splits."""
    h = hashlib.md5(Path(path).name.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class SSLPretrainDataset(Dataset):
    """Load soundscape + focal audio files, split into 30s non-overlapping chunks.

    Soundscapes (60s): split into two 30s halves.
    Focal recordings: one 30s chunk per file (padded if shorter).
    """

    def __init__(
        self,
        soundscape_dir: str | None = None,
        focal_dir: str | None = None,
        sample_rate: int = 32000,
        chunk_duration: float = 30.0,
        val_fraction: float = 0.0,
        is_val: bool = False,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)

        self.chunks = []

        # Soundscapes: split into 30s halves, with train/val split by file hash
        if soundscape_dir:
            soundscape_path = Path(soundscape_dir).expanduser()
            audio_files = sorted(
                list(soundscape_path.glob("*.ogg"))
                + list(soundscape_path.glob("*.wav"))
                + list(soundscape_path.glob("*.flac"))
            )
            n_sc_files = 0
            for audio_path in audio_files:
                # Deterministic train/val split by filename hash
                frac = _file_hash_fraction(str(audio_path))
                file_is_val = frac < val_fraction
                if file_is_val != is_val:
                    continue
                n_sc_files += 1

                try:
                    info = sf.info(str(audio_path))
                except Exception as e:
                    logger.warning(f"Skipping {audio_path}: {e}")
                    continue
                total_samples = int(info.frames * sample_rate / info.samplerate)

                # Split into non-overlapping chunks
                pos = 0
                while pos < total_samples:
                    real_samples = min(self.chunk_samples, total_samples - pos)
                    self.chunks.append((str(audio_path), pos, real_samples))
                    pos += self.chunk_samples

            split = "val" if is_val else "train"
            logger.info(
                f"Soundscapes ({split}): {len(self.chunks)} chunks from {n_sc_files} files"
            )

        # Focal recordings: one chunk per file (no train/val split — all go to train)
        n_focal_chunks = 0
        if focal_dir and not is_val:
            focal_path = Path(focal_dir).expanduser()
            focal_files = sorted(focal_path.rglob("*.ogg"))
            if not focal_files:
                focal_files = sorted(
                    list(focal_path.rglob("*.wav"))
                    + list(focal_path.rglob("*.flac"))
                )
            # Skip very short recordings (<2s) — they produce too few tokens
            # for meaningful masking and drag down the batch min_real count.
            min_focal_samples = int(2.0 * sample_rate)
            n_skipped = 0
            for audio_path in focal_files:
                try:
                    info = sf.info(str(audio_path))
                except Exception as e:
                    logger.warning(f"Skipping {audio_path}: {e}")
                    continue
                total_samples = int(info.frames * sample_rate / info.samplerate)
                if total_samples < min_focal_samples:
                    n_skipped += 1
                    continue
                real_samples = min(self.chunk_samples, total_samples)
                self.chunks.append((str(audio_path), 0, real_samples))
                n_focal_chunks += 1

            logger.info(f"Focal recordings: {n_focal_chunks} chunks (skipped {n_skipped} under 2s)")

        logger.info(f"SSLPretrainDataset total: {len(self.chunks)} chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, real_samples = self.chunks[idx]
        waveform = load_audio_chunk(
            audio_path, self.sample_rate,
            start_sample=start_sample, num_samples=real_samples,
        )
        # Zero-pad short recordings. Pass real_samples so the model can
        # exclude padded positions from masking and loss computation.
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )
        return {"waveform": waveform, "real_samples": real_samples}


class SSLDataModule(pl.LightningDataModule):
    """Lightning DataModule for SSL pretraining with train/val split."""

    def __init__(
        self,
        soundscape_dir: str | None = None,
        focal_dir: str | None = None,
        sample_rate: int = 32000,
        chunk_duration: float = 30.0,
        batch_size: int = 16,
        num_workers: int = 6,
        val_fraction: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.soundscape_dir = soundscape_dir
        self.focal_dir = focal_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.train_dataset = SSLPretrainDataset(
                soundscape_dir=self.soundscape_dir,
                focal_dir=self.focal_dir,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                val_fraction=self.val_fraction,
                is_val=False,
            )
        if self.val_dataset is None and self.val_fraction > 0:
            self.val_dataset = SSLPretrainDataset(
                soundscape_dir=self.soundscape_dir,
                focal_dir=None,  # Val is soundscapes only
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
                val_fraction=self.val_fraction,
                is_val=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pretrain,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_pretrain,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=False,
        )
