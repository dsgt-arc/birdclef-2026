"""Soundscape pretraining data: unlabeled 10s chunks from OGG files."""

import logging
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import collate_pretrain, load_audio_chunk

logger = logging.getLogger(__name__)


class SoundscapePretrainDataset(Dataset):
    """Load soundscape OGG files, split into 10s non-overlapping chunks. No labels."""

    def __init__(
        self,
        soundscape_dir: str,
        sample_rate: int = 32000,
        chunk_duration: float = 10.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)

        soundscape_path = Path(soundscape_dir).expanduser()
        audio_files = sorted(soundscape_path.glob("*.ogg"))
        if not audio_files:
            # Also try .wav and .flac
            audio_files = sorted(
                list(soundscape_path.glob("*.wav"))
                + list(soundscape_path.glob("*.flac"))
                + list(soundscape_path.glob("*.mp3"))
            )

        # Build chunk index: (file_path, start_sample, num_real_samples)
        self.chunks = []
        for audio_path in audio_files:
            try:
                info = sf.info(str(audio_path))
            except Exception as e:
                logger.warning(f"Skipping {audio_path}: {e}")
                continue
            total_samples = int(info.frames * sample_rate / info.samplerate)
            pos = 0
            while pos < total_samples:
                real_samples = min(self.chunk_samples, total_samples - pos)
                self.chunks.append((str(audio_path), pos, real_samples))
                pos += self.chunk_samples

        logger.info(
            f"SoundscapePretrainDataset: {len(self.chunks)} chunks "
            f"from {len(audio_files)} files"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, real_samples = self.chunks[idx]
        waveform = load_audio_chunk(
            audio_path, self.sample_rate,
            start_sample=start_sample, num_samples=real_samples,
        )
        # Pad to chunk_samples if short
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )
        return {"waveform": waveform}


class SoundscapeDataModule(pl.LightningDataModule):
    """Lightning DataModule for soundscape pretraining."""

    def __init__(
        self,
        soundscape_dir: str,
        sample_rate: int = 32000,
        chunk_duration: float = 10.0,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.soundscape_dir = soundscape_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = None

    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = SoundscapePretrainDataset(
                soundscape_dir=self.soundscape_dir,
                sample_rate=self.sample_rate,
                chunk_duration=self.chunk_duration,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pretrain,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
        )
