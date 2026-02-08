"""BirdSet data pipeline for SSAMBA training.

Self-contained data loading: HuggingFace datasets -> resampled audio -> on-the-fly
mel-spectrogram via Kaldi fbank. Outputs (B, 1, 128, target_frames) spectrograms
with padding masks for variable-length audio.

Training recordings are split into non-overlapping 30s chunks. Shorter/last chunks
are zero-padded, and padding masks track which tokens are real vs padded.
"""

import logging
import math

import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Number of species classes per BirdSet dataset
BIRDSET_NUM_CLASSES = {
    "HSN": 21,
    "PER": 132,
    "NES": 43,
    "UHH": 27,
    "POW": 48,
    "SSW": 81,
    "SNE": 52,
    "NBP": 51,
}


class MelSpectrogramTransform:
    """Kaldi fbank mel-spectrogram with AudioSet normalization.

    Follows the EAT / SSAMBA convention:
      - torchaudio.compliance.kaldi.fbank
      - Normalize: (fbank - mean) / (std * 2)
      - Pad or center-crop to target_frames

    Returns (spectrogram, real_frames) to support padding masks.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 3000,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_shift = frame_shift_ms
        self.target_frames = target_frames
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.frame_shift_samples = int(sample_rate * frame_shift_ms / 1000)

    def __call__(self, waveform: torch.Tensor, real_samples: int = None):
        """Convert waveform to normalized mel-spectrogram.

        Args:
            waveform: (samples,) float tensor
            real_samples: Number of real audio samples (before padding).
                If None, all samples are considered real.

        Returns:
            Tuple of:
                (1, n_mels, target_frames) normalized mel-spectrogram
                int: number of real spectrogram frames
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.n_mels,
            dither=0.0,
            frame_shift=self.frame_shift,
        )  # (num_frames, n_mels)

        fbank = fbank.transpose(0, 1)  # (n_mels, num_frames)
        total_frames = fbank.shape[1]

        # Compute real frames from real_samples
        if real_samples is not None and real_samples < waveform.shape[0]:
            real_frames = min(
                (real_samples - 1) // self.frame_shift_samples + 1,
                total_frames,
            )
        else:
            real_frames = total_frames

        # Pad or crop to target_frames
        if fbank.shape[1] < self.target_frames:
            pad_length = self.target_frames - fbank.shape[1]
            fbank = torch.nn.functional.pad(fbank, (0, pad_length), value=0)
        elif fbank.shape[1] > self.target_frames:
            fbank = fbank[:, : self.target_frames]
            real_frames = min(real_frames, self.target_frames)

        # AudioSet normalization
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        return fbank.unsqueeze(0), real_frames  # (1, n_mels, target_frames), int


class BirdSetDataset(Dataset):
    """PyTorch Dataset wrapping a HuggingFace BirdSet split.

    For training: splits each recording into non-overlapping chunks of
    target_duration seconds. A 90s recording with target_duration=30s yields
    3 chunks. The last chunk is zero-padded if shorter.

    For validation (test_5s): each clip is already 5s, loaded as-is and
    padded to target_duration.
    """

    def __init__(
        self,
        hf_dataset,
        num_classes: int,
        target_sr: int = 32000,
        target_duration: float = 30.0,
        chunk: bool = False,
    ):
        self.dataset = hf_dataset
        self.num_classes = num_classes
        self.target_sr = target_sr
        self.target_samples = int(target_duration * target_sr)

        if chunk:
            # Build chunk index: [(dataset_idx, chunk_start, chunk_real_samples)]
            self.chunks = self._build_chunk_index()
        else:
            # One-to-one: each dataset sample is one item, padded to target_duration
            self.chunks = None

    def _build_chunk_index(self):
        """Pre-compute non-overlapping chunk offsets for all recordings."""
        chunks = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            # Get audio duration from metadata or file info
            audio_info = sample["audio"]
            audio_path = audio_info["path"]
            info = sf.info(audio_path)
            total_samples = int(info.frames * self.target_sr / info.samplerate)

            n_chunks = max(1, math.ceil(total_samples / self.target_samples))
            for c in range(n_chunks):
                start = c * self.target_samples
                real = min(self.target_samples, total_samples - start)
                chunks.append((i, start, real))
        return chunks

    def __len__(self):
        if self.chunks is not None:
            return len(self.chunks)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.chunks is not None:
            dataset_idx, chunk_start_target, chunk_real = self.chunks[idx]
        else:
            dataset_idx = idx
            chunk_start_target = 0
            chunk_real = None  # determined after loading

        sample = self.dataset[dataset_idx]

        # Load audio from file path
        audio_path = sample["audio"]["path"]
        audio, source_sr = sf.read(audio_path, dtype="float32")

        # Stereo -> mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio_tensor = torch.from_numpy(audio).float()

        # Resample if needed
        if source_sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, orig_freq=source_sr, new_freq=self.target_sr
            )

        if self.chunks is not None:
            # Extract chunk
            start = chunk_start_target
            end = start + self.target_samples
            audio_tensor = audio_tensor[start:end]
            real_samples = min(chunk_real, audio_tensor.shape[0])
        else:
            real_samples = audio_tensor.shape[0]

        # Pad to target_samples if needed
        if audio_tensor.shape[0] < self.target_samples:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, self.target_samples - audio_tensor.shape[0])
            )

        # Multi-hot labels
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        label_indices = sample.get("ebird_code_multilabel", [])
        if label_indices:
            for li in label_indices:
                if 0 <= li < self.num_classes:
                    labels[li] = 1.0

        return {"audio": audio_tensor, "labels": labels, "real_samples": real_samples}


class BirdSetDataModule(pl.LightningDataModule):
    """Lightning DataModule for BirdSet with on-the-fly mel-spectrogram generation."""

    def __init__(
        self,
        dataset_name: str = "HSN",
        batch_size: int = 16,
        num_workers: int = 8,
        sample_rate: int = 32000,
        target_duration: float = 30.0,
        n_mels: int = 128,
        frame_shift_ms: int = 10,
        target_frames: int = 3000,
        norm_mean: float = -4.268,
        norm_std: float = 4.569,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.num_classes = BIRDSET_NUM_CLASSES[dataset_name]

        self.mel_transform = MelSpectrogramTransform(
            sample_rate=sample_rate,
            n_mels=n_mels,
            frame_shift_ms=frame_shift_ms,
            target_frames=target_frames,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            logger.info(f"Loading {self.dataset_name} train split...")
            hf_train = load_dataset(
                "DBD-research-group/BirdSet",
                self.dataset_name,
                split="train",
                trust_remote_code=True,
            )
            self.train_dataset = BirdSetDataset(
                hf_train,
                num_classes=self.num_classes,
                target_sr=self.sample_rate,
                target_duration=self.target_duration,
                chunk=True,
            )
            logger.info(f"Train: {len(self.train_dataset)} chunks from {len(hf_train)} recordings")

            logger.info(f"Loading {self.dataset_name} test_5s split for validation...")
            hf_val = load_dataset(
                "DBD-research-group/BirdSet",
                self.dataset_name,
                split="test_5s",
                trust_remote_code=True,
            )
            self.val_dataset = BirdSetDataset(
                hf_val,
                num_classes=self.num_classes,
                target_sr=self.sample_rate,
                target_duration=self.target_duration,
                chunk=False,
            )
            logger.info(f"Val: {len(self.val_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_with_melspec,
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
            collate_fn=self._collate_with_melspec,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def _collate_with_melspec(self, batch):
        """Collate batch and apply mel-spectrogram transform.

        Returns:
            (mel_specs, labels, padding_mask) where:
                mel_specs: (B, 1, n_mels, target_frames)
                labels: (B, num_classes)
                padding_mask: (B, num_tokens) boolean, True for real tokens
        """
        audio = torch.stack([item["audio"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        real_samples_list = [item["real_samples"] for item in batch]

        mel_specs = []
        real_frames_list = []
        for i in range(audio.shape[0]):
            spec, real_frames = self.mel_transform(audio[i], real_samples_list[i])
            mel_specs.append(spec)
            real_frames_list.append(real_frames)

        mel_specs = torch.stack(mel_specs)

        # Build token-level padding mask
        # With patch_size=[128, 1], num_tokens = target_frames (1:1 mapping)
        num_tokens = self.mel_transform.target_frames
        padding_mask = torch.zeros(len(batch), num_tokens, dtype=torch.bool)
        for i, rf in enumerate(real_frames_list):
            padding_mask[i, :rf] = True

        return mel_specs, labels, padding_mask
