"""AudioSet WebDataset loader for streaming large-scale audio data.

Uses the confit/audioset-16khz-wds HuggingFace dataset:
https://huggingface.co/datasets/confit/audioset-16khz-wds

WebDataset format allows efficient streaming without loading all data into memory.
"""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from ..audio import pad_or_trim


class AudioSetWebDataset(IterableDataset):
    """AudioSet dataset using WebDataset format for streaming.

    Expects tar files in WebDataset format with structure:
        {sample_id}.flac - Audio file (16kHz)
        {sample_id}.json - Metadata with 'labels' field

    Args:
        shard_paths: List of paths to tar shards or glob pattern
        target_sr: Target sample rate (default 16kHz, data is already 16kHz)
        target_length: Target length in samples (default 10 seconds at 16kHz)
        transform: Optional transform to apply to waveforms
        shuffle: Whether to shuffle samples within shards
    """

    def __init__(
        self,
        shard_paths: list[str | Path] | str,
        target_sr: int = 16000,
        target_length: int | None = None,
        transform=None,
        shuffle: bool = False,
    ):
        import webdataset as wds

        if isinstance(shard_paths, (str, Path)):
            shard_paths = str(shard_paths)

        self.shard_paths = shard_paths
        self.target_sr = target_sr
        self.target_length = target_length or (10 * target_sr)  # 10 seconds default
        self.transform = transform
        self.shuffle = shuffle

        # Create base dataset
        self.dataset = wds.WebDataset(shard_paths)
        if shuffle:
            self.dataset = self.dataset.shuffle(1000)

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes to numpy array."""
        import io
        import soundfile as sf

        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        if sr != self.target_sr:
            import torchaudio.functional as F
            audio_tensor = torch.from_numpy(audio_data).float()
            audio_tensor = F.resample(audio_tensor, sr, self.target_sr)
            audio_data = audio_tensor.numpy()
        return audio_data.astype(np.float32)

    def _process_sample(self, sample: dict) -> tuple[torch.Tensor, dict]:
        """Process a single sample from WebDataset."""
        # Get audio (try multiple common extensions)
        audio_bytes = sample.get("flac") or sample.get("mp3") or sample.get("wav")
        if audio_bytes is None:
            raise ValueError(f"No audio found in sample: {list(sample.keys())}")

        # Decode audio
        waveform = self._decode_audio(audio_bytes)

        # Pad or trim to target length
        waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dim
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Mono

        waveform = pad_or_trim(waveform.squeeze(0), self.target_length)

        if self.transform is not None:
            waveform = self.transform(waveform)

        # Get metadata
        meta = {}
        if "json" in sample:
            meta = json.loads(sample["json"])

        return waveform, meta

    def __iter__(self) -> Iterator[tuple[torch.Tensor, dict]]:
        for sample in self.dataset:
            try:
                yield self._process_sample(sample)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue


def get_audioset_shards(
    base_dir: str | Path,
    split: str = "train",
    version: str = "20k",
) -> str:
    """Get shard paths for AudioSet.

    Args:
        base_dir: Base directory containing audioset-wds
        split: Split name ('train' or 'test')
        version: Dataset version ('20k', '500k', '2m')

    Returns:
        Glob pattern for shards
    """
    base_dir = Path(base_dir)
    return str(base_dir / version / split / "*.tar")


class AudioSetDistillationDataset(IterableDataset):
    """AudioSet dataset for distillation training.

    Yields (waveform, sample_id) tuples for generating teacher targets.
    Supports worker sharding for multi-GPU training.

    Args:
        shard_pattern: Glob pattern for tar shards
        target_length: Target length in samples (default 10 seconds at 16kHz)
        shuffle: Whether to shuffle within shards
    """

    def __init__(
        self,
        shard_pattern: str,
        target_length: int = 160000,  # 10 seconds at 16kHz
        shuffle: bool = True,
    ):
        import webdataset as wds

        self.shard_pattern = shard_pattern
        self.target_length = target_length

        # Build pipeline
        self.dataset = (
            wds.WebDataset(shard_pattern, shardshuffle=shuffle)
            .shuffle(1000 if shuffle else 0)
            .decode("pil")  # Handles audio decoding
        )

    def _process(self, sample: dict) -> tuple[torch.Tensor, str]:
        """Process sample to (waveform, sample_id)."""
        import io
        import soundfile as sf

        # Get sample ID from __key__
        sample_id = sample.get("__key__", "unknown")

        # Get audio bytes
        audio_bytes = sample.get("flac") or sample.get("mp3") or sample.get("wav")
        if audio_bytes is None:
            raise ValueError(f"No audio in sample {sample_id}")

        # Decode
        if isinstance(audio_bytes, bytes):
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        else:
            # Already decoded by webdataset
            audio_data = np.array(audio_bytes)
            sr = 16000  # Assume 16kHz

        waveform = torch.from_numpy(audio_data.astype(np.float32))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=-1)  # Mono

        # Pad or trim
        waveform = pad_or_trim(waveform, self.target_length)

        return waveform, sample_id

    def __iter__(self):
        for sample in self.dataset:
            try:
                yield self._process(sample)
            except Exception as e:
                print(f"Error: {e}")
                continue
