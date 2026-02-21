"""PyTorch Dataset and DataLoader for BirdSet data.

This module provides PyTorch-native data loading utilities for transfer learning
where we extract embeddings from external models and train linear probes.

Used by:
- Experiment 001: BirdSet Bioacoustics Benchmark (transfer learning with BirdNET, Perch)

Usage:
    from birdclef.torch_data import BirdSetDataset, create_dataloader
    from birdclef.datasets import load_birdset_dataset, get_num_classes

    # Load HuggingFace dataset
    hf_dataset = load_birdset_dataset("HSN", split="train")
    num_classes = get_num_classes("HSN")

    # Create PyTorch Dataset
    dataset = BirdSetDataset(
        hf_dataset,
        num_classes=num_classes,
        target_sr=32000,
        target_duration=5.0,
    )

    # Create DataLoader
    dataloader = create_dataloader(dataset, batch_size=32, num_workers=4)
"""

import logging
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def resample_audio(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        audio: Audio array [samples]
        source_sr: Source sample rate (Hz)
        target_sr: Target sample rate (Hz)

    Returns:
        Resampled audio array
    """
    if source_sr == target_sr:
        return audio

    audio_resampled = librosa.resample(
        audio,
        orig_sr=source_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
    )
    return audio_resampled


def pad_or_crop(
    audio: np.ndarray,
    target_duration_sec: float,
    sr: int,
    mode: str = "pad",
) -> np.ndarray:
    """Pad or crop audio to target duration.

    Args:
        audio: Audio array [samples]
        target_duration_sec: Target duration in seconds
        sr: Sample rate (Hz)
        mode: "pad" (zero-pad if short), "crop" (center crop if long), or "repeat" (repeat if short)

    Returns:
        Audio array with target length [target_samples]
    """
    target_samples = int(target_duration_sec * sr)
    current_samples = len(audio)

    if current_samples == target_samples:
        return audio

    elif current_samples < target_samples:
        # Audio is too short
        if mode == "pad":
            # Zero-pad at the end
            padding = target_samples - current_samples
            audio = np.pad(audio, (0, padding), mode="constant", constant_values=0)
        elif mode == "repeat":
            # Repeat audio to fill duration
            repeats = int(np.ceil(target_samples / current_samples))
            audio = np.tile(audio, repeats)[:target_samples]
        else:
            raise ValueError(f"Unknown padding mode: {mode}")

    else:
        # Audio is too long - center crop
        start = (current_samples - target_samples) // 2
        audio = audio[start : start + target_samples]

    return audio


class BirdSetDataset(Dataset):
    """PyTorch Dataset for BirdSet data.

    Handles audio loading, resampling, and padding/cropping.
    Used for Phase 2B (Transfer Learning) where we need to:
    - Extract embeddings from external models
    - Train linear probes on embeddings

    For Phase 2A (Direct Inference), use the BMZ model's .predict() API directly.

    Args:
        dataset: HuggingFace Dataset object (from load_birdset_dataset)
        num_classes: Number of classes for multi-hot encoding (required if return_labels=True)
        target_sr: Target sample rate (Hz)
        target_duration: Target duration (seconds)
        return_labels: If True, return multi-hot labels
        return_metadata: If True, return metadata dict
    """

    def __init__(
        self,
        dataset,
        num_classes: int | None = None,
        target_sr: int = 32000,
        target_duration: float = 5.0,
        return_labels: bool = True,
        return_metadata: bool = False,
    ):
        self.dataset = dataset
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.return_labels = return_labels
        self.return_metadata = return_metadata

        # Determine number of classes
        if return_labels:
            if num_classes is not None:
                self.num_classes = num_classes
            else:
                # Auto-detect from first few samples (quick heuristic)
                logger.info("Auto-detecting number of classes from dataset...")
                max_label = 0
                check_samples = min(100, len(dataset))
                for i in range(check_samples):
                    sample = dataset[i]
                    if "ebird_code_multilabel" in sample and sample["ebird_code_multilabel"]:
                        max_label = max(max_label, max(sample["ebird_code_multilabel"]))

                self.num_classes = max_label + 1  # +1 because indices start at 0
                logger.warning(
                    f"Auto-detected {self.num_classes} classes from first {check_samples} samples. "
                    f"For accuracy, specify num_classes explicitly."
                )
        else:
            self.num_classes = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            dict with keys:
                - audio: Tensor [samples]
                - labels: Tensor [num_classes] (if return_labels=True)
                - metadata: dict (if return_metadata=True)
        """
        sample = self.dataset[idx]

        # Extract audio from file path
        audio_path = sample["audio"]["path"]
        audio, source_sr = sf.read(audio_path, dtype="float32")

        # Convert stereo to mono if needed
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # Average across channels

        original_length = len(audio) / source_sr  # Save before processing

        # Resample if needed
        if source_sr != self.target_sr:
            audio = resample_audio(audio, source_sr, self.target_sr)

        # Pad or crop to target duration
        audio = pad_or_crop(audio, self.target_duration, self.target_sr, mode="pad")

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        result = {"audio": audio_tensor}

        # Add labels if requested
        if self.return_labels:
            # Create multi-hot vector from label indices
            labels = torch.zeros(self.num_classes, dtype=torch.float32)

            if "ebird_code_multilabel" in sample:
                # Convert indices to multi-hot
                label_indices = sample["ebird_code_multilabel"]
                if label_indices:  # Check if not empty
                    for label_idx in label_indices:
                        if 0 <= label_idx < self.num_classes:
                            labels[label_idx] = 1.0

            result["labels"] = labels

        # Add metadata if requested
        if self.return_metadata:
            metadata = {
                "filename": sample.get("filepath", ""),
                "ebird_code": sample.get("ebird_code", []),
                "original_sr": source_sr,
                "original_length": original_length,
            }
            result["metadata"] = metadata

        return result


def collate_fn(batch):
    """Collate function for DataLoader.

    Handles variable-length metadata fields gracefully.

    Args:
        batch: List of sample dicts from BirdSetDataset

    Returns:
        Batched dict with stacked tensors
    """
    # Stack audio
    audio = torch.stack([item["audio"] for item in batch])

    result = {"audio": audio}

    # Stack labels if present
    if "labels" in batch[0]:
        labels = torch.stack([item["labels"] for item in batch])
        result["labels"] = labels

    # Collect metadata if present
    if "metadata" in batch[0]:
        # Keep as list since metadata fields vary
        result["metadata"] = [item["metadata"] for item in batch]

    return result


def create_dataloader(
    dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for BirdSet data.

    Args:
        dataset: BirdSetDataset instance
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DataLoader

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Faster GPU transfer
        **kwargs,
    )
