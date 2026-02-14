"""Shared audio loading and collation utilities."""

import logging

import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio_chunk(audio_path, target_sr, start_sample=0, num_samples=None):
    """Load a chunk of audio and resample to target sample rate.

    Uses sf.read(start, frames) for partial reads to avoid loading entire files.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        start_sample: Start position in target sample rate
        num_samples: Number of samples to read in target sample rate (None = to end)
    """
    info = sf.info(audio_path)
    source_sr = info.samplerate

    if not hasattr(load_audio_chunk, "_logged"):
        logger.info(f"Audio source sample rate: {source_sr}, target: {target_sr}")
        load_audio_chunk._logged = True

    # Convert target-domain positions to source-domain
    src_start = int(start_sample * source_sr / target_sr)
    if num_samples is not None:
        src_frames = int(num_samples * source_sr / target_sr) + 2
    else:
        src_frames = -1

    audio, _ = sf.read(audio_path, dtype="float32", start=src_start, frames=src_frames)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).float()
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=source_sr, new_freq=target_sr
        )

    if num_samples is not None and audio_tensor.shape[0] > num_samples:
        audio_tensor = audio_tensor[:num_samples]

    return audio_tensor


def collate_pretrain(batch):
    """Collate pretraining batch (waveforms only).

    Returns:
        dict with "waveform": (B, num_samples)
    """
    waveforms = torch.stack([item["waveform"] for item in batch])
    return {"waveform": waveforms}


def collate_finetune(batch):
    """Collate finetuning batch with padding masks.

    Returns:
        dict with waveform, clip_labels, padding_mask
    """
    waveforms = torch.stack([item["waveform"] for item in batch])
    clip_labels = torch.stack([item["clip_labels"] for item in batch])

    num_tokens = batch[0]["num_tokens"]
    masks = []
    for item in batch:
        real_tokens = item["real_tokens"]
        mask = torch.arange(num_tokens) < real_tokens
        masks.append(mask)
    padding_mask = torch.stack(masks)

    return {
        "waveform": waveforms,
        "clip_labels": clip_labels,
        "padding_mask": padding_mask,
    }
