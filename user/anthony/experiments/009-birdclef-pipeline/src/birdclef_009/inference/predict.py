"""Inference: process soundscapes and produce per-5s-window predictions.

Each 10s chunk maps to two 5s submission rows via forward_windows():
  - window 0: mean-pool tokens 0-499 → head → row for {soundscape_id}_{chunk_start+5}
  - window 1: mean-pool tokens 500-999 → head → row for {soundscape_id}_{chunk_start+10}
"""

import logging
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from ..data.audio import load_audio_chunk

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_soundscape(
    model,
    audio_path: str,
    species_list: list[str],
    sample_rate: int = 32000,
    chunk_duration: float = 10.0,
    device: str = "cpu",
) -> list[dict]:
    """Process a soundscape file and produce per-5s-window predictions.

    Args:
        model: SSAMBAClassifier in eval mode
        audio_path: Path to soundscape audio file
        species_list: Ordered list of species names
        sample_rate: Target sample rate
        chunk_duration: Model input duration in seconds
        device: Device for inference

    Returns:
        List of dicts with row_id and species probabilities
    """
    model.eval()
    chunk_samples = int(chunk_duration * sample_rate)

    info = sf.info(audio_path)
    total_samples = int(info.frames * sample_rate / info.samplerate)
    soundscape_id = Path(audio_path).stem

    predictions = []
    chunk_start_sample = 0

    while chunk_start_sample < total_samples:
        real_samples = min(chunk_samples, total_samples - chunk_start_sample)
        waveform = load_audio_chunk(
            audio_path, sample_rate,
            start_sample=chunk_start_sample, num_samples=real_samples,
        )

        if waveform.shape[0] < chunk_samples:
            waveform = F.pad(waveform, (0, chunk_samples - waveform.shape[0]))

        waveform = waveform.unsqueeze(0).to(device)  # (1, num_samples)

        # Get per-window logits: (1, 2, C) for two 5s windows
        window_logits = model.forward_windows(waveform, window_size=500)
        window_probs = torch.sigmoid(window_logits).squeeze(0).cpu()  # (2, C)

        chunk_start_s = chunk_start_sample / sample_rate

        for w in range(window_probs.shape[0]):
            end_time = int(chunk_start_s + (w + 1) * 5)
            row_id = f"{soundscape_id}_{end_time}"
            row = {"row_id": row_id}
            for i, species in enumerate(species_list):
                row[species] = float(window_probs[w, i])
            predictions.append(row)

        chunk_start_sample += chunk_samples

    return predictions


def predict_directory(
    model,
    soundscape_dir: str,
    species_list: list[str],
    sample_rate: int = 32000,
    chunk_duration: float = 10.0,
    device: str = "cpu",
) -> list[dict]:
    """Process all soundscapes in a directory.

    Args:
        model: SSAMBAClassifier in eval mode
        soundscape_dir: Directory containing soundscape audio files
        species_list: Ordered list of species names
        sample_rate: Target sample rate
        chunk_duration: Model input duration
        device: Device for inference

    Returns:
        List of all prediction dicts across all soundscapes
    """
    soundscape_path = Path(soundscape_dir).expanduser()
    audio_files = sorted(
        list(soundscape_path.glob("*.ogg"))
        + list(soundscape_path.glob("*.wav"))
        + list(soundscape_path.glob("*.flac"))
    )

    logger.info(f"Processing {len(audio_files)} soundscapes from {soundscape_path}")

    all_predictions = []
    for i, audio_path in enumerate(audio_files):
        logger.info(f"  [{i+1}/{len(audio_files)}] {audio_path.name}")
        preds = predict_soundscape(
            model, str(audio_path), species_list,
            sample_rate=sample_rate, chunk_duration=chunk_duration,
            device=device,
        )
        all_predictions.extend(preds)

    logger.info(f"Total prediction rows: {len(all_predictions)}")
    return all_predictions
