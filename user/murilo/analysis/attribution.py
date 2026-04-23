"""Gradient saliency, LSE effective attention, and per-frame logit extraction.

Pure computation module — no Marimo dependency. Works on CPU.

Usage:
    from analysis.model_loader import load_model
    from analysis.attribution import compute_attribution, load_audio

    model, mel_transform, species_list = load_model(config, ckpt)
    waveform = load_audio("path/to/file.ogg", sample_rate=32000, duration=5.0)
    result = compute_attribution(model, mel_transform, waveform, species_idx=42,
                                  species_name="abethr1")
"""

from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AttributionResult:
    """Container for all attribution outputs for a single sample + species."""

    mel: np.ndarray  # (n_mels, time_frames) — input mel spectrogram
    gradient_saliency: np.ndarray  # (n_mels, time_frames) — |grad(logit w.r.t. mel)|
    frame_logits: np.ndarray  # (num_classes, T_coarse) — per-frame logits before pooling
    frame_probs: np.ndarray  # (num_classes, T_coarse) — sigmoid(frame_logits)
    clipwise_logits: np.ndarray  # (num_classes,) — final clipwise logits
    clipwise_probs: np.ndarray  # (num_classes,) — sigmoid(clipwise_logits)
    effective_attention: np.ndarray  # (T_coarse,) — which frames dominate pooling
    species_idx: int
    species_name: str


def load_audio(
    filepath: str,
    sample_rate: int = 32000,
    duration: float | None = 5.0,
    offset: float = 0.0,
) -> torch.Tensor:
    """Load audio file and return as float32 tensor.

    Args:
        filepath: Path to audio file (.ogg, .wav, .mp3, etc.)
        sample_rate: Target sample rate (resamples if needed).
        duration: Duration in seconds to load. None for full file.
        offset: Start time in seconds.

    Returns:
        (num_samples,) float32 tensor in [-1, 1].
    """
    y, _ = librosa.load(filepath, sr=sample_rate, duration=duration, offset=offset, mono=True)
    return torch.from_numpy(y).float()


def compute_gradient_saliency(
    model: torch.nn.Module,
    mel: torch.Tensor,
    species_idx: int,
) -> np.ndarray:
    """Compute gradient of clipwise logit w.r.t. mel input.

    Args:
        model: SEDFinetuneModule in eval mode.
        mel: (1, 1, n_mels, T) input mel spectrogram (detached from mel_transform).
        species_idx: Which species to compute saliency for.

    Returns:
        (n_mels, T) absolute gradient magnitude, normalized to [0, 1].
    """
    mel = mel.detach().clone().requires_grad_(True)

    clipwise_logits, _ = model(mel)
    target = clipwise_logits[0, species_idx]
    target.backward()

    grad = mel.grad[0, 0].abs()
    grad_max = grad.max()
    if grad_max > 0:
        grad = grad / grad_max
    return grad.detach().cpu().numpy()


def compute_lse_effective_attention(
    frame_logits: torch.Tensor,
    species_idx: int,
    temperature: float = 1.0,
) -> np.ndarray:
    """Compute LSE effective attention for a single species.

    The gradient of logsumexp(x/T) w.r.t. x is softmax(x/T).
    This shows which temporal frames dominate the LSE pooling.

    Args:
        frame_logits: (1, num_classes, T) from model forward.
        species_idx: Which species.
        temperature: LSE temperature from config (default 1.0).

    Returns:
        (T,) softmax weights — sums to 1.0.
    """
    logits = frame_logits[0, species_idx, :]  # (T,)
    attention = F.softmax(logits / temperature, dim=0)
    return attention.detach().cpu().numpy()


def compute_attribution(
    model: torch.nn.Module,
    mel_transform: torch.nn.Module,
    waveform: torch.Tensor,
    species_idx: int,
    species_name: str,
    lse_temperature: float = 1.0,
    device: str = "cpu",
) -> AttributionResult:
    """Compute all attributions for one sample.

    This is the main entry point. Computes mel spectrogram, runs the model,
    extracts gradient saliency, per-frame logits, and LSE effective attention.

    Args:
        model: SEDFinetuneModule in eval mode.
        mel_transform: Mel spectrogram transform.
        waveform: (num_samples,) raw audio tensor at the correct sample rate.
        species_idx: Target species index (0-233).
        species_name: Species code string for labeling.
        lse_temperature: From config sed_head.lse_temperature (default 1.0).
        device: "cpu" or "cuda".

    Returns:
        AttributionResult with all computed signals.
    """
    waveform = waveform.to(device)

    # Compute mel spectrogram (no gradient through mel_transform)
    with torch.no_grad():
        mel = mel_transform(waveform.unsqueeze(0))  # (1, 1, n_mels, T)
    mel_np = mel[0, 0].detach().cpu().numpy()  # (n_mels, T)

    # Forward pass (no gradient — just to get logits)
    with torch.no_grad():
        clipwise_logits, framewise_logits = model(mel)

    clipwise_logits_np = clipwise_logits[0].cpu().numpy()  # (num_classes,)
    clipwise_probs_np = torch.sigmoid(clipwise_logits[0]).cpu().numpy()
    frame_logits_np = framewise_logits[0].cpu().numpy()  # (num_classes, T_coarse)
    frame_probs_np = torch.sigmoid(framewise_logits[0]).cpu().numpy()

    # LSE effective attention
    effective_attention = compute_lse_effective_attention(
        framewise_logits, species_idx, lse_temperature
    )

    # Gradient saliency (requires a separate forward+backward pass)
    gradient_saliency = compute_gradient_saliency(model, mel, species_idx)

    return AttributionResult(
        mel=mel_np,
        gradient_saliency=gradient_saliency,
        frame_logits=frame_logits_np,
        frame_probs=frame_probs_np,
        clipwise_logits=clipwise_logits_np,
        clipwise_probs=clipwise_probs_np,
        effective_attention=effective_attention,
        species_idx=species_idx,
        species_name=species_name,
    )
