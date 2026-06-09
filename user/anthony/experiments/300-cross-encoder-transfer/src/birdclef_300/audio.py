"""16 kHz audio loading for the foundational encoders.

The shared `birdclef.encoders` wrappers (BEATs, AST, EAT, SSAMBA-HF) all expect
mono float32 waveforms at 16 kHz, in contrast to exp 106's 32 kHz Perch loader.
This module is the 16 kHz analogue of `birdclef_106.audio`.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

ENCODER_SR = 16000
ENCODER_WINDOW_SEC = 5.0
ENCODER_WINDOW_SAMPLES = int(ENCODER_SR * ENCODER_WINDOW_SEC)  # 80000


def load_audio_16k(
    path: str,
    start_sec: float | None = None,
    duration_sec: float | None = None,
) -> np.ndarray:
    """Load audio, resample to 16 kHz mono, return a 1-D float32 array.

    Args:
        path: Path to audio file (wav, ogg, flac, ...).
        start_sec: Optional start offset in seconds (whole file if None).
        duration_sec: Optional duration in seconds to read (to EOF if None).
    """
    import torch
    import torchaudio.functional as taF

    if start_sec is not None or duration_sec is not None:
        info = sf.info(path)
        sr = info.samplerate
        start_frame = int((start_sec or 0.0) * sr)
        stop_frame = start_frame + int(duration_sec * sr) if duration_sec is not None else None
        audio, sr = sf.read(path, start=start_frame, stop=stop_frame, dtype="float32")
    else:
        audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != ENCODER_SR:
        audio_t = torch.from_numpy(audio).float()
        audio_t = taF.resample(audio_t, orig_freq=sr, new_freq=ENCODER_SR)
        audio = audio_t.numpy()

    return audio.astype(np.float32, copy=False)


def fit_window(wav: np.ndarray, window_samples: int = ENCODER_WINDOW_SAMPLES) -> np.ndarray:
    """Pad with zeros or truncate to exactly `window_samples`."""
    if len(wav) < window_samples:
        return np.pad(wav, (0, window_samples - len(wav))).astype(np.float32, copy=False)
    return wav[:window_samples].astype(np.float32, copy=False)


def window_audio(audio: np.ndarray, window_samples: int = ENCODER_WINDOW_SAMPLES) -> np.ndarray:
    """Split audio into non-overlapping fixed-size windows (zero-padded tail)."""
    n = max(1, int(np.ceil(len(audio) / window_samples)))
    padded_len = n * window_samples
    if len(audio) < padded_len:
        audio = np.pad(audio, (0, padded_len - len(audio)))
    return audio[: n * window_samples].reshape(n, window_samples).astype(np.float32, copy=False)
