"""Audio loading for Perch v2.

Perch v2 expects mono float32 waveforms at 32 kHz, passed in 5-second windows
(160000 samples). Adapted from exp 104's `_load_audio_32k`.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

PERCH_SR = 32000
PERCH_WINDOW_SEC = 5.0
PERCH_WINDOW_SAMPLES = int(PERCH_SR * PERCH_WINDOW_SEC)  # 160000


def load_audio_32k(path: str, start_sec: float | None = None, duration_sec: float | None = None) -> np.ndarray:
    """Load audio, resample to 32 kHz mono, return float32 numpy array.

    Args:
        path: Path to audio file (wav, ogg, flac, ...).
        start_sec: Optional start offset in seconds. Reads the whole file if None.
        duration_sec: Optional duration in seconds to read. Reads to EOF if None.

    Returns:
        1-D float32 array of 32 kHz mono samples.
    """
    import torch
    import torchaudio.functional as taF

    # Use partial reads when start/duration specified (efficient for large files).
    if start_sec is not None or duration_sec is not None:
        info = sf.info(path)
        sr = info.samplerate
        start_frame = int((start_sec or 0.0) * sr)
        if duration_sec is not None:
            stop_frame = start_frame + int(duration_sec * sr)
        else:
            stop_frame = None
        audio, sr = sf.read(path, start=start_frame, stop=stop_frame, dtype="float32")
    else:
        audio, sr = sf.read(path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != PERCH_SR:
        audio_t = torch.from_numpy(audio).float()
        audio_t = taF.resample(audio_t, orig_freq=sr, new_freq=PERCH_SR)
        audio = audio_t.numpy()

    return audio.astype(np.float32, copy=False)


def window_audio(audio: np.ndarray, window_samples: int = PERCH_WINDOW_SAMPLES) -> np.ndarray:
    """Split audio into non-overlapping fixed-size windows.

    Pads the tail to a full window with zeros. Returns shape `(n_windows, window_samples)`.
    Always returns at least one window.
    """
    n = max(1, int(np.ceil(len(audio) / window_samples)))
    padded_len = n * window_samples
    if len(audio) < padded_len:
        audio = np.pad(audio, (0, padded_len - len(audio)))
    return audio[: n * window_samples].reshape(n, window_samples).astype(np.float32, copy=False)
