"""PCEN spectrogram (ported from exp 200) + a soundscape audio-window loader.

The soundscape loader is the one piece that differs from exp 200's
``read_audio_window`` (which reads focal ``train_audio``); here we read from
``train_soundscapes`` and return the realized clip bounds so callers can map
clip-local times back to recording time.
"""

from __future__ import annotations

import librosa
import numpy as np

from birdclef_201.paths import TRAIN_SOUNDSCAPES_DIR

MEL_SAMPLE_RATE = 32_000
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512

# Wide band where focal frog/insect/bird energy concentrates; band discovery
# narrows within this.
WIDEBAND = (200.0, 6000.0)

# librosa.pcen conventions (same as exp 200).
PCEN_POWER_SCALE = 2 ** 31
PCEN_TIME_CONSTANT = 0.4


def hms_to_sec(value) -> float:
    """``"00:00:05"`` -> ``5.0``; numeric seconds pass through unchanged."""
    if isinstance(value, (int, float)):
        return float(value)
    sec = 0.0
    for part in str(value).split(":"):
        sec = sec * 60.0 + float(part)
    return sec


def read_soundscape_window(
    filename: str,
    start_sec: float,
    end_sec: float | None,
    *,
    pad_sec: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Mono 32 kHz window from ``train_soundscapes`` with optional ±pad context.

    Returns ``(audio, clip_start_sec, clip_end_sec)`` where ``clip_start`` is the
    realized start (``start_sec - pad_sec`` clamped at 0) and ``clip_end`` is the
    realized end (clamped at EOF). ``end_sec=None`` reads to EOF (whole file).
    Mean-subtract + peak-normalize, as exp 200.
    """
    clip_start = max(0.0, float(start_sec) - pad_sec)
    want = None if end_sec is None else (float(end_sec) + pad_sec - clip_start)
    path = TRAIN_SOUNDSCAPES_DIR / filename
    audio, _ = librosa.load(
        path, sr=MEL_SAMPLE_RATE, mono=True, offset=clip_start, duration=want
    )
    audio = audio.astype(np.float32, copy=False)
    if audio.size:
        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak
    clip_end = clip_start + audio.size / MEL_SAMPLE_RATE
    return audio, clip_start, clip_end


def wideband_pcen_spectrogram(
    audio: np.ndarray,
    *,
    band: tuple[float, float] = WIDEBAND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCEN spectrogram over a frequency band.

    Returns ``(db_band, pcen_band, freqs_band, times)`` — ported verbatim from
    exp 200's ``common.spectrogram``.
    """
    power = (
        np.abs(
            librosa.stft(
                audio, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, window="hann"
            )
        )
        ** 2
    )
    freqs = librosa.fft_frequencies(sr=MEL_SAMPLE_RATE, n_fft=MEL_N_FFT)
    times = librosa.frames_to_time(
        np.arange(power.shape[1]), sr=MEL_SAMPLE_RATE, hop_length=MEL_HOP_LENGTH
    )
    db = librosa.power_to_db(power, ref=np.max)
    pcen = librosa.pcen(
        power * PCEN_POWER_SCALE,
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        time_constant=PCEN_TIME_CONSTANT,
    )
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return db[mask], pcen[mask], freqs[mask], times
