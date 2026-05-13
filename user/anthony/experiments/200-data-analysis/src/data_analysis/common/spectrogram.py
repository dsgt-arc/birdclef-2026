"""Shared mel/STFT spectrogram constants and helpers used across analyses."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter

from data_analysis.common.paths import TRAIN_AUDIO_DIR

MEL_SAMPLE_RATE = 32_000
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128
MEL_FMIN = 50
MEL_FMAX = 8_000

# Display band used by every spectrogram visualization in the Marbled Frog
# pipeline; kept here so motif_boxes, motif_review, and motif_mining all share
# the same y-limit.
DISPLAY_BAND = (500.0, 3000.0)

# Wider band used by the species-agnostic auto-mining pipeline. Spans the range
# where focal frog/insect/bird energy is typically concentrated; band discovery
# narrows down within this.
WIDEBAND = (200.0, 6000.0)

# Robust excess-z spectrogram tuning; used by `robust_excess_spectrogram`
# (marbled_frog manual pipeline). Auto-motif uses PCEN.
MAD_DB_FLOOR = 0.5
ROBUST_Z_CLIP = 8.0
GAUSSIAN_SIGMA_FREQ_BINS = 1.5
GAUSSIAN_SIGMA_TIME_FRAMES = 1.0

# librosa.pcen defaults used by `wideband_pcen_spectrogram`. Scaling factor
# applied to power before PCEN is the librosa convention so values land in a
# usable dynamic range; see librosa.pcen docs.
PCEN_POWER_SCALE = 2 ** 31
PCEN_TIME_CONSTANT = 0.4


def read_audio_window(filename: str, start_sec: float, end_sec: float) -> np.ndarray:
    """Read a mono 32 kHz audio window from train_audio with librosa."""
    path = TRAIN_AUDIO_DIR / filename
    audio, _ = librosa.load(
        path,
        sr=MEL_SAMPLE_RATE,
        mono=True,
        offset=float(start_sec),
        duration=float(end_sec - start_sec),
    )
    if audio.size == 0:
        return audio.astype(np.float32)
    audio = audio.astype(np.float32, copy=False)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return audio


def wideband_pcen_spectrogram(
    audio: np.ndarray,
    *,
    band: tuple[float, float] = WIDEBAND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCEN spectrogram over an arbitrary frequency band.

    Per-channel energy normalization (Wang et al. 2017) replaces the per-bin
    MAD + Gaussian-smoothing robust-z recipe. PCEN's AGC + dynamic-range
    compression gives a perceptually-flat representation that auto-motif's
    downstream stages (percentile-based hysteresis, patch-normalized NCC,
    envelope-relative onset anchor) consume scale-invariantly.

    Returns the same 4-tuple as the robust-z variants — ``(db_band, s_band,
    freqs_band, times)`` — where ``s_band`` is the PCEN spectrogram in the
    band of interest.
    """
    power = (
        np.abs(librosa.stft(audio, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, window="hann"))
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


def robust_excess_spectrogram(
    audio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Robust excess-z spectrogram restricted to :data:`DISPLAY_BAND`.

    Used by the Marbled Frog manual pipeline (motif_boxes, motif_review,
    motif_mining, motif_template_calibration, motif_recording_diagnostics).
    Auto-motif moved to :func:`wideband_pcen_spectrogram`.
    """
    power = (
        np.abs(librosa.stft(audio, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, window="hann"))
        ** 2
    )
    freqs = librosa.fft_frequencies(sr=MEL_SAMPLE_RATE, n_fft=MEL_N_FFT)
    times = librosa.frames_to_time(
        np.arange(power.shape[1]), sr=MEL_SAMPLE_RATE, hop_length=MEL_HOP_LENGTH
    )
    db = librosa.power_to_db(power, ref=np.max)

    mask = (freqs >= DISPLAY_BAND[0]) & (freqs <= DISPLAY_BAND[1])
    db_band = db[mask]
    freqs_band = freqs[mask]
    med = np.median(db_band, axis=1, keepdims=True)
    mad = np.median(np.abs(db_band - med), axis=1, keepdims=True)
    denom = 1.4826 * np.maximum(mad, MAD_DB_FLOOR)
    z = np.maximum((db_band - med) / denom, 0.0)
    z = np.clip(z, 0.0, ROBUST_Z_CLIP)
    z = gaussian_filter(z, sigma=(GAUSSIAN_SIGMA_FREQ_BINS, GAUSSIAN_SIGMA_TIME_FRAMES))
    return db_band, z, freqs_band, times
