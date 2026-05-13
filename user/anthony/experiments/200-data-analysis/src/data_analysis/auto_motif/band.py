"""Auto frequency-band discovery for the species-agnostic motif pipeline.

Given the robust-z spectrograms of every focal recording for a species, we want
to discover up to ``max_bands`` contiguous frequency bands where the species'
energy is concentrated. Strategy:

1. For each recording, compute the time-mean of robust z across frequency,
   then divide by its own MAD-sigma so loud heterospecifics in one recording
   don't dominate the consensus.
2. Take the *median* across recordings (not the mean) — robust to a single
   recording dominated by a non-target.
3. Find contiguous frequency runs where the consensus exceeds
   ``median(consensus) + k * MAD(consensus)``.
4. Pad each run, enforce a minimum width, return the top ``max_bands`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandDiscoveryConfig:
    """Parameters for auto frequency-band discovery."""

    k_band_mad: float = 2.0
    min_band_width_hz: float = 150.0
    max_bands: int = 3
    band_pad_hz: float = 40.0


def per_recording_mean_spectrum(z: np.ndarray) -> np.ndarray:
    """Time-mean of a robust-z spectrogram, normalized by its own MAD-sigma.

    Dividing by per-recording MAD-sigma puts every recording's spectrum on a
    comparable scale before the cross-recording median consensus.
    """
    mean = z.mean(axis=1)
    med = float(np.median(mean))
    mad = float(np.median(np.abs(mean - med)))
    sigma = 1.4826 * max(mad, 1e-6)
    return (mean - med) / sigma


def consensus_spectrum(spectra: list[np.ndarray]) -> np.ndarray:
    """Median of per-recording-normalized spectra across recordings.

    Median (not mean) protects against one recording dominated by a heterospecific.
    """
    stack = np.stack(spectra, axis=0)
    return np.median(stack, axis=0)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs for True runs in ``mask``."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def discover_bands(
    consensus: np.ndarray,
    freqs: np.ndarray,
    *,
    config: BandDiscoveryConfig = BandDiscoveryConfig(),
) -> list[tuple[float, float]]:
    """Return up to ``config.max_bands`` (fmin_hz, fmax_hz) intervals."""
    med = float(np.median(consensus))
    mad = float(np.median(np.abs(consensus - med)))
    sigma = 1.4826 * max(mad, 1e-6)
    threshold = med + config.k_band_mad * sigma
    mask = consensus > threshold
    runs = _contiguous_runs(mask)

    bands: list[tuple[float, float, float]] = []
    for i0, i1 in runs:
        fmin = float(freqs[i0]) - config.band_pad_hz
        fmax = float(freqs[i1]) + config.band_pad_hz
        if fmax - fmin < config.min_band_width_hz:
            center = 0.5 * (fmin + fmax)
            half = 0.5 * config.min_band_width_hz
            fmin = center - half
            fmax = center + half
        fmin = max(fmin, float(freqs[0]))
        fmax = min(fmax, float(freqs[-1]))
        peak_score = float(consensus[i0 : i1 + 1].max())
        bands.append((fmin, fmax, peak_score))

    bands.sort(key=lambda x: x[2], reverse=True)
    return [(fmin, fmax) for fmin, fmax, _ in bands[: config.max_bands]]
