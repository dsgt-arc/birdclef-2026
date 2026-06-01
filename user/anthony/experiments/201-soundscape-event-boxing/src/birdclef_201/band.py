"""Frequency-band discovery (ported lean from exp 200's auto_motif.band).

Kept: the per-recording spectrum summary, the cross-recording median consensus,
the hysteresis ``discover_bands``, and ``merge_overlapping_bands``. Dropped the
exemplar-weighted recall extension (auto_motif's ``exemplar_weights`` /
``weighted_consensus`` / ``bands_outside``) — a clean stationary insect band
needs only the robust core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandDiscoveryConfig:
    """Hysteresis band-discovery parameters.

    Defaults match exp 200; for stationary insects we typically pass
    ``core_bands=1, max_bands=2`` (one dominant band, allow a harmonic).
    """

    k_band_mad: float = 2.0
    k_band_mad_lower: float = 1.0
    min_band_width_hz: float = 150.0
    core_bands: int = 3
    max_bands: int = 5
    band_pad_hz: float = 40.0


def per_recording_mean_spectrum(z: np.ndarray, *, time_quantile: float = 0.95) -> np.ndarray:
    """Per-bin high-quantile of a spectrogram, normalized by its own MAD-sigma.

    A high time-quantile (p95) rather than the time-mean: PCEN's AGC flattens
    long-running averages, but the high quantile preserves bins with
    concentrated activity. Dividing by per-recording MAD-sigma puts every window
    on a comparable scale before the cross-window median consensus.
    """
    summary = np.quantile(z, time_quantile, axis=1)
    med = float(np.median(summary))
    mad = float(np.median(np.abs(summary - med)))
    sigma = 1.4826 * max(mad, 1e-6)
    return (summary - med) / sigma


def consensus_spectrum(spectra: list[np.ndarray]) -> np.ndarray:
    """Median of per-recording-normalized spectra (robust to a heterospecific)."""
    stack = np.stack(spectra, axis=0)
    return np.median(stack, axis=0)


def merge_overlapping_bands(bands: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union overlapping frequency intervals; disjoint ones kept apart."""
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(bands):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs for True runs in ``mask``."""
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
    """Hysteresis band discovery; up to ``config.max_bands`` (fmin, fmax) intervals.

    Seed runs where consensus > ``med + k_band_mad * sigma``; grow while
    > ``med + k_band_mad_lower * sigma`` (clamped at adjacent runs); pad, enforce
    a minimum width, return the top runs by peak score.
    """
    med = float(np.median(consensus))
    mad = float(np.median(np.abs(consensus - med)))
    sigma = 1.4826 * max(mad, 1e-6)
    primary = med + config.k_band_mad * sigma
    secondary = med + config.k_band_mad_lower * sigma
    runs = _contiguous_runs(consensus > primary)

    n = len(consensus)
    grown: list[tuple[int, int, float]] = []
    for idx, (i0, i1) in enumerate(runs):
        left_bound = 0 if idx == 0 else runs[idx - 1][1] + 1
        right_bound = n - 1 if idx == len(runs) - 1 else runs[idx + 1][0] - 1
        j0 = i0
        while j0 > left_bound and consensus[j0 - 1] > secondary:
            j0 -= 1
        j1 = i1
        while j1 < right_bound and consensus[j1 + 1] > secondary:
            j1 += 1
        grown.append((j0, j1, float(consensus[i0 : i1 + 1].max())))

    bands: list[tuple[float, float, float]] = []
    for j0, j1, peak in grown:
        fmin = float(freqs[j0]) - config.band_pad_hz
        fmax = float(freqs[j1]) + config.band_pad_hz
        if fmax - fmin < config.min_band_width_hz:
            center = 0.5 * (fmin + fmax)
            half = 0.5 * config.min_band_width_hz
            fmin, fmax = center - half, center + half
        fmin = max(fmin, float(freqs[0]))
        fmax = min(fmax, float(freqs[-1]))
        bands.append((fmin, fmax, peak))

    bands.sort(key=lambda x: x[2], reverse=True)
    return [(fmin, fmax) for fmin, fmax, _ in bands[: config.max_bands]]
