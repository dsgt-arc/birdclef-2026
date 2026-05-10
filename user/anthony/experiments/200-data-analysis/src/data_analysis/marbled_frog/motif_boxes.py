"""Prototype motif bounding boxes for Marbled White-lipped Frog calls.

This detects short call-like motifs from a primary band envelope, then draws
primary/secondary frequency-component boxes on a spectrogram. It is intended as
an inspection aid, not ground-truth annotation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import polars as pl
from scipy.ndimage import label as label_components
from scipy.signal import find_peaks, peak_widths

from data_analysis.common.paths import TRAIN_AUDIO_DIR, artifact_dir, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_N_FFT,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)

DEFAULT_FILENAME = "22967/iNat1448422.ogg"
DEFAULT_START_SEC = 0.0
DEFAULT_DURATION_SEC = 5.0

PRIMARY_BAND = (600.0, 1300.0)
# Default/seed secondary band. Some motif families, especially iNat1697350,
# use a broader recording-specific secondary validation band below.
SECONDARY_BAND = (1500.0, 2300.0)
TEMPLATE_BAND = (800.0, 2300.0)
TEMPLATE_HALF_FRAMES = 15
TEMPLATE_MATCH_THRESHOLD = 0.35
TEMPLATE_MATCH_MIN_SPACING_SEC = 0.35


@dataclass(frozen=True)
class MotifBox:
    motif_id: int
    component: str
    start_sec: float
    end_sec: float
    fmin_hz: float
    fmax_hz: float
    peak_time_sec: float
    peak_score: float
    match_score: float | None = None


@dataclass(frozen=True)
class CuratedAnnotation:
    """A 'current-best' motif-box set, detected and rendered under curated/."""

    name: str
    motif_family: str
    status: str
    notes: str
    source_table: str  # filename within curated/tables/
    primary_band: tuple[float, float]
    secondary_band: tuple[float, float] | None
    output_plot_name: str


CURATED_ANNOTATIONS: list[CuratedAnnotation] = [
    CuratedAnnotation(
        name="iNat1448422_0-5s_low_click_secondary",
        motif_family="low_click_with_secondary",
        status="manual_current_best",
        notes=(
            "Four short calls in 0-5s seed window; primary around "
            "600-1300 Hz with secondary validation band around 1500-2300 Hz."
        ),
        source_table="iNat1448422_0-5s_motif_boxes.csv",
        primary_band=(600.0, 1300.0),
        secondary_band=(1500.0, 2300.0),
        output_plot_name="iNat1448422_0-5s_curated.png",
    ),
    CuratedAnnotation(
        name="iNat1697350_independent_broad_sweep",
        motif_family="broad_upper_sweep_with_low_tail",
        status="manual_current_best",
        notes=(
            "Ten paired-looking events; independent primary lower boxes plus "
            "broad secondary 1000-3000 Hz boxes. Treat as motif-event evidence, "
            "not precise boxes."
        ),
        source_table="iNat1697350_independent_band_motif_boxes.csv",
        primary_band=(600.0, 1300.0),
        secondary_band=(1000.0, 3000.0),
        output_plot_name="iNat1697350_curated.png",
    ),
    CuratedAnnotation(
        name="iNat1698104_1k_1p8k_rising_chips",
        motif_family="rising_chip_1k_1p8k",
        status="manual_current_best",
        notes=(
            "Five rising chips around 1-1.8 kHz; separate motif family not "
            "matched by the 1448422 seed template."
        ),
        source_table="iNat1698104_1k_1p8k_chip_boxes.csv",
        primary_band=(1000.0, 1800.0),
        secondary_band=None,
        output_plot_name="iNat1698104_curated.png",
    ),
]


def smooth(x: np.ndarray, n: int = 9) -> np.ndarray:
    """Moving-average smooth."""
    if n <= 1:
        return x
    return np.convolve(x, np.ones(n) / n, mode="same")


def component_tracks(
    z: np.ndarray, freqs: np.ndarray, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return envelope, centroid, tonality, and peak-picking score for a band.

    Tonality is 1 - normalized entropy of each time column. Multiplying the
    envelope by tonality favors concentrated ridges/chips over broadband noise.
    """
    mask = (freqs >= band[0]) & (freqs <= band[1])
    z_band = z[mask]
    band_freqs = freqs[mask]
    if z_band.size == 0:
        zeros = np.zeros(z.shape[1], dtype=np.float32)
        return zeros, zeros, zeros, zeros
    envelope = smooth(z_band.mean(axis=0), n=7)
    weights = np.maximum(z_band, 0.0)
    sums = weights.sum(axis=0) + 1e-9
    centroid = (weights * band_freqs[:, None]).sum(axis=0) / sums
    probs = weights / sums[None, :]
    entropy = -(probs * np.log(probs + 1e-9)).sum(axis=0)
    max_entropy = np.log(max(2, z_band.shape[0]))
    tonality = np.clip(1.0 - entropy / max_entropy, 0.0, 1.0)
    tonality = smooth(tonality, n=5)
    score = smooth(envelope * (0.35 + 0.65 * tonality), n=5)
    return envelope, centroid, tonality, score


def component_envelope(z: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Mean robust excess energy envelope for a frequency band."""
    return component_tracks(z, freqs, band)[0]


def select_peaks(
    score: np.ndarray,
    *,
    min_spacing_frames: int,
    expected_count: int | None = None,
    peak_height: float = 1.0,
    peak_prominence: float = 0.5,
) -> np.ndarray:
    """Select peaks from a score track, optionally using an expected-count prior.

    When expected_count is supplied, over-detect candidate peaks with loose
    thresholds, then solve a small dynamic program to choose exactly K peaks
    maximizing total score subject to minimum spacing.
    """
    peaks, _props = find_peaks(
        score,
        height=peak_height,
        prominence=peak_prominence,
        distance=max(1, min_spacing_frames // 2),
    )
    if expected_count is None or peaks.size <= expected_count:
        if expected_count is None:
            return peaks
        return peaks[np.argsort(score[peaks])[::-1]][:expected_count][::-1]

    frames = peaks.astype(int)
    values = score[frames].astype(float)
    n = frames.size
    k = expected_count
    prev = np.searchsorted(frames, frames - min_spacing_frames, side="right") - 1
    dp = np.full((n + 1, k + 1), -np.inf, dtype=float)
    take = np.zeros((n + 1, k + 1), dtype=bool)
    dp[:, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(0, k + 1):
            best = dp[i - 1, j]
            if j > 0:
                prior_i = prev[i - 1] + 1
                cand = dp[prior_i, j - 1] + values[i - 1]
                if cand > best:
                    best = cand
                    take[i, j] = True
            dp[i, j] = best

    # If exact K is impossible, reconstruct the best available count <= K.
    j = int(np.nanargmax(dp[n, : k + 1])) if not np.isfinite(dp[n, k]) else k
    selected: list[int] = []
    i = n
    while i > 0 and j > 0:
        if take[i, j]:
            selected.append(frames[i - 1])
            i = prev[i - 1] + 1
            j -= 1
        else:
            i -= 1
    return np.array(sorted(selected), dtype=int)


def time_bounds_from_peak(
    envelope: np.ndarray,
    times: np.ndarray,
    peak_idx: int,
    *,
    min_threshold: float = 0.8,
    frac_of_peak: float = 0.35,
    pre_pad_sec: float = 0.10,
    post_pad_sec: float = 0.04,
) -> tuple[int, int, float, float]:
    """Find contiguous envelope support around a peak.

    The onset side is padded more aggressively because the visible motif attack
    often starts before the smoothed band envelope crosses threshold.
    """
    threshold = max(min_threshold, float(envelope[peak_idx]) * frac_of_peak)
    left = peak_idx
    while left > 0 and envelope[left] >= threshold:
        left -= 1
    right = peak_idx
    while right < envelope.size - 1 and envelope[right] >= threshold:
        right += 1
    t0 = max(0.0, float(times[left]) - pre_pad_sec)
    t1 = float(times[min(right, times.size - 1)]) + post_pad_sec
    return left, right, t0, t1


def frequency_bounds(
    z: np.ndarray,
    freqs: np.ndarray,
    frame_slice: slice,
    band: tuple[float, float],
    *,
    min_threshold: float = 0.9,
) -> tuple[float, float]:
    """Estimate active frequency extent for a component within a time span."""
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs = freqs[band_mask]
    if frame_slice.start == frame_slice.stop:
        profile = z[band_mask, frame_slice.start]
    else:
        profile = z[band_mask, frame_slice].mean(axis=1)
    active = profile >= max(min_threshold, 0.35 * float(profile.max(initial=0.0)))
    if not np.any(active):
        return band
    return float(band_freqs[active].min()), float(band_freqs[active].max())


def pad_frequency_bounds(
    f0: float, f1: float, band: tuple[float, float], *, lower_pad_hz: float = 0.0, upper_pad_hz: float = 0.0
) -> tuple[float, float]:
    """Pad estimated frequency bounds while staying inside the component search band."""
    return max(band[0], f0 - lower_pad_hz), min(band[1], f1 + upper_pad_hz)


def connected_component_box(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    peak_idx: int,
    band: tuple[float, float],
    left_frame: int,
    right_frame: int,
    min_threshold: float = 0.75,
    frac_of_peak: float = 0.35,
    pre_pad_sec: float = 0.02,
    post_pad_sec: float = 0.02,
    lower_pad_hz: float = 0.0,
    upper_pad_hz: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return a tight signal-derived box from the connected component at a peak."""
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs = freqs[band_mask]
    if not np.any(band_mask):
        return float(times[left_frame]), float(times[right_frame]), band[0], band[1]
    left_frame = max(0, int(left_frame))
    right_frame = min(z.shape[1] - 1, int(right_frame))
    crop = z[band_mask, left_frame : right_frame + 1]
    if crop.size == 0:
        return float(times[left_frame]), float(times[right_frame]), band[0], band[1]

    local_peak_t = int(np.clip(peak_idx - left_frame, 0, crop.shape[1] - 1))
    local_peak_f = int(np.argmax(crop[:, local_peak_t]))
    peak_value = float(crop[local_peak_f, local_peak_t])
    threshold = max(min_threshold, frac_of_peak * peak_value)
    active = crop >= threshold
    labels, n_labels = label_components(active)
    chosen_label = int(labels[local_peak_f, local_peak_t])
    if chosen_label == 0 and n_labels:
        # If the exact ridge pixel is just below threshold, choose the component
        # with highest total energy near the peak time.
        label_scores = []
        for component_id in range(1, n_labels + 1):
            component_mask = labels == component_id
            time_distance = np.abs(np.where(component_mask)[1] - local_peak_t).min()
            label_scores.append((float(crop[component_mask].sum()) / (1.0 + time_distance), component_id))
        chosen_label = max(label_scores)[1]
    if chosen_label == 0:
        f0, f1 = frequency_bounds(z, freqs, slice(left_frame, right_frame + 1), band, min_threshold=min_threshold)
        return float(times[left_frame]), float(times[right_frame]), f0, f1

    ff, tt = np.where(labels == chosen_label)
    t0 = max(0.0, float(times[left_frame + tt.min()]) - pre_pad_sec)
    t1 = float(times[left_frame + tt.max()]) + post_pad_sec
    f0 = float(band_freqs[ff.min()])
    f1 = float(band_freqs[ff.max()])
    f0, f1 = pad_frequency_bounds(f0, f1, band, lower_pad_hz=lower_pad_hz, upper_pad_hz=upper_pad_hz)
    return t0, t1, f0, f1


def detect_motif_boxes(
    audio: np.ndarray,
    *,
    global_start_sec: float,
    peak_height: float = 0.55,
    peak_prominence: float = 0.20,
    min_call_spacing_sec: float = 0.35,
    expected_count: int | None = 4,
) -> tuple[list[MotifBox], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect primary motifs and optional secondary component boxes."""
    db_band, z, freqs, times = robust_excess_spectrogram(audio)
    primary_env, _primary_centroid, _primary_tonality, primary_score = component_tracks(z, freqs, PRIMARY_BAND)
    secondary_env, _secondary_centroid, _secondary_tonality, secondary_score = component_tracks(z, freqs, SECONDARY_BAND)
    frame_rate = MEL_SAMPLE_RATE / MEL_HOP_LENGTH
    peaks = select_peaks(
        primary_score,
        min_spacing_frames=max(1, int(min_call_spacing_sec * frame_rate)),
        expected_count=expected_count,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
    )

    boxes: list[MotifBox] = []
    for motif_id, peak_idx in enumerate(peaks, start=1):
        left, right, _t0, _t1 = time_bounds_from_peak(primary_score, times, int(peak_idx), min_threshold=0.35)
        t0, t1, p_f0, p_f1 = connected_component_box(
            z,
            freqs,
            times,
            peak_idx=int(peak_idx),
            band=PRIMARY_BAND,
            left_frame=left,
            right_frame=right,
            min_threshold=0.45,
            frac_of_peak=0.32,
            pre_pad_sec=0.045,
            post_pad_sec=0.025,
            lower_pad_hz=70.0,
        )
        boxes.append(
            MotifBox(
                motif_id=motif_id,
                component="primary",
                start_sec=global_start_sec + t0,
                end_sec=global_start_sec + t1,
                fmin_hz=p_f0,
                fmax_hz=p_f1,
                peak_time_sec=global_start_sec + float(times[peak_idx]),
                peak_score=float(primary_score[peak_idx]),
            )
        )

        # Look for a secondary/harmonic-like component close to the same primary event.
        pad_frames = max(1, int(0.12 * frame_rate))
        search_left = max(0, left - pad_frames)
        search_right = min(secondary_env.size - 1, right + pad_frames)
        local = secondary_score[search_left : search_right + 1]
        if local.size and float(local.max()) >= 0.35:
            sec_peak_idx = search_left + int(local.argmax())
            sec_left, sec_right, _st0, _st1 = time_bounds_from_peak(
                secondary_score,
                times,
                sec_peak_idx,
                min_threshold=0.25,
                frac_of_peak=0.35,
                pre_pad_sec=0.05,
                post_pad_sec=0.025,
            )
            st0, st1, s_f0, s_f1 = connected_component_box(
                z,
                freqs,
                times,
                peak_idx=int(sec_peak_idx),
                band=SECONDARY_BAND,
                left_frame=sec_left,
                right_frame=sec_right,
                min_threshold=0.35,
                frac_of_peak=0.32,
                pre_pad_sec=0.035,
                post_pad_sec=0.02,
                lower_pad_hz=80.0,
            )
            boxes.append(
                MotifBox(
                    motif_id=motif_id,
                    component="secondary",
                    start_sec=global_start_sec + st0,
                    end_sec=global_start_sec + st1,
                    fmin_hz=s_f0,
                    fmax_hz=s_f1,
                    peak_time_sec=global_start_sec + float(times[sec_peak_idx]),
                    peak_score=float(secondary_score[sec_peak_idx]),
                )
            )

    return boxes, db_band, z, freqs, times, primary_score, secondary_score


def boxes_to_frame(boxes: list[MotifBox], filename: str, window_start_sec: float, window_end_sec: float) -> pl.DataFrame:
    """Convert motif boxes to a Polars table."""
    return pl.DataFrame(
        [
            {
                "filename": filename,
                "window_start_sec": window_start_sec,
                "window_end_sec": window_end_sec,
                "motif_id": box.motif_id,
                "component": box.component,
                "start_sec": box.start_sec,
                "end_sec": box.end_sec,
                "duration_sec": box.end_sec - box.start_sec,
                "fmin_hz": box.fmin_hz,
                "fmax_hz": box.fmax_hz,
                "bandwidth_hz": box.fmax_hz - box.fmin_hz,
                "peak_time_sec": box.peak_time_sec,
                "peak_score": box.peak_score,
                "match_score": box.match_score,
            }
            for box in boxes
        ]
    )


def normalized_patch(patch: np.ndarray) -> np.ndarray:
    """Flatten and z-normalize a spectrogram patch."""
    vec = patch.astype(np.float32).ravel()
    return (vec - vec.mean()) / (vec.std() + 1e-9)


def build_seed_template(
    *,
    filename: str = DEFAULT_FILENAME,
    start_sec: float = DEFAULT_START_SEC,
    duration_sec: float = DEFAULT_DURATION_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[MotifBox]]:
    """Build a mean motif template from detected primary boxes in the seed window."""
    audio, _ = librosa.load(
        TRAIN_AUDIO_DIR / filename,
        sr=MEL_SAMPLE_RATE,
        mono=True,
        offset=start_sec,
        duration=duration_sec,
    )
    boxes, _db, z, freqs, times, _primary_env, _secondary_env = detect_motif_boxes(
        audio, global_start_sec=start_sec
    )
    primary = [box for box in boxes if box.component == "primary"]
    if not primary:
        raise ValueError("No primary seed motifs detected; cannot build template")

    band_mask = (freqs >= TEMPLATE_BAND[0]) & (freqs <= TEMPLATE_BAND[1])
    patches = []
    for box in primary:
        center_frame = int(np.argmin(np.abs((times + start_sec) - box.peak_time_sec)))
        left = center_frame - TEMPLATE_HALF_FRAMES
        right = center_frame + TEMPLATE_HALF_FRAMES + 1
        if left < 0 or right > z.shape[1]:
            continue
        patch = z[band_mask, left:right]
        patches.append(normalized_patch(patch).reshape(patch.shape))
    if not patches:
        raise ValueError("Seed motifs were too close to the edge to build a template")
    template = np.mean(np.stack(patches), axis=0)
    template = normalized_patch(template).reshape(template.shape)
    return template, freqs[band_mask], times, boxes


def template_match_scores(z: np.ndarray, freqs: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized sliding template-match scores over time."""
    band_mask = (freqs >= TEMPLATE_BAND[0]) & (freqs <= TEMPLATE_BAND[1])
    z_band = z[band_mask]
    width = template.shape[1]
    half = width // 2
    centers = np.arange(half, z_band.shape[1] - half)
    scores = np.empty(centers.size, dtype=np.float32)
    template_vec = template.ravel()
    for idx, center in enumerate(centers):
        patch = z_band[:, center - half : center + half + 1]
        scores[idx] = float(np.mean(normalized_patch(patch) * template_vec))
    return centers, scores


def detect_template_match_boxes(
    filename: str,
    template: np.ndarray,
    *,
    threshold: float = TEMPLATE_MATCH_THRESHOLD,
) -> tuple[list[MotifBox], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect seed-template-like motifs and refine each match to tight component boxes.

    The template is used only to find candidate time centers. For each candidate,
    the primary and secondary component boxes are re-estimated from local band
    envelopes and active frequency extents, so the final boxes are much tighter
    than the broad template search band.
    """
    audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
    db_band, z, freqs, times = robust_excess_spectrogram(audio)
    primary_env = component_envelope(z, freqs, PRIMARY_BAND)
    secondary_env = component_envelope(z, freqs, SECONDARY_BAND)
    centers, scores = template_match_scores(z, freqs, template)
    frame_rate = MEL_SAMPLE_RATE / MEL_HOP_LENGTH
    peaks, _props = find_peaks(
        scores,
        height=threshold,
        prominence=0.03,
        distance=max(1, int(TEMPLATE_MATCH_MIN_SPACING_SEC * frame_rate)),
    )

    boxes: list[MotifBox] = []
    half = template.shape[1] // 2
    for motif_id, peak_idx in enumerate(peaks, start=1):
        center_frame = int(centers[peak_idx])
        left_frame = max(0, center_frame - half)
        right_frame = min(times.size - 1, center_frame + half)
        match_score = float(scores[peak_idx])

        # Keep a faint template extent for provenance, but downstream users should
        # treat primary/secondary rows as the tight pseudo-boxes.
        boxes.append(
            MotifBox(
                motif_id=motif_id,
                component="template_extent",
                start_sec=float(times[left_frame]),
                end_sec=float(times[right_frame]),
                fmin_hz=TEMPLATE_BAND[0],
                fmax_hz=TEMPLATE_BAND[1],
                peak_time_sec=float(times[center_frame]),
                peak_score=match_score,
                match_score=match_score,
            )
        )

        search_left = max(0, left_frame - int(0.06 * frame_rate))
        search_right = min(primary_env.size - 1, right_frame + int(0.06 * frame_rate))
        primary_local = primary_env[search_left : search_right + 1]
        primary_peak = search_left + int(primary_local.argmax())
        p_left, p_right, p_t0, p_t1 = time_bounds_from_peak(
            primary_env,
            times,
            primary_peak,
            min_threshold=0.75,
            frac_of_peak=0.45,
            pre_pad_sec=0.055,
            post_pad_sec=0.025,
        )
        p_f0, p_f1 = pad_frequency_bounds(
            *frequency_bounds(z, freqs, slice(p_left, p_right + 1), PRIMARY_BAND, min_threshold=0.65),
            PRIMARY_BAND,
            lower_pad_hz=140.0,
        )
        boxes.append(
            MotifBox(
                motif_id=motif_id,
                component="primary",
                start_sec=float(p_t0),
                end_sec=float(p_t1),
                fmin_hz=p_f0,
                fmax_hz=p_f1,
                peak_time_sec=float(times[primary_peak]),
                peak_score=float(primary_env[primary_peak]),
                match_score=match_score,
            )
        )

        secondary_local = secondary_env[search_left : search_right + 1]
        if secondary_local.size and float(secondary_local.max()) >= 0.80:
            secondary_peak = search_left + int(secondary_local.argmax())
            s_left, s_right, s_t0, s_t1 = time_bounds_from_peak(
                secondary_env,
                times,
                secondary_peak,
                min_threshold=0.65,
                frac_of_peak=0.40,
                pre_pad_sec=0.05,
                post_pad_sec=0.025,
            )
            s_f0, s_f1 = pad_frequency_bounds(
                *frequency_bounds(z, freqs, slice(s_left, s_right + 1), SECONDARY_BAND, min_threshold=0.55),
                SECONDARY_BAND,
                lower_pad_hz=250.0,
            )
            boxes.append(
                MotifBox(
                    motif_id=motif_id,
                    component="secondary",
                    start_sec=float(s_t0),
                    end_sec=float(s_t1),
                    fmin_hz=s_f0,
                    fmax_hz=s_f1,
                    peak_time_sec=float(times[secondary_peak]),
                    peak_score=float(secondary_env[secondary_peak]),
                    match_score=match_score,
                )
            )
    return boxes, db_band, z, freqs, times, scores


def template_boxes_to_frame(boxes: list[MotifBox], filename: str) -> pl.DataFrame:
    """Convert template match boxes to a table."""
    return pl.DataFrame(
        [
            {
                "filename": filename,
                "match_id": box.motif_id,
                "component": box.component,
                "start_sec": box.start_sec,
                "end_sec": box.end_sec,
                "duration_sec": box.end_sec - box.start_sec,
                "fmin_hz": box.fmin_hz,
                "fmax_hz": box.fmax_hz,
                "peak_time_sec": box.peak_time_sec,
                "template_score": box.match_score if box.match_score is not None else box.peak_score,
                "component_score": box.peak_score,
            }
            for box in boxes
        ]
    )


def plot_template_match_boxes(
    db_band: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    boxes: list[MotifBox],
    output_path: Path,
    *,
    filename: str,
) -> None:
    """Plot full-recording spectrogram with template-match boxes."""
    width = max(10.0, min(18.0, 0.38 * float(times[-1]) + 7.0))
    fig, ax = plt.subplots(figsize=(width, 5.6), dpi=180)
    librosa.display.specshow(
        db_band,
        x_coords=times,
        y_coords=freqs,
        x_axis="time",
        y_axis="linear",
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        ax=ax,
        cmap="magma",
    )
    ax.axhspan(PRIMARY_BAND[0], PRIMARY_BAND[1], color="lime", alpha=0.07, label="primary seed band")
    ax.axhspan(SECONDARY_BAND[0], SECONDARY_BAND[1], color="cyan", alpha=0.07, label="secondary seed band")
    colors = {"template_extent": "magenta", "primary": "lime", "secondary": "cyan"}
    linestyles = {"template_extent": ":", "primary": "-", "secondary": "--"}
    linewidths = {"template_extent": 1.0, "primary": 2.0, "secondary": 1.8}
    alphas = {"template_extent": 0.55, "primary": 0.98, "secondary": 0.95}
    for box in boxes:
        rect = Rectangle(
            (box.start_sec, box.fmin_hz),
            box.end_sec - box.start_sec,
            box.fmax_hz - box.fmin_hz,
            fill=False,
            edgecolor=colors.get(box.component, "white"),
            linewidth=linewidths.get(box.component, 1.5),
            linestyle=linestyles.get(box.component, "-"),
            alpha=alphas.get(box.component, 1.0),
        )
        ax.add_patch(rect)
        if box.component != "template_extent":
            ax.text(
                box.start_sec,
                box.fmax_hz + 35,
                f"{box.motif_id}:{box.component[0]}\n{box.peak_score:.1f}",
                color=colors[box.component],
                fontsize=7,
                ha="left",
                weight="bold",
            )
    ax.set_ylim(DISPLAY_BAND)
    ax.set_title(
        f"Template motif matches for {filename}\n"
        f"seed template={DEFAULT_FILENAME} {DEFAULT_START_SEC:.0f}-{DEFAULT_START_SEC + DEFAULT_DURATION_SEC:.0f}s, "
        f"threshold={TEMPLATE_MATCH_THRESHOLD}; magenta=template extent, green/cyan=tight components"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_template_match_summary_grid(
    matches: pl.DataFrame,
    output_path: Path,
    *,
    max_cols: int = 4,
    max_matches_per_file: int = 1,
) -> None:
    """Grid top template matches per recording for quick visual review."""
    if matches.is_empty():
        return
    rows = []
    primary_matches = matches.filter(pl.col("component") == "primary")
    for filename in primary_matches.select("filename").unique().sort("filename").to_series().to_list():
        rows.extend(
            primary_matches.filter(pl.col("filename") == filename)
            .sort("template_score", descending=True)
            .head(max_matches_per_file)
            .iter_rows(named=True)
        )
    n_items = len(rows)
    n_cols = min(max_cols, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.9 * n_cols, 3.35 * n_rows),
        dpi=180,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for idx, row in enumerate(rows):
        row_idx, col_idx = divmod(idx, n_cols)
        ax = axes[row_idx, col_idx]
        pad = 0.15
        start = max(0.0, float(row["start_sec"]) - pad)
        duration = float(row["end_sec"] - row["start_sec"] + 2 * pad)
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / row["filename"], sr=MEL_SAMPLE_RATE, mono=True, offset=start, duration=duration)
        db_band, _z, freqs, times = robust_excess_spectrogram(audio)
        display_times = times + start
        librosa.display.specshow(
            db_band,
            x_coords=display_times,
            y_coords=freqs,
            x_axis="time",
            y_axis="linear",
            sr=MEL_SAMPLE_RATE,
            hop_length=MEL_HOP_LENGTH,
            ax=ax,
            cmap="magma",
        )
        ax.add_patch(
            Rectangle(
                (row["start_sec"], row["fmin_hz"]),
                row["end_sec"] - row["start_sec"],
                row["fmax_hz"] - row["fmin_hz"],
                fill=False,
                edgecolor="magenta",
                linewidth=2.0,
            )
        )
        ax.set_ylim(DISPLAY_BAND)
        ax.set_title(
            f"{Path(row['filename']).stem} {row['start_sec']:.2f}-{row['end_sec']:.2f}s\n"
            f"template score={row['template_score']:.2f}",
            fontsize=7.5,
            color="magenta",
        )
        if col_idx != 0:
            ax.set_ylabel("")
    for idx in range(n_items, n_rows * n_cols):
        row_idx, col_idx = divmod(idx, n_cols)
        axes[row_idx, col_idx].axis("off")
    fig.suptitle("Top seed-template motif match per recording", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def detect_band_motif_boxes(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    band: tuple[float, float],
    component: str,
    peak_height: float = 0.40,
    peak_prominence: float = 0.12,
    min_call_spacing_sec: float = 0.35,
    expected_count: int | None = None,
) -> list[MotifBox]:
    """Detect tight boxes independently from a single frequency band."""
    lower_pad_hz = 70.0 if component.startswith("primary") else 120.0
    return detect_custom_band_boxes(
        z,
        freqs,
        times,
        band=band,
        component=component,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
        min_call_spacing_sec=min_call_spacing_sec,
        lower_pad_hz=lower_pad_hz,
        expected_count=expected_count,
    )


def detect_custom_band_boxes(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    band: tuple[float, float],
    component: str,
    peak_height: float,
    peak_prominence: float,
    min_call_spacing_sec: float = 0.35,
    lower_pad_hz: float = 0.0,
    upper_pad_hz: float = 0.0,
    time_pre_pad_sec: float = 0.04,
    time_post_pad_sec: float = 0.02,
    expected_count: int | None = None,
) -> list[MotifBox]:
    """Detect boxes from a custom band using score-track peaks + connected components."""
    envelope, _centroid, _tonality, score = component_tracks(z, freqs, band)
    frame_rate = MEL_SAMPLE_RATE / MEL_HOP_LENGTH
    peaks = select_peaks(
        score,
        min_spacing_frames=max(1, int(min_call_spacing_sec * frame_rate)),
        expected_count=expected_count,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
    )
    boxes: list[MotifBox] = []
    for motif_id, peak_idx in enumerate(peaks, start=1):
        left, right, _t0, _t1 = time_bounds_from_peak(
            score,
            times,
            int(peak_idx),
            min_threshold=max(0.25, peak_height * 0.5),
            frac_of_peak=0.40,
            pre_pad_sec=time_pre_pad_sec,
            post_pad_sec=time_post_pad_sec,
        )
        t0, t1, f0, f1 = connected_component_box(
            z,
            freqs,
            times,
            peak_idx=int(peak_idx),
            band=band,
            left_frame=left,
            right_frame=right,
            min_threshold=0.35,
            frac_of_peak=0.30,
            pre_pad_sec=max(0.015, time_pre_pad_sec * 0.6),
            post_pad_sec=max(0.015, time_post_pad_sec),
            lower_pad_hz=lower_pad_hz,
            upper_pad_hz=upper_pad_hz,
        )
        boxes.append(
            MotifBox(
                motif_id=motif_id,
                component=component,
                start_sec=float(t0),
                end_sec=float(t1),
                fmin_hz=f0,
                fmax_hz=f1,
                peak_time_sec=float(times[peak_idx]),
                peak_score=float(score[peak_idx]),
            )
        )
    return boxes


def detect_seed_window_boxes(
    filename: str, *, start_sec: float, duration_sec: float
) -> pl.DataFrame:
    """Detect primary + secondary motif boxes inside a fixed window of one recording."""
    audio, _ = librosa.load(
        TRAIN_AUDIO_DIR / filename,
        sr=MEL_SAMPLE_RATE,
        mono=True,
        offset=start_sec,
        duration=duration_sec,
    )
    boxes, *_ = detect_motif_boxes(audio, global_start_sec=start_sec)
    return boxes_to_frame(boxes, filename, start_sec, start_sec + duration_sec)


def detect_independent_band_boxes_for_recording(
    filename: str,
    *,
    primary_band: tuple[float, float] = PRIMARY_BAND,
    secondary_band: tuple[float, float] = SECONDARY_BAND,
    expected_primary_count: int | None = None,
    expected_secondary_count: int | None = None,
) -> pl.DataFrame:
    """Detect independent primary + secondary band-envelope boxes across a full recording."""
    audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
    _db_band, z, freqs, times = robust_excess_spectrogram(audio)
    primary_boxes = detect_band_motif_boxes(
        z,
        freqs,
        times,
        band=primary_band,
        component="primary_independent",
        expected_count=expected_primary_count,
    )
    secondary_boxes = detect_band_motif_boxes(
        z,
        freqs,
        times,
        band=secondary_band,
        component="secondary_independent",
        expected_count=expected_secondary_count,
    )
    return boxes_to_frame(primary_boxes + secondary_boxes, filename, 0.0, float(times[-1]))


def detect_single_component_boxes_for_recording(
    filename: str,
    *,
    band: tuple[float, float],
    component: str,
    peak_height: float,
    peak_prominence: float,
    lower_pad_hz: float = 0.0,
    upper_pad_hz: float = 0.0,
    expected_count: int | None = None,
) -> pl.DataFrame:
    """Detect single-band custom-component boxes across a full recording."""
    audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
    _db_band, z, freqs, times = robust_excess_spectrogram(audio)
    boxes = detect_custom_band_boxes(
        z,
        freqs,
        times,
        band=band,
        component=component,
        peak_height=peak_height,
        peak_prominence=peak_prominence,
        lower_pad_hz=lower_pad_hz,
        upper_pad_hz=upper_pad_hz,
        expected_count=expected_count,
    )
    return boxes_to_frame(boxes, filename, 0.0, float(times[-1]))


def _component_style(component: str) -> tuple[str, str, float, str]:
    if "primary" in component or "chip" in component:
        return "lime", "-", 2.0, "p"
    if "secondary" in component:
        return "cyan", "--", 1.9, "s"
    return "white", ":", 1.2, "?"


def render_curated_plot(item: CuratedAnnotation, table_path: Path, output_path: Path) -> None:
    """Render a standardized primary/secondary validation-band plot for a curated set."""
    table = pl.read_csv(table_path)
    if table.is_empty():
        return
    rows = table.to_dicts()
    filename = rows[0]["filename"]
    window_start = float(min((row.get("window_start_sec") or 0.0) for row in rows))
    window_end = float(max((row.get("window_end_sec") or row.get("end_sec") or 0.0) for row in rows))
    duration = max(0.1, window_end - window_start)

    audio, _ = librosa.load(
        TRAIN_AUDIO_DIR / filename,
        sr=MEL_SAMPLE_RATE,
        mono=True,
        offset=window_start,
        duration=duration,
    )
    db_band, _z, freqs, times = robust_excess_spectrogram(audio)
    display_times = times + window_start

    fig, ax = plt.subplots(figsize=(12.0, 6.2), dpi=180)
    librosa.display.specshow(
        db_band,
        x_coords=display_times,
        y_coords=freqs,
        x_axis="time",
        y_axis="linear",
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        ax=ax,
        cmap="magma",
    )

    ax.axhspan(
        item.primary_band[0],
        item.primary_band[1],
        color="lime",
        alpha=0.075,
        label=f"primary validation band {item.primary_band[0]:.0f}-{item.primary_band[1]:.0f} Hz",
    )
    if item.secondary_band is not None:
        ax.axhspan(
            item.secondary_band[0],
            item.secondary_band[1],
            color="cyan",
            alpha=0.07,
            label=f"secondary validation band {item.secondary_band[0]:.0f}-{item.secondary_band[1]:.0f} Hz",
        )

    for row in rows:
        color, linestyle, linewidth, suffix = _component_style(row.get("component") or "")
        start_sec = float(row.get("start_sec") or 0.0)
        end_sec = float(row.get("end_sec") or 0.0)
        fmin_hz = float(row.get("fmin_hz") or 0.0)
        fmax_hz = float(row.get("fmax_hz") or 0.0)
        ax.add_patch(
            Rectangle(
                (start_sec, fmin_hz),
                end_sec - start_sec,
                fmax_hz - fmin_hz,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        )
        ax.text(
            start_sec,
            fmax_hz + 35,
            f"{row.get('motif_id', '')}:{suffix}",
            color=color,
            fontsize=8,
            weight="bold",
        )

    ax.set_ylim(DISPLAY_BAND)
    ax.set_xlim(window_start, window_end)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(
        f"Curated motif annotations: {item.name}\n"
        f"{item.motif_family}; green=primary boxes, cyan=secondary boxes; shaded bands are validation bands"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def emit_curated(output_dir: Path) -> None:
    """Detect, save, and render the curated current-best motif sets under output_dir/curated/.

    Each curated entry has its own detector configuration (seed-window for
    iNat1448422, independent primary/secondary bands for iNat1697350,
    single-component chip detector for iNat1698104). Tables go under
    curated/tables/ and standardized validation plots under curated/plots/.
    """
    curated_dir = output_dir / "curated"
    plots_dir = fresh_subdir(curated_dir, "plots")
    tables_dir = fresh_subdir(curated_dir, "tables")

    detect_seed_window_boxes(
        "22967/iNat1448422.ogg", start_sec=0.0, duration_sec=5.0
    ).write_csv(tables_dir / "iNat1448422_0-5s_motif_boxes.csv")

    detect_independent_band_boxes_for_recording(
        "22967/iNat1697350.ogg",
        secondary_band=(1000.0, 3000.0),
        expected_primary_count=10,
        expected_secondary_count=10,
    ).write_csv(tables_dir / "iNat1697350_independent_band_motif_boxes.csv")

    detect_single_component_boxes_for_recording(
        "22967/iNat1698104.ogg",
        band=(1000.0, 1800.0),
        component="chip_1k_1p8k",
        peak_height=0.7,
        peak_prominence=0.3,
        lower_pad_hz=100.0,
        upper_pad_hz=80.0,
        expected_count=5,
    ).write_csv(tables_dir / "iNat1698104_1k_1p8k_chip_boxes.csv")

    manifest_rows: list[dict[str, str]] = []
    combined_tables: list[pl.DataFrame] = []
    for item in CURATED_ANNOTATIONS:
        source_table = tables_dir / item.source_table
        plot_dst = plots_dir / item.output_plot_name
        render_curated_plot(item, source_table, plot_dst)

        manifest_rows.append(
            {
                "name": item.name,
                "motif_family": item.motif_family,
                "status": item.status,
                "source_table": str(source_table.relative_to(output_dir)),
                "curated_plot": str(plot_dst.relative_to(output_dir)),
                "primary_band_hz": f"{item.primary_band[0]:.0f}-{item.primary_band[1]:.0f}",
                "secondary_band_hz": (
                    "" if item.secondary_band is None else f"{item.secondary_band[0]:.0f}-{item.secondary_band[1]:.0f}"
                ),
                "notes": item.notes,
            }
        )

        tagged = pl.read_csv(source_table).with_columns(
            pl.lit(item.name).alias("annotation_set"),
            pl.lit(item.motif_family).alias("motif_family"),
            pl.lit(item.status).alias("annotation_status"),
            pl.lit(item.notes).alias("annotation_notes"),
        )
        combined_tables.append(tagged)

    pl.DataFrame(manifest_rows).write_csv(curated_dir / "manifest.csv")
    if combined_tables:
        pl.concat(combined_tables, how="diagonal_relaxed").write_csv(
            tables_dir / "curated_motif_boxes.csv"
        )


def run_template_matching(output_dir: Path, *, seed_filename: str = DEFAULT_FILENAME) -> None:
    """Build a seed template and match it across all Marbled Frog train recordings."""
    template, _template_freqs, _template_times, seed_boxes = build_seed_template(filename=seed_filename)
    template_dir = output_dir / "template_matches"
    template_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = fresh_subdir(template_dir, "plots")
    tables_dir = fresh_subdir(template_dir, "tables")

    all_tables = []
    for audio_path in sorted((TRAIN_AUDIO_DIR / "22967").glob("*.ogg")):
        filename = f"22967/{audio_path.name}"
        boxes, db_band, _z, freqs, times, _scores = detect_template_match_boxes(filename, template)
        table = template_boxes_to_frame(boxes, filename)
        if not table.is_empty():
            all_tables.append(table)
        stem = Path(filename).stem
        table.write_csv(tables_dir / f"{stem}_template_match_boxes.csv")
        plot_template_match_boxes(
            db_band,
            freqs,
            times,
            boxes,
            plots_dir / f"{stem}_template_match_boxes.png",
            filename=filename,
        )

    if all_tables:
        matches = pl.concat(all_tables, how="diagonal_relaxed")
        matches.write_csv(tables_dir / "all_template_match_boxes.csv")
        plot_template_match_summary_grid(matches, plots_dir / "top_template_match_grid.png")


def run(output_dir: Path | None = None, *, seed_filename: str = DEFAULT_FILENAME) -> Path:
    """Run template matching across recordings and emit the curated motif sets."""
    output_dir = output_dir or artifact_dir("marbled_frog/motif_boxes")
    run_template_matching(output_dir, seed_filename=seed_filename)
    emit_curated(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed-filename", default=DEFAULT_FILENAME)
    args = parser.parse_args()
    output_dir = run(args.output_dir, seed_filename=args.seed_filename)
    print(f"Wrote Marbled Frog motif box analysis to {output_dir}")


if __name__ == "__main__":
    main()
