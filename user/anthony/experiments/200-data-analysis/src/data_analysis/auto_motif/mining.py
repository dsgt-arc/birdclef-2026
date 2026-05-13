"""Species-agnostic automated motif mining.

Pipeline (single ``run()`` entry point, takes a species label):

1. Load wideband robust-z spectrograms for every focal recording of the label.
2. Auto-discover up to ``max_bands`` frequency bands via per-recording-normalized
   median consensus across recordings (``auto_motif.band``).
3. Per band: propose loose ROIs via scikit-maad hysteresis at in-band percentile
   thresholds (no absolute z-thresholds), then refine to the dominant
   (log duration, log bandwidth) KDE mode.
4. Per band: EM-style template refinement. Init from the median patch of refined
   ROIs; E-step picks score-peaks above a bimodal split of the score
   distribution; M-step re-medians patches anchored at the band-mean envelope
   argmax. Stop when ``1 - cosine`` between iterations < ``template_change_eps``.
5. Final detection: scan with all converged family templates; reuse the existing
   ``motif_template_calibration`` NMS + cross-family consolidation so the output
   schema matches the manual pipeline.
6. Emit the same 2-panel diagnostic plot per recording as
   ``motif_recording_diagnostics``, using the auto-discovered primary band.

Assumes "most focal recordings are relatively clean" — the median consensus
spectrum across recordings will surface the species' true frequency bands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import polars as pl
from maad import rois, util
from scipy.ndimage import binary_closing, gaussian_filter1d, zoom
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from data_analysis.auto_motif.band import (
    BandDiscoveryConfig,
    consensus_spectrum,
    discover_bands,
    per_recording_mean_spectrum,
)
from data_analysis.common.curated_eval import (
    curated_recovery,
    detection_stage_metrics,
    interval_iou,
)
from data_analysis.common.paths import (
    ARTIFACTS_DIR,
    TRAIN_AUDIO_DIR,
    artifact_dir,
    fresh_subdir,
    recording_files_for_label,
)
from data_analysis.common.recording_plot import plot_recording_diagnostic
from data_analysis.common.spectrogram import (
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    WIDEBAND,
    wideband_robust_excess_spectrogram,
)
from data_analysis.marbled_frog.motif_bank import canonical_support, fixed_support_crop
from data_analysis.marbled_frog.motif_recording_diagnostics import (
    DEFAULT_ENVELOPE_POST_SEC,
    DEFAULT_ENVELOPE_PRE_SEC,
    band_envelope_peaks,
    compare_peaks_to_events,
    envelope_events_from_peaks,
)
from data_analysis.marbled_frog.motif_template_calibration import (
    consolidate_events,
    nms_events,
    normalize_resize_patch,
    summarize_events,
)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"maad\..*")

DEFAULT_LABEL = "22967"
LABEL_TO_DIR = {"22967": "marbled_frog"}
TEMPLATE_SHAPE = (64, 48)


@dataclass(frozen=True)
class AutoConfig:
    """All knobs for the auto pipeline. Defaults are species-independent."""

    band: BandDiscoveryConfig = field(default_factory=BandDiscoveryConfig)
    seed_percentile: float = 0.99
    grow_percentile: float = 0.90
    loose_min_duration_sec: float = 0.02
    loose_max_duration_sec: float = 2.0
    loose_min_bandwidth_hz: float = 50.0
    loose_max_bandwidth_hz: float = 3000.0
    min_area_pixels: int = 6
    closing_freq_bins: int = 1
    closing_time_bins: int = 3
    kde_mode_k_mad: float = 2.5
    max_em_iter: int = 5
    template_change_eps: float = 0.01
    min_repetitions: int = 5
    nms_time_iou: float = 0.35
    nms_peak_distance_sec: float = 0.12
    consolidate_peak_distance_sec: float = 0.25
    consolidate_min_time_iou: float = 0.05
    threshold_floor: float = 0.45
    threshold_ceiling: float = 0.90
    fallback_quantile: float = 0.95
    seed: int = 42


# -----------------------------------------------------------------------------
# Spectrogram cache
# -----------------------------------------------------------------------------


def load_focal_z(
    files: list[str],
    *,
    band: tuple[float, float] = WIDEBAND,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load every focal recording once and cache the wideband z spectrogram."""
    cache: dict[str, tuple] = {}
    for filename in files:
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
        cache[filename] = wideband_robust_excess_spectrogram(audio, band=band)
    return cache


# -----------------------------------------------------------------------------
# Band discovery
# -----------------------------------------------------------------------------


def auto_discover_bands(
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    config: BandDiscoveryConfig,
) -> tuple[list[tuple[float, float]], pl.DataFrame]:
    """Discover species bands; also return the per-recording + consensus spectra."""
    spectra: list[np.ndarray] = []
    rows: list[dict] = []
    freqs_ref: np.ndarray | None = None
    for filename, (_db, z, freqs, _times) in cache.items():
        spectrum = per_recording_mean_spectrum(z)
        spectra.append(spectrum)
        freqs_ref = freqs
        for f, v in zip(freqs, spectrum):
            rows.append({"filename": filename, "freq_hz": float(f), "z_normalized": float(v)})
    if freqs_ref is None:
        return [], pl.DataFrame()
    consensus = consensus_spectrum(spectra)
    bands = discover_bands(consensus, freqs_ref, config=config)
    for f, v in zip(freqs_ref, consensus):
        rows.append({"filename": "__consensus__", "freq_hz": float(f), "z_normalized": float(v)})
    return bands, pl.DataFrame(rows)


# -----------------------------------------------------------------------------
# Loose ROI proposal (per-band, percentile thresholds)
# -----------------------------------------------------------------------------


def _propose_band_rois(
    filename: str,
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    band: tuple[float, float],
    band_id: int,
    *,
    config: AutoConfig,
    candidate_offset: int,
) -> list[dict]:
    """Hysteresis ROIs in a single band, using per-recording in-band percentiles."""
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return []
    z_band = z[band_mask]
    freqs_band = freqs[band_mask]
    seed_thr = float(np.quantile(z_band, config.seed_percentile))
    grow_thr = float(np.quantile(z_band, config.grow_percentile))
    if seed_thr <= grow_thr:
        seed_thr = grow_thr + 1e-3
    im_bin = rois.create_mask(z_band, mode_bin="absolute", bin_h=seed_thr, bin_l=grow_thr)
    if config.closing_freq_bins > 1 or config.closing_time_bins > 1:
        im_bin = binary_closing(
            im_bin.astype(bool),
            structure=np.ones((config.closing_freq_bins, config.closing_time_bins), dtype=bool),
        )
    im_rois, rois_df = rois.select_rois(im_bin.astype(bool), min_roi=config.min_area_pixels)
    if rois_df.empty:
        return []
    rois_df = util.format_features(rois_df, times, freqs_band)

    rows: list[dict] = []
    for roi_row in rois_df.itertuples(index=False):
        label_id = int(getattr(roi_row, "labelID"))
        rr, cc = np.where(im_rois == label_id)
        if rr.size < config.min_area_pixels:
            continue
        f_lo, f_hi = int(rr.min()), int(rr.max())
        t_lo, t_hi = int(cc.min()), int(cc.max())
        start_sec = float(times[t_lo])
        end_sec = float(times[t_hi])
        fmin_hz = float(freqs_band[f_lo])
        fmax_hz = float(freqs_band[f_hi])
        duration = end_sec - start_sec
        bw = fmax_hz - fmin_hz
        if not (config.loose_min_duration_sec <= duration <= config.loose_max_duration_sec):
            continue
        if not (config.loose_min_bandwidth_hz <= bw <= config.loose_max_bandwidth_hz):
            continue
        values = z_band[rr, cc]
        peak_idx = int(np.argmax(values))
        bbox_area = (f_hi - f_lo + 1) * (t_hi - t_lo + 1)
        candidate_id = candidate_offset + len(rows) + 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "filename": filename,
                "band_id": band_id,
                "band_fmin_hz": band[0],
                "band_fmax_hz": band[1],
                "seed_threshold_z": seed_thr,
                "grow_threshold_z": grow_thr,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration,
                "fmin_hz": fmin_hz,
                "fmax_hz": fmax_hz,
                "bandwidth_hz": bw,
                "peak_time_sec": float(times[cc[peak_idx]]),
                "peak_freq_hz": float(freqs_band[rr[peak_idx]]),
                "peak_z": float(values.max()),
                "mean_z": float(values.mean()),
                "area_pixels": int(rr.size),
                "fill_ratio": float(rr.size / bbox_area),
            }
        )
    return rows


def propose_loose_rois(
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    bands: list[tuple[float, float]],
    *,
    config: AutoConfig,
) -> pl.DataFrame:
    """Per-band hysteresis ROI proposal at in-band percentile thresholds."""
    rows: list[dict] = []
    offset = 0
    for filename, (_db, z, freqs, times) in cache.items():
        for band_id, band in enumerate(bands):
            band_rows = _propose_band_rois(
                filename, z, freqs, times, band, band_id,
                config=config, candidate_offset=offset,
            )
            rows.extend(band_rows)
            offset += len(band_rows)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


# -----------------------------------------------------------------------------
# KDE mode refinement (per band)
# -----------------------------------------------------------------------------


def kde_mode_refine(
    rois_band: pl.DataFrame,
    *,
    k_mad: float,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Filter ROIs to the dominant (log dur, log bw) KDE mode within k_mad.

    Returns the filtered DataFrame plus the mode estimates and bounds for
    artifact logging.
    """
    if rois_band.is_empty():
        return rois_band, {}
    log_dur = np.log10(rois_band["duration_sec"].to_numpy())
    log_bw = np.log10(rois_band["bandwidth_hz"].to_numpy())
    n = len(log_dur)
    if n < 5:
        return rois_band, {
            "n_input": int(n),
            "n_kept": int(n),
            "mode_log_duration": float(np.median(log_dur)),
            "mode_log_bandwidth": float(np.median(log_bw)),
        }
    pts = np.vstack([log_dur, log_bw])
    try:
        kde = gaussian_kde(pts)
        densities = kde(pts)
        mode_idx = int(np.argmax(densities))
        mode_log_dur = float(log_dur[mode_idx])
        mode_log_bw = float(log_bw[mode_idx])
    except (np.linalg.LinAlgError, ValueError):
        mode_log_dur = float(np.median(log_dur))
        mode_log_bw = float(np.median(log_bw))
    mad_dur = max(float(np.median(np.abs(log_dur - mode_log_dur))), 0.05)
    mad_bw = max(float(np.median(np.abs(log_bw - mode_log_bw))), 0.05)
    sigma_dur = 1.4826 * mad_dur
    sigma_bw = 1.4826 * mad_bw
    keep = (
        (np.abs(log_dur - mode_log_dur) <= k_mad * sigma_dur)
        & (np.abs(log_bw - mode_log_bw) <= k_mad * sigma_bw)
    )
    refined = rois_band.with_columns(pl.Series("_keep", keep)).filter(pl.col("_keep")).drop("_keep")
    info = {
        "n_input": int(n),
        "n_kept": int(refined.height),
        "mode_log_duration": mode_log_dur,
        "mode_log_bandwidth": mode_log_bw,
        "mode_duration_sec": float(10 ** mode_log_dur),
        "mode_bandwidth_hz": float(10 ** mode_log_bw),
        "log_duration_sigma": float(sigma_dur),
        "log_bandwidth_sigma": float(sigma_bw),
    }
    return refined, info


# -----------------------------------------------------------------------------
# Score-distribution thresholding (bimodal split with fallbacks)
# -----------------------------------------------------------------------------


def _is_bimodal(scores: np.ndarray) -> bool:
    """Cheap KDE-valley check: True if there is a clear dip between two modes."""
    if scores.size < 30:
        return False
    try:
        kde = gaussian_kde(scores)
    except (np.linalg.LinAlgError, ValueError):
        return False
    grid = np.linspace(float(scores.min()), float(scores.max()), 200)
    densities = kde(grid)
    peaks, _ = find_peaks(densities)
    if peaks.size < 2:
        return False
    top_two = peaks[np.argsort(densities[peaks])[-2:]]
    lo, hi = sorted(top_two)
    valley = float(densities[lo:hi].min())
    peak = float(densities[top_two].min())
    return valley < 0.7 * peak


def bimodal_split(
    scores: np.ndarray,
    *,
    fallback_quantile: float = 0.95,
    floor: float = 0.45,
    ceiling: float = 0.90,
) -> tuple[float, str]:
    """Return (threshold, method) splitting a score distribution into bg/fg."""
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return floor, "fallback_empty"
    if _is_bimodal(finite):
        try:
            gmm = GaussianMixture(n_components=2, random_state=0).fit(finite.reshape(-1, 1))
            means = gmm.means_.flatten()
            lo_idx = int(np.argmin(means))
            hi_idx = 1 - lo_idx
            grid = np.linspace(float(finite.min()), float(finite.max()), 400)
            lo_density = gmm.weights_[lo_idx] * np.exp(
                -0.5 * ((grid - means[lo_idx]) ** 2) / gmm.covariances_[lo_idx].flatten()[0]
            )
            hi_density = gmm.weights_[hi_idx] * np.exp(
                -0.5 * ((grid - means[hi_idx]) ** 2) / gmm.covariances_[hi_idx].flatten()[0]
            )
            crossover_idx = int(np.argmin(np.abs(lo_density - hi_density)))
            thr = float(grid[crossover_idx])
            return float(np.clip(thr, floor, ceiling)), "gmm2_crossover"
        except (np.linalg.LinAlgError, ValueError):
            pass
        hist, edges = np.histogram(finite, bins=64)
        total = hist.sum()
        if total > 0:
            cum_weight = np.cumsum(hist)
            cum_value = np.cumsum(hist * 0.5 * (edges[:-1] + edges[1:]))
            mean_total = cum_value[-1] / total
            best_var, best_thr = -1.0, float(np.median(finite))
            for i in range(1, len(hist)):
                w0 = cum_weight[i - 1] / total
                w1 = 1.0 - w0
                if w0 == 0 or w1 == 0:
                    continue
                m0 = cum_value[i - 1] / cum_weight[i - 1]
                m1 = (cum_value[-1] - cum_value[i - 1]) / (total - cum_weight[i - 1])
                between = w0 * w1 * (m0 - m1) ** 2
                if between > best_var:
                    best_var = between
                    best_thr = float(edges[i])
            return float(np.clip(best_thr, floor, ceiling)), "otsu"
    thr = float(np.quantile(finite, fallback_quantile))
    return float(np.clip(thr, floor, ceiling)), "fallback_quantile"


# -----------------------------------------------------------------------------
# Template scanning (single-template version of scan_cluster)
# -----------------------------------------------------------------------------


def _score_track(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    template: np.ndarray,
    support: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Slide one template over a recording's z; return (center_times, scores)."""
    band_mask = (freqs >= support["template_fmin_hz"]) & (freqs <= support["template_fmax_hz"])
    if not np.any(band_mask):
        return np.empty(0), np.empty(0)
    z_band = z[band_mask]
    dt = float(np.median(np.diff(times))) if times.size > 1 else MEL_HOP_LENGTH / MEL_SAMPLE_RATE
    pre_frames = max(1, int(round(support["template_pre_sec"] / dt)))
    post_frames = max(1, int(round(support["template_post_sec"] / dt)))
    centers = np.arange(pre_frames, z_band.shape[1] - post_frames)
    if centers.size == 0:
        return np.empty(0), np.empty(0)
    template_vec = template.ravel()
    scores = np.empty(centers.size, dtype=np.float32)
    for out_idx, center in enumerate(centers):
        patch = z_band[:, center - pre_frames : center + post_frames + 1]
        patch_vec = normalize_resize_patch(patch).ravel()
        scores[out_idx] = float(template_vec @ patch_vec / patch_vec.size)
    return times[centers], scores


# -----------------------------------------------------------------------------
# EM template refinement
# -----------------------------------------------------------------------------


def _crop_at_peak(
    z: np.ndarray, freqs: np.ndarray, times: np.ndarray,
    peak_time_sec: float, support: dict[str, float],
) -> np.ndarray:
    return fixed_support_crop(z, freqs, times, peak_time_sec=peak_time_sec, support=support)


def init_template(
    rois_band: pl.DataFrame,
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    band: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Median patch of refined ROIs anchored at peak_time."""
    support = canonical_support(rois_band, freq_clip=band)
    patches: list[np.ndarray] = []
    for row in rois_band.iter_rows(named=True):
        _db, z, freqs, times = cache[row["filename"]]
        patches.append(_crop_at_peak(z, freqs, times, float(row["peak_time_sec"]), support))
    template = np.median(np.stack(patches, axis=0), axis=0).astype(np.float32)
    template = (template - template.mean()) / (template.std() + 1e-6)
    return template, support


def _refine_anchor_via_envelope(
    z: np.ndarray, freqs: np.ndarray, times: np.ndarray,
    peak_time_sec: float, support: dict[str, float],
) -> float:
    """Re-anchor a peak at the band-mean envelope argmax inside its support window."""
    band_mask = (freqs >= support["template_fmin_hz"]) & (freqs <= support["template_fmax_hz"])
    if not np.any(band_mask):
        return peak_time_sec
    envelope = z[band_mask].mean(axis=0)
    t0 = peak_time_sec - support["template_pre_sec"]
    t1 = peak_time_sec + support["template_post_sec"]
    window = (times >= t0) & (times <= t1)
    if not np.any(window):
        return peak_time_sec
    sub_env = envelope[window]
    sub_times = times[window]
    return float(sub_times[int(np.argmax(sub_env))])


def em_step(
    template: np.ndarray,
    support: dict[str, float],
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    config: AutoConfig,
) -> tuple[np.ndarray, pl.DataFrame, dict]:
    """One E + M iteration."""
    rows: list[dict] = []
    all_scores: list[float] = []
    per_file_tracks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for filename, (_db, z, freqs, times) in cache.items():
        center_times, scores = _score_track(z, freqs, times, template, support)
        if center_times.size == 0:
            continue
        per_file_tracks[filename] = (center_times, scores)
        all_scores.extend(scores.tolist())

    threshold, method = bimodal_split(
        np.asarray(all_scores, dtype=np.float64),
        fallback_quantile=config.fallback_quantile,
        floor=config.threshold_floor,
        ceiling=config.threshold_ceiling,
    )
    dt = MEL_HOP_LENGTH / MEL_SAMPLE_RATE
    min_spacing = max(0.12, 0.60 * float(support["template_duration_sec"]))
    distance = max(1, int(round(min_spacing / dt)))

    for filename, (center_times, scores) in per_file_tracks.items():
        peaks, _ = find_peaks(scores, height=threshold, distance=distance, prominence=0.02)
        _db, z, freqs, times = cache[filename]
        for peak_idx in peaks:
            raw_time = float(center_times[peak_idx])
            anchored_time = _refine_anchor_via_envelope(z, freqs, times, raw_time, support)
            rows.append(
                {
                    "filename": filename,
                    "raw_peak_time_sec": raw_time,
                    "peak_time_sec": anchored_time,
                    "score": float(scores[peak_idx]),
                }
            )

    kept = pl.DataFrame(rows) if rows else pl.DataFrame()
    if kept.is_empty():
        return template, kept, {"threshold": threshold, "method": method, "n_kept": 0}

    patches = []
    for row in kept.iter_rows(named=True):
        _db, z, freqs, times = cache[row["filename"]]
        patches.append(_crop_at_peak(z, freqs, times, float(row["peak_time_sec"]), support))
    new_template = np.median(np.stack(patches, axis=0), axis=0).astype(np.float32)
    new_template = (new_template - new_template.mean()) / (new_template.std() + 1e-6)
    return new_template, kept, {
        "threshold": threshold,
        "method": method,
        "n_kept": int(kept.height),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    num = float(av @ bv)
    den = float(np.linalg.norm(av) * np.linalg.norm(bv) + 1e-9)
    return num / den


def em_loop(
    bands: list[tuple[float, float]],
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    refined_rois: pl.DataFrame,
    *,
    config: AutoConfig,
) -> tuple[list[dict], pl.DataFrame]:
    """For each band, run EM until convergence; return accepted families + trace."""
    families: list[dict] = []
    trace_rows: list[dict] = []
    for band_id, band in enumerate(bands):
        if refined_rois.is_empty():
            continue
        rois_band = refined_rois.filter(pl.col("band_id") == band_id)
        if rois_band.height < config.min_repetitions:
            trace_rows.append({
                "band_id": band_id, "iter": -1, "n_kept": rois_band.height,
                "threshold": float("nan"), "method": "skipped_too_few_rois",
                "cosine_to_prev": float("nan"),
            })
            continue
        template, support = init_template(rois_band, cache, band)
        prev_template = template.copy()
        last_kept: pl.DataFrame | None = None
        last_info: dict | None = None
        converged = False
        for iteration in range(config.max_em_iter):
            new_template, kept, info = em_step(template, support, cache, config=config)
            cosine = _cosine(new_template, prev_template)
            trace_rows.append({
                "band_id": band_id, "iter": iteration,
                "n_kept": info["n_kept"], "threshold": info["threshold"],
                "method": info["method"], "cosine_to_prev": cosine,
            })
            template = new_template
            last_kept = kept
            last_info = info
            if 1.0 - cosine < config.template_change_eps and iteration > 0:
                converged = True
                break
            prev_template = template.copy()
        if last_kept is None or last_kept.is_empty():
            continue
        if last_kept.height < config.min_repetitions:
            trace_rows.append({
                "band_id": band_id, "iter": -2, "n_kept": last_kept.height,
                "threshold": float("nan"), "method": "dropped_below_min_repetitions",
                "cosine_to_prev": float("nan"),
            })
            continue
        families.append({
            "band_id": band_id,
            "band": band,
            "template": template,
            "support": support,
            "kept_peaks": last_kept,
            "threshold": last_info["threshold"] if last_info else float("nan"),
            "threshold_method": last_info["method"] if last_info else "unknown",
            "converged": converged,
        })
    return families, pl.DataFrame(trace_rows)


# -----------------------------------------------------------------------------
# Final detection (reuses motif_template_calibration NMS + consolidation)
# -----------------------------------------------------------------------------


def _member_vs_background_threshold(
    member_scores: np.ndarray,
    background_scores: np.ndarray,
    *,
    floor: float,
    ceiling: float,
) -> tuple[float, str]:
    """Threshold = min(Q25 members, Q99.5 background), clipped.

    Mirrors ``motif_template_calibration.estimate_thresholds`` so the auto and
    manual pipelines apply the same recipe for accepting score-track peaks.
    """
    if member_scores.size == 0:
        thr = float(np.quantile(background_scores, 0.99)) if background_scores.size else floor
        return float(np.clip(thr, floor, ceiling)), "background_only"
    q25_pos = float(np.quantile(member_scores, 0.25))
    q995_bg = float(np.quantile(background_scores, 0.995)) if background_scores.size else q25_pos
    raw = min(q25_pos, q995_bg) if q25_pos >= floor else q995_bg
    return float(np.clip(raw, floor, ceiling)), "members_q25_or_bg_q995"


def detect_with_auto_templates(
    families: list[dict],
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    config: AutoConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Final scan + NMS + cross-family consolidation; mirrors the manual pipeline."""
    score_rows: list[dict] = []
    event_rows: list[dict] = []
    for family in families:
        band_id = family["band_id"]
        support = family["support"]
        template = family["template"]

        # Re-derive the final-detection threshold from EM-kept members vs
        # background score-track values, exactly like motif_template_calibration.
        member_times = {
            (row["filename"], float(row["peak_time_sec"]))
            for row in family["kept_peaks"].iter_rows(named=True)
        }
        member_files = {f for f, _ in member_times}
        member_score_list: list[float] = []
        background_score_list: list[float] = []
        per_file_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for filename, (_db, z, freqs, times) in cache.items():
            ct, sc = _score_track(z, freqs, times, template, support)
            per_file_scores[filename] = (ct, sc)
            mem_times_file = [t for (f, t) in member_times if f == filename]
            if mem_times_file and ct.size:
                keep_bg = np.ones(ct.size, dtype=bool)
                for t in mem_times_file:
                    idx = int(np.argmin(np.abs(ct - t)))
                    member_score_list.append(float(sc[idx]))
                    keep_bg &= np.abs(ct - t) > 0.25
                background_score_list.extend(sc[keep_bg].tolist())
            elif filename not in member_files and ct.size:
                background_score_list.extend(sc.tolist())
        threshold, threshold_method = _member_vs_background_threshold(
            np.asarray(member_score_list),
            np.asarray(background_score_list),
            floor=config.threshold_floor,
            ceiling=config.threshold_ceiling,
        )
        family["threshold"] = threshold
        family["threshold_method"] = threshold_method
        dt = MEL_HOP_LENGTH / MEL_SAMPLE_RATE
        min_spacing = max(0.12, 0.60 * float(support["template_duration_sec"]))
        distance = max(1, int(round(min_spacing / dt)))
        for filename, (center_times, scores) in per_file_scores.items():
            for t, s in zip(center_times, scores):
                score_rows.append({
                    "filename": filename, "motif_cluster": band_id,
                    "time_sec": float(t), "score": float(s),
                    "best_template_index": 0,
                })
            if center_times.size == 0:
                continue
            peaks, props = find_peaks(scores, height=threshold, distance=distance, prominence=0.02)
            for peak in peaks:
                peak_time = float(center_times[peak])
                event_rows.append({
                    "filename": filename,
                    "motif_cluster": band_id,
                    "peak_time_sec": peak_time,
                    "start_sec": max(0.0, peak_time - float(support["template_pre_sec"])),
                    "end_sec": peak_time + float(support["template_post_sec"]),
                    "fmin_hz": float(support["template_fmin_hz"]),
                    "fmax_hz": float(support["template_fmax_hz"]),
                    "score": float(scores[peak]),
                    "threshold": threshold,
                    "prominence": float(props["prominences"][np.where(peaks == peak)[0][0]])
                        if "prominences" in props else float("nan"),
                    "best_template_index": 0,
                })
    score_tracks = pl.DataFrame(score_rows)
    raw_events = pl.DataFrame(event_rows)
    nms = nms_events(
        raw_events,
        time_iou_threshold=config.nms_time_iou,
        peak_distance_sec=config.nms_peak_distance_sec,
    )
    consolidated = consolidate_events(
        nms,
        merge_peak_distance_sec=config.consolidate_peak_distance_sec,
        min_time_iou=config.consolidate_min_time_iou,
    )
    return score_tracks, raw_events, nms, consolidated


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def _plot_band_consensus(consensus_df: pl.DataFrame, bands: list[tuple[float, float]], output_path: Path) -> None:
    cons = consensus_df.filter(pl.col("filename") == "__consensus__").sort("freq_hz")
    others = consensus_df.filter(pl.col("filename") != "__consensus__")
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=180)
    for filename, group in others.group_by("filename", maintain_order=True):
        group = group.sort("freq_hz")
        ax.plot(group["freq_hz"], group["z_normalized"], color="0.7", linewidth=0.7, alpha=0.6, label="_nolegend_")
    ax.plot(cons["freq_hz"], cons["z_normalized"], color="black", linewidth=2.0, label="median consensus")
    for i, (fmin, fmax) in enumerate(bands):
        ax.axvspan(fmin, fmax, color="C{}".format(i % 10), alpha=0.18, label=f"band {i}: {fmin:.0f}-{fmax:.0f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("normalized z (per-recording MAD-sigma)")
    ax.set_title("Auto band discovery: per-recording spectra + median consensus")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_kde_modes(refined_info: list[dict], rois: pl.DataFrame, output_path: Path) -> None:
    if rois.is_empty():
        return
    bands = sorted(rois["band_id"].unique().to_list())
    n_cols = min(3, max(1, len(bands)))
    n_rows = int(np.ceil(len(bands) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.3 * n_rows), dpi=180, squeeze=False)
    info_by_band = {item["band_id"]: item for item in refined_info}
    for idx, band_id in enumerate(bands):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        df = rois.filter(pl.col("band_id") == band_id)
        ax.scatter(np.log10(df["duration_sec"]), np.log10(df["bandwidth_hz"]),
                   s=12, alpha=0.55, color=f"C{band_id % 10}")
        info = info_by_band.get(band_id)
        if info:
            ax.axvline(info["mode_log_duration"], color="black", linestyle="--", linewidth=0.8)
            ax.axhline(info["mode_log_bandwidth"], color="black", linestyle="--", linewidth=0.8)
            ax.set_title(f"band {band_id}: kept {info['n_kept']}/{info['n_input']}", fontsize=9)
        ax.set_xlabel("log10 duration (s)")
        ax.set_ylabel("log10 bandwidth (Hz)")
        ax.grid(alpha=0.25)
    for idx in range(len(bands), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_em_convergence(trace: pl.DataFrame, output_path: Path) -> None:
    if trace.is_empty():
        return
    df = trace.filter(pl.col("iter") >= 0)
    if df.is_empty():
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    for band_id in sorted(df["band_id"].unique().to_list()):
        sub = df.filter(pl.col("band_id") == band_id).sort("iter")
        ax.plot(sub["iter"], 1.0 - sub["cosine_to_prev"], marker="o", label=f"band {band_id}")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("1 - cosine(template, prev)")
    ax.set_title("EM template convergence")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_templates(families: list[dict], output_dir: Path) -> None:
    for family in families:
        template = family["template"]
        band_id = family["band_id"]
        n_kept = family["kept_peaks"].height
        support = family["support"]
        fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=180)
        ax.imshow(template, origin="lower", aspect="auto", cmap="magma",
                  extent=[
                      -support["template_pre_sec"], support["template_post_sec"],
                      support["template_fmin_hz"], support["template_fmax_hz"],
                  ])
        ax.set_xlabel("Time relative to peak (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"band {band_id} template (n={n_kept}, conv={family['converged']})", fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / f"template_family_{band_id:02d}.png")
        plt.close(fig)


def _plot_full_recording_events(
    events: pl.DataFrame,
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    bands: list[tuple[float, float]],
    display_ylim: tuple[float, float],
    output_dir: Path,
) -> None:
    if events.is_empty():
        return
    cmap = plt.cm.tab10
    for filename, (db_band, _z, freqs, times) in cache.items():
        width = max(12.0, min(24.0, 0.42 * float(times[-1]) + 6.0))
        fig, ax = plt.subplots(figsize=(width, 6.2), dpi=180)
        import librosa.display as _ld
        _ld.specshow(db_band, x_coords=times, y_coords=freqs,
                     x_axis="time", y_axis="linear",
                     sr=MEL_SAMPLE_RATE, hop_length=MEL_HOP_LENGTH,
                     ax=ax, cmap="magma")
        df = events.filter(pl.col("filename") == filename).sort("peak_time_sec")
        for row in df.iter_rows(named=True):
            cluster = int(row["best_motif_cluster"] if "best_motif_cluster" in row else row["motif_cluster"])
            color = cmap(cluster % 10)
            ax.add_patch(Rectangle(
                (row["start_sec"], row["fmin_hz"]),
                row["end_sec"] - row["start_sec"],
                row["fmax_hz"] - row["fmin_hz"],
                fill=False, edgecolor=color, linewidth=1.6, alpha=0.95,
            ))
            ax.text(row["peak_time_sec"], min(display_ylim[1] - 45, row["fmax_hz"] + 35),
                    f"b{cluster}\n{row['score']:.2f}",
                    color=color, fontsize=6.5, ha="center", va="bottom", weight="bold")
        ax.set_ylim(display_ylim)
        ax.set_xlim(0.0, float(times[-1]))
        ax.set_title(
            f"Auto motif events: {Path(filename).stem}\n"
            f"n={df.height} consolidated events; b#=auto band id"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{Path(filename).stem}_full_recording_events.png")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def species_output_dir(label: str) -> Path:
    """Map a label to its artifact subdir (falls back to the label itself)."""
    subdir = LABEL_TO_DIR.get(label, label)
    return ARTIFACTS_DIR / subdir / "motif_mining_auto"


def run(
    label: str = DEFAULT_LABEL,
    output_dir: Path | None = None,
    *,
    config: AutoConfig | None = None,
    curated_path: Path | None = None,
) -> Path:
    config = config or AutoConfig()
    output_dir = output_dir or species_output_dir(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = fresh_subdir(output_dir, "tables")
    plots_dir = fresh_subdir(output_dir, "plots")
    templates_dir = fresh_subdir(output_dir, "templates")

    files = recording_files_for_label(label)
    if not files:
        raise RuntimeError(f"No focal recordings found for label {label}")

    # 1. Load wideband spectrograms.
    cache = load_focal_z(files, band=WIDEBAND)

    # 2. Discover bands.
    bands, consensus_df = auto_discover_bands(cache, config=config.band)
    bands_df = pl.DataFrame([
        {"band_id": i, "fmin_hz": float(fmin), "fmax_hz": float(fmax), "bandwidth_hz": float(fmax - fmin)}
        for i, (fmin, fmax) in enumerate(bands)
    ])
    bands_df.write_csv(tables_dir / "auto_bands.csv")
    consensus_df.write_csv(tables_dir / "auto_band_consensus.csv")
    _plot_band_consensus(consensus_df, bands, plots_dir / "auto_band_consensus.png")
    if not bands:
        raise RuntimeError("Auto band discovery returned no bands; check k_band_mad / spectra")

    # 3. Loose ROIs + KDE mode refinement (per band).
    loose = propose_loose_rois(cache, bands, config=config)
    loose.write_csv(tables_dir / "loose_rois.csv")
    refined_rows: list[pl.DataFrame] = []
    refined_info_all: list[dict] = []
    for band_id in range(len(bands)):
        band_loose = loose.filter(pl.col("band_id") == band_id) if not loose.is_empty() else loose
        refined_band, info = kde_mode_refine(band_loose, k_mad=config.kde_mode_k_mad)
        if not refined_band.is_empty():
            refined_rows.append(refined_band)
        refined_info_all.append({"band_id": band_id, **info})
    refined = pl.concat(refined_rows, how="diagonal_relaxed") if refined_rows else pl.DataFrame()
    refined.write_csv(tables_dir / "kde_refined_rois.csv")
    pl.DataFrame(refined_info_all).write_csv(tables_dir / "kde_modes.csv")
    _plot_kde_modes(refined_info_all, refined, plots_dir / "kde_duration_bandwidth.png")

    # 4. EM template refinement.
    families, em_trace = em_loop(bands, cache, refined, config=config)
    em_trace.write_csv(tables_dir / "em_trace.csv")
    _plot_em_convergence(em_trace, plots_dir / "em_convergence.png")
    _plot_templates(families, plots_dir)

    if not families:
        raise RuntimeError("EM produced no accepted motif families; check min_repetitions / thresholds")

    family_meta = pl.DataFrame([
        {
            "band_id": f["band_id"],
            "band_fmin_hz": f["band"][0],
            "band_fmax_hz": f["band"][1],
            "threshold": f["threshold"],
            "threshold_method": f["threshold_method"],
            "converged": f["converged"],
            "n_kept_peaks": f["kept_peaks"].height,
            **f["support"],
        } for f in families
    ])
    family_meta.write_csv(tables_dir / "auto_templates_metadata.csv")
    for f in families:
        np.savez_compressed(
            templates_dir / f"family_{f['band_id']:02d}_template.npz",
            template=f["template"],
            **{k: np.array([v], dtype=np.float32) for k, v in f["support"].items()},
        )

    # 5. Final detection (reuse manual pipeline NMS + consolidation).
    score_tracks, raw_events, nms, consolidated = detect_with_auto_templates(
        families, cache, config=config,
    )
    score_tracks.write_csv(tables_dir / "template_score_tracks.csv")
    raw_events.write_csv(tables_dir / "template_match_events_raw.csv")
    nms.write_csv(tables_dir / "template_match_events_nms.csv")
    consolidated.write_csv(tables_dir / "consolidated_events.csv")
    if not consolidated.is_empty():
        consolidated_summary, intervals = summarize_events(consolidated, cluster_col="best_motif_cluster")
        consolidated_summary.write_csv(tables_dir / "event_summary.csv")
        intervals.write_csv(tables_dir / "inter_event_intervals.csv")

    # 6. Curated recovery (optional — only Marbled Frog has these today).
    if curated_path is not None and curated_path.exists():
        raw_recovery = curated_recovery(raw_events, curated_path)
        nms_recovery = curated_recovery(nms, curated_path)
        cons_recovery = curated_recovery(consolidated, curated_path)
        stage_metrics = detection_stage_metrics(
            raw_events, nms, consolidated, raw_recovery, nms_recovery, cons_recovery,
        )
        stage_metrics.write_csv(tables_dir / "detection_stage_metrics.csv")
        if not cons_recovery.is_empty():
            cons_recovery.write_csv(tables_dir / "curated_recovery.csv")
            cons_summary = cons_recovery.group_by("component").agg(
                pl.len().alias("n_curated_boxes"),
                pl.col("recovered").sum().alias("n_recovered"),
                pl.col("recovered").mean().alias("recovery_rate"),
                pl.col("best_detection_score").median().alias("median_best_detection_score"),
            ).sort("component")
            cons_summary.write_csv(tables_dir / "curated_recovery_summary.csv")

    # 7. Per-recording full-recording event plot.
    _plot_full_recording_events(consolidated, cache, bands, WIDEBAND, plots_dir)

    # 8. Per-recording band-envelope-vs-template diagnostic plot
    #    (same 2-panel layout as motif_recording_diagnostics).
    if bands:
        primary_band = bands[0]
        for filename, (db_band, z, freqs, times) in cache.items():
            peak_idx, envelope, _ = band_envelope_peaks(z, freqs, times, band=primary_band)
            peak_times = times[peak_idx]
            env_events = envelope_events_from_peaks(filename, peak_times, envelope, times, band=primary_band)
            peak_rows, _ev_rows, _metrics = compare_peaks_to_events(filename, peak_times, consolidated)
            if not env_events.is_empty() and peak_rows:
                agreement = pl.DataFrame(peak_rows).select(
                    pl.col("envelope_peak_time_sec").alias("peak_time_sec"),
                    "matched_template_event",
                    "nearest_event_distance_sec",
                    "nearest_event_id",
                    "nearest_event_cluster",
                    "nearest_event_score",
                )
                env_events = env_events.join(agreement, on="peak_time_sec", how="left")
            plot_recording_diagnostic(
                filename, db_band, z, freqs, times, envelope, peak_rows,
                consolidated, env_events,
                plots_dir / f"{Path(filename).stem}_band_envelope_vs_template_events.png",
                band=primary_band,
                display_ylim=WIDEBAND,
                title_extra=f"auto bands: {', '.join(f'{a:.0f}-{b:.0f}' for a, b in bands)}",
            )

    # 9. Summary markdown.
    (output_dir / "summary.md").write_text(
        f"""# Auto motif mining: {label}

Species-agnostic pipeline (no per-species hand-tuning).

Discovered bands ({len(bands)}):

```text
{bands_df.write_csv(separator='|').strip()}
```

Accepted motif families ({len(families)}):

```text
{family_meta.write_csv(separator='|').strip()}
```

Artifacts:

- `tables/auto_bands.csv` — discovered frequency bands.
- `tables/auto_band_consensus.csv` — per-recording + median consensus spectra.
- `tables/loose_rois.csv` — loose hysteresis ROIs at in-band percentile thresholds.
- `tables/kde_refined_rois.csv` — ROIs surviving (log dur, log bw) KDE mode filter.
- `tables/kde_modes.csv` — per-band KDE mode estimates.
- `tables/em_trace.csv` — EM iteration trace (n_kept, threshold, threshold_method, 1-cosine).
- `tables/auto_templates_metadata.csv` — final accepted families.
- `tables/template_score_tracks.csv` — dense score tracks per recording/family.
- `tables/template_match_events_raw.csv` — raw peak-picked detections.
- `tables/template_match_events_nms.csv` — within-family NMS.
- `tables/consolidated_events.csv` — cross-family consolidated event boxes.
- `tables/event_summary.csv` — consolidated event counts per family.
- `tables/inter_event_intervals.csv` — consecutive event intervals.
- `tables/curated_recovery.csv` — best consolidated event per curated box (if curated provided).
- `tables/detection_stage_metrics.csv` — raw vs NMS vs consolidated reduction + recovery.
- `templates/family_XX_template.npz` — fixed-support templates per band.
- `plots/auto_band_consensus.png` — band discovery visualization.
- `plots/kde_duration_bandwidth.png` — per-band KDE mode + kept ROIs.
- `plots/em_convergence.png` — per-family 1-cosine vs iter.
- `plots/template_family_XX.png` — final templates.
- `plots/{{stem}}_full_recording_events.png` — full focal recordings with auto event boxes.
- `plots/{{stem}}_band_envelope_vs_template_events.png` — 2-panel diagnostic (same layout as motif_recording_diagnostics).
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Species label (default: Marbled Frog)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--curated", type=Path, default=None,
                        help="Optional curated_motif_boxes.csv for recovery evaluation")
    parser.add_argument("--max-bands", type=int, default=BandDiscoveryConfig.max_bands)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = AutoConfig(band=BandDiscoveryConfig(max_bands=args.max_bands), seed=args.seed)
    output_dir = run(
        label=args.label,
        output_dir=args.output_dir,
        config=config,
        curated_path=args.curated,
    )
    print(f"Wrote auto motif mining to {output_dir}")


if __name__ == "__main__":
    main()
