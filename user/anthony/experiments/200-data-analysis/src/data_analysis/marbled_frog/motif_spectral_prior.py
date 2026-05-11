"""Spectral-mode priors for Marbled Frog motif candidates.

For each candidate box, average robust-excess-z over the candidate time span to
obtain a 1D frequency profile, detect spectral modes, estimate frequency support
boundaries, and score whether the box looks like a plausible motif component
rather than a top/bottom display-edge artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from data_analysis.common.paths import TRAIN_AUDIO_DIR, fresh_subdir
from data_analysis.common.spectrogram import DISPLAY_BAND, MEL_SAMPLE_RATE, robust_excess_spectrogram

DEFAULT_CANDIDATES = Path("artifacts/marbled_frog/motif_mining_maad_loose/tables/candidates.csv")
DEFAULT_OUTPUT = Path("artifacts/marbled_frog/motif_mining_maad_loose/spectral_prior")
EDGE_MARGIN_HZ = 80.0


def load_spectrogram_cache(candidates: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load each recording once and compute robust-excess-z spectrograms."""
    cache = {}
    for filename in candidates["filename"].unique().sort().to_list():
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
        cache[filename] = robust_excess_spectrogram(audio)
    return cache


def candidate_frequency_profile(
    z: np.ndarray,
    times: np.ndarray,
    row: dict,
    *,
    time_pad_sec: float = 0.03,
) -> np.ndarray:
    """Average robust-excess-z over the candidate time span."""
    t0 = max(0.0, float(row["start_sec"]) - time_pad_sec)
    t1 = min(float(times[-1]), float(row["end_sec"]) + time_pad_sec)
    t_mask = (times >= t0) & (times <= t1)
    if not np.any(t_mask):
        return np.zeros(z.shape[0], dtype=np.float32)
    profile = z[:, t_mask].mean(axis=1).astype(np.float32)
    # Remove broad floor so modes reflect local spectral components.
    floor = float(np.percentile(profile, 20))
    profile = np.maximum(profile - floor, 0.0)
    return profile


def detect_modes(profile: np.ndarray, freqs: np.ndarray, *, max_modes: int = 4) -> dict[str, object]:
    """Detect spectral modes and support intervals from a frequency profile."""
    if profile.size == 0 or float(profile.max(initial=0.0)) <= 1e-9:
        return {
            "n_modes": 0,
            "dominant_mode_hz": np.nan,
            "dominant_mode_prominence": 0.0,
            "dominant_support_fmin_hz": np.nan,
            "dominant_support_fmax_hz": np.nan,
            "dominant_support_width_hz": np.nan,
            "dominant_support_touches_edge": True,
            "modes": [],
            "smooth_profile": profile,
        }

    smooth = gaussian_filter1d(profile.astype(float), sigma=2.0)
    dynamic = float(smooth.max() - np.percentile(smooth, 20))
    prominence = max(0.10, 0.18 * dynamic)
    df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 15.625
    min_distance_bins = max(1, int(round(90.0 / df)))
    peaks, props = find_peaks(smooth, prominence=prominence, distance=min_distance_bins)
    if peaks.size == 0:
        peak = int(np.argmax(smooth))
        peaks = np.array([peak], dtype=int)
        prominences = np.array([float(smooth[peak])], dtype=float)
    else:
        prominences = props.get("prominences", np.zeros(peaks.size, dtype=float))

    order = np.argsort(prominences)[::-1][:max_modes]
    modes = []
    baseline = float(np.percentile(smooth, 20))
    for idx in order:
        peak_idx = int(peaks[idx])
        peak_value = float(smooth[peak_idx])
        support_threshold = baseline + 0.35 * max(0.0, peak_value - baseline)
        left = peak_idx
        while left > 0 and smooth[left] >= support_threshold:
            left -= 1
        right = peak_idx
        while right < smooth.size - 1 and smooth[right] >= support_threshold:
            right += 1
        fmin = float(freqs[left])
        fmax = float(freqs[right])
        modes.append(
            {
                "mode_hz": float(freqs[peak_idx]),
                "prominence": float(prominences[idx]),
                "support_fmin_hz": fmin,
                "support_fmax_hz": fmax,
                "support_width_hz": fmax - fmin,
                "touches_edge": fmin <= DISPLAY_BAND[0] + EDGE_MARGIN_HZ or fmax >= DISPLAY_BAND[1] - EDGE_MARGIN_HZ,
            }
        )

    dom = modes[0]
    return {
        "n_modes": len(modes),
        "dominant_mode_hz": dom["mode_hz"],
        "dominant_mode_prominence": dom["prominence"],
        "dominant_support_fmin_hz": dom["support_fmin_hz"],
        "dominant_support_fmax_hz": dom["support_fmax_hz"],
        "dominant_support_width_hz": dom["support_width_hz"],
        "dominant_support_touches_edge": dom["touches_edge"],
        "modes": modes,
        "smooth_profile": smooth,
    }


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    """1D IoU for frequency intervals."""
    if not np.isfinite([a0, a1, b0, b1]).all():
        return 0.0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return float(inter / union) if union > 0 else 0.0


def spectral_prior_score(row: dict, mode: dict[str, object], profile: np.ndarray, freqs: np.ndarray) -> dict[str, float | bool]:
    """Score candidate plausibility from spectral-mode information."""
    support_fmin = float(mode["dominant_support_fmin_hz"]) if np.isfinite(mode["dominant_support_fmin_hz"]) else np.nan
    support_fmax = float(mode["dominant_support_fmax_hz"]) if np.isfinite(mode["dominant_support_fmax_hz"]) else np.nan
    support_width = float(mode["dominant_support_width_hz"]) if np.isfinite(mode["dominant_support_width_hz"]) else np.nan
    bbox_iou = interval_iou(float(row["fmin_hz"]), float(row["fmax_hz"]), support_fmin, support_fmax)

    profile_sum = float(profile.sum()) + 1e-9
    bbox_mask = (freqs >= float(row["fmin_hz"])) & (freqs <= float(row["fmax_hz"]))
    support_mask = (freqs >= support_fmin) & (freqs <= support_fmax) if np.isfinite(support_fmin + support_fmax) else np.zeros_like(freqs, dtype=bool)
    bbox_energy_fraction = float(profile[bbox_mask].sum() / profile_sum) if np.any(bbox_mask) else 0.0
    support_energy_fraction = float(profile[support_mask].sum() / profile_sum) if np.any(support_mask) else 0.0

    edge_touch_bbox = float(row["fmin_hz"]) <= DISPLAY_BAND[0] + EDGE_MARGIN_HZ or float(row["fmax_hz"]) >= DISPLAY_BAND[1] - EDGE_MARGIN_HZ
    edge_touch_mode = bool(mode["dominant_support_touches_edge"])
    plausible_width = np.isfinite(support_width) and 80.0 <= support_width <= 1800.0
    prominence_score = min(1.0, float(mode["dominant_mode_prominence"]) / 1.2)
    overlap_score = min(1.0, bbox_iou / 0.45)
    energy_score = min(1.0, support_energy_fraction / 0.35)
    width_score = 1.0 if plausible_width else 0.35
    edge_score = 0.25 if edge_touch_mode else (0.55 if edge_touch_bbox else 1.0)
    score = 0.30 * prominence_score + 0.25 * overlap_score + 0.20 * energy_score + 0.15 * width_score + 0.10 * edge_score
    return {
        "bbox_mode_freq_iou": bbox_iou,
        "bbox_energy_fraction": bbox_energy_fraction,
        "support_energy_fraction": support_energy_fraction,
        "bbox_touches_display_edge": edge_touch_bbox,
        "spectral_width_plausible": bool(plausible_width),
        "spectral_prior_score": float(score),
    }


def compute_candidate_priors(candidates: pl.DataFrame) -> tuple[pl.DataFrame, dict[int, np.ndarray], np.ndarray]:
    """Compute spectral-mode prior features for every candidate."""
    cache = load_spectrogram_cache(candidates)
    rows = []
    profiles = {}
    freqs_ref = None
    for row in candidates.iter_rows(named=True):
        _db, z, freqs, times = cache[row["filename"]]
        if freqs_ref is None:
            freqs_ref = freqs
        profile = candidate_frequency_profile(z, times, row)
        mode = detect_modes(profile, freqs)
        score = spectral_prior_score(row, mode, profile, freqs)
        candidate_id = int(row["candidate_id"])
        profiles[candidate_id] = profile / (float(profile.sum()) + 1e-9)
        modes = mode["modes"]
        mode_string = ";".join(
            f"{m['mode_hz']:.1f}:{m['support_fmin_hz']:.1f}-{m['support_fmax_hz']:.1f}"
            for m in modes
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "filename": row["filename"],
                "motif_cluster": int(row["motif_cluster"]),
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "fmin_hz": row["fmin_hz"],
                "fmax_hz": row["fmax_hz"],
                "bandwidth_hz": row["bandwidth_hz"],
                "peak_z": row["peak_z"],
                "n_spectral_modes": int(mode["n_modes"]),
                "dominant_mode_hz": mode["dominant_mode_hz"],
                "dominant_mode_prominence": mode["dominant_mode_prominence"],
                "dominant_support_fmin_hz": mode["dominant_support_fmin_hz"],
                "dominant_support_fmax_hz": mode["dominant_support_fmax_hz"],
                "dominant_support_width_hz": mode["dominant_support_width_hz"],
                "dominant_support_touches_edge": mode["dominant_support_touches_edge"],
                "spectral_modes": mode_string,
                **score,
            }
        )
    assert freqs_ref is not None
    return pl.DataFrame(rows), profiles, freqs_ref


def recording_priors(
    candidate_priors: pl.DataFrame, profiles: dict[int, np.ndarray], freqs: np.ndarray
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate spectral priors from actual recordings/local clips, not clusters.

    For each recording, average candidate-local frequency profiles from that
    recording to estimate recurring spectral bands. Then score every candidate
    by how well its box and its local spectral support align with those
    recording-derived modes.
    """
    recording_rows = []
    candidate_rows = []
    for filename in sorted(candidate_priors["filename"].unique().to_list()):
        df = candidate_priors.filter(pl.col("filename") == filename).sort("start_sec")
        ids = [int(x) for x in df["candidate_id"].to_list()]
        recording_profile = np.mean(np.stack([profiles[i] for i in ids], axis=0), axis=0)
        rec_mode = detect_modes(recording_profile, freqs, max_modes=6)
        rec_modes = rec_mode["modes"]
        mode_string = ";".join(
            f"{m['mode_hz']:.1f}:{m['support_fmin_hz']:.1f}-{m['support_fmax_hz']:.1f}"
            for m in rec_modes
        )
        recording_rows.append(
            {
                "filename": filename,
                "n_candidates": df.height,
                "recording_dominant_mode_hz": rec_mode["dominant_mode_hz"],
                "recording_support_fmin_hz": rec_mode["dominant_support_fmin_hz"],
                "recording_support_fmax_hz": rec_mode["dominant_support_fmax_hz"],
                "recording_support_width_hz": rec_mode["dominant_support_width_hz"],
                "recording_support_touches_edge": rec_mode["dominant_support_touches_edge"],
                "recording_spectral_modes": mode_string,
            }
        )
        for row in df.iter_rows(named=True):
            best_bbox_iou = 0.0
            best_local_support_iou = 0.0
            best_mode_hz = np.nan
            best_fmin = np.nan
            best_fmax = np.nan
            for m in rec_modes:
                bbox_iou = interval_iou(row["fmin_hz"], row["fmax_hz"], m["support_fmin_hz"], m["support_fmax_hz"])
                support_iou = interval_iou(
                    row["dominant_support_fmin_hz"],
                    row["dominant_support_fmax_hz"],
                    m["support_fmin_hz"],
                    m["support_fmax_hz"],
                )
                combined = max(bbox_iou, support_iou)
                if combined > max(best_bbox_iou, best_local_support_iou):
                    best_bbox_iou = bbox_iou
                    best_local_support_iou = support_iou
                    best_mode_hz = m["mode_hz"]
                    best_fmin = m["support_fmin_hz"]
                    best_fmax = m["support_fmax_hz"]
            # Candidate-level prior from actual evidence: local spectral mode,
            # recording recurrence mode, and edge penalty. This is deliberately
            # independent of cluster membership.
            local_score = float(row["spectral_prior_score"])
            rec_align = max(best_bbox_iou, best_local_support_iou)
            edge_penalty = 0.45 if bool(row["bbox_touches_display_edge"]) else 1.0
            recording_aware_score = (0.60 * local_score + 0.40 * min(1.0, rec_align / 0.45)) * edge_penalty
            candidate_rows.append(
                {
                    "candidate_id": int(row["candidate_id"]),
                    "filename": filename,
                    "best_recording_mode_hz": best_mode_hz,
                    "best_recording_support_fmin_hz": best_fmin,
                    "best_recording_support_fmax_hz": best_fmax,
                    "bbox_recording_mode_iou": best_bbox_iou,
                    "local_support_recording_mode_iou": best_local_support_iou,
                    "recording_aware_spectral_prior_score": float(recording_aware_score),
                    "recording_aware_prior_label": (
                        "good"
                        if recording_aware_score >= 0.60
                        else ("edge_or_misaligned" if bool(row["bbox_touches_display_edge"]) or rec_align < 0.20 else "weak")
                    ),
                }
            )
    return pl.DataFrame(recording_rows), pl.DataFrame(candidate_rows)


def cluster_priors(candidate_priors: pl.DataFrame, profiles: dict[int, np.ndarray], freqs: np.ndarray) -> pl.DataFrame:
    """Aggregate candidate priors by cluster for diagnostics only.

    Do not use these cluster-average modes as source-of-truth priors. The actual
    priors are candidate-local and recording-derived.
    """
    rows = []
    for cluster in sorted(candidate_priors["motif_cluster"].unique().to_list()):
        df = candidate_priors.filter(pl.col("motif_cluster") == cluster)
        ids = [int(x) for x in df["candidate_id"].to_list()]
        cluster_profile = np.mean(np.stack([profiles[i] for i in ids], axis=0), axis=0)
        mode = detect_modes(cluster_profile, freqs)
        modes = mode["modes"]
        mode_string = ";".join(
            f"{m['mode_hz']:.1f}:{m['support_fmin_hz']:.1f}-{m['support_fmax_hz']:.1f}"
            for m in modes
        )
        edge_rate = float(df["dominant_support_touches_edge"].mean())
        bbox_edge_rate = float(df["bbox_touches_display_edge"].mean())
        score_med = float(df["spectral_prior_score"].median())
        # Cluster-level validity prior: many candidates can be individually modal,
        # but a motif family should not be mostly display-edge support.
        cluster_spectral_score = score_med * (1.0 - 0.55 * edge_rate) * (1.0 - 0.25 * bbox_edge_rate)
        rows.append(
            {
                "motif_cluster": int(cluster),
                "n_candidates": df.height,
                "n_recordings": df["filename"].n_unique(),
                "median_candidate_spectral_prior_score": score_med,
                "min_candidate_spectral_prior_score": float(df["spectral_prior_score"].min()),
                "edge_mode_rate": edge_rate,
                "bbox_edge_rate": bbox_edge_rate,
                "median_bbox_mode_freq_iou": float(df["bbox_mode_freq_iou"].median()),
                "median_support_width_hz": float(df["dominant_support_width_hz"].median()),
                "cluster_dominant_mode_hz": mode["dominant_mode_hz"],
                "cluster_support_fmin_hz": mode["dominant_support_fmin_hz"],
                "cluster_support_fmax_hz": mode["dominant_support_fmax_hz"],
                "cluster_support_width_hz": mode["dominant_support_width_hz"],
                "cluster_support_touches_edge": mode["dominant_support_touches_edge"],
                "cluster_spectral_modes": mode_string,
                "cluster_spectral_prior_score": float(cluster_spectral_score),
                "spectral_prior_label": "good" if cluster_spectral_score >= 0.55 else ("edge_or_mixed" if edge_rate >= 0.5 or bbox_edge_rate >= 0.75 else "weak"),
            }
        )
    return pl.DataFrame(rows).sort("cluster_spectral_prior_score", descending=True)


def plot_recording_profiles(candidate_priors: pl.DataFrame, profiles: dict[int, np.ndarray], freqs: np.ndarray, output_dir: Path) -> None:
    """Plot recording-level average candidate-clip spectral profiles."""
    for filename in sorted(candidate_priors["filename"].unique().to_list()):
        df = candidate_priors.filter(pl.col("filename") == filename)
        ids = [int(x) for x in df["candidate_id"].to_list()]
        stack = np.stack([profiles[i] for i in ids], axis=0)
        mean = stack.mean(axis=0)
        lo = np.percentile(stack, 20, axis=0)
        hi = np.percentile(stack, 80, axis=0)
        mode = detect_modes(mean, freqs, max_modes=6)
        fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
        ax.fill_between(freqs, lo, hi, color="tab:green", alpha=0.18, label="20-80% local candidate profiles")
        ax.plot(freqs, mean, color="tab:green", linewidth=2.0, label="recording mean candidate-clip profile")
        for idx, m in enumerate(mode["modes"]):
            color = "red" if m["touches_edge"] else "lime"
            ax.axvspan(m["support_fmin_hz"], m["support_fmax_hz"], color=color, alpha=0.14)
            ax.axvline(m["mode_hz"], color=color, linestyle="--", linewidth=1.4, label="mode" if idx == 0 else None)
            ax.text(m["mode_hz"], float(mean.max(initial=0.0)) * (0.92 - 0.08 * idx), f"{m['mode_hz']:.0f} Hz", rotation=90, va="top", ha="right", fontsize=8, color=color)
        ax.axvline(DISPLAY_BAND[0] + EDGE_MARGIN_HZ, color="gray", linestyle=":", linewidth=1.0)
        ax.axvline(DISPLAY_BAND[1] - EDGE_MARGIN_HZ, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlim(DISPLAY_BAND)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("normalized mean robust-excess-z")
        ax.set_title(f"Recording {Path(filename).stem}: candidate-clip spectral modes")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{Path(filename).stem}_recording_spectral_profile.png")
        plt.close(fig)


def plot_cluster_profiles(candidate_priors: pl.DataFrame, profiles: dict[int, np.ndarray], freqs: np.ndarray, output_dir: Path) -> None:
    """Plot cluster-average profiles for diagnostics only."""
    for cluster in sorted(candidate_priors["motif_cluster"].unique().to_list()):
        df = candidate_priors.filter(pl.col("motif_cluster") == cluster)
        ids = [int(x) for x in df["candidate_id"].to_list()]
        stack = np.stack([profiles[i] for i in ids], axis=0)
        mean = stack.mean(axis=0)
        lo = np.percentile(stack, 20, axis=0)
        hi = np.percentile(stack, 80, axis=0)
        mode = detect_modes(mean, freqs)
        fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
        ax.fill_between(freqs, lo, hi, color="tab:blue", alpha=0.18, label="20-80% candidate profiles")
        ax.plot(freqs, mean, color="tab:blue", linewidth=2.0, label="cluster mean spectral profile")
        for idx, m in enumerate(mode["modes"]):
            color = "red" if m["touches_edge"] else "lime"
            ax.axvspan(m["support_fmin_hz"], m["support_fmax_hz"], color=color, alpha=0.14)
            ax.axvline(m["mode_hz"], color=color, linestyle="--", linewidth=1.4, label="mode" if idx == 0 else None)
            ax.text(m["mode_hz"], float(mean.max(initial=0.0)) * (0.92 - 0.08 * idx), f"{m['mode_hz']:.0f} Hz", rotation=90, va="top", ha="right", fontsize=8, color=color)
        ax.axvline(DISPLAY_BAND[0] + EDGE_MARGIN_HZ, color="gray", linestyle=":", linewidth=1.0)
        ax.axvline(DISPLAY_BAND[1] - EDGE_MARGIN_HZ, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlim(DISPLAY_BAND)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("normalized mean robust-excess-z")
        ax.set_title(f"Cluster {cluster}: time-averaged spectral modes")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{cluster:02d}_spectral_profile.png")
        plt.close(fig)


def run(candidates_path: Path = DEFAULT_CANDIDATES, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = fresh_subdir(output_dir, "tables")
    plots_dir = fresh_subdir(output_dir, "plots")
    candidates = pl.read_csv(candidates_path)
    candidate_priors, profiles, freqs = compute_candidate_priors(candidates)
    recording_table, recording_candidate_scores = recording_priors(candidate_priors, profiles, freqs)
    candidate_priors = candidate_priors.join(recording_candidate_scores, on=["candidate_id", "filename"], how="left")
    cluster_table = cluster_priors(candidate_priors, profiles, freqs)
    candidate_priors.write_csv(tables_dir / "candidate_spectral_priors.csv")
    recording_table.write_csv(tables_dir / "recording_spectral_priors.csv")
    cluster_table.write_csv(tables_dir / "cluster_spectral_priors_diagnostic.csv")
    plot_recording_profiles(candidate_priors, profiles, freqs, plots_dir)
    plot_cluster_profiles(candidate_priors, profiles, freqs, plots_dir)
    (output_dir / "summary.md").write_text(
        f"""# Spectral-mode motif priors

Input candidates: `{candidates_path}`

Method: compute spectral-mode priors from actual evidence before using clusters:

1. Candidate-local prior: average robust-excess-z over each candidate's actual time clip.
2. Recording prior: average candidate-local profiles within each source recording to find
   recurring recording-level spectral modes.
3. Candidate score: local modal strength + alignment to recording-level modes, with
   edge-touch penalties.

Cluster-average profiles are emitted only as diagnostics; they are not the source
of truth for the prior.

Artifacts:

- `tables/candidate_spectral_priors.csv` — candidate-level local + recording-aware priors.
- `tables/recording_spectral_priors.csv` — recording-level spectral modes.
- `tables/cluster_spectral_priors_diagnostic.csv` — cluster summaries for diagnostics only.
- `plots/*_recording_spectral_profile.png` — recording-level mean candidate-clip profiles.
- `plots/cluster_XX_spectral_profile.png` — diagnostic cluster profiles.

Recording spectral priors:

```text
{recording_table.write_csv(separator='|').strip()}
```

Diagnostic cluster spectral priors:

```text
{cluster_table.write_csv(separator='|').strip()}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = run(args.candidates, args.output_dir)
    print(f"Wrote spectral-mode priors to {output_dir}")


if __name__ == "__main__":
    main()
