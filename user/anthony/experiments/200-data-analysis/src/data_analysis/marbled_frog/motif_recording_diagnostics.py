"""Recording-level motif diagnostics for Marbled Frog focal clips.

This is a sanity check for low-SNR or overlapping recordings where template-bank
matching may miss visible repeated structure. It peak-picks a simple band envelope
from the actual recording and compares those envelope peaks with consolidated
motif-template events from calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from data_analysis.common.paths import TRAIN_AUDIO_DIR, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)
from data_analysis.marbled_frog.motif_mining import recording_files

DEFAULT_EVENTS = Path("artifacts/marbled_frog/motif_template_calibration/tables/consolidated_events.csv")
DEFAULT_OUTPUT = Path("artifacts/marbled_frog/motif_recording_diagnostics")
DEFAULT_BAND = (850.0, 1600.0)
DEFAULT_ENVELOPE_PRE_SEC = 0.12
DEFAULT_ENVELOPE_POST_SEC = 0.16


def band_envelope_peaks(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
    smooth_sigma_frames: float = 2.0,
    height_mad: float = 0.75,
    prominence_mad: float = 0.5,
    min_spacing_sec: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Peak-pick a robust-z band envelope from the actual recording."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.empty(0), np.zeros(times.size, dtype=np.float32), {}
    envelope = z[mask].mean(axis=0).astype(np.float32)
    envelope = gaussian_filter1d(envelope, sigma=smooth_sigma_frames)
    med = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - med)))
    robust_sigma = 1.4826 * max(mad, 1e-6)
    threshold = med + height_mad * robust_sigma
    prominence = prominence_mad * robust_sigma
    dt = float(np.median(np.diff(times))) if times.size > 1 else MEL_HOP_LENGTH / MEL_SAMPLE_RATE
    distance = max(1, int(round(min_spacing_sec / dt)))
    peaks, _props = find_peaks(envelope, height=threshold, prominence=prominence, distance=distance)
    params = {
        "band_fmin_hz": band[0],
        "band_fmax_hz": band[1],
        "envelope_median": med,
        "envelope_mad_sigma": robust_sigma,
        "peak_height_threshold": threshold,
        "peak_prominence_threshold": prominence,
        "min_spacing_sec": min_spacing_sec,
    }
    return peaks.astype(int), envelope, params


def envelope_events_from_peaks(
    filename: str,
    peak_times: np.ndarray,
    envelope: np.ndarray,
    times: np.ndarray,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
    pre_sec: float = DEFAULT_ENVELOPE_PRE_SEC,
    post_sec: float = DEFAULT_ENVELOPE_POST_SEC,
) -> pl.DataFrame:
    """Convert band-envelope peaks into fixed-support high-recall motif events."""
    rows = []
    for idx, peak_time in enumerate(peak_times, start=1):
        nearest_idx = int(np.argmin(np.abs(times - float(peak_time))))
        rows.append(
            {
                "filename": filename,
                "envelope_event_id": idx,
                "detector": "band_envelope",
                "peak_time_sec": float(peak_time),
                "start_sec": max(0.0, float(peak_time) - pre_sec),
                "end_sec": float(peak_time) + post_sec,
                "fmin_hz": float(band[0]),
                "fmax_hz": float(band[1]),
                "score": float(envelope[nearest_idx]),
                "support_pre_sec": pre_sec,
                "support_post_sec": post_sec,
            }
        )
    return pl.DataFrame(rows)


def compare_peaks_to_events(
    filename: str,
    peak_times: np.ndarray,
    events: pl.DataFrame,
    *,
    match_tolerance_sec: float = 0.25,
) -> tuple[list[dict], list[dict], dict[str, float]]:
    """Compare envelope peaks to consolidated template events."""
    event_rows = events.filter(pl.col("filename") == filename).sort("peak_time_sec").to_dicts()
    peak_rows = []
    for peak_time in peak_times:
        if event_rows:
            distances = np.array([abs(float(ev["peak_time_sec"]) - float(peak_time)) for ev in event_rows])
            best_idx = int(np.argmin(distances))
            best_event = event_rows[best_idx]
            best_distance = float(distances[best_idx])
            matched = best_distance <= match_tolerance_sec
        else:
            best_event = None
            best_distance = float("nan")
            matched = False
        peak_rows.append(
            {
                "filename": filename,
                "envelope_peak_time_sec": float(peak_time),
                "matched_template_event": matched,
                "nearest_event_distance_sec": best_distance,
                "nearest_event_id": None if best_event is None else int(best_event["event_id"]),
                "nearest_event_cluster": None if best_event is None else int(best_event["best_motif_cluster"]),
                "nearest_event_score": None if best_event is None else float(best_event["score"]),
            }
        )

    event_match_rows = []
    for ev in event_rows:
        if peak_times.size:
            distances = np.abs(peak_times.astype(float) - float(ev["peak_time_sec"]))
            best_idx = int(np.argmin(distances))
            best_distance = float(distances[best_idx])
            matched = best_distance <= match_tolerance_sec
            nearest_peak_time = float(peak_times[best_idx])
        else:
            best_distance = float("nan")
            matched = False
            nearest_peak_time = None
        event_match_rows.append(
            {
                "filename": filename,
                "event_id": int(ev["event_id"]),
                "event_peak_time_sec": float(ev["peak_time_sec"]),
                "best_motif_cluster": int(ev["best_motif_cluster"]),
                "score": float(ev["score"]),
                "matched_envelope_peak": matched,
                "nearest_envelope_peak_time_sec": nearest_peak_time,
                "nearest_peak_distance_sec": best_distance,
            }
        )

    n_peaks = len(peak_rows)
    n_events = len(event_match_rows)
    n_peak_matched = sum(1 for r in peak_rows if r["matched_template_event"])
    n_event_matched = sum(1 for r in event_match_rows if r["matched_envelope_peak"])
    metrics = {
        "filename": filename,
        "n_envelope_peaks": n_peaks,
        "n_template_events": n_events,
        "n_envelope_peaks_matched": n_peak_matched,
        "n_template_events_matched": n_event_matched,
        "envelope_peak_recall_by_template_events": n_peak_matched / n_peaks if n_peaks else float("nan"),
        "template_event_precision_vs_envelope": n_event_matched / n_events if n_events else float("nan"),
        "median_envelope_peak_interval_sec": float(np.median(np.diff(peak_times))) if peak_times.size > 1 else float("nan"),
    }
    return peak_rows, event_match_rows, metrics


def plot_recording_diagnostic(
    filename: str,
    db_band: np.ndarray,
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    envelope: np.ndarray,
    peak_rows: list[dict],
    events: pl.DataFrame,
    envelope_events: pl.DataFrame,
    output_path: Path,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
) -> None:
    """Plot spectrogram, consolidated template events, and band-envelope peaks."""
    cmap = plt.cm.tab10
    width = max(12.0, min(24.0, 0.42 * float(times[-1]) + 6.0))
    fig, (ax_spec, ax_env) = plt.subplots(
        2,
        1,
        figsize=(width, 8.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    librosa.display.specshow(
        db_band,
        x_coords=times,
        y_coords=freqs,
        x_axis="time",
        y_axis="linear",
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        ax=ax_spec,
        cmap="magma",
    )
    ax_spec.axhspan(band[0], band[1], color="white", alpha=0.08, label=f"diagnostic band {band[0]:.0f}-{band[1]:.0f} Hz")
    env_df = envelope_events.filter(pl.col("filename") == filename).sort("peak_time_sec")
    for row in env_df.iter_rows(named=True):
        matched = any(abs(float(row["peak_time_sec"]) - float(p["envelope_peak_time_sec"])) < 1e-6 and p["matched_template_event"] for p in peak_rows)
        ax_spec.add_patch(
            Rectangle(
                (row["start_sec"], row["fmin_hz"]),
                row["end_sec"] - row["start_sec"],
                row["fmax_hz"] - row["fmin_hz"],
                fill=False,
                edgecolor="cyan" if matched else "red",
                linewidth=1.0,
                linestyle=":" if matched else "-",
                alpha=0.85,
            )
        )
    df = events.filter(pl.col("filename") == filename).sort("peak_time_sec")
    for row in df.iter_rows(named=True):
        cluster = int(row["best_motif_cluster"])
        color = cmap(cluster % 10)
        ax_spec.add_patch(
            Rectangle(
                (row["start_sec"], row["fmin_hz"]),
                row["end_sec"] - row["start_sec"],
                row["fmax_hz"] - row["fmin_hz"],
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                alpha=0.95,
            )
        )
    for peak in peak_rows:
        color = "cyan" if peak["matched_template_event"] else "red"
        ax_spec.axvline(peak["envelope_peak_time_sec"], color=color, linewidth=0.8, alpha=0.65)
    ax_spec.set_ylim(DISPLAY_BAND)
    ax_spec.set_title(
        f"Recording diagnostic: {Path(filename).stem}\n"
        "cyan/red boxes=band-envelope motif events (matched/unmatched to template); colored boxes=template events"
    )
    ax_spec.legend(loc="upper right", fontsize=8)

    env = envelope.astype(float)
    env_norm = (env - np.nanmin(env)) / (np.nanmax(env) - np.nanmin(env) + 1e-9)
    ax_env.plot(times, env_norm, color="black", linewidth=1.1)
    for peak in peak_rows:
        color = "cyan" if peak["matched_template_event"] else "red"
        ax_env.axvline(peak["envelope_peak_time_sec"], color=color, linewidth=0.8, alpha=0.7)
    for row in df.iter_rows(named=True):
        ax_env.axvline(row["peak_time_sec"], color=cmap(int(row["best_motif_cluster"]) % 10), linewidth=1.0, alpha=0.65)
    ax_env.set_ylim(-0.02, 1.05)
    ax_env.set_ylabel("norm envelope")
    ax_env.set_xlabel("Time (s)")
    ax_env.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(
    events_path: Path = DEFAULT_EVENTS,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    band: tuple[float, float] = DEFAULT_BAND,
    only_filename: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = fresh_subdir(output_dir, "tables")
    plots_dir = fresh_subdir(output_dir, "plots")
    events = pl.read_csv(events_path)
    files = recording_files()
    if only_filename is not None:
        files = [only_filename]

    peak_rows_all = []
    event_rows_all = []
    envelope_event_tables = []
    metrics_rows = []
    params_rows = []
    for filename in files:
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
        db_band, z, freqs, times = robust_excess_spectrogram(audio)
        peak_idx, envelope, params = band_envelope_peaks(z, freqs, times, band=band)
        peak_times = times[peak_idx]
        envelope_events = envelope_events_from_peaks(filename, peak_times, envelope, times, band=band)
        peak_rows, event_rows, metrics = compare_peaks_to_events(filename, peak_times, events)
        if not envelope_events.is_empty():
            # Attach template-match agreement fields to the promoted envelope events.
            agreement = pl.DataFrame(peak_rows).select(
                pl.col("envelope_peak_time_sec").alias("peak_time_sec"),
                "matched_template_event",
                "nearest_event_distance_sec",
                "nearest_event_id",
                "nearest_event_cluster",
                "nearest_event_score",
            )
            envelope_events = envelope_events.join(agreement, on="peak_time_sec", how="left")
            envelope_event_tables.append(envelope_events)
        peak_rows_all.extend(peak_rows)
        event_rows_all.extend(event_rows)
        metrics_rows.append(metrics)
        params_rows.append({"filename": filename, **params})
        plot_recording_diagnostic(
            filename,
            db_band,
            z,
            freqs,
            times,
            envelope,
            peak_rows,
            events,
            envelope_events,
            plots_dir / f"{Path(filename).stem}_band_envelope_vs_template_events.png",
            band=band,
        )

    peaks_df = pl.DataFrame(peak_rows_all)
    event_df = pl.DataFrame(event_rows_all)
    envelope_events_df = pl.concat(envelope_event_tables, how="diagonal_relaxed") if envelope_event_tables else pl.DataFrame()
    metrics_df = pl.DataFrame(metrics_rows)
    params_df = pl.DataFrame(params_rows)
    peaks_df.write_csv(tables_dir / "band_envelope_peaks.csv")
    envelope_events_df.write_csv(tables_dir / "band_envelope_events.csv")
    event_df.write_csv(tables_dir / "template_events_vs_envelope_peaks.csv")
    metrics_df.write_csv(tables_dir / "recording_diagnostic_summary.csv")
    if not envelope_events_df.is_empty():
        envelope_summary = envelope_events_df.group_by("filename").agg(
            pl.len().alias("n_band_envelope_events"),
            pl.col("matched_template_event").sum().alias("n_matched_template_events"),
            pl.col("matched_template_event").mean().alias("template_match_rate"),
            pl.col("peak_time_sec").diff().median().alias("median_interval_sec"),
        ).sort("filename")
        envelope_summary.write_csv(tables_dir / "band_envelope_event_summary.csv")
    else:
        envelope_summary = pl.DataFrame()
    params_df.write_csv(tables_dir / "band_envelope_parameters.csv")
    (output_dir / "summary.md").write_text(
        f"""# Motif recording diagnostics

Compared consolidated motif-template events against simple band-envelope peaks.

Events: `{events_path}`

Diagnostic band: {band[0]:.0f}-{band[1]:.0f} Hz

Artifacts:

- `tables/band_envelope_peaks.csv` — visible/repeated envelope peaks and nearest template event.
- `tables/band_envelope_events.csv` — promoted high-recall motif event boxes from envelope peaks.
- `tables/band_envelope_event_summary.csv` — per-recording counts/rates for envelope events.
- `tables/template_events_vs_envelope_peaks.csv` — template events and nearest envelope peak.
- `tables/recording_diagnostic_summary.csv` — per-recording peak/event agreement.
- `tables/band_envelope_parameters.csv` — peak-picking thresholds.
- `plots/*_band_envelope_vs_template_events.png` — full recording diagnostic plots.

Summary:

```text
{metrics_df.write_csv(separator='|').strip()}
```

Band-envelope event summary:

```text
{envelope_summary.write_csv(separator='|').strip() if not envelope_summary.is_empty() else 'none'}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--band-fmin", type=float, default=DEFAULT_BAND[0])
    parser.add_argument("--band-fmax", type=float, default=DEFAULT_BAND[1])
    parser.add_argument("--filename", default=None, help="Optional single filename, e.g. 22967/iNat56387.ogg")
    args = parser.parse_args()
    output_dir = run(
        args.events,
        args.output_dir,
        band=(args.band_fmin, args.band_fmax),
        only_filename=args.filename,
    )
    print(f"Wrote motif recording diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
