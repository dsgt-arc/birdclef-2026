"""Calibrate motif-bank template matching on focal Marbled Frog recordings.

This script scans the original `22967` focal train recordings with the simplified
motif bank, estimates per-family match thresholds from discovered members vs.
background, detects repeated events, summarizes inter-event intervals, and checks
recovery of the manually curated motif-box annotations.
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
from scipy.ndimage import zoom
from scipy.signal import find_peaks

from data_analysis.common.paths import TRAIN_AUDIO_DIR, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)
from data_analysis.marbled_frog.motif_mining import TARGET_LABEL, recording_files

DEFAULT_BANK_DIR = Path("artifacts/marbled_frog/motif_mining_simplified/motif_bank_approved")
DEFAULT_MINING_DIR = Path("artifacts/marbled_frog/motif_mining_simplified")
DEFAULT_CURATED = Path("artifacts/marbled_frog/motif_boxes/curated/tables/curated_motif_boxes.csv")
DEFAULT_OUTPUT = Path("artifacts/marbled_frog/motif_template_calibration")
TEMPLATE_SHAPE = (64, 48)


def load_bank(bank_dir: Path) -> tuple[pl.DataFrame, dict[int, dict]]:
    """Load motif-bank cluster metadata and template arrays."""
    clusters = pl.read_csv(bank_dir / "tables" / "motif_bank_clusters.csv")
    bank = {}
    for row in clusters.iter_rows(named=True):
        cluster = int(row["motif_cluster"])
        npz = np.load(bank_dir / "templates" / f"cluster_{cluster:02d}_templates.npz")
        bank[cluster] = {
            "templates": npz["templates"].astype(np.float32),
            "candidate_ids": [int(x) for x in npz["candidate_ids"]],
            "support": {
                "template_pre_sec": float(row["template_pre_sec"]),
                "template_post_sec": float(row["template_post_sec"]),
                "template_duration_sec": float(row["template_duration_sec"]),
                "template_fmin_hz": float(row["template_fmin_hz"]),
                "template_fmax_hz": float(row["template_fmax_hz"]),
                "template_bandwidth_hz": float(row["template_bandwidth_hz"]),
            },
        }
    return clusters, bank


def normalize_resize_patch(patch: np.ndarray, shape: tuple[int, int] = TEMPLATE_SHAPE) -> np.ndarray:
    """Normalize and resize a spectrogram patch to template shape."""
    if patch.size == 0:
        return np.zeros(shape, dtype=np.float32)
    patch = patch.astype(np.float32)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    resized = zoom(patch, (shape[0] / patch.shape[0], shape[1] / patch.shape[1]), order=1)
    out = np.zeros(shape, dtype=np.float32)
    h = min(shape[0], resized.shape[0])
    w = min(shape[1], resized.shape[1])
    out[:h, :w] = resized[:h, :w]
    return out.astype(np.float32)


def scan_cluster(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    templates: np.ndarray,
    support: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide one motif-family template set over a recording.

    Returns center times, best score over templates, and best template index.
    """
    band_mask = (freqs >= support["template_fmin_hz"]) & (freqs <= support["template_fmax_hz"])
    if not np.any(band_mask):
        return np.empty(0), np.empty(0), np.empty(0, dtype=int)
    z_band = z[band_mask]
    dt = float(np.median(np.diff(times))) if times.size > 1 else MEL_HOP_LENGTH / MEL_SAMPLE_RATE
    pre_frames = max(1, int(round(support["template_pre_sec"] / dt)))
    post_frames = max(1, int(round(support["template_post_sec"] / dt)))
    centers = np.arange(pre_frames, z_band.shape[1] - post_frames)
    if centers.size == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=int)
    template_vecs = templates.reshape(templates.shape[0], -1)
    scores = np.empty(centers.size, dtype=np.float32)
    best_template = np.empty(centers.size, dtype=np.int32)
    for out_idx, center in enumerate(centers):
        patch = z_band[:, center - pre_frames : center + post_frames + 1]
        patch_vec = normalize_resize_patch(patch).ravel()
        template_scores = template_vecs @ patch_vec / patch_vec.size
        best = int(np.argmax(template_scores))
        scores[out_idx] = float(template_scores[best])
        best_template[out_idx] = best
    return times[centers], scores, best_template


def estimate_thresholds(
    score_tracks: pl.DataFrame,
    candidates: pl.DataFrame,
    clusters: list[int],
    *,
    exclusion_sec: float = 0.25,
) -> pl.DataFrame:
    """Estimate per-cluster match thresholds from member peaks vs background."""
    rows = []
    for cluster in clusters:
        cluster_scores = score_tracks.filter(pl.col("motif_cluster") == cluster)
        cluster_members = candidates.filter(pl.col("motif_cluster") == cluster)
        positive_scores = []
        background_scores = []
        for filename in recording_files():
            scores_f = cluster_scores.filter(pl.col("filename") == filename).sort("time_sec")
            if scores_f.is_empty():
                continue
            times = scores_f["time_sec"].to_numpy()
            vals = scores_f["score"].to_numpy()
            member_times = cluster_members.filter(pl.col("filename") == filename)["peak_time_sec"].to_numpy()
            if member_times.size:
                for t in member_times:
                    idx = int(np.argmin(np.abs(times - float(t))))
                    positive_scores.append(float(vals[idx]))
                keep_bg = np.ones(times.size, dtype=bool)
                for t in member_times:
                    keep_bg &= np.abs(times - float(t)) > exclusion_sec
                background_scores.extend(vals[keep_bg].astype(float).tolist())
            else:
                background_scores.extend(vals.astype(float).tolist())
        pos = np.array(positive_scores, dtype=float)
        bg = np.array(background_scores, dtype=float)
        if pos.size:
            q25_pos = float(np.quantile(pos, 0.25))
            median_pos = float(np.median(pos))
        else:
            q25_pos = 0.55
            median_pos = float("nan")
        q995_bg = float(np.quantile(bg, 0.995)) if bg.size else 0.40
        # If fixed-support templates do not score their own discovery members
        # well, do not let a low positive quantile collapse the threshold. In
        # that case use the high background quantile and flag the family via the
        # threshold table for review.
        raw_threshold = min(q25_pos, q995_bg) if q25_pos >= 0.45 else q995_bg
        threshold = float(np.clip(raw_threshold, 0.45, 0.90))
        rows.append(
            {
                "motif_cluster": cluster,
                "threshold": threshold,
                "n_positive_members": int(pos.size),
                "median_positive_score": median_pos,
                "q25_positive_score": q25_pos,
                "n_background_samples": int(bg.size),
                "q995_background_score": q995_bg,
            }
        )
    return pl.DataFrame(rows)


def detect_events(score_tracks: pl.DataFrame, thresholds: pl.DataFrame, bank_clusters: pl.DataFrame) -> pl.DataFrame:
    """Peak-pick template match score tracks into candidate events."""
    threshold_by_cluster = {int(r["motif_cluster"]): float(r["threshold"]) for r in thresholds.iter_rows(named=True)}
    support_by_cluster = {int(r["motif_cluster"]): r for r in bank_clusters.iter_rows(named=True)}
    rows = []
    for cluster in sorted(threshold_by_cluster):
        support = support_by_cluster[cluster]
        min_spacing_sec = max(0.12, 0.60 * float(support["template_duration_sec"]))
        frame_dt = MEL_HOP_LENGTH / MEL_SAMPLE_RATE
        distance = max(1, int(round(min_spacing_sec / frame_dt)))
        for filename in recording_files():
            track = score_tracks.filter((pl.col("motif_cluster") == cluster) & (pl.col("filename") == filename)).sort("time_sec")
            if track.is_empty():
                continue
            scores = track["score"].to_numpy()
            times = track["time_sec"].to_numpy()
            best_template = track["best_template_index"].to_numpy()
            peaks, props = find_peaks(scores, height=threshold_by_cluster[cluster], distance=distance, prominence=0.02)
            for peak in peaks:
                peak_time = float(times[peak])
                rows.append(
                    {
                        "filename": filename,
                        "motif_cluster": cluster,
                        "peak_time_sec": peak_time,
                        "start_sec": max(0.0, peak_time - float(support["template_pre_sec"])),
                        "end_sec": peak_time + float(support["template_post_sec"]),
                        "fmin_hz": float(support["template_fmin_hz"]),
                        "fmax_hz": float(support["template_fmax_hz"]),
                        "score": float(scores[peak]),
                        "threshold": threshold_by_cluster[cluster],
                        "prominence": float(props["prominences"][np.where(peaks == peak)[0][0]]) if "prominences" in props else float("nan"),
                        "best_template_index": int(best_template[peak]),
                    }
                )
    return pl.DataFrame(rows)


def summarize_events(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize event rates and inter-event intervals."""
    if events.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    interval_rows = []
    for (filename, cluster), df in events.group_by("filename", "motif_cluster", maintain_order=True):
        df = df.sort("peak_time_sec")
        times = df["peak_time_sec"].to_numpy()
        intervals = np.diff(times)
        for i, interval in enumerate(intervals):
            interval_rows.append(
                {
                    "filename": filename,
                    "motif_cluster": int(cluster),
                    "event_a_time_sec": float(times[i]),
                    "event_b_time_sec": float(times[i + 1]),
                    "interval_sec": float(interval),
                }
            )
    intervals = pl.DataFrame(interval_rows)
    summary = events.group_by("motif_cluster").agg(
        pl.len().alias("n_events"),
        pl.col("filename").n_unique().alias("n_recordings"),
        pl.col("score").median().alias("median_score"),
        pl.col("score").max().alias("max_score"),
    )
    if not intervals.is_empty():
        interval_summary = intervals.group_by("motif_cluster").agg(
            pl.col("interval_sec").median().alias("median_inter_event_interval_sec"),
            pl.col("interval_sec").quantile(0.25).alias("q25_inter_event_interval_sec"),
            pl.col("interval_sec").quantile(0.75).alias("q75_inter_event_interval_sec"),
        )
        summary = summary.join(interval_summary, on="motif_cluster", how="left")
    return summary.sort("motif_cluster"), intervals


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return float(inter / union) if union > 0 else 0.0


def curated_recovery(events: pl.DataFrame, curated_path: Path) -> pl.DataFrame:
    """Compare motif-bank detections against curated motif-box annotations."""
    if not curated_path.exists() or events.is_empty():
        return pl.DataFrame()
    curated = pl.read_csv(curated_path)
    rows = []
    for idx, cur in enumerate(curated.iter_rows(named=True), start=1):
        ev = events.filter(pl.col("filename") == cur["filename"])
        best = None
        for det in ev.iter_rows(named=True):
            time_iou = interval_iou(cur["start_sec"], cur["end_sec"], det["start_sec"], det["end_sec"])
            freq_iou = interval_iou(cur["fmin_hz"], cur["fmax_hz"], det["fmin_hz"], det["fmax_hz"])
            peak_inside = cur["start_sec"] <= det["peak_time_sec"] <= cur["end_sec"]
            score = max(time_iou, 1.0 if peak_inside else 0.0) * max(freq_iou, 0.05)
            if best is None or score > best[0]:
                best = (score, time_iou, freq_iou, peak_inside, det)
        if best is None:
            rows.append({"curated_row_id": idx, **cur, "recovered": False})
        else:
            _score, time_iou, freq_iou, peak_inside, det = best
            recovered = bool((peak_inside or time_iou >= 0.20) and freq_iou >= 0.10)
            rows.append(
                {
                    "curated_row_id": idx,
                    **cur,
                    "recovered": recovered,
                    "best_motif_cluster": det["motif_cluster"],
                    "best_detection_peak_time_sec": det["peak_time_sec"],
                    "best_detection_score": det["score"],
                    "best_time_iou": time_iou,
                    "best_freq_iou": freq_iou,
                    "best_peak_inside_curated": peak_inside,
                }
            )
    return pl.DataFrame(rows)


def plot_curated_recovery(events: pl.DataFrame, curated_path: Path, output_dir: Path) -> None:
    """Plot curated annotation windows with matched motif-bank detections overlaid."""
    if not curated_path.exists() or events.is_empty():
        return
    curated = pl.read_csv(curated_path)
    groups = curated.group_by("filename", "window_start_sec", "window_end_sec", maintain_order=True)
    for keys, df in groups:
        filename, window_start, window_end = keys
        window_start = float(window_start)
        window_end = float(window_end)
        audio, _ = librosa.load(
            TRAIN_AUDIO_DIR / filename,
            sr=MEL_SAMPLE_RATE,
            mono=True,
            offset=window_start,
            duration=window_end - window_start,
        )
        db_band, _z, freqs, times = robust_excess_spectrogram(audio)
        display_times = times + window_start
        fig, ax = plt.subplots(figsize=(13.0, 6.0), dpi=180)
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
        for row in df.iter_rows(named=True):
            ax.add_patch(
                Rectangle(
                    (row["start_sec"], row["fmin_hz"]),
                    row["end_sec"] - row["start_sec"],
                    row["fmax_hz"] - row["fmin_hz"],
                    fill=False,
                    edgecolor="lime",
                    linestyle="--",
                    linewidth=1.8,
                )
            )
        dets = events.filter(
            (pl.col("filename") == filename)
            & (pl.col("peak_time_sec") >= window_start)
            & (pl.col("peak_time_sec") <= window_end)
        ).sort("peak_time_sec")
        cmap = plt.cm.tab10
        for det in dets.iter_rows(named=True):
            color = cmap(int(det["motif_cluster"]) % 10)
            ax.add_patch(
                Rectangle(
                    (det["start_sec"], det["fmin_hz"]),
                    det["end_sec"] - det["start_sec"],
                    det["fmax_hz"] - det["fmin_hz"],
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                    alpha=0.9,
                )
            )
            ax.text(
                det["peak_time_sec"],
                det["fmax_hz"] + 35,
                f"c{det['motif_cluster']} {det['score']:.2f}",
                color=color,
                fontsize=7,
                weight="bold",
            )
        ax.set_ylim(DISPLAY_BAND)
        ax.set_xlim(window_start, window_end)
        ax.set_title(
            f"Curated motif recovery: {Path(filename).stem} {window_start:.1f}-{window_end:.1f}s\n"
            "lime dashed=curated boxes; colored solid=motif-bank template detections"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{Path(filename).stem}_{window_start:.1f}_{window_end:.1f}_curated_recovery.png")
        plt.close(fig)


def plot_interval_histograms(intervals: pl.DataFrame, output_path: Path) -> None:
    if intervals.is_empty():
        return
    clusters = sorted(intervals["motif_cluster"].unique().to_list())
    n_cols = min(4, len(clusters))
    n_rows = int(np.ceil(len(clusters) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), dpi=180, squeeze=False)
    for idx, cluster in enumerate(clusters):
        r, c = divmod(idx, n_cols)
        vals = intervals.filter(pl.col("motif_cluster") == cluster)["interval_sec"].to_numpy()
        axes[r, c].hist(vals, bins=20, color="tab:blue", alpha=0.8)
        axes[r, c].set_title(f"cluster {cluster} intervals")
        axes[r, c].set_xlabel("seconds")
        axes[r, c].set_ylabel("count")
        axes[r, c].grid(alpha=0.2)
    for idx in range(len(clusters), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(
    bank_dir: Path = DEFAULT_BANK_DIR,
    mining_dir: Path = DEFAULT_MINING_DIR,
    curated_path: Path = DEFAULT_CURATED,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = fresh_subdir(output_dir, "tables")
    plots_dir = fresh_subdir(output_dir, "plots")

    bank_clusters, bank = load_bank(bank_dir)
    candidates = pl.read_csv(mining_dir / "tables" / "candidates.csv")
    score_rows = []
    for filename in recording_files():
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
        _db, z, freqs, times = robust_excess_spectrogram(audio)
        for cluster, item in bank.items():
            center_times, scores, best_template = scan_cluster(
                z,
                freqs,
                times,
                templates=item["templates"],
                support=item["support"],
            )
            for t, score, template_idx in zip(center_times, scores, best_template, strict=True):
                score_rows.append(
                    {
                        "filename": filename,
                        "motif_cluster": cluster,
                        "time_sec": float(t),
                        "score": float(score),
                        "best_template_index": int(template_idx),
                    }
                )
    score_tracks = pl.DataFrame(score_rows)
    thresholds = estimate_thresholds(score_tracks, candidates, sorted(bank.keys()))
    events = detect_events(score_tracks, thresholds, bank_clusters)
    event_summary, intervals = summarize_events(events)
    recovery = curated_recovery(events, curated_path)

    score_tracks.write_csv(tables_dir / "template_score_tracks.csv")
    thresholds.write_csv(tables_dir / "template_thresholds.csv")
    events.write_csv(tables_dir / "template_match_events.csv")
    event_summary.write_csv(tables_dir / "event_summary.csv")
    intervals.write_csv(tables_dir / "inter_event_intervals.csv")
    if not recovery.is_empty():
        recovery.write_csv(tables_dir / "curated_recovery.csv")
        recovery_summary = recovery.group_by("component").agg(
            pl.len().alias("n_curated_boxes"),
            pl.col("recovered").sum().alias("n_recovered"),
            pl.col("recovered").mean().alias("recovery_rate"),
            pl.col("best_detection_score").median().alias("median_best_detection_score"),
        ).sort("component")
        recovery_summary.write_csv(tables_dir / "curated_recovery_summary.csv")
    else:
        recovery_summary = pl.DataFrame()

    plot_curated_recovery(events, curated_path, plots_dir)
    plot_interval_histograms(intervals, plots_dir / "inter_event_interval_histograms.png")

    (output_dir / "summary.md").write_text(
        f"""# Motif-bank template calibration on focal recordings

Motif bank: `{bank_dir}`

Input recordings: `{TARGET_LABEL}` focal train audio ({len(recording_files())} recordings)

Method:

1. Scan each focal recording with each approved motif-family template set.
2. Estimate a simple per-family threshold from discovered member scores vs. background score tracks.
3. Peak-pick detections and summarize event counts / inter-event intervals.
4. Compare detections to curated motif-box annotations.

Artifacts:

- `tables/template_score_tracks.csv` — dense score tracks per recording/family.
- `tables/template_thresholds.csv` — per-family calibration thresholds.
- `tables/template_match_events.csv` — peak-picked motif detections.
- `tables/event_summary.csv` — event counts and interval summaries by family.
- `tables/inter_event_intervals.csv` — consecutive-event intervals per recording/family.
- `tables/curated_recovery.csv` — best detection for each curated motif box.
- `tables/curated_recovery_summary.csv` — recovery by curated component.
- `plots/*_curated_recovery.png` — curated windows with detections overlaid.
- `plots/inter_event_interval_histograms.png` — interval distributions.

Event summary:

```text
{event_summary.write_csv(separator='|').strip() if not event_summary.is_empty() else 'none'}
```

Curated recovery summary:

```text
{recovery_summary.write_csv(separator='|').strip() if not recovery_summary.is_empty() else 'none'}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, default=DEFAULT_BANK_DIR)
    parser.add_argument("--mining-dir", type=Path, default=DEFAULT_MINING_DIR)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = run(args.bank_dir, args.mining_dir, args.curated, args.output_dir)
    print(f"Wrote motif template calibration to {output_dir}")


if __name__ == "__main__":
    main()
