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


def nms_events(
    events: pl.DataFrame,
    *,
    time_iou_threshold: float = 0.35,
    peak_distance_sec: float = 0.12,
) -> pl.DataFrame:
    """Suppress duplicate detections within each recording/family.

    Peak-picking already enforces a minimum distance, but template score tracks can
    still produce duplicate boxes around the same event. Keep the highest-scoring
    detection and suppress lower-scoring boxes that overlap substantially or have
    nearly identical peak times.
    """
    if events.is_empty():
        return events
    kept_rows = []
    for (_filename, _cluster), df in events.group_by("filename", "motif_cluster", maintain_order=True):
        rows = sorted(df.to_dicts(), key=lambda r: float(r["score"]), reverse=True)
        kept: list[dict] = []
        for row in rows:
            duplicate = False
            for prev in kept:
                tiou = interval_iou(row["start_sec"], row["end_sec"], prev["start_sec"], prev["end_sec"])
                close_peak = abs(row["peak_time_sec"] - prev["peak_time_sec"]) <= peak_distance_sec
                if tiou >= time_iou_threshold or close_peak:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(row)
        kept_rows.extend(sorted(kept, key=lambda r: (r["filename"], r["motif_cluster"], r["peak_time_sec"])))
    return pl.DataFrame(kept_rows).sort("filename", "motif_cluster", "peak_time_sec")


def consolidate_events(
    events: pl.DataFrame,
    *,
    merge_peak_distance_sec: float = 0.25,
    min_time_iou: float = 0.05,
) -> pl.DataFrame:
    """Merge cross-family component detections into one biological-event box."""
    if events.is_empty():
        return pl.DataFrame()
    rows = []
    event_id = 0
    for filename_key, df in events.group_by("filename", maintain_order=True):
        filename = filename_key[0] if isinstance(filename_key, tuple) else filename_key
        detections = sorted(df.to_dicts(), key=lambda r: float(r["peak_time_sec"]))
        groups: list[list[dict]] = []
        for det in detections:
            if not groups:
                groups.append([det])
                continue
            group = groups[-1]
            g_start = min(float(x["start_sec"]) for x in group)
            g_end = max(float(x["end_sec"]) for x in group)
            g_peak_max = max(float(x["peak_time_sec"]) for x in group)
            close_peak = float(det["peak_time_sec"]) - g_peak_max <= merge_peak_distance_sec
            overlaps_group = interval_iou(g_start, g_end, float(det["start_sec"]), float(det["end_sec"])) >= min_time_iou
            if close_peak or overlaps_group:
                group.append(det)
            else:
                groups.append([det])
        for group in groups:
            event_id += 1
            best = max(group, key=lambda r: float(r["score"]))
            clusters = sorted({int(x["motif_cluster"]) for x in group})
            rows.append(
                {
                    "event_id": event_id,
                    "filename": filename,
                    "start_sec": min(float(x["start_sec"]) for x in group),
                    "end_sec": max(float(x["end_sec"]) for x in group),
                    "peak_time_sec": float(best["peak_time_sec"]),
                    "fmin_hz": min(float(x["fmin_hz"]) for x in group),
                    "fmax_hz": max(float(x["fmax_hz"]) for x in group),
                    "score": float(best["score"]),
                    "best_score": float(best["score"]),
                    "motif_cluster": int(best["motif_cluster"]),
                    "best_motif_cluster": int(best["motif_cluster"]),
                    "motif_clusters_present": ";".join(str(x) for x in clusters),
                    "n_component_detections": len(group),
                    "mean_component_score": float(np.mean([float(x["score"]) for x in group])),
                }
            )
    return pl.DataFrame(rows).sort("filename", "peak_time_sec")


def summarize_events(events: pl.DataFrame, *, cluster_col: str = "motif_cluster") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize event rates and inter-event intervals."""
    if events.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    interval_rows = []
    for (filename, cluster), df in events.group_by("filename", cluster_col, maintain_order=True):
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
    summary = events.group_by(cluster_col).agg(
        pl.len().alias("n_events"),
        pl.col("filename").n_unique().alias("n_recordings"),
        pl.col("score").median().alias("median_score"),
        pl.col("score").max().alias("max_score"),
    )
    if cluster_col != "motif_cluster":
        summary = summary.rename({cluster_col: "motif_cluster"})
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


def detection_stage_metrics(
    raw_events: pl.DataFrame,
    nms_events_df: pl.DataFrame,
    consolidated: pl.DataFrame,
    raw_recovery: pl.DataFrame,
    nms_recovery: pl.DataFrame,
    consolidated_recovery: pl.DataFrame,
) -> pl.DataFrame:
    """Quantify whether NMS/consolidation reduces duplicates while preserving recovery."""
    def recovery_rate(df: pl.DataFrame) -> float:
        return float(df["recovered"].mean()) if not df.is_empty() and "recovered" in df.columns else float("nan")

    def median_components(df: pl.DataFrame) -> float:
        return float(df["n_component_detections"].median()) if not df.is_empty() and "n_component_detections" in df.columns else float("nan")

    raw_n = raw_events.height
    nms_n = nms_events_df.height
    cons_n = consolidated.height
    return pl.DataFrame(
        [
            {
                "stage": "raw_template_peaks",
                "n_boxes_or_events": raw_n,
                "reduction_vs_raw": 0.0,
                "curated_recovery_rate": recovery_rate(raw_recovery),
                "median_components_per_event": float("nan"),
            },
            {
                "stage": "within_family_nms",
                "n_boxes_or_events": nms_n,
                "reduction_vs_raw": 1.0 - nms_n / raw_n if raw_n else float("nan"),
                "curated_recovery_rate": recovery_rate(nms_recovery),
                "median_components_per_event": float("nan"),
            },
            {
                "stage": "cross_family_consolidated_events",
                "n_boxes_or_events": cons_n,
                "reduction_vs_raw": 1.0 - cons_n / raw_n if raw_n else float("nan"),
                "curated_recovery_rate": recovery_rate(consolidated_recovery),
                "median_components_per_event": median_components(consolidated),
            },
        ]
    )


def plot_curated_recovery(events: pl.DataFrame, curated_path: Path, output_dir: Path, *, suffix: str = "curated_recovery") -> None:
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
        fig.savefig(output_dir / f"{Path(filename).stem}_{window_start:.1f}_{window_end:.1f}_{suffix}.png")
        plt.close(fig)


def plot_full_recording_events(events: pl.DataFrame, output_dir: Path) -> None:
    """Plot each full focal recording with consolidated motif events overlaid."""
    if events.is_empty():
        return
    cmap = plt.cm.tab10
    for filename in recording_files():
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
        db_band, _z, freqs, times = robust_excess_spectrogram(audio)
        width = max(12.0, min(24.0, 0.42 * float(times[-1]) + 6.0))
        fig, ax = plt.subplots(figsize=(width, 6.2), dpi=180)
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
        df = events.filter(pl.col("filename") == filename).sort("peak_time_sec")
        for row in df.iter_rows(named=True):
            cluster = int(row["best_motif_cluster"] if "best_motif_cluster" in row else row["motif_cluster"])
            color = cmap(cluster % 10)
            ax.add_patch(
                Rectangle(
                    (row["start_sec"], row["fmin_hz"]),
                    row["end_sec"] - row["start_sec"],
                    row["fmax_hz"] - row["fmin_hz"],
                    fill=False,
                    edgecolor=color,
                    linewidth=1.6,
                    alpha=0.95,
                )
            )
            ax.text(
                row["peak_time_sec"],
                min(DISPLAY_BAND[1] - 45, row["fmax_hz"] + 35),
                f"c{cluster}\n{row['score']:.2f}",
                color=color,
                fontsize=6.5,
                ha="center",
                va="bottom",
                weight="bold",
            )
        ax.set_ylim(DISPLAY_BAND)
        ax.set_xlim(0.0, float(times[-1]))
        ax.set_title(
            f"Full focal recording motif-template detections: {Path(filename).stem}\n"
            f"consolidated events: n={df.height}; colored boxes = event boxes, label = best cluster / score"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{Path(filename).stem}_full_recording_events.png")
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
    raw_events = detect_events(score_tracks, thresholds, bank_clusters)
    nms_events_df = nms_events(raw_events)
    consolidated = consolidate_events(nms_events_df)

    raw_event_summary, raw_intervals = summarize_events(raw_events)
    nms_event_summary, nms_intervals = summarize_events(nms_events_df)
    consolidated_summary, consolidated_intervals = summarize_events(consolidated, cluster_col="best_motif_cluster")

    raw_recovery = curated_recovery(raw_events, curated_path)
    nms_recovery = curated_recovery(nms_events_df, curated_path)
    consolidated_recovery = curated_recovery(consolidated, curated_path)
    stage_metrics = detection_stage_metrics(
        raw_events,
        nms_events_df,
        consolidated,
        raw_recovery,
        nms_recovery,
        consolidated_recovery,
    )

    score_tracks.write_csv(tables_dir / "template_score_tracks.csv")
    thresholds.write_csv(tables_dir / "template_thresholds.csv")
    raw_events.write_csv(tables_dir / "template_match_events_raw.csv")
    nms_events_df.write_csv(tables_dir / "template_match_events_nms.csv")
    consolidated.write_csv(tables_dir / "consolidated_events.csv")
    raw_event_summary.write_csv(tables_dir / "event_summary_raw.csv")
    nms_event_summary.write_csv(tables_dir / "event_summary_nms.csv")
    consolidated_summary.write_csv(tables_dir / "event_summary_consolidated.csv")
    raw_intervals.write_csv(tables_dir / "inter_event_intervals_raw.csv")
    nms_intervals.write_csv(tables_dir / "inter_event_intervals_nms.csv")
    consolidated_intervals.write_csv(tables_dir / "inter_event_intervals_consolidated.csv")
    stage_metrics.write_csv(tables_dir / "detection_stage_metrics.csv")

    def write_recovery(prefix: str, recovery: pl.DataFrame) -> pl.DataFrame:
        if recovery.is_empty():
            return pl.DataFrame()
        recovery.write_csv(tables_dir / f"curated_recovery_{prefix}.csv")
        summary = recovery.group_by("component").agg(
            pl.len().alias("n_curated_boxes"),
            pl.col("recovered").sum().alias("n_recovered"),
            pl.col("recovered").mean().alias("recovery_rate"),
            pl.col("best_detection_score").median().alias("median_best_detection_score"),
        ).sort("component")
        summary.write_csv(tables_dir / f"curated_recovery_summary_{prefix}.csv")
        return summary

    raw_recovery_summary = write_recovery("raw", raw_recovery)
    nms_recovery_summary = write_recovery("nms", nms_recovery)
    consolidated_recovery_summary = write_recovery("consolidated", consolidated_recovery)

    # Backward-compatible aliases for the recommended consolidated view.
    consolidated.write_csv(tables_dir / "template_match_events.csv")
    consolidated_summary.write_csv(tables_dir / "event_summary.csv")
    consolidated_intervals.write_csv(tables_dir / "inter_event_intervals.csv")
    if not consolidated_recovery.is_empty():
        consolidated_recovery.write_csv(tables_dir / "curated_recovery.csv")
        consolidated_recovery_summary.write_csv(tables_dir / "curated_recovery_summary.csv")

    plot_curated_recovery(raw_events, curated_path, plots_dir, suffix="curated_recovery_raw")
    plot_curated_recovery(consolidated, curated_path, plots_dir, suffix="curated_recovery_consolidated")
    plot_full_recording_events(consolidated, plots_dir)
    plot_interval_histograms(consolidated_intervals, plots_dir / "inter_event_interval_histograms.png")

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
- `tables/template_match_events_raw.csv` — raw peak-picked motif detections.
- `tables/template_match_events_nms.csv` — within-family NMS detections.
- `tables/consolidated_events.csv` — cross-family merged biological-event boxes.
- `tables/detection_stage_metrics.csv` — duplicate reduction vs. curated recovery.
- `tables/event_summary.csv` — consolidated event counts and interval summaries by family.
- `tables/inter_event_intervals.csv` — consecutive consolidated-event intervals.
- `tables/curated_recovery.csv` — best consolidated event for each curated motif box.
- `tables/curated_recovery_summary.csv` — consolidated recovery by curated component.
- `plots/*_curated_recovery_raw.png` — curated windows with raw detections.
- `plots/*_curated_recovery_consolidated.png` — curated windows with consolidated event boxes.
- `plots/*_full_recording_events.png` — full focal clips with consolidated event boxes.
- `plots/inter_event_interval_histograms.png` — interval distributions.

Event summary:

```text
{consolidated_summary.write_csv(separator='|').strip() if not consolidated_summary.is_empty() else 'none'}
```

Detection stage metrics:

```text
{stage_metrics.write_csv(separator='|').strip()}
```

Curated recovery summary, consolidated events:

```text
{consolidated_recovery_summary.write_csv(separator='|').strip() if not consolidated_recovery_summary.is_empty() else 'none'}
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
