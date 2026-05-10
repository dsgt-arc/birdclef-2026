"""Motif mining for Marbled White-lipped Frog train recordings.

Detects motif candidates as connected components in the robust-excess-z
spectrogram, gated by physical event properties (duration, bandwidth, peak
salience, area). Each motif is exactly one connected component; bounding boxes
are the component's time-frequency extent. Crops around each component are
normalized, embedded via PCA, and clustered with KMeans for review grids.
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
from scipy.ndimage import label, zoom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_analysis.common.paths import TRAIN_AUDIO_DIR, artifact_dir, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)

TARGET_LABEL = "22967"
PATCH_SHAPE = (64, 48)  # frequency bins x time bins
MAX_GRID_ITEMS = 24


@dataclass(frozen=True)
class ComponentGate:
    """Physical-property gates applied to each connected component.

    Fields:
        z_threshold: pixel-level robust-excess-z threshold used to binarize the
            spectrogram before connected-component labeling.
        min_duration_sec / max_duration_sec: bbox time-extent must fall in this
            range (filters salt-and-pepper bursts and sustained noise).
        min_bandwidth_hz / max_bandwidth_hz: bbox frequency-extent range.
        min_peak_z: component must contain at least one pixel this bright.
        min_area_pixels: minimum component size in time-frequency pixels.
    """

    z_threshold: float = 2.5
    min_duration_sec: float = 0.03
    max_duration_sec: float = 1.5
    min_bandwidth_hz: float = 80.0
    max_bandwidth_hz: float = 2200.0
    min_peak_z: float = 3.5
    min_area_pixels: int = 8


@dataclass(frozen=True)
class ComponentBox:
    """Bounding box for a single connected component above the z threshold."""

    motif_id: int
    start_sec: float
    end_sec: float
    fmin_hz: float
    fmax_hz: float
    peak_time_sec: float
    peak_freq_hz: float
    peak_z: float
    mean_z: float
    area_pixels: int
    fill_ratio: float


def recording_files() -> list[str]:
    """Return all Marbled Frog train recording filenames."""
    return [f"{TARGET_LABEL}/{path.name}" for path in sorted((TRAIN_AUDIO_DIR / TARGET_LABEL).glob("*.ogg"))]


def detect_components(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    gate: ComponentGate = ComponentGate(),
) -> list[ComponentBox]:
    """Find motif candidates as 8-connected components above z_threshold.

    Each surviving component yields exactly one bounding box, sized to its own
    time-frequency extent rather than to a hand-picked frequency band.
    """
    binary = z >= gate.z_threshold
    if not binary.any():
        return []
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, n_components = label(binary, structure=structure)
    if n_components == 0:
        return []
    boxes: list[ComponentBox] = []
    motif_id = 0
    for component_idx in range(1, n_components + 1):
        rows, cols = np.where(labeled == component_idx)
        if rows.size < gate.min_area_pixels:
            continue
        f_lo, f_hi = int(rows.min()), int(rows.max())
        t_lo, t_hi = int(cols.min()), int(cols.max())
        start_sec = float(times[t_lo])
        end_sec = float(times[t_hi])
        fmin_hz = float(freqs[f_lo])
        fmax_hz = float(freqs[f_hi])
        duration_sec = end_sec - start_sec
        bandwidth_hz = fmax_hz - fmin_hz
        if not (gate.min_duration_sec <= duration_sec <= gate.max_duration_sec):
            continue
        if not (gate.min_bandwidth_hz <= bandwidth_hz <= gate.max_bandwidth_hz):
            continue
        component_values = z[rows, cols]
        peak_z = float(component_values.max())
        if peak_z < gate.min_peak_z:
            continue
        peak_in_component = int(np.argmax(component_values))
        bbox_area = (f_hi - f_lo + 1) * (t_hi - t_lo + 1)
        motif_id += 1
        boxes.append(
            ComponentBox(
                motif_id=motif_id,
                start_sec=start_sec,
                end_sec=end_sec,
                fmin_hz=fmin_hz,
                fmax_hz=fmax_hz,
                peak_time_sec=float(times[cols[peak_in_component]]),
                peak_freq_hz=float(freqs[rows[peak_in_component]]),
                peak_z=peak_z,
                mean_z=float(component_values.mean()),
                area_pixels=int(rows.size),
                fill_ratio=float(rows.size / bbox_area),
            )
        )
    return boxes


def crop_patch(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    box: ComponentBox,
    *,
    time_pad_sec: float = 0.12,
    freq_pad_hz: float = 180.0,
) -> np.ndarray:
    """Extract and resize a normalized robust-excess spectrogram crop."""
    t0 = max(0.0, box.start_sec - time_pad_sec)
    t1 = min(float(times[-1]), box.end_sec + time_pad_sec)
    f0 = max(DISPLAY_BAND[0], box.fmin_hz - freq_pad_hz)
    f1 = min(DISPLAY_BAND[1], box.fmax_hz + freq_pad_hz)
    t_mask = (times >= t0) & (times <= t1)
    f_mask = (freqs >= f0) & (freqs <= f1)
    if not np.any(t_mask) or not np.any(f_mask):
        return np.zeros(PATCH_SHAPE, dtype=np.float32)
    patch = z[f_mask][:, t_mask]
    if patch.size == 0:
        return np.zeros(PATCH_SHAPE, dtype=np.float32)
    patch = patch.astype(np.float32)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    scale = (PATCH_SHAPE[0] / patch.shape[0], PATCH_SHAPE[1] / patch.shape[1])
    resized = zoom(patch, scale, order=1)
    if resized.shape != PATCH_SHAPE:
        out = np.zeros(PATCH_SHAPE, dtype=np.float32)
        h = min(PATCH_SHAPE[0], resized.shape[0])
        w = min(PATCH_SHAPE[1], resized.shape[1])
        out[:h, :w] = resized[:h, :w]
        resized = out
    return resized.astype(np.float32)


def box_features(
    z: np.ndarray, freqs: np.ndarray, times: np.ndarray, box: ComponentBox
) -> dict[str, float]:
    """Compute interpretable features over the component bbox."""
    t_mask = (times >= box.start_sec) & (times <= box.end_sec)
    f_mask = (freqs >= box.fmin_hz) & (freqs <= box.fmax_hz)
    if not np.any(t_mask) or not np.any(f_mask):
        return {
            "freq_centroid_hz": float("nan"),
            "freq_slope_hz_per_s": float("nan"),
        }
    crop = z[f_mask][:, t_mask]
    crop_freqs = freqs[f_mask]
    crop_times = times[t_mask]
    weights = np.maximum(crop, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-9:
        centroid = float(np.mean(crop_freqs))
        slope = float("nan")
    else:
        freq_grid = crop_freqs[:, None]
        time_grid = crop_times[None, :]
        centroid = float((weights * freq_grid).sum() / weight_sum)
        t_mean = float((weights * time_grid).sum() / weight_sum)
        f_mean = centroid
        cov = float((weights * (time_grid - t_mean) * (freq_grid - f_mean)).sum() / weight_sum)
        var_t = float((weights * (time_grid - t_mean) ** 2).sum() / weight_sum)
        slope = cov / (var_t + 1e-9)
    return {
        "freq_centroid_hz": centroid,
        "freq_slope_hz_per_s": slope,
    }


def generate_candidates(
    *, gate: ComponentGate = ComponentGate()
) -> tuple[pl.DataFrame, np.ndarray]:
    """Detect connected-component motif candidates over all recordings."""
    rows = []
    patches = []
    for filename in recording_files():
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
        _db_band, z, freqs, times = robust_excess_spectrogram(audio)
        boxes = detect_components(z, freqs, times, gate=gate)
        for box in boxes:
            patch = crop_patch(z, freqs, times, box)
            patches.append(patch.ravel())
            feats = box_features(z, freqs, times, box)
            rows.append(
                {
                    "candidate_id": len(rows) + 1,
                    "filename": filename,
                    "motif_id": box.motif_id,
                    "start_sec": box.start_sec,
                    "end_sec": box.end_sec,
                    "duration_sec": box.end_sec - box.start_sec,
                    "fmin_hz": box.fmin_hz,
                    "fmax_hz": box.fmax_hz,
                    "bandwidth_hz": box.fmax_hz - box.fmin_hz,
                    "peak_time_sec": box.peak_time_sec,
                    "peak_freq_hz": box.peak_freq_hz,
                    "peak_z": box.peak_z,
                    "mean_z": box.mean_z,
                    "area_pixels": box.area_pixels,
                    "fill_ratio": box.fill_ratio,
                    **feats,
                }
            )
    if not rows:
        return pl.DataFrame(), np.empty((0, PATCH_SHAPE[0] * PATCH_SHAPE[1]), dtype=np.float32)
    return pl.DataFrame(rows), np.vstack(patches).astype(np.float32)


def cluster_candidates(
    candidates: pl.DataFrame, patches: np.ndarray, *, n_clusters: int = 8, seed: int = 42
) -> tuple[pl.DataFrame, np.ndarray]:
    """Cluster normalized motif crops and add embedding coordinates."""
    x = StandardScaler().fit_transform(patches)
    n_pca = min(24, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=max(2, n_pca), random_state=seed)
    xp = pca.fit_transform(x)
    n_clusters = min(n_clusters, x.shape[0])
    labels = KMeans(n_clusters=n_clusters, n_init=50, random_state=seed).fit_predict(xp)
    coords = xp[:, :2]
    return candidates.with_columns(
        pl.Series("motif_cluster", labels).cast(pl.Int64),
        pl.Series("motif_x", coords[:, 0]),
        pl.Series("motif_y", coords[:, 1]),
    ), xp


def summarize_clusters(candidates: pl.DataFrame) -> pl.DataFrame:
    """Cluster-level summary for review."""
    return candidates.group_by("motif_cluster").agg(
        pl.len().alias("n_candidates"),
        pl.col("filename").n_unique().alias("n_recordings"),
        pl.col("duration_sec").median().alias("median_duration_sec"),
        pl.col("bandwidth_hz").median().alias("median_bandwidth_hz"),
        pl.col("fmin_hz").median().alias("median_fmin_hz"),
        pl.col("fmax_hz").median().alias("median_fmax_hz"),
        pl.col("freq_centroid_hz").median().alias("median_centroid_hz"),
        pl.col("freq_slope_hz_per_s").median().alias("median_slope_hz_per_s"),
        pl.col("peak_z").median().alias("median_peak_z"),
        pl.col("area_pixels").median().alias("median_area_pixels"),
    ).sort("motif_cluster")


def plot_candidate_embedding(candidates: pl.DataFrame, output_path: Path) -> None:
    """Scatter plot of motif-crop PCA coordinates colored by cluster."""
    fig, ax = plt.subplots(figsize=(9.0, 7.0), dpi=180)
    clusters = sorted(candidates["motif_cluster"].unique().to_list())
    cmap = plt.cm.tab10
    for cluster in clusters:
        df = candidates.filter(pl.col("motif_cluster") == cluster)
        ax.scatter(
            df["motif_x"],
            df["motif_y"],
            s=np.clip(df["peak_z"].to_numpy() * 14, 18, 95),
            color=cmap(int(cluster) % 10),
            alpha=0.78,
            edgecolors="black",
            linewidths=0.25,
            label=f"cluster {cluster} (n={df.height})",
        )
    ax.set_title("Marbled Frog motif candidates: PCA of normalized spectrogram crops")
    ax.set_xlabel("motif PCA 1")
    ax.set_ylabel("motif PCA 2")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cluster_grid(
    candidates: pl.DataFrame, output_dir: Path, *, max_items: int = MAX_GRID_ITEMS
) -> None:
    """Render one spectrogram crop grid per cluster."""
    for cluster in sorted(candidates["motif_cluster"].unique().to_list()):
        df = (
            candidates.filter(pl.col("motif_cluster") == cluster)
            .sort("filename", "start_sec")
            .head(max_items)
        )
        n_items = df.height
        if n_items == 0:
            continue
        n_cols = min(4, n_items)
        n_rows = int(np.ceil(n_items / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.7 * n_cols, 3.2 * n_rows),
            dpi=180,
            sharex=False,
            sharey=True,
            squeeze=False,
        )
        for idx, row in enumerate(df.iter_rows(named=True)):
            row_idx, col_idx = divmod(idx, n_cols)
            ax = axes[row_idx, col_idx]
            pad = 0.18
            offset = max(0.0, float(row["start_sec"]) - pad)
            duration = float(row["end_sec"] - row["start_sec"] + 2 * pad)
            audio, _ = librosa.load(
                TRAIN_AUDIO_DIR / row["filename"],
                sr=MEL_SAMPLE_RATE,
                mono=True,
                offset=offset,
                duration=duration,
            )
            db_band, _z, freqs, times = robust_excess_spectrogram(audio)
            display_times = times + offset
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
                    edgecolor="lime",
                    linewidth=2.0,
                )
            )
            ax.set_ylim(DISPLAY_BAND)
            ax.set_title(
                f"{Path(row['filename']).stem} {row['start_sec']:.2f}-{row['end_sec']:.2f}s\n"
                f"peak_z={row['peak_z']:.2f} bw={row['bandwidth_hz']:.0f}Hz",
                fontsize=7.0,
            )
            if col_idx != 0:
                ax.set_ylabel("")
        for idx in range(n_items, n_rows * n_cols):
            row_idx, col_idx = divmod(idx, n_cols)
            axes[row_idx, col_idx].axis("off")
        fig.suptitle(f"Motif cluster {cluster}: review grid", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{int(cluster):02d}_grid.png")
        plt.close(fig)


def run(
    output_dir: Path | None = None,
    *,
    n_clusters: int = 8,
    seed: int = 42,
    gate: ComponentGate = ComponentGate(),
) -> Path:
    """Run motif mining."""
    output_dir = output_dir or artifact_dir("marbled_frog/motif_mining")
    plots_dir = fresh_subdir(output_dir, "plots")
    tables_dir = fresh_subdir(output_dir, "tables")
    cluster_dir = fresh_subdir(output_dir, "clusters")

    candidates, patches = generate_candidates(gate=gate)
    if candidates.is_empty():
        raise RuntimeError("No motif candidates detected")
    candidates, _xp = cluster_candidates(candidates, patches, n_clusters=n_clusters, seed=seed)
    summary = summarize_clusters(candidates)

    candidates.write_csv(tables_dir / "candidates.csv")
    candidates.write_csv(tables_dir / "boxes_by_cluster.csv")
    summary.write_csv(tables_dir / "cluster_summary.csv")
    plot_candidate_embedding(candidates, plots_dir / "candidate_motif_pca.png")
    plot_cluster_grid(candidates, cluster_dir)

    (output_dir / "summary.md").write_text(
        f"""# Marbled Frog Motif Mining

Each motif candidate is a single 8-connected component in the robust-excess-z
spectrogram, gated on physical event properties:

```text
z_threshold       = {gate.z_threshold}
min/max duration  = {gate.min_duration_sec}-{gate.max_duration_sec} s
min/max bandwidth = {gate.min_bandwidth_hz}-{gate.max_bandwidth_hz} Hz
min peak z        = {gate.min_peak_z}
min area (pixels) = {gate.min_area_pixels}
```

Candidates were extracted from {len(recording_files())} `{TARGET_LABEL}` train recordings.

Artifacts:

- `tables/candidates.csv` — every component box with features and cluster assignment.
- `tables/cluster_summary.csv` — cluster-level statistics.
- `plots/candidate_motif_pca.png` — PCA scatter of normalized spectrogram crops.
- `clusters/cluster_XX_grid.png` — review grids for each motif cluster.

Cluster summary:

```text
{summary.write_csv(separator='|').strip()}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-threshold", type=float, default=ComponentGate.z_threshold)
    parser.add_argument("--min-duration-sec", type=float, default=ComponentGate.min_duration_sec)
    parser.add_argument("--max-duration-sec", type=float, default=ComponentGate.max_duration_sec)
    parser.add_argument("--min-bandwidth-hz", type=float, default=ComponentGate.min_bandwidth_hz)
    parser.add_argument("--max-bandwidth-hz", type=float, default=ComponentGate.max_bandwidth_hz)
    parser.add_argument("--min-peak-z", type=float, default=ComponentGate.min_peak_z)
    parser.add_argument("--min-area-pixels", type=int, default=ComponentGate.min_area_pixels)
    args = parser.parse_args()
    gate = ComponentGate(
        z_threshold=args.z_threshold,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        min_bandwidth_hz=args.min_bandwidth_hz,
        max_bandwidth_hz=args.max_bandwidth_hz,
        min_peak_z=args.min_peak_z,
        min_area_pixels=args.min_area_pixels,
    )
    output_dir = run(args.output_dir, n_clusters=args.n_clusters, seed=args.seed, gate=gate)
    print(f"Wrote Marbled Frog motif mining analysis to {output_dir}")


if __name__ == "__main__":
    main()
