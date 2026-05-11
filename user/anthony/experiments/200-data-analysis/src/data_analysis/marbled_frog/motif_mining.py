"""Simplified motif mining for Marbled White-lipped Frog recordings.

Pipeline:

1. Use scikit-maad hysteresis ROIs on the robust-excess-z spectrogram to propose
   time-frequency event boxes.
2. Represent each ROI with scikit-maad shape and centroid features, plus a small
   set of physical box descriptors.
3. Cluster non-edge candidates with sklearn HDBSCAN. HDBSCAN noise/low-probability
   points are the outlier mechanism; no custom pairwise-template scoring.

Raw ROI boxes are detection/provenance boxes. Motif-bank templates are built in
``motif_bank.py`` using fixed physical support per cluster.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

from maad import features as maad_features
from maad import rois, util
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import polars as pl
from scipy.ndimage import binary_closing
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_analysis.common.paths import TRAIN_AUDIO_DIR, artifact_dir, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)

warnings.filterwarnings("ignore", category=FutureWarning, module=r"maad\..*")

TARGET_LABEL = "22967"
EDGE_MARGIN_HZ = 80.0
MAX_GRID_ITEMS = 24


@dataclass(frozen=True)
class RoiConfig:
    """Parameters for scikit-maad ROI proposal and physical filtering."""

    z_threshold: float = 3.2
    low_z_threshold: float = 1.4
    closing_freq_bins: int = 1
    closing_time_bins: int = 3
    min_duration_sec: float = 0.03
    max_duration_sec: float = 1.5
    min_bandwidth_hz: float = 80.0
    max_bandwidth_hz: float = 2200.0
    min_peak_z: float = 3.2
    min_area_pixels: int = 8


@dataclass(frozen=True)
class ClusterConfig:
    """Parameters for HDBSCAN clustering in maad-feature space."""

    min_cluster_size: int = 5
    min_samples: int = 3
    edge_margin_hz: float = EDGE_MARGIN_HZ


def recording_files() -> list[str]:
    """Return all Marbled Frog train recording filenames."""
    return [f"{TARGET_LABEL}/{path.name}" for path in sorted((TRAIN_AUDIO_DIR / TARGET_LABEL).glob("*.ogg"))]


def _candidate_rows_from_rois(
    im_rois: np.ndarray,
    rois_df: pd.DataFrame,
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    filename: str,
    config: RoiConfig,
    candidate_offset: int,
) -> tuple[list[dict], pd.DataFrame]:
    """Build candidate metadata rows and return filtered rois_df in the same order."""
    rows: list[dict] = []
    kept_rois = []
    motif_id = 0
    for roi_row in rois_df.itertuples(index=False):
        label_id = int(getattr(roi_row, "labelID"))
        rr, cc = np.where(im_rois == label_id)
        if rr.size < config.min_area_pixels:
            continue
        f_lo, f_hi = int(rr.min()), int(rr.max())
        t_lo, t_hi = int(cc.min()), int(cc.max())
        start_sec = float(times[t_lo])
        end_sec = float(times[t_hi])
        fmin_hz = float(freqs[f_lo])
        fmax_hz = float(freqs[f_hi])
        duration_sec = end_sec - start_sec
        bandwidth_hz = fmax_hz - fmin_hz
        if not (config.min_duration_sec <= duration_sec <= config.max_duration_sec):
            continue
        if not (config.min_bandwidth_hz <= bandwidth_hz <= config.max_bandwidth_hz):
            continue
        values = z[rr, cc]
        peak_z = float(values.max())
        if peak_z < config.min_peak_z:
            continue
        peak_idx = int(np.argmax(values))
        bbox_area = (f_hi - f_lo + 1) * (t_hi - t_lo + 1)
        motif_id += 1
        candidate_id = candidate_offset + len(rows) + 1
        rows.append(
            {
                "candidate_id": candidate_id,
                "filename": filename,
                "motif_id": motif_id,
                "maad_label_id": label_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
                "fmin_hz": fmin_hz,
                "fmax_hz": fmax_hz,
                "bandwidth_hz": bandwidth_hz,
                "peak_time_sec": float(times[cc[peak_idx]]),
                "peak_freq_hz": float(freqs[rr[peak_idx]]),
                "peak_z": peak_z,
                "mean_z": float(values.mean()),
                "area_pixels": int(rr.size),
                "fill_ratio": float(rr.size / bbox_area),
                "edge_touch": bool(
                    fmin_hz <= DISPLAY_BAND[0] + EDGE_MARGIN_HZ
                    or fmax_hz >= DISPLAY_BAND[1] - EDGE_MARGIN_HZ
                ),
            }
        )
        kept_rois.append(roi_row._asdict())
    return rows, pd.DataFrame(kept_rois)


def detect_rois_for_recording(
    filename: str,
    *,
    config: RoiConfig = RoiConfig(),
    candidate_offset: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Detect maad ROIs and compute maad feature rows for one recording."""
    audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
    _db_band, z, freqs, times = robust_excess_spectrogram(audio)

    im_bin = rois.create_mask(
        z,
        mode_bin="absolute",
        bin_h=config.z_threshold,
        bin_l=config.low_z_threshold,
    )
    if config.closing_freq_bins > 1 or config.closing_time_bins > 1:
        im_bin = binary_closing(
            im_bin.astype(bool),
            structure=np.ones((config.closing_freq_bins, config.closing_time_bins), dtype=bool),
        )
    im_rois, rois_df = rois.select_rois(im_bin.astype(bool), min_roi=config.min_area_pixels)
    if rois_df.empty:
        return pl.DataFrame(), pl.DataFrame()
    rois_df = util.format_features(rois_df, times, freqs)
    rows, kept_rois_df = _candidate_rows_from_rois(
        im_rois,
        rois_df,
        z,
        freqs,
        times,
        filename=filename,
        config=config,
        candidate_offset=candidate_offset,
    )
    if not rows:
        return pl.DataFrame(), pl.DataFrame()

    # maad shape/centroid features replace flattened pixel PCA crops as the
    # cluster representation. They are computed on the same robust-z image used
    # for ROI detection; this keeps the visualization and features aligned.
    shape_df, _shape_params = maad_features.shape_features(z, resolution="low", rois=kept_rois_df)
    centroid_df = maad_features.centroid_features(z, rois=kept_rois_df)
    shape_cols = [c for c in shape_df.columns if c.startswith("shp_")]
    centroid_cols = [c for c in centroid_df.columns if c in {"centroid_y", "centroid_x", "duration_x", "bandwidth_y", "area_xy"}]
    feature_pd = pd.concat(
        [
            pd.Series([row["candidate_id"] for row in rows], name="candidate_id"),
            shape_df[shape_cols].reset_index(drop=True),
            centroid_df[centroid_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    # Add a few physically interpretable descriptors; StandardScaler handles units.
    meta_pd = pd.DataFrame(rows)
    for col in ["duration_sec", "bandwidth_hz", "peak_z", "area_pixels", "fill_ratio", "peak_freq_hz"]:
        feature_pd[col] = meta_pd[col].to_numpy()

    return pl.DataFrame(rows), pl.from_pandas(feature_pd)


def generate_candidates(*, config: RoiConfig = RoiConfig()) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Detect candidate ROIs and maad feature matrix over all recordings."""
    candidate_tables = []
    feature_tables = []
    offset = 0
    for filename in recording_files():
        candidates, feature_matrix = detect_rois_for_recording(
            filename, config=config, candidate_offset=offset
        )
        if not candidates.is_empty():
            candidate_tables.append(candidates)
            feature_tables.append(feature_matrix)
            offset += candidates.height
    if not candidate_tables:
        return pl.DataFrame(), pl.DataFrame()
    return pl.concat(candidate_tables, how="diagonal_relaxed"), pl.concat(feature_tables, how="diagonal_relaxed")


def cluster_candidates(
    candidates: pl.DataFrame,
    feature_matrix: pl.DataFrame,
    *,
    config: ClusterConfig = ClusterConfig(),
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cluster non-edge candidates with HDBSCAN in maad-feature space."""
    del seed  # HDBSCAN is deterministic for fixed inputs.
    feature_cols = [c for c in feature_matrix.columns if c != "candidate_id"]
    joined = candidates.join(feature_matrix, on="candidate_id", how="left")
    eligible = joined.filter(~pl.col("edge_touch"))

    labels_by_id = {int(row["candidate_id"]): -1 for row in candidates.iter_rows(named=True)}
    probs_by_id = {int(row["candidate_id"]): 0.0 for row in candidates.iter_rows(named=True)}
    coords_by_id = {int(row["candidate_id"]): (np.nan, np.nan) for row in candidates.iter_rows(named=True)}

    if eligible.height >= config.min_cluster_size:
        x = eligible.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
        x = StandardScaler().fit_transform(x)
        model = HDBSCAN(min_cluster_size=config.min_cluster_size, min_samples=config.min_samples)
        labels = model.fit_predict(x)
        probs = getattr(model, "probabilities_", np.where(labels >= 0, 1.0, 0.0))
        n_pca = min(2, x.shape[0], x.shape[1])
        coords = PCA(n_components=n_pca, random_state=42).fit_transform(x)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        for candidate_id, label, prob, coord in zip(
            eligible["candidate_id"].to_list(), labels, probs, coords, strict=True
        ):
            labels_by_id[int(candidate_id)] = int(label)
            probs_by_id[int(candidate_id)] = float(prob)
            coords_by_id[int(candidate_id)] = (float(coord[0]), float(coord[1]))

    ordered_ids = candidates["candidate_id"].to_list()
    candidates = candidates.with_columns(
        pl.Series("motif_cluster", [labels_by_id[int(i)] for i in ordered_ids]).cast(pl.Int64),
        pl.Series("cluster_probability", [probs_by_id[int(i)] for i in ordered_ids]),
        pl.Series("hdbscan_outlier_score", [1.0 - probs_by_id[int(i)] for i in ordered_ids]),
        pl.Series("motif_x", [coords_by_id[int(i)][0] for i in ordered_ids]),
        pl.Series("motif_y", [coords_by_id[int(i)][1] for i in ordered_ids]),
    )
    return candidates, feature_matrix


def summarize_clusters(candidates: pl.DataFrame) -> pl.DataFrame:
    """Cluster-level summary for human review."""
    return candidates.group_by("motif_cluster").agg(
        pl.len().alias("n_candidates"),
        pl.col("filename").n_unique().alias("n_recordings"),
        pl.col("edge_touch").mean().alias("edge_touch_rate"),
        pl.col("cluster_probability").median().alias("median_cluster_probability"),
        pl.col("duration_sec").median().alias("median_duration_sec"),
        pl.col("bandwidth_hz").median().alias("median_bandwidth_hz"),
        pl.col("fmin_hz").median().alias("median_fmin_hz"),
        pl.col("fmax_hz").median().alias("median_fmax_hz"),
        pl.col("peak_z").median().alias("median_peak_z"),
        pl.col("area_pixels").median().alias("median_area_pixels"),
    ).sort("motif_cluster")


def plot_candidate_embedding(candidates: pl.DataFrame, output_path: Path) -> None:
    """Scatter plot of feature-space PCA coordinates colored by HDBSCAN label."""
    fig, ax = plt.subplots(figsize=(9.0, 7.0), dpi=180)
    clusters = sorted(candidates["motif_cluster"].unique().to_list())
    cmap = plt.cm.tab20
    for cluster in clusters:
        df = candidates.filter(pl.col("motif_cluster") == cluster)
        color = "lightgray" if int(cluster) < 0 else cmap(int(cluster) % 20)
        ax.scatter(
            df["motif_x"],
            df["motif_y"],
            s=np.clip(df["peak_z"].to_numpy() * 14, 18, 95),
            color=color,
            alpha=0.78,
            edgecolors="black",
            linewidths=0.25,
            label=f"cluster {cluster} (n={df.height})",
        )
    ax.set_title("Marbled Frog maad ROI features: PCA view, HDBSCAN labels")
    ax.set_xlabel("feature PCA 1")
    ax.set_ylabel("feature PCA 2")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cluster_grid(candidates: pl.DataFrame, output_dir: Path, *, max_items: int = MAX_GRID_ITEMS) -> None:
    """Render one spectrogram review grid per cluster."""
    for cluster in sorted(candidates["motif_cluster"].unique().to_list()):
        df = candidates.filter(pl.col("motif_cluster") == cluster).sort("filename", "start_sec").head(max_items)
        if df.is_empty():
            continue
        n_items = df.height
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
            edge = "red" if row["edge_touch"] else "lime"
            ax.add_patch(
                Rectangle(
                    (row["start_sec"], row["fmin_hz"]),
                    row["end_sec"] - row["start_sec"],
                    row["fmax_hz"] - row["fmin_hz"],
                    fill=False,
                    edgecolor=edge,
                    linewidth=2.0,
                )
            )
            ax.set_ylim(DISPLAY_BAND)
            ax.set_title(
                f"id={row['candidate_id']} {Path(row['filename']).stem} {row['start_sec']:.2f}-{row['end_sec']:.2f}s\n"
                f"p={row['cluster_probability']:.2f} peak_z={row['peak_z']:.2f} bw={row['bandwidth_hz']:.0f}Hz",
                fontsize=7.0,
            )
            if col_idx != 0:
                ax.set_ylabel("")
        for idx in range(n_items, n_rows * n_cols):
            row_idx, col_idx = divmod(idx, n_cols)
            axes[row_idx, col_idx].axis("off")
        title = "noise / edge / unclustered" if int(cluster) < 0 else f"motif cluster {cluster}"
        fig.suptitle(f"{title}: maad-feature HDBSCAN review grid", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{int(cluster):02d}_grid.png")
        plt.close(fig)


def run(
    output_dir: Path | None = None,
    *,
    seed: int = 42,
    roi_config: RoiConfig = RoiConfig(),
    cluster_config: ClusterConfig = ClusterConfig(),
) -> Path:
    """Run simplified motif mining."""
    output_dir = output_dir or artifact_dir("marbled_frog/motif_mining_simplified")
    plots_dir = fresh_subdir(output_dir, "plots")
    tables_dir = fresh_subdir(output_dir, "tables")
    cluster_dir = fresh_subdir(output_dir, "clusters")

    candidates, feature_matrix = generate_candidates(config=roi_config)
    if candidates.is_empty():
        raise RuntimeError("No motif candidates detected")
    candidates, feature_matrix = cluster_candidates(
        candidates, feature_matrix, config=cluster_config, seed=seed
    )
    summary = summarize_clusters(candidates)

    candidates.write_csv(tables_dir / "candidates.csv")
    feature_matrix.write_csv(tables_dir / "feature_matrix.csv")
    summary.write_csv(tables_dir / "cluster_summary.csv")
    plot_candidate_embedding(candidates, plots_dir / "candidate_feature_pca.png")
    plot_cluster_grid(candidates, cluster_dir)

    (output_dir / "summary.md").write_text(
        f"""# Marbled Frog Simplified Motif Mining

Candidates are scikit-maad hysteresis ROIs on the robust-excess-z spectrogram.
Clusters use scikit-maad shape + centroid features plus physical ROI descriptors,
standardized and clustered with sklearn HDBSCAN.

ROI proposal parameters:

```text
high/seed z threshold = {roi_config.z_threshold}
low/growth z threshold = {roi_config.low_z_threshold}
closing freq x time    = {roi_config.closing_freq_bins} x {roi_config.closing_time_bins} bins
min/max duration       = {roi_config.min_duration_sec}-{roi_config.max_duration_sec} s
min/max bandwidth      = {roi_config.min_bandwidth_hz}-{roi_config.max_bandwidth_hz} Hz
min peak z             = {roi_config.min_peak_z}
min area (pixels)      = {roi_config.min_area_pixels}
```

Clustering parameters:

```text
HDBSCAN min_cluster_size = {cluster_config.min_cluster_size}
HDBSCAN min_samples      = {cluster_config.min_samples}
edge margin              = {cluster_config.edge_margin_hz} Hz
```

Artifacts:

- `tables/candidates.csv` — candidate ROI boxes, HDBSCAN labels, probabilities.
- `tables/feature_matrix.csv` — maad shape/centroid feature matrix.
- `tables/cluster_summary.csv` — cluster-level review summary.
- `plots/candidate_feature_pca.png` — PCA view of maad feature space.
- `clusters/cluster_XX_grid.png` — review grids for HDBSCAN labels (`-1` = noise/edge/unclustered).

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-threshold", type=float, default=RoiConfig.z_threshold)
    parser.add_argument("--low-z-threshold", type=float, default=RoiConfig.low_z_threshold)
    parser.add_argument("--min-duration-sec", type=float, default=RoiConfig.min_duration_sec)
    parser.add_argument("--max-duration-sec", type=float, default=RoiConfig.max_duration_sec)
    parser.add_argument("--min-bandwidth-hz", type=float, default=RoiConfig.min_bandwidth_hz)
    parser.add_argument("--max-bandwidth-hz", type=float, default=RoiConfig.max_bandwidth_hz)
    parser.add_argument("--min-peak-z", type=float, default=RoiConfig.min_peak_z)
    parser.add_argument("--min-area-pixels", type=int, default=RoiConfig.min_area_pixels)
    parser.add_argument("--min-cluster-size", type=int, default=ClusterConfig.min_cluster_size)
    parser.add_argument("--min-samples", type=int, default=ClusterConfig.min_samples)
    args = parser.parse_args()
    output_dir = run(
        args.output_dir,
        seed=args.seed,
        roi_config=RoiConfig(
            z_threshold=args.z_threshold,
            low_z_threshold=args.low_z_threshold,
            min_duration_sec=args.min_duration_sec,
            max_duration_sec=args.max_duration_sec,
            min_bandwidth_hz=args.min_bandwidth_hz,
            max_bandwidth_hz=args.max_bandwidth_hz,
            min_peak_z=args.min_peak_z,
            min_area_pixels=args.min_area_pixels,
        ),
        cluster_config=ClusterConfig(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        ),
    )
    print(f"Wrote simplified motif mining analysis to {output_dir}")


if __name__ == "__main__":
    main()
