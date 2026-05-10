"""Heuristic window relevance labels for Marbled White-lipped Frog.

Uses the frequency-band + Perch genus-logit analysis to assign each train-audio
5-second window a weak `likely_relevant` / `likely_not_relevant` label, then
plots the windows in Perch embedding PaCMAP space and creates sample mel
spectrogram grids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

from data_analysis.common.paths import (
    PERCH_TRAIN_AUDIO_EMBEDDINGS,
    TRAIN_AUDIO_DIR,
    artifact_dir,
    fresh_subdir,
)
from data_analysis.common.spectrogram import (
    MEL_FMAX,
    MEL_FMIN,
    MEL_HOP_LENGTH,
    MEL_N_FFT,
    MEL_N_MELS,
    MEL_SAMPLE_RATE,
    read_audio_window,
)
from data_analysis.marbled_frog.frequency_perch import (
    DEFAULT_LABEL,
    compute_window_scores,
)
from data_analysis.soundscapes.pacmap_embeddings import (
    embedding_columns,
    matrix,
    preprocess_for_pacmap,
    run_pacmap,
    with_coords,
)

RELEVANT_LOGIT_THRESHOLD = 8.5
RELEVANT_BAND_ENERGY_THRESHOLD = 0.5
CANDIDATE_BAND = (1000, 3000)
PLOT_CIRCLE_CENTER = (-4.0, 0.0)
PLOT_CIRCLE_RADIUS = 3.0
ROBUST_Z_THRESHOLD = 2.0


def add_relevance_labels(scores: pl.DataFrame) -> pl.DataFrame:
    """Assign weak relevance labels from Perch genus logit + 1-3 kHz mel energy."""
    return (
        scores.with_columns(
            (
                pl.col("band_1000_1500_mel_energy_frac")
                + pl.col("band_1500_2000_mel_energy_frac")
                + pl.col("band_2000_3000_mel_energy_frac")
            ).alias("mel_1_3k_energy_frac")
        )
        .with_columns(
            (
                (pl.col("max_genus_logit") >= RELEVANT_LOGIT_THRESHOLD)
                & (pl.col("mel_1_3k_energy_frac") >= RELEVANT_BAND_ENERGY_THRESHOLD)
            ).alias("is_likely_relevant"),
            (
                pl.col("max_genus_logit")
                + 2.0 * pl.col("mel_1_3k_energy_frac")
                - 0.25 * (pl.col("mel_centroid_hz") / 1000.0)
            ).alias("relevance_score"),
        )
        .with_columns(
            pl.when(pl.col("is_likely_relevant"))
            .then(pl.lit("likely_relevant"))
            .otherwise(pl.lit("likely_not_relevant"))
            .alias("relevance_label")
        )
        .sort("relevance_score", descending=True)
    )


def load_embeddings_for_scores(scores: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Load Perch embeddings for the scored windows."""
    lf = pl.scan_parquet(PERCH_TRAIN_AUDIO_EMBEDDINGS)
    emb_cols = embedding_columns(lf.collect_schema())
    keys = scores.select("filename", "chunk_idx", "start_sec", "end_sec").unique()
    embeddings = (
        lf.join(keys.lazy(), on=["filename", "chunk_idx", "start_sec", "end_sec"], how="inner")
        .select("filename", "chunk_idx", "start_sec", "end_sec", *emb_cols)
        .collect()
    )
    return embeddings, emb_cols


def build_pacmap(scores: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    """Build window-level PaCMAP coordinates with relevance labels attached."""
    embeddings, emb_cols = load_embeddings_for_scores(scores)
    coords = run_pacmap(matrix(embeddings, emb_cols), seed=seed, max_pca_dims=50)
    metadata = embeddings.select("filename", "chunk_idx", "start_sec", "end_sec").join(
        scores,
        on=["filename", "chunk_idx", "start_sec", "end_sec"],
        how="left",
    )
    return with_coords(metadata, coords).sort("relevance_score", descending=True)


def add_circle_membership(coords: pl.DataFrame) -> pl.DataFrame:
    """Add distance/membership columns for the manually selected PaCMAP circle."""
    cx, cy = PLOT_CIRCLE_CENTER
    return coords.with_columns(
        (
            ((pl.col("pacmap_x") - cx) ** 2 + (pl.col("pacmap_y") - cy) ** 2).sqrt()
        ).alias("circle_distance"),
    ).with_columns((pl.col("circle_distance") <= PLOT_CIRCLE_RADIUS).alias("in_circle_r3_at_minus4_0"))


def circle_centroid(coords: pl.DataFrame) -> dict[str, float | int]:
    """Compute plotted-coordinate centroid and count for points inside the selected circle."""
    inside = coords.filter(pl.col("in_circle_r3_at_minus4_0"))
    if inside.is_empty():
        return {"n": 0, "centroid_x": float("nan"), "centroid_y": float("nan")}
    return {
        "n": inside.height,
        "centroid_x": float(inside["pacmap_x"].mean()),
        "centroid_y": float(inside["pacmap_y"].mean()),
    }


def add_pacmap_mad_boundary(coords: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute centroid/radius in current PaCMAP coordinates from seed-circle points."""
    seed = coords.filter(pl.col("in_circle_r3_at_minus4_0"))
    if seed.height < 3:
        raise ValueError("Need at least 3 seed points inside the manual PaCMAP circle")
    seed_xy = seed.select("pacmap_x", "pacmap_y").to_numpy()
    all_xy = coords.select("pacmap_x", "pacmap_y").to_numpy()
    centroid = seed_xy.mean(axis=0)
    distances = np.linalg.norm(all_xy - centroid, axis=1)
    seed_distances = np.linalg.norm(seed_xy - centroid, axis=1)
    median_distance = float(np.median(seed_distances))
    raw_mad = float(np.median(np.abs(seed_distances - median_distance)))
    scaled_mad = 1.4826 * raw_mad
    if scaled_mad <= 1e-12:
        scaled_mad = float(np.std(seed_distances)) or 1e-12
    radius = median_distance + ROBUST_Z_THRESHOLD * scaled_mad
    robust_z = (distances - median_distance) / scaled_mad
    result = coords.with_columns(
        pl.Series("pacmap_cluster_distance", distances),
        pl.Series("pacmap_cluster_robust_z", robust_z),
        pl.lit(float(centroid[0])).alias("pacmap_seed_centroid_x"),
        pl.lit(float(centroid[1])).alias("pacmap_seed_centroid_y"),
        pl.lit(median_distance).alias("pacmap_seed_median_distance"),
        pl.lit(raw_mad).alias("pacmap_seed_raw_mad"),
        pl.lit(scaled_mad).alias("pacmap_seed_scaled_mad"),
        pl.lit(radius).alias("pacmap_cluster_radius_z2"),
        pl.Series("in_pacmap_mad_z2", distances <= radius),
    )
    summary = pl.DataFrame(
        {
            "space": ["pacmap_2d"],
            "seed_source": [f"PaCMAP circle center={PLOT_CIRCLE_CENTER}, r={PLOT_CIRCLE_RADIUS}"],
            "seed_n": [seed.height],
            "centroid_x": [float(centroid[0])],
            "centroid_y": [float(centroid[1])],
            "median_seed_distance": [median_distance],
            "raw_mad_seed_distance": [raw_mad],
            "scaled_mad_seed_distance": [scaled_mad],
            "robust_z_threshold": [ROBUST_Z_THRESHOLD],
            "radius": [radius],
            "n_within_radius": [int((distances <= radius).sum())],
        }
    )
    return result, summary


def add_embedding_mad_boundary(coords: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Use the plotted circle as a seed, then define a robust embedding-space boundary.

    The seed is selected in the current PaCMAP view, but the centroid, distances,
    MAD radius, and final membership are computed in standardized/PCA-compressed
    Perch embedding space rather than in PaCMAP coordinates.
    """
    coords_with_id = coords.with_row_index("row_id")
    embeddings, emb_cols = load_embeddings_for_scores(coords_with_id)
    embeddings = embeddings.join(
        coords_with_id.select("row_id", "filename", "chunk_idx", "start_sec", "end_sec"),
        on=["filename", "chunk_idx", "start_sec", "end_sec"],
        how="inner",
    ).sort("row_id")

    x = preprocess_for_pacmap(matrix(embeddings, emb_cols), max_pca_dims=50, seed=42)
    seed_mask = coords_with_id["in_circle_r3_at_minus4_0"].to_numpy()
    if seed_mask.sum() < 3:
        raise ValueError("Need at least 3 seed points inside the manual PaCMAP circle")

    seed_x = x[seed_mask]
    centroid = seed_x.mean(axis=0)
    distances = np.linalg.norm(x - centroid, axis=1)
    seed_distances = distances[seed_mask]
    median_distance = float(np.median(seed_distances))
    raw_mad = float(np.median(np.abs(seed_distances - median_distance)))
    scaled_mad = 1.4826 * raw_mad
    if scaled_mad <= 1e-12:
        scaled_mad = float(np.std(seed_distances)) or 1e-12
    radius = median_distance + ROBUST_Z_THRESHOLD * scaled_mad
    robust_z = (distances - median_distance) / scaled_mad

    result = coords_with_id.with_columns(
        pl.Series("embedding_cluster_distance", distances),
        pl.Series("embedding_cluster_robust_z", robust_z),
        pl.lit(median_distance).alias("embedding_seed_median_distance"),
        pl.lit(raw_mad).alias("embedding_seed_raw_mad"),
        pl.lit(scaled_mad).alias("embedding_seed_scaled_mad"),
        pl.lit(radius).alias("embedding_cluster_radius_z2"),
        pl.Series("in_embedding_mad_z2", distances <= radius),
    )

    summary = pl.DataFrame(
        {
            "space": ["embedding_pca"],
            "seed_source": [f"PaCMAP circle center={PLOT_CIRCLE_CENTER}, r={PLOT_CIRCLE_RADIUS}"],
            "seed_n": [int(seed_mask.sum())],
            "embedding_dims": [int(x.shape[1])],
            "median_seed_distance": [median_distance],
            "raw_mad_seed_distance": [raw_mad],
            "scaled_mad_seed_distance": [scaled_mad],
            "robust_z_threshold": [ROBUST_Z_THRESHOLD],
            "radius": [radius],
            "n_within_radius": [int((distances <= radius).sum())],
        }
    )
    return result.drop("row_id"), summary


def plot_pacmap(coords: pl.DataFrame, output_path: Path) -> None:
    """Plot PaCMAP colored by heuristic relevance label."""
    colors = {
        "likely_relevant": "#d62728",
        "likely_not_relevant": "#1f77b4",
    }
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    for label in ["likely_not_relevant", "likely_relevant"]:
        df = coords.filter(pl.col("relevance_label") == label)
        ax.scatter(
            df["pacmap_x"],
            df["pacmap_y"],
            s=np.clip(df["mel_1_3k_energy_frac"].to_numpy() * 140, 28, 180),
            c=colors[label],
            label=f"{label} (n={df.height})",
            alpha=0.82,
            edgecolors="black",
            linewidths=0.35,
        )

    if "in_embedding_mad_z2" in coords.columns:
        robust_df = coords.filter(pl.col("in_embedding_mad_z2"))
        ax.scatter(
            robust_df["pacmap_x"],
            robust_df["pacmap_y"],
            s=np.clip(robust_df["mel_1_3k_energy_frac"].to_numpy() * 180, 45, 220),
            facecolors="none",
            edgecolors="#ffd700",
            linewidths=1.8,
            label=f"embedding MAD z≤{ROBUST_Z_THRESHOLD:g} (n={robust_df.height})",
        )

    for row in coords.head(6).iter_rows(named=True):
        ax.annotate(
            f"{Path(row['filename']).stem}\n{int(row['start_sec'])}-{int(row['end_sec'])}s",
            (row["pacmap_x"], row["pacmap_y"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
        )

    cx, cy = PLOT_CIRCLE_CENTER
    circle = plt.Circle(
        PLOT_CIRCLE_CENTER,
        PLOT_CIRCLE_RADIUS,
        fill=False,
        color="#111111",
        linestyle="--",
        linewidth=1.6,
        alpha=0.8,
        label=f"manual circle r={PLOT_CIRCLE_RADIUS:g} @ ({cx:g},{cy:g})",
    )
    ax.add_patch(circle)
    if "pacmap_seed_centroid_x" in coords.columns:
        mad_cx = float(coords["pacmap_seed_centroid_x"][0])
        mad_cy = float(coords["pacmap_seed_centroid_y"][0])
        mad_radius = float(coords["pacmap_cluster_radius_z2"][0])
        ax.add_patch(
            plt.Circle(
                (mad_cx, mad_cy),
                mad_radius,
                fill=False,
                color="#ffd700",
                linestyle="-",
                linewidth=1.6,
                alpha=0.95,
                label=f"PaCMAP centroid+MAD z≤{ROBUST_Z_THRESHOLD:g} r={mad_radius:.2f}",
            )
        )
        ax.scatter(
            [mad_cx],
            [mad_cy],
            marker="+",
            s=150,
            color="#7f6000",
            linewidths=2.2,
            label=f"PaCMAP MAD centroid ({mad_cx:.2f},{mad_cy:.2f})",
        )

    centroid = circle_centroid(coords)
    if centroid["n"]:
        ax.scatter(
            [centroid["centroid_x"]],
            [centroid["centroid_y"]],
            marker="x",
            s=130,
            color="black",
            linewidths=2.0,
            label=f"circle centroid (n={centroid['n']})",
        )
        ax.annotate(
            f"centroid\n({centroid['centroid_x']:.2f}, {centroid['centroid_y']:.2f})",
            (centroid["centroid_x"], centroid["centroid_y"]),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=8,
            alpha=0.85,
        )

    ax.set_title("Marbled White-lipped Frog train-audio windows: heuristic relevance labels")
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.grid(alpha=0.22)
    ax.legend(title="weak label", loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_file_frequency_spectra(
    filename: str,
    output_path: Path,
    *,
    window_starts: list[float] | None = None,
    window_seconds: float = 5.0,
    fmin: float = 500.0,
    fmax: float = 3500.0,
) -> pl.DataFrame:
    """Plot mean log-power spectra for several windows in a recording and return detected peaks."""
    audio_path = TRAIN_AUDIO_DIR / filename
    audio, _ = librosa.load(audio_path, sr=MEL_SAMPLE_RATE, mono=True)
    duration = audio.size / MEL_SAMPLE_RATE
    if window_starts is None:
        window_starts = list(np.arange(0, max(duration - 0.25, 0), window_seconds))

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(window_starts)))
    peak_rows = []
    for color, start_sec in zip(colors, window_starts):
        end_sec = min(start_sec + window_seconds, duration)
        segment = audio[int(start_sec * MEL_SAMPLE_RATE) : int(end_sec * MEL_SAMPLE_RATE)]
        if segment.size < MEL_SAMPLE_RATE:
            continue
        spectrum = np.abs(
            librosa.stft(segment, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, window="hann")
        ) ** 2
        freqs = librosa.fft_frequencies(sr=MEL_SAMPLE_RATE, n_fft=MEL_N_FFT)
        mask = (freqs >= fmin) & (freqs <= fmax)
        mean_power = spectrum.mean(axis=1)
        log_power = 10.0 * np.log10(mean_power + 1e-12)
        smoothed = np.convolve(log_power[mask], np.ones(7) / 7.0, mode="same")
        band_freqs = freqs[mask]
        label = f"{int(start_sec)}-{int(end_sec)}s"
        ax.plot(band_freqs, smoothed, color=color, linewidth=1.5, alpha=0.88, label=label)

        peaks, _props = find_peaks(smoothed, prominence=3.0, distance=8)
        if peaks.size:
            top = peaks[np.argsort(smoothed[peaks])[-3:]][::-1]
            for rank, peak in enumerate(top, start=1):
                peak_rows.append(
                    {
                        "filename": filename,
                        "window": label,
                        "start_sec": float(start_sec),
                        "end_sec": float(end_sec),
                        "peak_rank": rank,
                        "frequency_hz": float(band_freqs[peak]),
                        "log_power_db": float(smoothed[peak]),
                    }
                )
            strongest = top[0]
            ax.scatter(
                [band_freqs[strongest]],
                [smoothed[strongest]],
                color=color,
                edgecolors="black",
                linewidths=0.4,
                s=34,
                zorder=5,
            )
            ax.annotate(
                f"{band_freqs[strongest]:.0f} Hz",
                (band_freqs[strongest], smoothed[strongest]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

    ax.axvspan(CANDIDATE_BAND[0], CANDIDATE_BAND[1], color="cyan", alpha=0.08, label="1-3 kHz prior")
    for harmonic_hz in [1000, 1500, 2000, 3000]:
        ax.axvline(harmonic_hz, color="black", linestyle=":", linewidth=0.7, alpha=0.35)
    ax.set_title(f"Frequency spectra for {filename}\nmean log-power per 5s window")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean log power (dB, arbitrary reference)")
    ax.grid(alpha=0.22)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    peaks_df = pl.DataFrame(peak_rows) if peak_rows else pl.DataFrame()
    if peak_rows:
        peaks_df.write_csv(output_path.with_suffix(".peaks.csv"))
    return peaks_df


def plot_peak_frequency_summary(peaks: pl.DataFrame, output_path: Path) -> None:
    """Summarize detected peak frequencies across recordings/windows."""
    if peaks.is_empty():
        return
    top = peaks.filter(pl.col("peak_rank") <= 3).with_columns(
        pl.col("filename").str.replace("22967/", "").str.replace(".ogg", "", literal=True).alias("recording")
    )
    recordings = top.select("recording").unique(maintain_order=True).to_series().to_list()
    y_positions = {recording: idx for idx, recording in enumerate(recordings)}
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.55 * len(recordings) + 1.5)), dpi=180)
    for rank, marker, size in [(1, "o", 62), (2, "^", 46), (3, "s", 34)]:
        df = top.filter(pl.col("peak_rank") == rank)
        if df.is_empty():
            continue
        y = [y_positions[r] for r in df["recording"].to_list()]
        ax.scatter(
            df["frequency_hz"],
            y,
            s=size,
            marker=marker,
            alpha=0.72,
            edgecolors="black",
            linewidths=0.25,
            label=f"peak rank {rank}",
        )
    ax.axvspan(CANDIDATE_BAND[0], CANDIDATE_BAND[1], color="cyan", alpha=0.08, label="1-3 kHz prior")
    for guide in [1000, 1500, 2000, 3000]:
        ax.axvline(guide, color="black", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(recordings)
    ax.set_xlabel("Detected peak frequency (Hz)")
    ax.set_ylabel("Recording")
    ax.set_title("Marbled White-lipped Frog: detected spectral peaks by recording/window")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_recording_frequency_spectra(coords: pl.DataFrame, spectra_dir: Path, plots_dir: Path, tables_dir: Path) -> None:
    """Generate spectroscopy-style frequency plots and peak summaries for every train recording."""
    spectra_dir.mkdir(parents=True, exist_ok=True)
    filenames = coords.select("filename").unique().sort("filename").to_series().to_list()
    peak_tables = []
    for filename in filenames:
        stem = Path(filename).stem
        peaks = plot_file_frequency_spectra(filename, spectra_dir / f"{stem}_frequency_spectra.png")
        if not peaks.is_empty():
            peak_tables.append(peaks)
    if not peak_tables:
        return
    all_peaks = pl.concat(peak_tables, how="diagonal_relaxed")
    window_scores = coords.select(
        "filename",
        "start_sec",
        "end_sec",
        "relevance_label",
        "in_pacmap_mad_z2",
        "outside_kmeans_cluster",
        "max_genus_logit",
        "mel_1_3k_energy_frac",
        "mel_centroid_hz",
    )
    all_peaks = all_peaks.join(window_scores, on=["filename", "start_sec", "end_sec"], how="left")
    all_peaks.write_csv(tables_dir / "all_recording_peak_summary.csv")
    plot_peak_frequency_summary(all_peaks, plots_dir / "peak_frequency_by_recording.png")


def mel_db_for_window(filename: str, start_sec: float, end_sec: float) -> np.ndarray:
    """Compute a mel spectrogram in dB for display."""
    audio = read_audio_window(filename, start_sec, end_sec)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=MEL_SAMPLE_RATE,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH,
        n_mels=MEL_N_MELS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def audio_duration_s(filename: str, start_sec: float, end_sec: float) -> float:
    """Return actual loaded audio duration for a requested window."""
    audio = read_audio_window(filename, start_sec, end_sec)
    return float(audio.size / MEL_SAMPLE_RATE)


def select_fullish_windows(df: pl.DataFrame, *, n: int, descending: bool) -> pl.DataFrame:
    """Select windows by score, preferring windows with at least 4s of audio."""
    rows = []
    for row in df.sort("relevance_score", descending=descending).iter_rows(named=True):
        duration_s = audio_duration_s(row["filename"], float(row["start_sec"]), float(row["end_sec"]))
        if duration_s >= 4.0:
            row = dict(row)
            row["audio_duration_s"] = duration_s
            rows.append(row)
        if len(rows) >= n:
            break
    if len(rows) < n:
        for row in df.sort("relevance_score", descending=descending).iter_rows(named=True):
            if any(existing["filename"] == row["filename"] and existing["start_sec"] == row["start_sec"] for existing in rows):
                continue
            row = dict(row)
            row["audio_duration_s"] = audio_duration_s(row["filename"], float(row["start_sec"]), float(row["end_sec"]))
            rows.append(row)
            if len(rows) >= n:
                break
    return pl.DataFrame(rows)


def add_outside_cluster_kmeans(coords: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    """Cluster points outside the PaCMAP MAD seed cluster into two visual groups."""
    outside = coords.filter(~pl.col("in_pacmap_mad_z2"))
    if outside.height < 2:
        return coords.with_columns(pl.lit(None).cast(pl.Int64).alias("outside_kmeans_cluster"))

    features = outside.select("pacmap_x", "pacmap_y").to_numpy()
    n_clusters = min(2, outside.height)
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(features)
    outside_with_clusters = outside.with_columns(pl.Series("outside_kmeans_cluster", labels).cast(pl.Int64))
    return coords.join(
        outside_with_clusters.select(
            "filename", "chunk_idx", "start_sec", "end_sec", "outside_kmeans_cluster"
        ),
        on=["filename", "chunk_idx", "start_sec", "end_sec"],
        how="left",
    )


def plot_outside_kmeans_pacmap(coords: pl.DataFrame, output_path: Path) -> None:
    """Plot only/outside-cluster emphasis with two KMeans groups colored."""
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    inside = coords.filter(pl.col("in_pacmap_mad_z2"))
    outside = coords.filter(~pl.col("in_pacmap_mad_z2"))

    ax.scatter(
        inside["pacmap_x"],
        inside["pacmap_y"],
        s=36,
        c="#d0d0d0",
        alpha=0.45,
        edgecolors="none",
        label=f"inside selected cluster (n={inside.height})",
    )

    colors = {0: "#e41a1c", 1: "#377eb8"}
    for cluster_id in sorted(outside["outside_kmeans_cluster"].drop_nulls().unique().to_list()):
        df = outside.filter(pl.col("outside_kmeans_cluster") == cluster_id)
        ax.scatter(
            df["pacmap_x"],
            df["pacmap_y"],
            s=np.clip(df["mel_1_3k_energy_frac"].to_numpy() * 170, 45, 220),
            c=colors.get(int(cluster_id), "#4daf4a"),
            alpha=0.88,
            edgecolors="black",
            linewidths=0.45,
            label=f"outside k-means {cluster_id} (n={df.height})",
        )
        for row in df.iter_rows(named=True):
            ax.annotate(
                f"{Path(row['filename']).stem}\n{int(row['start_sec'])}-{int(row['end_sec'])}s",
                (row["pacmap_x"], row["pacmap_y"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.5,
                alpha=0.78,
            )

    if "pacmap_seed_centroid_x" in coords.columns:
        cx = float(coords["pacmap_seed_centroid_x"][0])
        cy = float(coords["pacmap_seed_centroid_y"][0])
        radius = float(coords["pacmap_cluster_radius_z2"][0])
        ax.add_patch(
            plt.Circle((cx, cy), radius, fill=False, color="#555555", linestyle="--", linewidth=1.5)
        )
        ax.scatter([cx], [cy], marker="+", s=140, color="#555555", linewidths=2.0)

    ax.set_title("Marbled White-lipped Frog: points outside selected cluster, split by 2-means")
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_outside_kmeans_spectrogram_grid(
    coords: pl.DataFrame, output_path: Path, *, max_cols: int = 4
) -> None:
    """Grid outside-cluster windows as mel spectrograms, grouped by KMeans cluster."""
    outside = coords.filter(~pl.col("in_pacmap_mad_z2")).sort(
        "outside_kmeans_cluster", "filename", "start_sec"
    )
    cluster_ids = sorted(outside["outside_kmeans_cluster"].drop_nulls().unique().to_list())
    if not cluster_ids:
        return

    layout: list[tuple[int, dict]] = []
    for cluster_id in cluster_ids:
        df = outside.filter(pl.col("outside_kmeans_cluster") == cluster_id)
        for row in df.iter_rows(named=True):
            layout.append((int(cluster_id), row))

    n_items = len(layout)
    n_cols = min(max_cols, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.25 * n_rows),
        dpi=180,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    cluster_colors = {0: "#e41a1c", 1: "#377eb8"}
    for idx, (cluster_id, row) in enumerate(layout):
        row_idx, col_idx = divmod(idx, n_cols)
        ax = axes[row_idx, col_idx]
        mel_db = mel_db_for_window(row["filename"], float(row["start_sec"]), float(row["end_sec"]))
        librosa.display.specshow(
            mel_db,
            sr=MEL_SAMPLE_RATE,
            hop_length=MEL_HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
            ax=ax,
            cmap="magma",
        )
        ax.axhspan(CANDIDATE_BAND[0], CANDIDATE_BAND[1], color="cyan", alpha=0.13)
        color = cluster_colors.get(cluster_id, "#4daf4a")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_title(
            f"k={cluster_id} | {Path(row['filename']).stem} {int(row['start_sec'])}-{int(row['end_sec'])}s\n"
            f"{row['relevance_label'].replace('likely_', '')}; "
            f"G={row['max_genus_logit']:.2f}, B={row['mel_1_3k_energy_frac']:.2f}",
            fontsize=7.5,
            color=color,
        )
        if col_idx != 0:
            ax.set_ylabel("")

    for idx in range(n_items, n_rows * n_cols):
        row_idx, col_idx = divmod(idx, n_cols)
        axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "Marbled White-lipped Frog: outside selected cluster, 2-means spectrogram grid\n"
        "cyan band = 1-3 kHz prior; border/title color = outside k-means cluster",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def best_window_per_recording(coords: pl.DataFrame) -> pl.DataFrame:
    """Select the highest relevance-score window from each recording."""
    rows = []
    for filename in coords.select("filename").unique().sort("filename").to_series().to_list():
        df = coords.filter(pl.col("filename") == filename).sort("relevance_score", descending=True)
        # Prefer non-truncated windows when possible.
        selected = None
        for row in df.iter_rows(named=True):
            if audio_duration_s(row["filename"], float(row["start_sec"]), float(row["end_sec"])) >= 4.0:
                selected = row
                break
        if selected is None:
            selected = df.row(0, named=True)
        rows.append(selected)
    return pl.DataFrame(rows).sort("filename")


def plot_best_window_per_recording_grid(
    coords: pl.DataFrame, output_path: Path, *, max_cols: int = 4
) -> pl.DataFrame:
    """Plot the top relevance-score window from each recording as a spectrogram grid."""
    best = best_window_per_recording(coords)
    n_items = best.height
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

    label_colors = {"likely_relevant": "#d62728", "likely_not_relevant": "#1f77b4"}
    for idx, row in enumerate(best.iter_rows(named=True)):
        row_idx, col_idx = divmod(idx, n_cols)
        ax = axes[row_idx, col_idx]
        mel_db = mel_db_for_window(row["filename"], float(row["start_sec"]), float(row["end_sec"]))
        librosa.display.specshow(
            mel_db,
            sr=MEL_SAMPLE_RATE,
            hop_length=MEL_HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
            ax=ax,
            cmap="magma",
        )
        ax.axhspan(CANDIDATE_BAND[0], CANDIDATE_BAND[1], color="cyan", alpha=0.13)
        color = label_colors.get(row["relevance_label"], "#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        outside = row.get("outside_kmeans_cluster")
        cluster_text = "inside" if row["in_pacmap_mad_z2"] else f"outside k={outside}"
        ax.set_title(
            f"{Path(row['filename']).stem} {int(row['start_sec'])}-{int(row['end_sec'])}s\n"
            f"{row['relevance_label'].replace('likely_', '')}; {cluster_text}; "
            f"score={row['relevance_score']:.2f}\n"
            f"G={row['max_genus_logit']:.2f}, B={row['mel_1_3k_energy_frac']:.2f}",
            fontsize=7.4,
            color=color,
        )
        if col_idx != 0:
            ax.set_ylabel("")

    for idx in range(n_items, n_rows * n_cols):
        row_idx, col_idx = divmod(idx, n_cols)
        axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "Marbled White-lipped Frog: highest relevance-score window per recording\n"
        "cyan band = 1-3 kHz prior; blue=scalar not-relevant, red=scalar relevant",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return best


def plot_inside_cluster_spectrogram_grid(
    coords: pl.DataFrame, output_path: Path, *, max_cols: int = 4
) -> None:
    """Grid windows inside the selected PaCMAP MAD cluster for visual inspection."""
    inside = coords.filter(pl.col("in_pacmap_mad_z2")).sort("filename", "start_sec")
    if inside.is_empty():
        return

    n_items = inside.height
    n_cols = min(max_cols, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.25 * n_rows),
        dpi=180,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    label_colors = {"likely_relevant": "#d62728", "likely_not_relevant": "#1f77b4"}
    for idx, row in enumerate(inside.iter_rows(named=True)):
        row_idx, col_idx = divmod(idx, n_cols)
        ax = axes[row_idx, col_idx]
        mel_db = mel_db_for_window(row["filename"], float(row["start_sec"]), float(row["end_sec"]))
        librosa.display.specshow(
            mel_db,
            sr=MEL_SAMPLE_RATE,
            hop_length=MEL_HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            fmin=MEL_FMIN,
            fmax=MEL_FMAX,
            ax=ax,
            cmap="magma",
        )
        ax.axhspan(CANDIDATE_BAND[0], CANDIDATE_BAND[1], color="cyan", alpha=0.13)
        color = label_colors.get(row["relevance_label"], "#555555")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_title(
            f"{Path(row['filename']).stem} {int(row['start_sec'])}-{int(row['end_sec'])}s\n"
            f"{row['relevance_label'].replace('likely_', '')}; "
            f"G={row['max_genus_logit']:.2f}, B={row['mel_1_3k_energy_frac']:.2f}, "
            f"z={row['pacmap_cluster_robust_z']:.1f}",
            fontsize=7.5,
            color=color,
        )
        if col_idx != 0:
            ax.set_ylabel("")

    for idx in range(n_items, n_rows * n_cols):
        row_idx, col_idx = divmod(idx, n_cols)
        axes[row_idx, col_idx].axis("off")

    fig.suptitle(
        "Marbled White-lipped Frog: windows inside selected PaCMAP/MAD cluster\n"
        "cyan band = 1-3 kHz prior; blue=scalar not-relevant, red=scalar relevant",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_spectrogram_samples(labels: pl.DataFrame, output_path: Path, n_each: int = 4) -> None:
    """Plot sample relevant and not-relevant mel spectrograms."""
    relevant = select_fullish_windows(
        labels.filter(pl.col("is_likely_relevant")), n=n_each, descending=True
    )
    not_relevant = select_fullish_windows(
        labels.filter(~pl.col("is_likely_relevant")), n=n_each, descending=False
    )
    rows = [("likely relevant", relevant), ("likely not relevant", not_relevant)]

    fig, axes = plt.subplots(2, n_each, figsize=(4.2 * n_each, 6.8), dpi=180, sharex=True, sharey=True)
    for row_idx, (row_label, df) in enumerate(rows):
        for col_idx, row in enumerate(df.iter_rows(named=True)):
            ax = axes[row_idx, col_idx]
            mel_db = mel_db_for_window(row["filename"], float(row["start_sec"]), float(row["end_sec"]))
            librosa.display.specshow(
                mel_db,
                sr=MEL_SAMPLE_RATE,
                hop_length=MEL_HOP_LENGTH,
                x_axis="time",
                y_axis="mel",
                fmin=MEL_FMIN,
                fmax=MEL_FMAX,
                ax=ax,
                cmap="magma",
            )
            ax.axhspan(CANDIDATE_BAND[0], CANDIDATE_BAND[1], color="cyan", alpha=0.13)
            ax.set_title(
                f"{row_label}\n{Path(row['filename']).stem} {int(row['start_sec'])}-{int(row['end_sec'])}s\n"
                f"G={row['max_genus_logit']:.2f}, B={row['mel_1_3k_energy_frac']:.2f}",
                fontsize=8,
            )
            if col_idx != 0:
                ax.set_ylabel("")
    fig.suptitle("Marbled White-lipped Frog: sample mel spectrogram windows\ncyan band = 1-3 kHz prior", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_summary(
    labels: pl.DataFrame, coords: pl.DataFrame, boundary_summary: pl.DataFrame, output_dir: Path
) -> str:
    """Render a short Markdown summary."""
    counts = labels.group_by("relevance_label").agg(
        pl.len().alias("n_windows"),
        pl.col("max_genus_logit").mean().alias("mean_max_genus_logit"),
        pl.col("mel_1_3k_energy_frac").mean().alias("mean_mel_1_3k_energy_frac"),
    ).sort("relevance_label")
    robust_counts = coords.group_by("in_embedding_mad_z2").agg(
        pl.len().alias("n_windows"),
        pl.col("max_genus_logit").mean().alias("mean_max_genus_logit"),
        pl.col("mel_1_3k_energy_frac").mean().alias("mean_mel_1_3k_energy_frac"),
        pl.col("embedding_cluster_distance").mean().alias("mean_embedding_distance"),
    ).sort("in_embedding_mad_z2")
    pacmap_counts = coords.group_by("in_pacmap_mad_z2").agg(
        pl.len().alias("n_windows"),
        pl.col("max_genus_logit").mean().alias("mean_max_genus_logit"),
        pl.col("mel_1_3k_energy_frac").mean().alias("mean_mel_1_3k_energy_frac"),
        pl.col("pacmap_cluster_distance").mean().alias("mean_pacmap_distance"),
    ).sort("in_pacmap_mad_z2")
    table = coords.select(
        "filename",
        "start_sec",
        "end_sec",
        "relevance_label",
        "in_circle_r3_at_minus4_0",
        "in_pacmap_mad_z2",
        "pacmap_cluster_distance",
        "pacmap_cluster_robust_z",
        "in_embedding_mad_z2",
        "embedding_cluster_distance",
        "embedding_cluster_robust_z",
        "max_genus_logit",
        "target_logit",
        "mel_1_3k_energy_frac",
        "mel_centroid_hz",
        "max_genus_label",
    ).sort("embedding_cluster_distance")
    return f"""# Notebook: Marbled White-lipped Frog Window Relevance Labels

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| Label | `22967` |
| Species | Marbled White-lipped Frog / *Leptodactylus elenae* |
| Upstream code | `data_analysis.marbled_frog.frequency_perch.compute_window_scores` |
| Output directory | `{output_dir}` |

## Heuristic label rule

A window is labeled `likely_relevant` if both conditions hold:

- `max_genus_logit >= {RELEVANT_LOGIT_THRESHOLD}` where genus is Perch labels starting with `Leptodactylus`, and
- `mel_1_3k_energy_frac >= {RELEVANT_BAND_ENERGY_THRESHOLD}`.

Otherwise it is labeled `likely_not_relevant`.

This is a weak/pseudo label, not ground truth.

## Counts

```text
{counts.write_csv(separator='|').strip()}
```

## Embedding-space MAD boundary

The circle at `{PLOT_CIRCLE_CENTER}` with radius `{PLOT_CIRCLE_RADIUS}` is used only as a **seed selection in this plotted PaCMAP view**. From those seed windows, the code computes a centroid in standardized/PCA-compressed Perch embedding space, then defines a robust radius:

```text
radius = median(seed distances) + {ROBUST_Z_THRESHOLD} * 1.4826 * MAD(seed distances)
```

Boundary summary:

```text
{boundary_summary.write_csv(separator='|').strip()}
```

PaCMAP-coordinate membership counts:

```text
{pacmap_counts.write_csv(separator='|').strip()}
```

Embedding-space membership counts:

```text
{robust_counts.write_csv(separator='|').strip()}
```

## Artifacts

Generated artifacts are organized by type:

- `plots/window_relevance_pacmap.png` — Perch embedding PaCMAP of all 33 windows, colored by heuristic relevance. Dashed circle shows the seed region; gold outlines show embedding-space MAD z≤2 members.
- `plots/outside_cluster_kmeans_pacmap.png` — points outside the selected cluster, split into two PaCMAP-space k-means groups.
- `plots/outside_cluster_kmeans_spectrogram_grid.png` — spectrogram grid for outside-cluster windows, max 4 columns.
- `plots/inside_cluster_spectrogram_grid.png` — spectrogram grid for inside-cluster windows, max 4 columns.
- `plots/best_window_per_recording_spectrogram_grid.png` — highest relevance-score window from each train recording.
- `plots/peak_frequency_by_recording.png` — detected spectral peaks across all Marbled Frog train recordings.
- `plots/spectrogram_samples_relevant_vs_not.png` — 4 high-confidence relevant and 4 low-confidence not-relevant mel spectrograms.
- `spectra/*_frequency_spectra.png` — spectroscopy-style mean log-power line plots per recording.
- `tables/*.csv` — window labels, cluster membership, peak summaries, and boundary diagnostics.

## Window labels and distances

```text
{table.write_csv(separator='|').strip()}
```
"""


def run(output_dir: Path | None = None, *, seed: int = 42) -> Path:
    """Run window relevance analysis."""
    output_dir = output_dir or artifact_dir("marbled_frog/window_relevance")
    plots_dir = fresh_subdir(output_dir, "plots")
    tables_dir = fresh_subdir(output_dir, "tables")
    spectra_dir = fresh_subdir(output_dir, "spectra")

    scores, _ctx = compute_window_scores(label=DEFAULT_LABEL)
    labels = add_relevance_labels(scores)
    coords = add_circle_membership(build_pacmap(labels, seed=seed))
    coords, pacmap_boundary_summary = add_pacmap_mad_boundary(coords)
    coords, embedding_boundary_summary = add_embedding_mad_boundary(coords)
    coords = add_outside_cluster_kmeans(coords, seed=seed)
    boundary_summary = pl.concat(
        [pacmap_boundary_summary, embedding_boundary_summary], how="diagonal_relaxed"
    )

    labels.write_csv(tables_dir / "window_relevance_labels.csv")
    coords.write_csv(tables_dir / "window_relevance_pacmap.csv")
    boundary_summary.write_csv(tables_dir / "embedding_mad_boundary_summary.csv")
    coords.filter(pl.col("in_circle_r3_at_minus4_0")).write_csv(tables_dir / "seed_circle_members.csv")
    coords.filter(pl.col("in_embedding_mad_z2")).write_csv(tables_dir / "embedding_mad_z2_members.csv")
    coords.filter(pl.col("in_pacmap_mad_z2")).write_csv(tables_dir / "inside_cluster_windows.csv")
    coords.filter(~pl.col("in_pacmap_mad_z2")).write_csv(tables_dir / "outside_cluster_kmeans_windows.csv")
    plot_pacmap(coords, plots_dir / "window_relevance_pacmap.png")
    plot_outside_kmeans_pacmap(coords, plots_dir / "outside_cluster_kmeans_pacmap.png")
    plot_outside_kmeans_spectrogram_grid(coords, plots_dir / "outside_cluster_kmeans_spectrogram_grid.png")
    plot_inside_cluster_spectrogram_grid(coords, plots_dir / "inside_cluster_spectrogram_grid.png")
    best_windows = plot_best_window_per_recording_grid(
        coords, plots_dir / "best_window_per_recording_spectrogram_grid.png"
    )
    best_windows.write_csv(tables_dir / "best_window_per_recording.csv")
    plot_all_recording_frequency_spectra(coords, spectra_dir, plots_dir, tables_dir)
    plot_spectrogram_samples(labels, plots_dir / "spectrogram_samples_relevant_vs_not.png")
    (output_dir / "summary.md").write_text(render_summary(labels, coords, boundary_summary, output_dir))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = run(args.output_dir, seed=args.seed)
    print(f"Wrote Marbled Frog relevance analysis to {output_dir}")


if __name__ == "__main__":
    main()
