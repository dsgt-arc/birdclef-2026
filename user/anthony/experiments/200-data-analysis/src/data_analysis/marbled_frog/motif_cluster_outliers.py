"""Pairwise template-similarity outlier review for motif-mining clusters.

Reads motif candidates from a motif_mining run, extracts a normalized crop for
selected clusters, computes all-pairs shift-tolerant normalized cross-correlation
within each cluster, and emits outlier scores for human review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import polars as pl
from scipy.ndimage import zoom

from data_analysis.common.paths import TRAIN_AUDIO_DIR, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)

PATCH_SHAPE = (64, 48)
DEFAULT_CLUSTERS = (1, 5, 6, 7)
DEFAULT_CANDIDATES = Path("artifacts/marbled_frog/motif_mining_maad_loose/tables/candidates.csv")
DEFAULT_OUTPUT = Path("artifacts/marbled_frog/motif_mining_maad_loose/outlier_review")


def normalized_crop(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    row: dict,
    *,
    time_pad_sec: float = 0.12,
    freq_pad_hz: float = 180.0,
) -> np.ndarray:
    """Extract a row-normalized, fixed-size crop around one candidate box."""
    t0 = max(0.0, float(row["start_sec"]) - time_pad_sec)
    t1 = min(float(times[-1]), float(row["end_sec"]) + time_pad_sec)
    f0 = max(DISPLAY_BAND[0], float(row["fmin_hz"]) - freq_pad_hz)
    f1 = min(DISPLAY_BAND[1], float(row["fmax_hz"]) + freq_pad_hz)
    t_mask = (times >= t0) & (times <= t1)
    f_mask = (freqs >= f0) & (freqs <= f1)
    if not np.any(t_mask) or not np.any(f_mask):
        return np.zeros(PATCH_SHAPE, dtype=np.float32)
    patch = z[f_mask][:, t_mask].astype(np.float32)
    if patch.size == 0:
        return np.zeros(PATCH_SHAPE, dtype=np.float32)
    patch = patch - np.median(patch, axis=1, keepdims=True)
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    resized = zoom(patch, (PATCH_SHAPE[0] / patch.shape[0], PATCH_SHAPE[1] / patch.shape[1]), order=1)
    out = np.zeros(PATCH_SHAPE, dtype=np.float32)
    h = min(PATCH_SHAPE[0], resized.shape[0])
    w = min(PATCH_SHAPE[1], resized.shape[1])
    out[:h, :w] = resized[:h, :w]
    out = (out - out.mean()) / (out.std() + 1e-6)
    return out.astype(np.float32)


def shift_score(a: np.ndarray, b: np.ndarray, *, max_freq_shift: int = 4, max_time_shift: int = 4) -> float:
    """Return max normalized cross-correlation over small time/frequency shifts."""
    best = -1.0
    h, w = a.shape
    for dy in range(-max_freq_shift, max_freq_shift + 1):
        if dy >= 0:
            ay = slice(dy, h)
            by = slice(0, h - dy)
        else:
            ay = slice(0, h + dy)
            by = slice(-dy, h)
        for dx in range(-max_time_shift, max_time_shift + 1):
            if dx >= 0:
                ax = slice(dx, w)
                bx = slice(0, w - dx)
            else:
                ax = slice(0, w + dx)
                bx = slice(-dx, w)
            aa = a[ay, ax]
            bb = b[by, bx]
            if aa.size < 16:
                continue
            av = (aa - aa.mean()) / (aa.std() + 1e-6)
            bv = (bb - bb.mean()) / (bb.std() + 1e-6)
            score = float(np.mean(av * bv))
            if score > best:
                best = score
    return best


def load_recording_spectrograms(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load each needed recording once."""
    cache = {}
    for filename in sorted({str(row["filename"]) for row in rows}):
        audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
        cache[filename] = robust_excess_spectrogram(audio)
    return cache


def compute_cluster_scores(candidates: pl.DataFrame, *, clusters: tuple[int, ...]) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, np.ndarray]]:
    """Compute pairwise similarities and per-candidate outlier scores."""
    selected = candidates.filter(pl.col("motif_cluster").is_in(list(clusters))).sort("motif_cluster", "filename", "start_sec")
    rows = selected.to_dicts()
    spec_cache = load_recording_spectrograms(rows)

    patches = []
    for row in rows:
        _db, z, freqs, times = spec_cache[str(row["filename"])]
        patches.append(normalized_crop(z, freqs, times, row))

    row_by_id = {int(row["candidate_id"]): row for row in rows}
    patch_by_id = {int(row["candidate_id"]): patch for row, patch in zip(rows, patches)}

    pair_rows = []
    score_rows = []
    matrices: dict[int, np.ndarray] = {}
    for cluster in clusters:
        ids = [int(row["candidate_id"]) for row in rows if int(row["motif_cluster"]) == cluster]
        n = len(ids)
        if n == 0:
            continue
        sim = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                s = shift_score(patch_by_id[ids[i]], patch_by_id[ids[j]])
                sim[i, j] = sim[j, i] = s
                pair_rows.append(
                    {
                        "motif_cluster": cluster,
                        "candidate_id_a": ids[i],
                        "candidate_id_b": ids[j],
                        "similarity": s,
                        "distance": 1.0 - s,
                    }
                )
        matrices[cluster] = sim
        off_diag = sim[~np.eye(n, dtype=bool)] if n > 1 else np.array([1.0], dtype=np.float32)
        cluster_median = float(np.median(off_diag))
        cluster_mad = float(np.median(np.abs(off_diag - cluster_median)))
        denom = 1.4826 * max(cluster_mad, 1e-6)
        medoid_idx = int(np.argmax((sim.sum(axis=1) - 1.0) / max(1, n - 1)))
        for i, candidate_id in enumerate(ids):
            others = np.delete(sim[i], i) if n > 1 else np.array([1.0], dtype=np.float32)
            k = min(5, others.size)
            knn = np.sort(others)[-k:]
            median_similarity = float(np.median(others))
            mean_similarity = float(np.mean(others))
            knn5_similarity = float(np.mean(knn))
            medoid_similarity = float(sim[i, medoid_idx]) if i != medoid_idx else 1.0
            low_similarity_z = (cluster_median - median_similarity) / denom
            # Relative outlierness alone can over-flag valid variants when a
            # cluster is dominated by near-duplicates from one recording. Only
            # auto-drop candidates that are both relatively isolated and low in
            # absolute pairwise-template similarity. High-absolute-similarity
            # cases remain likely valid even if their robust-z looks large.
            drop_candidate = low_similarity_z >= 2.5 and median_similarity < 0.65
            check_candidate = (low_similarity_z >= 2.5 and median_similarity < 0.75) or median_similarity < 0.45
            row = row_by_id[candidate_id]
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "filename": row["filename"],
                    "motif_cluster": cluster,
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "fmin_hz": row["fmin_hz"],
                    "fmax_hz": row["fmax_hz"],
                    "peak_z": row["peak_z"],
                    "cluster_n": n,
                    "cluster_pairwise_median_similarity": cluster_median,
                    "candidate_median_similarity": median_similarity,
                    "candidate_mean_similarity": mean_similarity,
                    "candidate_knn5_similarity": knn5_similarity,
                    "candidate_medoid_similarity": medoid_similarity,
                    "low_similarity_z": low_similarity_z,
                    "template_outlier_flag": drop_candidate,
                    "review_priority": "drop_candidate" if drop_candidate else ("check" if check_candidate else "keep_likely"),
                }
            )
    return pl.DataFrame(score_rows), pl.DataFrame(pair_rows), matrices


def plot_distance_matrix(scores: pl.DataFrame, matrices: dict[int, np.ndarray], output_dir: Path) -> None:
    """Save one pairwise similarity heatmap per cluster."""
    for cluster, sim in matrices.items():
        cluster_scores = scores.filter(pl.col("motif_cluster") == cluster).sort("candidate_median_similarity")
        ids_sorted = cluster_scores["candidate_id"].to_list()
        all_ids = scores.filter(pl.col("motif_cluster") == cluster).sort("filename", "start_sec")["candidate_id"].to_list()
        order = [all_ids.index(candidate_id) for candidate_id in ids_sorted]
        sim_sorted = sim[np.ix_(order, order)]
        fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=180)
        im = ax.imshow(sim_sorted, vmin=-0.2, vmax=1.0, cmap="viridis")
        ax.set_title(f"Cluster {cluster}: pairwise shift-template similarity\nlowest-similarity candidates first")
        ax.set_xticks(range(len(ids_sorted)))
        ax.set_yticks(range(len(ids_sorted)))
        ax.set_xticklabels(ids_sorted, rotation=90, fontsize=6)
        ax.set_yticklabels(ids_sorted, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{cluster:02d}_pairwise_similarity.png")
        plt.close(fig)


def plot_outlier_grid(scores: pl.DataFrame, output_dir: Path, *, max_items: int = 16) -> None:
    """Render candidate grids sorted by outlierness for each cluster."""
    for cluster in sorted(scores["motif_cluster"].unique().to_list()):
        df = scores.filter(pl.col("motif_cluster") == cluster).sort("candidate_median_similarity").head(max_items)
        if df.is_empty():
            continue
        n_items = df.height
        n_cols = min(4, n_items)
        n_rows = int(np.ceil(n_items / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.2 * n_rows), dpi=180, sharey=True, squeeze=False)
        for idx, row in enumerate(df.iter_rows(named=True)):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            pad = 0.18
            offset = max(0.0, float(row["start_sec"]) - pad)
            duration = float(row["end_sec"] - row["start_sec"] + 2 * pad)
            audio, _ = librosa.load(TRAIN_AUDIO_DIR / row["filename"], sr=MEL_SAMPLE_RATE, mono=True, offset=offset, duration=duration)
            db_band, _z, freqs, times = robust_excess_spectrogram(audio)
            display_times = times + offset
            librosa.display.specshow(db_band, x_coords=display_times, y_coords=freqs, x_axis="time", y_axis="linear", sr=MEL_SAMPLE_RATE, hop_length=MEL_HOP_LENGTH, ax=ax, cmap="magma")
            color = "red" if row["review_priority"] == "drop_candidate" else ("orange" if row["review_priority"] == "check" else "lime")
            ax.add_patch(Rectangle((row["start_sec"], row["fmin_hz"]), row["end_sec"] - row["start_sec"], row["fmax_hz"] - row["fmin_hz"], fill=False, edgecolor=color, linewidth=2.0))
            ax.set_ylim(DISPLAY_BAND)
            ax.set_title(
                f"id={row['candidate_id']} {Path(row['filename']).stem}\n"
                f"med_sim={row['candidate_median_similarity']:.2f} z={row['low_similarity_z']:.1f} {row['review_priority']}",
                fontsize=7,
            )
            if c != 0:
                ax.set_ylabel("")
        for idx in range(n_items, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].axis("off")
        fig.suptitle(f"Cluster {cluster}: lowest pairwise-template similarity first", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{cluster:02d}_outlier_grid.png")
        plt.close(fig)


def run(candidates_path: Path = DEFAULT_CANDIDATES, output_dir: Path = DEFAULT_OUTPUT, *, clusters: tuple[int, ...] = DEFAULT_CLUSTERS) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = fresh_subdir(output_dir, "plots")
    tables_dir = fresh_subdir(output_dir, "tables")
    candidates = pl.read_csv(candidates_path)
    scores, pairs, matrices = compute_cluster_scores(candidates, clusters=clusters)
    scores.sort("motif_cluster", "candidate_median_similarity").write_csv(tables_dir / "cluster_outlier_scores.csv")
    pairs.sort("motif_cluster", "distance", descending=[False, True]).write_csv(tables_dir / "pairwise_template_scores.csv")
    plot_distance_matrix(scores, matrices, plots_dir)
    plot_outlier_grid(scores, plots_dir)
    summary = scores.group_by("motif_cluster").agg(
        pl.len().alias("n_candidates"),
        (pl.col("review_priority") == "drop_candidate").sum().alias("n_drop_candidate"),
        (pl.col("review_priority") == "check").sum().alias("n_check"),
        pl.col("candidate_median_similarity").median().alias("median_candidate_similarity"),
        pl.col("candidate_median_similarity").min().alias("min_candidate_similarity"),
    ).sort("motif_cluster")
    summary.write_csv(tables_dir / "cluster_outlier_summary.csv")
    (output_dir / "summary.md").write_text(
        f"""# Motif cluster outlier review

Input candidates: `{candidates_path}`

Clusters reviewed: {', '.join(str(c) for c in clusters)}

Method: extract normalized robust-excess-z crop for each candidate, compute all-pairs
shift-tolerant normalized cross-correlation within each cluster, and flag candidates
whose median within-cluster similarity is a robust low outlier.

Artifacts:

- `tables/cluster_outlier_scores.csv` — one row per candidate with similarity/outlier fields.
- `tables/pairwise_template_scores.csv` — all pairwise template similarities.
- `tables/cluster_outlier_summary.csv` — counts per cluster.
- `plots/cluster_XX_pairwise_similarity.png` — pairwise similarity matrix.
- `plots/cluster_XX_outlier_grid.png` — lowest-similarity candidates first.

Summary:

```text
{summary.write_csv(separator='|').strip()}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clusters", type=int, nargs="+", default=list(DEFAULT_CLUSTERS))
    args = parser.parse_args()
    output_dir = run(args.candidates, args.output_dir, clusters=tuple(args.clusters))
    print(f"Wrote motif cluster outlier review to {output_dir}")


if __name__ == "__main__":
    main()
