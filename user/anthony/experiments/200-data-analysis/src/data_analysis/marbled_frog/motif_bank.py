"""Build a simple motif bank from simplified motif-mining clusters.

The bank is intentionally small and defensible:

- Input clusters come from ``motif_mining.py`` maad-feature + HDBSCAN output.
- A motif-family template set is the feature-space medoid plus nearest neighbors.
- Every template in a family is extracted with the same physical support:
  canonical pre/post time around the candidate peak and a canonical frequency band.
- Original maad ROI boxes are kept only as provenance.
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
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from data_analysis.common.paths import TRAIN_AUDIO_DIR, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)

DEFAULT_INPUT_DIR = Path("artifacts/marbled_frog/motif_mining_simplified")
DEFAULT_OUTPUT = Path("artifacts/marbled_frog/motif_mining_simplified/motif_bank")
TEMPLATE_SHAPE = (64, 48)


def canonical_support(members: pl.DataFrame) -> dict[str, float]:
    """Estimate fixed physical support for one motif family from member boxes.

    Time is anchored at each member's ``peak_time_sec``. Frequency support uses
    robust quantiles of member ROI bounds. This keeps the template extent stable
    while still being derived from detections.
    """
    pre = (members["peak_time_sec"] - members["start_sec"]).to_numpy()
    post = (members["end_sec"] - members["peak_time_sec"]).to_numpy()
    fmins = members["fmin_hz"].to_numpy()
    fmaxs = members["fmax_hz"].to_numpy()

    pre_sec = float(np.clip(np.quantile(pre, 0.75), 0.04, 0.75))
    post_sec = float(np.clip(np.quantile(post, 0.75), 0.04, 0.75))
    fmin_hz = float(np.clip(np.quantile(fmins, 0.25), DISPLAY_BAND[0], DISPLAY_BAND[1]))
    fmax_hz = float(np.clip(np.quantile(fmaxs, 0.75), DISPLAY_BAND[0], DISPLAY_BAND[1]))
    if fmax_hz - fmin_hz < 80.0:
        center = 0.5 * (fmin_hz + fmax_hz)
        fmin_hz = max(DISPLAY_BAND[0], center - 40.0)
        fmax_hz = min(DISPLAY_BAND[1], center + 40.0)
    return {
        "template_pre_sec": pre_sec,
        "template_post_sec": post_sec,
        "template_duration_sec": pre_sec + post_sec,
        "template_fmin_hz": fmin_hz,
        "template_fmax_hz": fmax_hz,
        "template_bandwidth_hz": fmax_hz - fmin_hz,
    }


def fixed_support_crop(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    peak_time_sec: float,
    support: dict[str, float],
    shape: tuple[int, int] = TEMPLATE_SHAPE,
) -> np.ndarray:
    """Extract a fixed-physical-support crop anchored at peak_time_sec."""
    t0 = max(0.0, peak_time_sec - support["template_pre_sec"])
    t1 = min(float(times[-1]), peak_time_sec + support["template_post_sec"])
    f0 = support["template_fmin_hz"]
    f1 = support["template_fmax_hz"]
    t_mask = (times >= t0) & (times <= t1)
    f_mask = (freqs >= f0) & (freqs <= f1)
    if not np.any(t_mask) or not np.any(f_mask):
        return np.zeros(shape, dtype=np.float32)
    patch = z[f_mask][:, t_mask].astype(np.float32)
    # Simple per-template normalization; no row-median subtraction so pitch/band
    # structure is preserved inside the canonical physical support.
    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
    resized = zoom(patch, (shape[0] / patch.shape[0], shape[1] / patch.shape[1]), order=1)
    out = np.zeros(shape, dtype=np.float32)
    h = min(shape[0], resized.shape[0])
    w = min(shape[1], resized.shape[1])
    out[:h, :w] = resized[:h, :w]
    return out.astype(np.float32)


def select_bank_members(
    members: pl.DataFrame,
    feature_matrix: pl.DataFrame,
    *,
    library_size: int,
) -> tuple[int, list[int], np.ndarray]:
    """Select feature-space medoid and nearest neighbors."""
    ids = [int(x) for x in members["candidate_id"].to_list()]
    feature_cols = [c for c in feature_matrix.columns if c != "candidate_id"]
    x = (
        feature_matrix.filter(pl.col("candidate_id").is_in(ids))
        .sort("candidate_id")
        .select(feature_cols)
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float32)
    )
    sorted_ids = [int(x) for x in feature_matrix.filter(pl.col("candidate_id").is_in(ids)).sort("candidate_id")["candidate_id"].to_list()]
    if len(sorted_ids) == 1:
        return sorted_ids[0], sorted_ids, np.zeros((1, 1), dtype=np.float32)
    x = StandardScaler().fit_transform(x)
    distances = pairwise_distances(x, metric="euclidean")
    medoid_idx = int(np.argmin(distances.sum(axis=1)))
    nearest = np.argsort(distances[medoid_idx])[: min(library_size, len(sorted_ids))]
    return sorted_ids[medoid_idx], [sorted_ids[i] for i in nearest], distances


def extract_templates(
    members: pl.DataFrame,
    library_ids: list[int],
    support: dict[str, float],
) -> np.ndarray:
    """Extract fixed-support templates for selected candidate IDs."""
    templates = []
    cache = {}
    for candidate_id in library_ids:
        row = members.filter(pl.col("candidate_id") == candidate_id).to_dicts()[0]
        filename = row["filename"]
        if filename not in cache:
            audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
            cache[filename] = robust_excess_spectrogram(audio)
        _db, z, freqs, times = cache[filename]
        templates.append(
            fixed_support_crop(
                z,
                freqs,
                times,
                peak_time_sec=float(row["peak_time_sec"]),
                support=support,
            )
        )
    return np.stack(templates).astype(np.float32)


def build_bank(
    candidates: pl.DataFrame,
    feature_matrix: pl.DataFrame,
    *,
    approved_clusters: tuple[int, ...] | None,
    library_size: int,
    templates_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build motif-bank cluster/member tables and template NPZ files."""
    if approved_clusters:
        clusters = list(approved_clusters)
        acceptance_source = "human_approved_cluster"
    else:
        clusters = sorted(c for c in candidates["motif_cluster"].unique().to_list() if int(c) >= 0)
        acceptance_source = "all_non_noise_for_review"

    cluster_rows = []
    member_rows = []
    for cluster in clusters:
        members = candidates.filter(pl.col("motif_cluster") == cluster).sort("candidate_id")
        if members.is_empty():
            continue
        support = canonical_support(members)
        medoid_id, library_ids, _distances = select_bank_members(
            members, feature_matrix, library_size=library_size
        )
        templates = extract_templates(members, library_ids, support)
        np.savez_compressed(
            templates_dir / f"cluster_{int(cluster):02d}_templates.npz",
            templates=templates,
            candidate_ids=np.array(library_ids, dtype=np.int64),
            medoid_candidate_id=np.array([medoid_id], dtype=np.int64),
            template_pre_sec=np.array([support["template_pre_sec"]], dtype=np.float32),
            template_post_sec=np.array([support["template_post_sec"]], dtype=np.float32),
            template_fmin_hz=np.array([support["template_fmin_hz"]], dtype=np.float32),
            template_fmax_hz=np.array([support["template_fmax_hz"]], dtype=np.float32),
        )

        cluster_rows.append(
            {
                "motif_cluster": int(cluster),
                "accepted": True,
                "acceptance_source": acceptance_source,
                "n_candidates": members.height,
                "n_recordings": members["filename"].n_unique(),
                "n_templates": len(library_ids),
                "medoid_candidate_id": medoid_id,
                "library_candidate_ids": ";".join(str(x) for x in library_ids),
                "median_cluster_probability": float(members["cluster_probability"].median()),
                "median_peak_z": float(members["peak_z"].median()),
                "median_duration_sec": float(members["duration_sec"].median()),
                "median_bandwidth_hz": float(members["bandwidth_hz"].median()),
                **support,
            }
        )
        for row in members.iter_rows(named=True):
            candidate_id = int(row["candidate_id"])
            if candidate_id == medoid_id:
                role = "medoid"
                rank = 0
            elif candidate_id in library_ids:
                role = "exemplar"
                rank = library_ids.index(candidate_id)
            else:
                role = "member"
                rank = None
            member_rows.append({**row, **support, "bank_role": role, "bank_rank": rank})

        members.filter(pl.col("candidate_id").is_in(library_ids)).sort("candidate_id").write_csv(
            templates_dir / f"cluster_{int(cluster):02d}_template_metadata.csv"
        )
    return pl.DataFrame(cluster_rows), pl.DataFrame(member_rows)


def plot_bank_grid(members: pl.DataFrame, output_dir: Path, *, max_items: int = 8) -> None:
    """Plot medoid/exemplar grids using fixed template support and original ROI provenance."""
    for cluster in sorted(members["motif_cluster"].unique().to_list()):
        df = (
            members.filter(
                (pl.col("motif_cluster") == cluster)
                & (pl.col("bank_role").is_in(["medoid", "exemplar"]))
            )
            .sort("bank_rank")
            .head(max_items)
        )
        if df.is_empty():
            continue
        n_items = df.height
        n_cols = min(4, n_items)
        n_rows = int(np.ceil(n_items / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.3 * n_rows),
            dpi=180,
            sharey=True,
            squeeze=False,
        )
        for idx, row in enumerate(df.iter_rows(named=True)):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            template_start = max(0.0, row["peak_time_sec"] - row["template_pre_sec"])
            template_end = row["peak_time_sec"] + row["template_post_sec"]
            pad = 0.08
            offset = max(0.0, template_start - pad)
            duration = template_end - template_start + 2 * pad
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
            # Canonical template support: solid cyan, same size for all rows in cluster.
            ax.add_patch(
                Rectangle(
                    (template_start, row["template_fmin_hz"]),
                    row["template_duration_sec"],
                    row["template_bandwidth_hz"],
                    fill=False,
                    edgecolor="cyan",
                    linewidth=2.2,
                )
            )
            # Raw maad ROI provenance: dashed lime, may vary by candidate.
            ax.add_patch(
                Rectangle(
                    (row["start_sec"], row["fmin_hz"]),
                    row["end_sec"] - row["start_sec"],
                    row["fmax_hz"] - row["fmin_hz"],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.85,
                )
            )
            ax.set_ylim(DISPLAY_BAND)
            ax.set_title(
                f"{row['bank_role']} id={row['candidate_id']} {Path(row['filename']).stem}\n"
                f"cyan=fixed support, green=ROI, p={row['cluster_probability']:.2f}",
                fontsize=7,
            )
            if c != 0:
                ax.set_ylabel("")
        for idx in range(n_items, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].axis("off")
        fig.suptitle(f"Cluster {cluster}: fixed-support motif bank", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{int(cluster):02d}_motif_bank.png")
        plt.close(fig)


def run(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    approved_clusters: tuple[int, ...] | None = None,
    library_size: int = 4,
) -> Path:
    """Build a simple fixed-support motif bank."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = fresh_subdir(output_dir, "tables")
    plots_dir = fresh_subdir(output_dir, "plots")
    templates_dir = fresh_subdir(output_dir, "templates")

    candidates = pl.read_csv(input_dir / "tables" / "candidates.csv")
    feature_matrix = pl.read_csv(input_dir / "tables" / "feature_matrix.csv")
    bank_clusters, bank_members = build_bank(
        candidates,
        feature_matrix,
        approved_clusters=approved_clusters,
        library_size=library_size,
        templates_dir=templates_dir,
    )
    bank_clusters.write_csv(tables_dir / "motif_bank_clusters.csv")
    bank_members.write_csv(tables_dir / "motif_bank_members.csv")
    plot_bank_grid(bank_members, plots_dir)

    (output_dir / "summary.md").write_text(
        f"""# Simplified Marbled Frog Motif Bank

Input motif-mining run: `{input_dir}`

Acceptance source: `{bank_clusters['acceptance_source'][0] if bank_clusters.height else 'none'}`

Method:

1. Use HDBSCAN clusters from maad shape/centroid feature space.
2. For each reviewed cluster, compute a canonical physical template support from
   robust ROI quantiles.
3. Select feature-space medoid plus nearest neighbors.
4. Extract all templates in a family with the same physical support. Original maad
   ROI boxes remain provenance only.

Plot convention: cyan = fixed family template support; dashed green = raw maad ROI.

Artifacts:

- `tables/motif_bank_clusters.csv` — family-level canonical support and medoids.
- `tables/motif_bank_members.csv` — candidate membership and bank roles.
- `templates/cluster_XX_templates.npz` — fixed-support template arrays.
- `plots/cluster_XX_motif_bank.png` — medoid/exemplar review grids.

Motif bank clusters:

```text
{bank_clusters.write_csv(separator='|').strip() if bank_clusters.height else 'none'}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approved-clusters", type=int, nargs="*", default=None)
    parser.add_argument("--library-size", type=int, default=4)
    args = parser.parse_args()
    output_dir = run(
        args.input_dir,
        args.output_dir,
        approved_clusters=None if args.approved_clusters is None else tuple(args.approved_clusters),
        library_size=args.library_size,
    )
    print(f"Wrote simplified motif bank to {output_dir}")


if __name__ == "__main__":
    main()
