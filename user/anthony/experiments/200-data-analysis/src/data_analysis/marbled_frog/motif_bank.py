"""Build a reviewed motif bank from motif-mining clusters.

The bank has two layers:

1. Cluster-level validity metrics/suggestions, computed from candidate metadata and
   pairwise template similarity.
2. A small template library per accepted motif cluster: medoid plus central
   high-quality exemplars selected from non-dropped, spectrally plausible candidates.
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

from data_analysis.common.paths import TRAIN_AUDIO_DIR, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)
from data_analysis.marbled_frog.motif_cluster_outliers import normalized_crop

DEFAULT_CANDIDATES = Path("artifacts/marbled_frog/motif_mining_maad_loose/tables/candidates.csv")
DEFAULT_OUTLIER_SCORES = Path(
    "artifacts/marbled_frog/motif_mining_maad_loose/all_cluster_similarity/tables/cluster_outlier_scores.csv"
)
DEFAULT_PAIRWISE = Path(
    "artifacts/marbled_frog/motif_mining_maad_loose/all_cluster_similarity/tables/pairwise_template_scores.csv"
)
DEFAULT_OUTPUT = Path("artifacts/marbled_frog/motif_mining_maad_loose/motif_bank")
DEFAULT_SPECTRAL_PRIORS = Path(
    "artifacts/marbled_frog/motif_mining_maad_loose/spectral_prior/tables/candidate_spectral_priors.csv"
)
DEFAULT_APPROVED_CLUSTERS = (1, 5, 6, 7)
MIN_TEMPLATE_SPECTRAL_PRIOR = 0.58


def cluster_validity_table(
    candidates: pl.DataFrame, scores: pl.DataFrame, spectral_priors: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Summarize cluster-level motif-likelihood diagnostics."""
    candidate_summary = candidates.group_by("motif_cluster").agg(
        pl.len().alias("n_candidates"),
        pl.col("filename").n_unique().alias("n_recordings"),
        pl.col("duration_sec").median().alias("median_duration_sec"),
        pl.col("bandwidth_hz").median().alias("median_bandwidth_hz"),
        pl.col("fmin_hz").median().alias("median_fmin_hz"),
        pl.col("fmax_hz").median().alias("median_fmax_hz"),
        pl.col("peak_z").median().alias("median_peak_z"),
        pl.col("area_pixels").median().alias("median_area_pixels"),
    )

    recording_conc = (
        candidates.group_by("motif_cluster", "filename")
        .len()
        .join(candidate_summary.select("motif_cluster", "n_candidates"), on="motif_cluster")
        .with_columns((pl.col("len") / pl.col("n_candidates")).alias("recording_fraction"))
        .group_by("motif_cluster")
        .agg(pl.col("recording_fraction").max().alias("max_recording_fraction"))
    )

    score_summary = scores.group_by("motif_cluster").agg(
        pl.col("candidate_median_similarity").median().alias("median_similarity"),
        pl.col("candidate_median_similarity").min().alias("min_similarity"),
        (pl.col("review_priority") == "drop_candidate").mean().alias("drop_rate"),
        (pl.col("review_priority") == "check").mean().alias("check_rate"),
    )

    summary = candidate_summary.join(recording_conc, on="motif_cluster").join(score_summary, on="motif_cluster")
    if spectral_priors is not None:
        spectral_summary = spectral_priors.group_by("motif_cluster").agg(
            pl.col("recording_aware_spectral_prior_score").median().alias("median_recording_spectral_prior"),
            pl.col("recording_aware_spectral_prior_score").min().alias("min_recording_spectral_prior"),
            (pl.col("recording_aware_prior_label") == "good").mean().alias("spectral_good_rate"),
            (pl.col("recording_aware_prior_label") == "edge_or_misaligned").mean().alias("spectral_edge_misaligned_rate"),
        )
        summary = summary.join(spectral_summary, on="motif_cluster", how="left")
    else:
        summary = summary.with_columns(
            pl.lit(0.5).alias("median_recording_spectral_prior"),
            pl.lit(0.5).alias("min_recording_spectral_prior"),
            pl.lit(0.5).alias("spectral_good_rate"),
            pl.lit(0.0).alias("spectral_edge_misaligned_rate"),
        )

    # Heuristic score. It is intentionally a suggestion, not a ground-truth label:
    # cohesive recurring spectrogram shapes can still be nuisance/background families.
    summary = summary.with_columns(
        (pl.min_horizontal(pl.col("n_candidates") / 20.0, pl.lit(1.0))).alias("count_score"),
        (pl.min_horizontal(pl.col("n_recordings") / 4.0, pl.lit(1.0))).alias("recording_score"),
        (((pl.col("median_similarity") - 0.25) / 0.65).clip(0.0, 1.0)).alias("cohesion_score"),
        (((pl.col("median_peak_z") - 3.2) / 3.0).clip(0.0, 1.0)).alias("salience_score"),
        (
            pl.when((pl.col("median_duration_sec") >= 0.04) & (pl.col("median_duration_sec") <= 0.8))
            .then(1.0)
            .otherwise(0.4)
        ).alias("duration_score"),
        (
            pl.when((pl.col("median_bandwidth_hz") >= 80.0) & (pl.col("median_bandwidth_hz") <= 1800.0))
            .then(1.0)
            .otherwise(0.5)
        ).alias("bandwidth_score"),
        (1.0 - pl.col("drop_rate")).alias("non_outlier_score"),
        (1.0 - (pl.col("max_recording_fraction") - 0.75).clip(0.0, 0.25) / 0.25 * 0.25).alias("recording_balance_score"),
        pl.col("median_recording_spectral_prior").fill_null(0.5).clip(0.0, 1.0).alias("spectral_prior_score"),
    ).with_columns(
        (
            0.15 * pl.col("count_score")
            + 0.10 * pl.col("recording_score")
            + 0.25 * pl.col("cohesion_score")
            + 0.15 * pl.col("salience_score")
            + 0.10 * pl.col("duration_score")
            + 0.10 * pl.col("bandwidth_score")
            + 0.10 * pl.col("non_outlier_score")
            + 0.05 * pl.col("recording_balance_score")
        ).alias("template_only_likelihood_score"),
        (
            0.11 * pl.col("count_score")
            + 0.08 * pl.col("recording_score")
            + 0.16 * pl.col("cohesion_score")
            + 0.11 * pl.col("salience_score")
            + 0.08 * pl.col("duration_score")
            + 0.08 * pl.col("bandwidth_score")
            + 0.08 * pl.col("non_outlier_score")
            + 0.05 * pl.col("recording_balance_score")
            + 0.25 * pl.col("spectral_prior_score")
        ).alias("motif_likelihood_score")
    ).with_columns(
        (
            (pl.col("motif_likelihood_score") >= 0.62)
            & (pl.col("median_recording_spectral_prior") >= 0.58)
            & (pl.col("spectral_edge_misaligned_rate") <= 0.45)
            & (pl.col("drop_rate") <= 0.25)
            & (pl.col("n_candidates") >= 5)
        ).alias("auto_valid_suggested")
    )
    return summary.sort("motif_likelihood_score", descending=True)


def similarity_matrix(ids: list[int], pairs: pl.DataFrame) -> np.ndarray:
    """Build a dense similarity matrix for selected candidate ids."""
    n = len(ids)
    index = {candidate_id: i for i, candidate_id in enumerate(ids)}
    sim = np.eye(n, dtype=np.float32)
    if n <= 1:
        return sim
    pair_df = pairs.filter(
        pl.col("candidate_id_a").is_in(ids) & pl.col("candidate_id_b").is_in(ids)
    )
    for row in pair_df.iter_rows(named=True):
        i = index[int(row["candidate_id_a"])]
        j = index[int(row["candidate_id_b"])]
        sim[i, j] = sim[j, i] = float(row["similarity"])
    return sim


def select_library_ids(
    cluster_scores: pl.DataFrame,
    pairs: pl.DataFrame,
    *,
    library_size: int,
    min_spectral_prior: float = MIN_TEMPLATE_SPECTRAL_PRIOR,
) -> tuple[int, list[int], np.ndarray]:
    """Select medoid and a conservative core template library for one cluster.

    Earlier versions selected far-apart exemplars to maximize coverage. That is
    the wrong objective for a motif bank: it pulls boundary cases and visually
    ugly outliers into the template set. For the bank we want clean prototypes,
    so choose the medoid plus the most central high-similarity candidates.
    """
    usable = cluster_scores.filter(
        (pl.col("review_priority") == "keep_likely")
        & (pl.col("recording_aware_prior_label") == "good")
        & (pl.col("recording_aware_spectral_prior_score") >= min_spectral_prior)
    )
    if usable.is_empty():
        usable = cluster_scores.filter(
            (pl.col("review_priority") != "drop_candidate")
            & (pl.col("recording_aware_spectral_prior_score") >= min_spectral_prior)
        )
    if usable.is_empty():
        usable = cluster_scores.filter(pl.col("review_priority") != "drop_candidate")
    if usable.is_empty():
        usable = cluster_scores
    usable = usable.with_columns(
        (
            0.55 * pl.col("recording_aware_spectral_prior_score").fill_null(0.0)
            + 0.30 * pl.col("candidate_median_similarity").fill_null(0.0)
            + 0.15 * pl.col("candidate_knn5_similarity").fill_null(0.0)
        ).alias("template_quality_score")
    ).sort(
        ["template_quality_score", "recording_aware_spectral_prior_score", "candidate_median_similarity"],
        descending=[True, True, True],
    )
    ids = [int(x) for x in usable["candidate_id"].to_list()]
    sim = similarity_matrix(ids, pairs)
    mean_sim = (sim.sum(axis=1) - 1.0) / np.maximum(1, len(ids) - 1)
    quality = usable["template_quality_score"].to_numpy()
    medoid_idx = int(np.argmax(0.65 * quality + 0.35 * mean_sim))
    medoid_id = ids[medoid_idx]

    central_ids = [int(x) for x in usable["candidate_id"].head(library_size).to_list()]
    if medoid_id not in central_ids:
        central_ids = [medoid_id] + central_ids[: max(0, library_size - 1)]
    else:
        central_ids = [medoid_id] + [candidate_id for candidate_id in central_ids if candidate_id != medoid_id]
    return medoid_id, central_ids[:library_size], sim


def extract_template(row: dict) -> np.ndarray:
    """Extract normalized template crop for one candidate row."""
    audio, _ = librosa.load(TRAIN_AUDIO_DIR / row["filename"], sr=MEL_SAMPLE_RATE, mono=True)
    _db, z, freqs, times = robust_excess_spectrogram(audio)
    return normalized_crop(z, freqs, times, row)


def plot_bank_grid(members: pl.DataFrame, output_dir: Path, *, max_items: int = 12) -> None:
    """Plot selected templates/boxes per motif cluster."""
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
            figsize=(3.8 * n_cols, 3.25 * n_rows),
            dpi=180,
            sharey=True,
            squeeze=False,
        )
        for idx, row in enumerate(df.iter_rows(named=True)):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
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
            color = "lime" if row["bank_role"] == "medoid" else "cyan"
            ax.add_patch(
                Rectangle(
                    (row["start_sec"], row["fmin_hz"]),
                    row["end_sec"] - row["start_sec"],
                    row["fmax_hz"] - row["fmin_hz"],
                    fill=False,
                    edgecolor=color,
                    linewidth=2.0,
                )
            )
            ax.set_ylim(DISPLAY_BAND)
            ax.set_title(
                f"{row['bank_role']} id={row['candidate_id']} {Path(row['filename']).stem}\n"
                f"sim={row['candidate_median_similarity']:.2f} spec={row['recording_aware_spectral_prior_score']:.2f} "
                f"{row['start_sec']:.2f}-{row['end_sec']:.2f}s",
                fontsize=7,
            )
            if c != 0:
                ax.set_ylabel("")
        for idx in range(n_items, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].axis("off")
        fig.suptitle(f"Motif bank cluster {cluster}: spectral-prior medoid + core exemplars", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / f"cluster_{cluster:02d}_motif_bank.png")
        plt.close(fig)


def build_bank(
    candidates: pl.DataFrame,
    scores: pl.DataFrame,
    pairs: pl.DataFrame,
    validity: pl.DataFrame,
    spectral_priors: pl.DataFrame | None,
    *,
    approved_clusters: tuple[int, ...] | None,
    library_size: int,
    templates_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build cluster and member motif-bank tables."""
    if approved_clusters is None:
        accepted = validity.filter(pl.col("auto_valid_suggested"))["motif_cluster"].to_list()
        acceptance_source = "auto_suggested"
    else:
        accepted = list(approved_clusters)
        acceptance_source = "human_approved_cluster"

    cluster_rows = []
    member_rows = []
    joined = candidates.join(
        scores.select(
            "candidate_id",
            "candidate_median_similarity",
            "candidate_knn5_similarity",
            "candidate_medoid_similarity",
            "low_similarity_z",
            "review_priority",
        ),
        on="candidate_id",
        how="left",
    )
    if spectral_priors is not None:
        joined = joined.join(
            spectral_priors.select(
                "candidate_id",
                "recording_aware_spectral_prior_score",
                "recording_aware_prior_label",
                "dominant_mode_hz",
                "dominant_support_fmin_hz",
                "dominant_support_fmax_hz",
                "best_recording_mode_hz",
                "best_recording_support_fmin_hz",
                "best_recording_support_fmax_hz",
                "bbox_recording_mode_iou",
                "local_support_recording_mode_iou",
            ),
            on="candidate_id",
            how="left",
        )
    else:
        joined = joined.with_columns(
            pl.lit(0.5).alias("recording_aware_spectral_prior_score"),
            pl.lit("unknown").alias("recording_aware_prior_label"),
            pl.lit(None).cast(pl.Float64).alias("dominant_mode_hz"),
            pl.lit(None).cast(pl.Float64).alias("dominant_support_fmin_hz"),
            pl.lit(None).cast(pl.Float64).alias("dominant_support_fmax_hz"),
            pl.lit(None).cast(pl.Float64).alias("best_recording_mode_hz"),
            pl.lit(None).cast(pl.Float64).alias("best_recording_support_fmin_hz"),
            pl.lit(None).cast(pl.Float64).alias("best_recording_support_fmax_hz"),
            pl.lit(None).cast(pl.Float64).alias("bbox_recording_mode_iou"),
            pl.lit(None).cast(pl.Float64).alias("local_support_recording_mode_iou"),
        )

    for cluster in accepted:
        cluster_scores = joined.filter(pl.col("motif_cluster") == cluster)
        if cluster_scores.is_empty():
            continue
        medoid_id, library_ids, _sim = select_library_ids(
            cluster_scores, pairs, library_size=library_size
        )
        cluster_validity = validity.filter(pl.col("motif_cluster") == cluster).to_dicts()[0]
        cluster_candidates = joined.filter(pl.col("motif_cluster") == cluster)
        kept_candidates = cluster_candidates.filter(
            (pl.col("review_priority") != "drop_candidate")
            & (pl.col("recording_aware_prior_label") == "good")
        )

        cluster_rows.append(
            {
                "motif_cluster": cluster,
                "accepted": True,
                "acceptance_source": acceptance_source,
                "motif_likelihood_score": cluster_validity["motif_likelihood_score"],
                "auto_valid_suggested": cluster_validity["auto_valid_suggested"],
                "n_candidates": cluster_validity["n_candidates"],
                "n_bank_members": kept_candidates.height,
                "n_templates": len(library_ids),
                "medoid_candidate_id": medoid_id,
                "library_candidate_ids": ";".join(str(x) for x in library_ids),
                "median_similarity": cluster_validity["median_similarity"],
                "median_recording_spectral_prior": cluster_validity.get("median_recording_spectral_prior"),
                "spectral_good_rate": cluster_validity.get("spectral_good_rate"),
                "spectral_edge_misaligned_rate": cluster_validity.get("spectral_edge_misaligned_rate"),
                "median_peak_z": cluster_validity["median_peak_z"],
                "median_duration_sec": cluster_validity["median_duration_sec"],
                "median_bandwidth_hz": cluster_validity["median_bandwidth_hz"],
                "median_fmin_hz": cluster_validity["median_fmin_hz"],
                "median_fmax_hz": cluster_validity["median_fmax_hz"],
            }
        )

        for row in cluster_candidates.sort("filename", "start_sec").iter_rows(named=True):
            candidate_id = int(row["candidate_id"])
            if candidate_id == medoid_id:
                role = "medoid"
                rank = 0
            elif candidate_id in library_ids:
                role = "exemplar"
                rank = library_ids.index(candidate_id)
            elif row["review_priority"] == "drop_candidate":
                role = "excluded_outlier"
                rank = None
            elif row.get("recording_aware_prior_label") != "good":
                role = "excluded_spectral_prior"
                rank = None
            else:
                role = "member"
                rank = None
            member_rows.append({**row, "bank_role": role, "bank_rank": rank})

        templates = []
        template_metadata = []
        for rank, candidate_id in enumerate(library_ids):
            row = cluster_candidates.filter(pl.col("candidate_id") == candidate_id).to_dicts()[0]
            templates.append(extract_template(row))
            template_metadata.append(row)
        if templates:
            np.savez_compressed(
                templates_dir / f"cluster_{cluster:02d}_templates.npz",
                templates=np.stack(templates).astype(np.float32),
                candidate_ids=np.array(library_ids, dtype=np.int64),
                medoid_candidate_id=np.array([medoid_id], dtype=np.int64),
            )
            pl.DataFrame(template_metadata).write_csv(
                templates_dir / f"cluster_{cluster:02d}_template_metadata.csv"
            )

    return pl.DataFrame(cluster_rows), pl.DataFrame(member_rows)


def run(
    candidates_path: Path = DEFAULT_CANDIDATES,
    outlier_scores_path: Path = DEFAULT_OUTLIER_SCORES,
    pairwise_path: Path = DEFAULT_PAIRWISE,
    spectral_priors_path: Path | None = DEFAULT_SPECTRAL_PRIORS,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    approved_clusters: tuple[int, ...] | None = DEFAULT_APPROVED_CLUSTERS,
    library_size: int = 4,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = fresh_subdir(output_dir, "tables")
    plots_dir = fresh_subdir(output_dir, "plots")
    templates_dir = fresh_subdir(output_dir, "templates")

    candidates = pl.read_csv(candidates_path)
    scores = pl.read_csv(outlier_scores_path)
    pairs = pl.read_csv(pairwise_path)
    spectral_priors = (
        pl.read_csv(spectral_priors_path)
        if spectral_priors_path is not None and spectral_priors_path.exists()
        else None
    )
    validity = cluster_validity_table(candidates, scores, spectral_priors)
    validity.write_csv(tables_dir / "cluster_validity_scores.csv")

    bank_clusters, bank_members = build_bank(
        candidates,
        scores,
        pairs,
        validity,
        spectral_priors,
        approved_clusters=approved_clusters,
        library_size=library_size,
        templates_dir=templates_dir,
    )
    bank_clusters.write_csv(tables_dir / "motif_bank_clusters.csv")
    bank_members.write_csv(tables_dir / "motif_bank_members.csv")
    plot_bank_grid(bank_members, plots_dir)

    (output_dir / "summary.md").write_text(
        f"""# Marbled Frog motif bank

Input candidates: `{candidates_path}`

Accepted clusters: {', '.join(str(c) for c in bank_clusters['motif_cluster'].to_list())}

Acceptance mode: `{('auto_suggested' if approved_clusters is None else 'human_approved_cluster')}`

Method:

1. Score every cluster using recurrence, recording spread, pairwise-template cohesion,
   salience, box plausibility, outlier rate, and recording-aware spectral priors.
2. For accepted clusters, exclude `drop_candidate` and spectrally misaligned candidates.
3. Choose one medoid candidate per cluster, then add central high-quality
   exemplars to create a small template library.

Artifacts:

- `tables/cluster_validity_scores.csv` — automatic motif-likelihood diagnostics for all clusters.
- `tables/motif_bank_clusters.csv` — accepted motif families and medoid/library ids.
- `tables/motif_bank_members.csv` — accepted/excluded candidates with bank roles.
- `templates/cluster_XX_templates.npz` — normalized template crops for each motif family.
- `plots/cluster_XX_motif_bank.png` — visual medoid/exemplar grids.

Accepted motif bank:

```text
{bank_clusters.write_csv(separator='|').strip()}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--outlier-scores", type=Path, default=DEFAULT_OUTLIER_SCORES)
    parser.add_argument("--pairwise", type=Path, default=DEFAULT_PAIRWISE)
    parser.add_argument("--spectral-priors", type=Path, default=DEFAULT_SPECTRAL_PRIORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approved-clusters", type=int, nargs="*", default=list(DEFAULT_APPROVED_CLUSTERS))
    parser.add_argument("--auto", action="store_true", help="Use automatic cluster suggestions instead of approved clusters")
    parser.add_argument("--library-size", type=int, default=4)
    args = parser.parse_args()
    approved = None if args.auto else tuple(args.approved_clusters)
    output_dir = run(
        args.candidates,
        args.outlier_scores,
        args.pairwise,
        args.spectral_priors,
        args.output_dir,
        approved_clusters=approved,
        library_size=args.library_size,
    )
    print(f"Wrote motif bank to {output_dir}")


if __name__ == "__main__":
    main()
