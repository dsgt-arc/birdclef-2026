"""PaCMAP plots for selected species' train-audio Perch embeddings.

For each representative labeled-soundscape species, this creates one figure with
window-level train-audio embeddings and file-pooled train-audio embeddings shown
side by side in a shared PaCMAP coordinate system.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from data_analysis.common.paths import (
    PERCH_TRAIN_AUDIO_EMBEDDINGS,
    TAXONOMY_CSV,
    TRAIN_CSV,
    TRAIN_SCHEMA_OVERRIDES,
    artifact_dir,
)
from data_analysis.common.selected_species import REPRESENTATIVE_LABELS
from data_analysis.soundscapes.pacmap_embeddings import (
    embedding_columns,
    matrix,
    run_pacmap,
    with_coords,
)

ROLE_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
}


def slugify(value: str) -> str:
    """Convert a label/name into a safe filename slug."""
    value = value.lower().replace("×", "x")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def load_species_metadata(labels: list[str]) -> pl.DataFrame:
    """Load taxonomy metadata for selected labels."""
    return (
        pl.read_csv(TAXONOMY_CSV, schema_overrides={"primary_label": pl.Utf8})
        .filter(pl.col("primary_label").is_in(labels))
        .select("primary_label", "scientific_name", "common_name", "class_name")
    )


def load_train_label_roles(labels: list[str]) -> pl.DataFrame:
    """Return one row per train_audio file/selected label with primary/secondary role."""
    train = pl.read_csv(TRAIN_CSV, schema_overrides=TRAIN_SCHEMA_OVERRIDES).select(
        "filename", "primary_label", "secondary_labels"
    )

    primary = train.filter(pl.col("primary_label").is_in(labels)).select(
        "filename",
        pl.col("primary_label").alias("label"),
        pl.lit("primary").alias("label_role"),
    )

    secondary = (
        train.select(
            "filename",
            pl.col("secondary_labels")
            .str.extract_all(r"[A-Za-z0-9]+")
            .alias("label"),
        )
        .explode("label")
        .drop_nulls("label")
        .filter(pl.col("label").is_in(labels))
        .with_columns(pl.lit("secondary").alias("label_role"))
        .select("filename", "label", "label_role")
    )

    return (
        pl.concat([primary, secondary])
        .with_columns(
            pl.when(pl.col("label_role") == "primary").then(0).otherwise(1).alias("role_rank")
        )
        .group_by("filename", "label")
        .agg(pl.col("role_rank").min().alias("role_rank"))
        .with_columns(
            pl.when(pl.col("role_rank") == 0)
            .then(pl.lit("primary"))
            .otherwise(pl.lit("secondary"))
            .alias("label_role")
        )
        .drop("role_rank")
    )


def load_label_windows(label_roles: pl.DataFrame, label: str) -> tuple[pl.DataFrame, list[str]]:
    """Load train-audio Perch windows for a selected label."""
    lf = pl.scan_parquet(PERCH_TRAIN_AUDIO_EMBEDDINGS)
    emb_cols = embedding_columns(lf.collect_schema())
    roles = label_roles.filter(pl.col("label") == label)

    windows = (
        lf.join(roles.lazy(), on="filename", how="inner")
        .select("filename", "chunk_idx", "start_sec", "end_sec", "label", "label_role", *emb_cols)
        .sort("filename", "chunk_idx")
        .collect()
    )
    return windows, emb_cols


def file_pool(windows: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    """Average selected species train-audio windows to one row per file."""
    return (
        windows.group_by("filename", "label", "label_role")
        .agg(
            pl.len().cast(pl.Int64).alias("n_windows"),
            pl.col("start_sec").min().alias("first_start_sec"),
            pl.col("end_sec").max().alias("last_end_sec"),
            *[pl.col(col).mean().alias(col) for col in emb_cols],
        )
        .sort("filename")
    )


def combined_projection(
    windows: pl.DataFrame,
    files: pl.DataFrame,
    emb_cols: list[str],
    *,
    seed: int,
    max_pca_dims: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Project windows and file-pooled embeddings jointly, then split them back out."""
    window_meta = windows.select(
        "filename", "chunk_idx", "start_sec", "end_sec", "label", "label_role"
    ).with_columns(
        pl.lit("window").alias("level"),
        pl.lit(None, dtype=pl.Int64).alias("n_windows"),
    )
    file_meta = files.select("filename", "label", "label_role", "n_windows").with_columns(
        pl.lit(None, dtype=pl.Int64).alias("chunk_idx"),
        pl.lit(None, dtype=pl.Float64).alias("start_sec"),
        pl.lit(None, dtype=pl.Float64).alias("end_sec"),
        pl.lit("file_pool").alias("level"),
    )
    file_meta = file_meta.select(window_meta.columns)

    combined_meta = pl.concat([window_meta, file_meta])
    combined_x = np.vstack([matrix(windows, emb_cols), matrix(files, emb_cols)])
    coords = run_pacmap(combined_x, seed=seed, max_pca_dims=max_pca_dims)
    combined = with_coords(combined_meta, coords)
    return (
        combined.filter(pl.col("level") == "window"),
        combined.filter(pl.col("level") == "file_pool"),
    )


def plot_species_projection(
    window_coords: pl.DataFrame,
    file_coords: pl.DataFrame,
    metadata: dict[str, str],
    output_path: Path,
) -> None:
    """Create a 1x2 window/file-pooled PaCMAP figure for one species."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=180, sharex=True, sharey=True)
    common_name = metadata.get("common_name") or metadata["primary_label"]
    class_name = metadata.get("class_name") or "unknown"
    label = metadata["primary_label"]

    for ax, df, title in [
        (axes[0], window_coords, f"Window embeddings ({window_coords.height:,} chunks)"),
        (axes[1], file_coords, f"File-pooled embeddings ({file_coords.height:,} files)"),
    ]:
        for role in ["primary", "secondary"]:
            role_df = df.filter(pl.col("label_role") == role)
            if role_df.is_empty():
                continue
            if "n_windows" in role_df.columns and title.startswith("File"):
                sizes = np.clip(role_df.get_column("n_windows").fill_null(1).to_numpy() * 4, 25, 220)
                alpha = 0.82
            else:
                sizes = 12 if df.height > 2000 else 22
                alpha = 0.34 if df.height > 2000 else 0.62
            ax.scatter(
                role_df.get_column("pacmap_x"),
                role_df.get_column("pacmap_y"),
                s=sizes,
                alpha=alpha,
                color=ROLE_COLORS[role],
                label=role,
                edgecolors="black" if title.startswith("File") else "none",
                linewidths=0.25,
            )
        ax.set_title(title)
        ax.set_xlabel("PaCMAP 1")
        ax.grid(alpha=0.22)
        ax.legend(title="train.csv role", loc="best", frameon=True)

    axes[0].set_ylabel("PaCMAP 2")
    all_x = np.concatenate([
        window_coords.get_column("pacmap_x").to_numpy(),
        file_coords.get_column("pacmap_x").to_numpy(),
    ])
    all_y = np.concatenate([
        window_coords.get_column("pacmap_y").to_numpy(),
        file_coords.get_column("pacmap_y").to_numpy(),
    ])
    x_pad = max((all_x.max() - all_x.min()) * 0.05, 0.5)
    y_pad = max((all_y.max() - all_y.min()) * 0.05, 0.5)
    for ax in axes:
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    fig.suptitle(f"{common_name} ({label}) — {class_name}: train-audio Perch embeddings", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_index(rows: list[dict[str, object]], output_dir: Path) -> str:
    """Render a compact Markdown index for generated selected-species figures."""
    table = pl.DataFrame(rows).select(
        "primary_label",
        "common_name",
        "class_name",
        "n_windows",
        "n_files",
        "plot",
        "coords",
    )
    return f"""# Selected Species Train-Audio PaCMAP Index

One figure is generated per representative train-supported labeled-soundscape species.
Each figure uses a shared PaCMAP projection for that species' train-audio window embeddings and file-pooled embeddings, then displays them side by side.

```text
{table.write_csv(separator='|').strip()}
```

Output directory: `{output_dir}`
"""


def run(
    output_dir: Path | None = None,
    *,
    labels: list[str] | None = None,
    seed: int = 42,
    max_pca_dims: int = 50,
) -> Path:
    """Run selected-species train-audio PaCMAP analysis."""
    output_dir = output_dir or artifact_dir("soundscapes/per_species_train_pacmap")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or REPRESENTATIVE_LABELS

    metadata = load_species_metadata(labels)
    label_roles = load_train_label_roles(labels)
    rows: list[dict[str, object]] = []

    for label in labels:
        meta_row = metadata.filter(pl.col("primary_label") == label).row(0, named=True)
        windows, emb_cols = load_label_windows(label_roles, label)
        if windows.height < 3:
            print(f"Skipping {label}: only {windows.height} train-audio windows")
            continue
        files = file_pool(windows, emb_cols)
        if files.height < 3:
            print(f"Skipping {label}: only {files.height} train-audio files")
            continue

        window_coords, file_coords = combined_projection(
            windows,
            files,
            emb_cols,
            seed=seed,
            max_pca_dims=max_pca_dims,
        )

        slug = slugify(f"{label}-{meta_row.get('common_name') or label}")
        coords_path = output_dir / f"{slug}_coords.csv"
        plot_path = output_dir / f"{slug}.png"
        pl.concat([window_coords, file_coords]).write_csv(coords_path)
        plot_species_projection(window_coords, file_coords, meta_row, plot_path)

        rows.append(
            {
                "primary_label": label,
                "common_name": meta_row.get("common_name"),
                "class_name": meta_row.get("class_name"),
                "n_windows": windows.height,
                "n_files": files.height,
                "plot": plot_path.name,
                "coords": coords_path.name,
            }
        )
        print(f"Wrote {plot_path.name}: {windows.height:,} windows, {files.height:,} files")

    (output_dir / "index.md").write_text(render_index(rows, output_dir))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/soundscapes/per_species_train_pacmap/.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=None,
        help="Specific label to plot; repeat for multiple. Defaults to the representative set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pca-dims", type=int, default=50)
    args = parser.parse_args()
    output_dir = run(
        args.output_dir,
        labels=args.labels,
        seed=args.seed,
        max_pca_dims=args.max_pca_dims,
    )
    print(f"Wrote selected species train-audio PaCMAP plots to {output_dir}")


if __name__ == "__main__":
    main()
