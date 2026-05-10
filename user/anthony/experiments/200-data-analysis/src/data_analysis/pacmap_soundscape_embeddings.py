"""PaCMAP plots for Perch soundscape embeddings.

By default this analyzes only the 66 files that appear in
`train_soundscapes_labels.csv`, matching the labeled-soundscape analysis focus.
It creates four PaCMAP projections:

1. every available 5-second Perch embedding window for labeled clips
2. one average-pooled embedding per clip/file
3. one average-pooled embedding per site, computed from clip means
4. one average-pooled embedding per labeled species/taxon
5. one average-pooled embedding per labeled species/taxon per site
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pacmap
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_analysis.paths import (
    PERCH_SOUNDSCAPE_EMBEDDINGS,
    TAXONOMY_CSV,
    TRAIN_SOUNDSCAPES_LABELS_CSV,
    artifact_dir,
)
from data_analysis.labeled_soundscapes import hms_to_seconds


@dataclass(frozen=True)
class ProjectionResult:
    """A named PaCMAP projection plus plotting metadata."""

    name: str
    title: str
    coords: np.ndarray
    metadata: pl.DataFrame
    point_size: float
    alpha: float
    annotate: bool = False


def embedding_columns(schema: pl.Schema) -> list[str]:
    """Return ordered `emb_*` columns from a Perch embedding parquet schema."""
    return sorted(
        [name for name in schema.names() if name.startswith("emb_")],
        key=lambda name: int(name.split("_", 1)[1]),
    )


def labeled_filenames(path: Path = TRAIN_SOUNDSCAPES_LABELS_CSV) -> pl.DataFrame:
    """Return unique labeled soundscape filenames."""
    return pl.read_csv(path).select("filename").unique()


def load_window_embeddings(
    embeddings_path: Path = PERCH_SOUNDSCAPE_EMBEDDINGS,
    *,
    scope: str = "labeled",
) -> tuple[pl.DataFrame, list[str]]:
    """Load window-level embeddings and metadata.

    Parameters
    ----------
    embeddings_path:
        Perch soundscape embeddings parquet.
    scope:
        `labeled` filters to files in `train_soundscapes_labels.csv`; `all`
        loads every train-soundscape embedding row.
    """
    lf = pl.scan_parquet(embeddings_path)
    emb_cols = embedding_columns(lf.collect_schema())

    if scope == "labeled":
        lf = lf.join(labeled_filenames().lazy(), on="filename", how="inner")
    elif scope != "all":
        raise ValueError(f"Unknown scope: {scope!r}; expected 'labeled' or 'all'")

    df = (
        lf.select(
            "filename",
            "chunk_idx",
            "start_sec",
            "end_sec",
            pl.col("filename").str.extract(r"_(S\d{2})_", 1).alias("site"),
            *emb_cols,
        )
        .sort("filename", "chunk_idx")
        .collect()
    )
    return df, emb_cols


def clip_pool(windows: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    """Average window embeddings to one row per clip/file."""
    return (
        windows.group_by("filename", "site")
        .agg(
            pl.len().alias("n_windows"),
            pl.col("start_sec").min().alias("first_start_sec"),
            pl.col("end_sec").max().alias("last_end_sec"),
            *[pl.col(col).mean().alias(col) for col in emb_cols],
        )
        .sort("filename")
    )


def site_pool(clips: pl.DataFrame, windows: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    """Average clip embeddings to one row per site, equally weighting clips."""
    window_counts = windows.group_by("site").agg(pl.len().alias("n_windows"))
    return (
        clips.group_by("site")
        .agg(
            pl.len().alias("n_clips"),
            *[pl.col(col).mean().alias(col) for col in emb_cols],
        )
        .join(window_counts, on="site", how="left")
        .sort("site")
    )


def load_exploded_labels() -> pl.DataFrame:
    """Load one row per labeled 5-second segment/species with taxonomy metadata."""
    labels = (
        pl.read_csv(TRAIN_SOUNDSCAPES_LABELS_CSV)
        .with_columns(
            hms_to_seconds("start").cast(pl.Float64).alias("start_sec"),
            hms_to_seconds("end").cast(pl.Float64).alias("end_sec"),
            pl.col("primary_label").str.split(";").alias("label"),
        )
        .explode("label")
    )
    taxonomy = pl.read_csv(TAXONOMY_CSV).with_columns(
        pl.col("primary_label").cast(pl.Utf8).alias("label")
    )
    return labels.join(
        taxonomy.drop("primary_label"),
        on="label",
        how="left",
    )


def labeled_windows_with_taxonomy(windows: pl.DataFrame) -> pl.DataFrame:
    """Join Perch windows to exploded segment labels and taxonomy metadata."""
    return windows.join(
        load_exploded_labels(),
        on=["filename", "start_sec", "end_sec"],
        how="inner",
    )


def species_pool(windows: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    """Average labeled window embeddings to one row per species/taxon label."""
    labeled_windows = labeled_windows_with_taxonomy(windows)
    return (
        labeled_windows.group_by("label", "scientific_name", "common_name", "class_name")
        .agg(
            pl.len().alias("n_windows"),
            pl.col("filename").n_unique().alias("n_files"),
            pl.col("site").n_unique().alias("n_sites"),
            *[pl.col(col).mean().alias(col) for col in emb_cols],
        )
        .sort("n_windows", descending=True)
    )


def species_site_pool(windows: pl.DataFrame, emb_cols: list[str]) -> pl.DataFrame:
    """Average labeled window embeddings to one row per species/taxon/site."""
    labeled_windows = labeled_windows_with_taxonomy(windows)
    return (
        labeled_windows.group_by("label", "scientific_name", "common_name", "class_name", "site")
        .agg(
            pl.len().alias("n_windows"),
            pl.col("filename").n_unique().alias("n_files"),
            *[pl.col(col).mean().alias(col) for col in emb_cols],
        )
        .sort(["site", "n_windows"], descending=[False, True])
    )


def matrix(df: pl.DataFrame, emb_cols: list[str]) -> np.ndarray:
    """Convert embedding columns to a float32 NumPy matrix."""
    return df.select(emb_cols).to_numpy().astype(np.float32, copy=False)


def preprocess_for_pacmap(
    x: np.ndarray,
    *,
    max_pca_dims: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Standardize and PCA-compress embeddings before PaCMAP."""
    if x.shape[0] < 3:
        raise ValueError("Need at least 3 rows for a 2D PaCMAP projection")

    x_scaled = StandardScaler().fit_transform(x).astype(np.float32, copy=False)
    n_components = min(max_pca_dims, x_scaled.shape[0] - 1, x_scaled.shape[1])
    if n_components <= 2:
        return x_scaled
    return PCA(n_components=n_components, random_state=seed).fit_transform(x_scaled).astype(
        np.float32, copy=False
    )


def run_pacmap(x: np.ndarray, *, seed: int = 42, max_pca_dims: int = 50) -> np.ndarray:
    """Project embeddings to 2D with PaCMAP."""
    x_pre = preprocess_for_pacmap(x, max_pca_dims=max_pca_dims, seed=seed)
    n_samples = x_pre.shape[0]
    n_neighbors = min(10, max(2, n_samples - 1))
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=seed,
    )
    return reducer.fit_transform(x_pre, init="pca").astype(np.float32, copy=False)


def with_coords(metadata: pl.DataFrame, coords: np.ndarray) -> pl.DataFrame:
    """Append PaCMAP coordinates to a metadata frame."""
    return metadata.with_columns(
        pl.Series("pacmap_x", coords[:, 0]),
        pl.Series("pacmap_y", coords[:, 1]),
    )


def plot_by_site(result: ProjectionResult, output_path: Path) -> None:
    """Scatter a PaCMAP projection, colored by site."""
    df = with_coords(result.metadata, result.coords)
    sites = df.get_column("site").unique().sort().to_list()
    cmap = plt.get_cmap("tab20", len(sites))

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for idx, site in enumerate(sites):
        site_df = df.filter(pl.col("site") == site)
        size = result.point_size
        if "n_clips" in site_df.columns:
            size = np.clip(site_df.get_column("n_clips").to_numpy() * 12, 80, 420)
        ax.scatter(
            site_df.get_column("pacmap_x"),
            site_df.get_column("pacmap_y"),
            s=size,
            alpha=result.alpha,
            color=cmap(idx),
            label=site,
            edgecolors="black" if result.annotate else "none",
            linewidths=0.5 if result.annotate else 0,
        )
        if result.annotate:
            label_col = "site"
            for row in site_df.iter_rows(named=True):
                extra = f"\n{row['n_clips']} clips" if "n_clips" in row else ""
                ax.annotate(
                    f"{row[label_col]}{extra}",
                    (row["pacmap_x"], row["pacmap_y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

    ax.set_title(result.title)
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.grid(alpha=0.2)
    ax.legend(title="Site", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_by_class(result: ProjectionResult, output_path: Path, *, annotate_top_n: int = 25) -> None:
    """Scatter a species-level PaCMAP projection, colored by taxonomy class."""
    df = with_coords(result.metadata, result.coords)
    classes = df.get_column("class_name").unique().sort().to_list()
    cmap = plt.get_cmap("tab10", len(classes))

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    for idx, class_name in enumerate(classes):
        class_df = df.filter(pl.col("class_name") == class_name)
        sizes = np.clip(class_df.get_column("n_windows").to_numpy() * 1.5, 35, 360)
        ax.scatter(
            class_df.get_column("pacmap_x"),
            class_df.get_column("pacmap_y"),
            s=sizes,
            alpha=0.78,
            color=cmap(idx),
            label=class_name,
            edgecolors="black",
            linewidths=0.35,
        )

    for row in df.sort("n_windows", descending=True).head(annotate_top_n).iter_rows(named=True):
        name = row.get("common_name") or row["label"]
        if "site" in row:
            name = f"{name} @ {row['site']}"
        ax.annotate(
            str(name)[:40],
            (row["pacmap_x"], row["pacmap_y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_title(result.title)
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.grid(alpha=0.2)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_summary(
    *,
    scope: str,
    windows: pl.DataFrame,
    clips: pl.DataFrame,
    sites: pl.DataFrame,
    species: pl.DataFrame,
    species_sites: pl.DataFrame,
    outputs: list[Path],
) -> str:
    """Render a Markdown summary for this PaCMAP analysis."""
    output_list = "\n".join(f"- `{path.name}`" for path in outputs)
    by_site = (
        windows.group_by("site")
        .agg(
            pl.col("filename").n_unique().alias("n_clips"),
            pl.len().alias("n_windows"),
        )
        .sort("site")
        .write_csv(separator="|")
        .strip()
    )
    return f"""# PaCMAP of Perch Soundscape Embeddings

Scope: `{scope}` soundscapes.

Inputs:

- Embeddings: `{PERCH_SOUNDSCAPE_EMBEDDINGS}`
- Labeled file filter: `{TRAIN_SOUNDSCAPES_LABELS_CSV}`

Rows projected:

- Window-level embeddings: {windows.height}
- Clip-pooled embeddings: {clips.height}
- Site-pooled embeddings: {sites.height}
- Species-pooled embeddings: {species.height}
- Species×site-pooled embeddings: {species_sites.height}

Pooling:

- Clip embeddings are arithmetic means across available Perch windows per file.
- Site embeddings are arithmetic means of clip embeddings, so each clip has equal weight.
- Species embeddings are arithmetic means across labeled matching 5-second windows for that species.
- Species×site embeddings use the same labeled-window average, but keep each site separate.
- Features are standardized and PCA-compressed to at most 50 dimensions before PaCMAP.

Site counts:

```text
{by_site}
```

Outputs:

{output_list}
"""


def run(
    *,
    output_dir: Path | None = None,
    scope: str = "labeled",
    seed: int = 42,
    max_pca_dims: int = 50,
) -> Path:
    """Run the three PaCMAP projections and write plot/coordinate artifacts."""
    output_dir = output_dir or artifact_dir("pacmap_soundscape_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    windows, emb_cols = load_window_embeddings(scope=scope)
    clips = clip_pool(windows, emb_cols)
    sites = site_pool(clips, windows, emb_cols)
    species = species_pool(windows, emb_cols)
    species_sites = species_site_pool(windows, emb_cols)

    projections = [
        ProjectionResult(
            name="windows",
            title=f"Perch embeddings: all {scope} soundscape windows",
            coords=run_pacmap(matrix(windows, emb_cols), seed=seed, max_pca_dims=max_pca_dims),
            metadata=windows.select("filename", "site", "chunk_idx", "start_sec", "end_sec"),
            point_size=14,
            alpha=0.55,
        ),
        ProjectionResult(
            name="clip_pooled",
            title=f"Perch embeddings: average-pooled by clip ({scope})",
            coords=run_pacmap(matrix(clips, emb_cols), seed=seed, max_pca_dims=max_pca_dims),
            metadata=clips.select("filename", "site", "n_windows", "first_start_sec", "last_end_sec"),
            point_size=42,
            alpha=0.8,
        ),
        ProjectionResult(
            name="site_pooled",
            title=f"Perch embeddings: average-pooled by site ({scope})",
            coords=run_pacmap(matrix(sites, emb_cols), seed=seed, max_pca_dims=max_pca_dims),
            metadata=sites.select("site", "n_clips", "n_windows"),
            point_size=120,
            alpha=0.95,
            annotate=True,
        ),
    ]
    species_projection = ProjectionResult(
        name="species_pooled",
        title=f"Perch embeddings: average-pooled by labeled species ({scope})",
        coords=run_pacmap(matrix(species, emb_cols), seed=seed, max_pca_dims=max_pca_dims),
        metadata=species.select(
            "label", "scientific_name", "common_name", "class_name", "n_windows", "n_files", "n_sites"
        ),
        point_size=60,
        alpha=0.8,
    )
    species_site_projection = ProjectionResult(
        name="species_site_pooled",
        title=f"Perch embeddings: average-pooled by labeled species×site ({scope})",
        coords=run_pacmap(matrix(species_sites, emb_cols), seed=seed, max_pca_dims=max_pca_dims),
        metadata=species_sites.select(
            "label", "scientific_name", "common_name", "class_name", "site", "n_windows", "n_files"
        ),
        point_size=55,
        alpha=0.78,
    )

    outputs: list[Path] = []
    for result in projections:
        coords_path = output_dir / f"{result.name}_pacmap.csv"
        plot_path = output_dir / f"{result.name}_pacmap_by_site.png"
        with_coords(result.metadata, result.coords).write_csv(coords_path)
        plot_by_site(result, plot_path)
        outputs.extend([coords_path, plot_path])

    coords_path = output_dir / "species_pooled_pacmap.csv"
    plot_path = output_dir / "species_pooled_pacmap_by_class.png"
    with_coords(species_projection.metadata, species_projection.coords).write_csv(coords_path)
    plot_by_class(species_projection, plot_path)
    outputs.extend([coords_path, plot_path])

    coords_path = output_dir / "species_site_pooled_pacmap.csv"
    class_plot_path = output_dir / "species_site_pooled_pacmap_by_class.png"
    site_plot_path = output_dir / "species_site_pooled_pacmap_by_site.png"
    with_coords(species_site_projection.metadata, species_site_projection.coords).write_csv(coords_path)
    plot_by_class(species_site_projection, class_plot_path, annotate_top_n=35)
    plot_by_site(species_site_projection, site_plot_path)
    outputs.extend([coords_path, class_plot_path, site_plot_path])

    summary = build_summary(
        scope=scope,
        windows=windows,
        clips=clips,
        sites=sites,
        species=species,
        species_sites=species_sites,
        outputs=outputs,
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/pacmap_soundscape_embeddings/.",
    )
    parser.add_argument(
        "--scope",
        choices=["labeled", "all"],
        default="labeled",
        help="Use only labeled soundscape files, or every train_soundscape embedding row.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pca-dims", type=int, default=50)
    args = parser.parse_args()

    output_dir = run(
        output_dir=args.output_dir,
        scope=args.scope,
        seed=args.seed,
        max_pca_dims=args.max_pca_dims,
    )
    print(f"Wrote PaCMAP embedding analysis to {output_dir}")


if __name__ == "__main__":
    main()
