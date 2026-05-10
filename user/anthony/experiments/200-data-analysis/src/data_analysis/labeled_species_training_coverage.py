"""Count train-audio file/window coverage for labels seen in labeled soundscapes.

Outputs lightweight CSV/Markdown artifacts under
`artifacts/labeled_species_training_coverage/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from data_analysis.labeled_soundscapes import (
    build_summaries,
    explode_segment_labels,
    load_segment_labels,
    load_taxonomy,
)
from data_analysis.paths import PERCH_TRAIN_AUDIO_EMBEDDINGS, TRAIN_CSV, artifact_dir

TRAIN_SCHEMA_OVERRIDES = {
    "primary_label": pl.Utf8,
    "secondary_labels": pl.Utf8,
}


def load_train_audio_metadata(train_csv: Path = TRAIN_CSV) -> pl.DataFrame:
    """Load train.csv columns needed for label coverage counts."""
    return pl.read_csv(train_csv, schema_overrides=TRAIN_SCHEMA_OVERRIDES).select(
        "filename",
        "primary_label",
        "secondary_labels",
        "scientific_name",
        "common_name",
        "class_name",
    )


def load_train_audio_window_counts(
    embeddings_parquet: Path = PERCH_TRAIN_AUDIO_EMBEDDINGS,
) -> pl.DataFrame:
    """Return one row per train_audio file with its number of 5-second Perch chunks."""
    return (
        pl.scan_parquet(embeddings_parquet)
        .select("filename")
        .group_by("filename")
        .agg(pl.len().alias("n_windows"))
        .collect()
    )


def labeled_soundscape_species_counts() -> pl.DataFrame:
    """Count labeled soundscape windows/files for each species/taxon."""
    segments = load_segment_labels()
    taxonomy = load_taxonomy()
    labels = explode_segment_labels(segments, taxonomy)
    return build_summaries(segments, labels)["by_species"].rename(
        {
            "label": "primary_label",
            "n_segment_occurrences": "soundscape_labeled_windows",
            "n_files": "soundscape_labeled_files",
            "n_sites": "soundscape_labeled_sites",
        }
    )


def train_audio_label_rows(train: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """Explode train_audio primary + secondary labels to one row per file/label.

    The window count is the number of available Perch 5-second chunks for that
    source file. Secondary-label window counts are weak: the label is known to
    be present somewhere in the file, not necessarily every 5-second chunk.
    """
    train_with_windows = train.join(windows, on="filename", how="left").with_columns(
        pl.col("n_windows").fill_null(0)
    )

    primary = train_with_windows.select(
        "filename",
        pl.col("primary_label").alias("label"),
        pl.lit("primary").alias("label_role"),
        "n_windows",
    )

    secondary = (
        train_with_windows.select(
            "filename",
            pl.col("secondary_labels")
            .str.extract_all(r"[A-Za-z0-9]+")
            .alias("label"),
            "n_windows",
        )
        .explode("label")
        .drop_nulls("label")
        .with_columns(pl.lit("secondary").alias("label_role"))
        .select("filename", "label", "label_role", "n_windows")
    )

    return pl.concat([primary, secondary]).unique(["filename", "label", "label_role"])


def summarize_train_audio(label_rows: pl.DataFrame) -> pl.DataFrame:
    """Aggregate primary, secondary, and any-label train_audio coverage."""
    primary = (
        label_rows.filter(pl.col("label_role") == "primary")
        .group_by("label")
        .agg(
            pl.col("filename").n_unique().alias("train_audio_primary_files"),
            pl.col("n_windows").sum().alias("train_audio_primary_windows"),
        )
    )
    secondary = (
        label_rows.filter(pl.col("label_role") == "secondary")
        .group_by("label")
        .agg(
            pl.col("filename").n_unique().alias("train_audio_secondary_files"),
            pl.col("n_windows").sum().alias("train_audio_secondary_windows"),
        )
    )
    any_label = (
        label_rows.unique(["filename", "label"])
        .group_by("label")
        .agg(
            pl.col("filename").n_unique().alias("train_audio_any_files"),
            pl.col("n_windows").sum().alias("train_audio_any_windows"),
        )
    )

    return (
        any_label.join(primary, on="label", how="left")
        .join(secondary, on="label", how="left")
        .rename({"label": "primary_label"})
    )


def build_coverage_table() -> pl.DataFrame:
    """Build the combined coverage table for labels present in soundscapes."""
    soundscape = labeled_soundscape_species_counts()
    train = load_train_audio_metadata()
    windows = load_train_audio_window_counts()
    train_counts = summarize_train_audio(train_audio_label_rows(train, windows))

    count_cols = [
        "train_audio_any_files",
        "train_audio_any_windows",
        "train_audio_primary_files",
        "train_audio_primary_windows",
        "train_audio_secondary_files",
        "train_audio_secondary_windows",
    ]

    return (
        soundscape.join(train_counts, on="primary_label", how="left")
        .with_columns(pl.col(count_cols).fill_null(0).cast(pl.Int64))
        .select(
            "primary_label",
            "scientific_name",
            "common_name",
            "class_name",
            "soundscape_labeled_windows",
            "soundscape_labeled_files",
            "soundscape_labeled_sites",
            "train_audio_any_files",
            "train_audio_any_windows",
            "train_audio_primary_files",
            "train_audio_primary_windows",
            "train_audio_secondary_files",
            "train_audio_secondary_windows",
        )
        .sort(
            ["soundscape_labeled_windows", "train_audio_any_files"],
            descending=[True, True],
        )
    )


def plot_coverage_scatter(coverage: pl.DataFrame, output_path: Path) -> None:
    """Plot soundscape support vs train-audio support for each label."""
    colors = {
        "Amphibia": "#1f77b4",
        "Aves": "#2ca02c",
        "Insecta": "#8c564b",
        "Mammalia": "#7f7f7f",
        "Reptilia": "#17becf",
    }

    fig, ax = plt.subplots(figsize=(13, 8), dpi=180)
    for class_name in sorted(coverage["class_name"].drop_nulls().unique().to_list()):
        subset = coverage.filter(pl.col("class_name") == class_name)
        x = (subset["train_audio_any_windows"] + 1).to_list()
        y = subset["soundscape_labeled_windows"].to_list()
        sizes = [35 + 12 * min(files, 20) for files in subset["soundscape_labeled_files"].to_list()]
        ax.scatter(
            x,
            y,
            s=sizes,
            label=class_name,
            color=colors.get(class_name),
            alpha=0.78,
            edgecolor="black",
            linewidth=0.45,
        )

    to_annotate = coverage.filter(
        (pl.col("soundscape_labeled_windows") >= 100)
        | ((pl.col("train_audio_any_windows") == 0) & (pl.col("soundscape_labeled_windows") >= 50))
    ).sort("soundscape_labeled_windows", descending=True)

    for row in to_annotate.iter_rows(named=True):
        name = row["common_name"] or row["primary_label"]
        ax.annotate(
            str(name)[:34],
            (row["train_audio_any_windows"] + 1, row["soundscape_labeled_windows"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    ax.axvline(1, color="black", linestyle="--", linewidth=1, alpha=0.35)
    ax.text(
        1.1,
        coverage["soundscape_labeled_windows"].max() * 0.72,
        "no train-audio windows",
        rotation=90,
        va="center",
        fontsize=9,
        alpha=0.7,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("train_audio primary-or-secondary Perch 5s windows + 1 (log scale)")
    ax.set_ylabel("labeled soundscape 5s windows (log scale)")
    ax.set_title("Training coverage for labels found in labeled soundscapes")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_train_audio_windows_bar(coverage: pl.DataFrame, output_path: Path) -> None:
    """Plot per-label train-audio window counts as a horizontal bar chart."""
    colors = {
        "Amphibia": "#1f77b4",
        "Aves": "#2ca02c",
        "Insecta": "#8c564b",
        "Mammalia": "#7f7f7f",
        "Reptilia": "#17becf",
    }
    df = coverage.sort("train_audio_any_windows", descending=True).with_columns(
        (
            pl.col("common_name").fill_null(pl.col("primary_label"))
            + " ("
            + pl.col("primary_label")
            + ")"
        ).alias("display_name")
    )

    names = df["display_name"].to_list()
    values = df["train_audio_any_windows"].to_list()
    bar_colors = [colors.get(class_name, "#333333") for class_name in df["class_name"].to_list()]

    fig_height = max(12, 0.26 * len(names))
    fig, ax = plt.subplots(figsize=(15, fig_height), dpi=180)
    y_pos = list(range(len(names)))
    ax.barh(y_pos, values, color=bar_colors, alpha=0.82, edgecolor="black", linewidth=0.25)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("train_audio primary-or-secondary Perch 5s windows (symlog scale)")
    ax.set_title("Train-audio window coverage for labels found in labeled soundscapes")
    ax.grid(True, axis="x", which="both", alpha=0.25)

    for y, value in enumerate(values):
        if value == 0:
            ax.text(0.15, y, "0", va="center", ha="left", fontsize=6, alpha=0.7)
        elif value < 25:
            ax.text(value + 0.3, y, str(value), va="center", ha="left", fontsize=6, alpha=0.7)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=class_name)
        for class_name, color in colors.items()
        if class_name in set(df["class_name"].to_list())
    ]
    ax.legend(handles=handles, title="Class", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_summary(coverage: pl.DataFrame) -> str:
    """Render a compact Markdown summary."""
    n_labels = coverage.height
    no_primary = coverage.filter(pl.col("train_audio_primary_files") == 0).height
    no_any = coverage.filter(pl.col("train_audio_any_files") == 0).height
    total_soundscape_windows = coverage["soundscape_labeled_windows"].sum()
    total_train_audio_files_any = coverage["train_audio_any_files"].sum()
    total_train_audio_windows_any = coverage["train_audio_any_windows"].sum()

    top_soundscape = coverage.select(
        "primary_label",
        "common_name",
        "class_name",
        "soundscape_labeled_windows",
        "soundscape_labeled_files",
        "train_audio_any_files",
        "train_audio_any_windows",
    ).head(20)

    scarce_train = coverage.filter(pl.col("train_audio_any_files") <= 5).select(
        "primary_label",
        "common_name",
        "class_name",
        "soundscape_labeled_windows",
        "train_audio_any_files",
        "train_audio_any_windows",
    )

    def table(df: pl.DataFrame) -> str:
        return df.write_csv(separator="|").strip()

    return f"""# Labeled Soundscape Species Training Coverage

## Overall

- Labels found in labeled soundscapes: {n_labels}
- Soundscape segment-label windows across those labels: {total_soundscape_windows}
- Labels with no primary-label `train_audio` files: {no_primary}
- Labels with no primary-or-secondary `train_audio` files: {no_any}
- Sum of per-label primary-or-secondary `train_audio` files: {total_train_audio_files_any}
- Sum of per-label primary-or-secondary 5-second Perch windows: {total_train_audio_windows_any}

## Count definitions

- `soundscape_labeled_windows`: number of labeled 5-second soundscape segments where the label appears.
- `soundscape_labeled_files`: number of labeled soundscape files containing the label.
- `train_audio_primary_files/windows`: `train.csv` source recordings where this is the primary label, plus their Perch 5-second chunk count.
- `train_audio_secondary_files/windows`: source recordings where this appears in `secondary_labels`; these are weak counts because the label may not occupy every chunk.
- `train_audio_any_files/windows`: primary or secondary source recordings, deduplicated per file/label.

## Top labels by labeled-soundscape window count

```text
{table(top_soundscape)}
```

## Labels with <=5 primary-or-secondary train-audio files

```text
{table(scarce_train)}
```
"""


def run(output_dir: Path | None = None) -> Path:
    """Run coverage analysis and return the output directory."""
    output_dir = output_dir or artifact_dir("labeled_species_training_coverage")
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage = build_coverage_table()
    coverage.write_csv(output_dir / "coverage.csv")
    plot_coverage_scatter(coverage, output_dir / "coverage_scatter.png")
    plot_train_audio_windows_bar(coverage, output_dir / "train_audio_windows_by_label.png")
    (output_dir / "summary.md").write_text(render_summary(coverage))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/labeled_species_training_coverage/.",
    )
    args = parser.parse_args()
    output_dir = run(args.output_dir)
    print(f"Wrote labeled species training coverage to {output_dir}")


if __name__ == "__main__":
    main()
