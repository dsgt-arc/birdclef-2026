"""Summarize BirdCLEF 2026 labeled soundscape annotations.

Outputs lightweight CSV/Markdown artifacts under
`artifacts/labeled_soundscapes/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from data_analysis.paths import (
    TAXONOMY_CSV,
    TRAIN_SOUNDSCAPES_LABELS_CSV,
    artifact_dir,
)


def hms_to_seconds(column: str) -> pl.Expr:
    """Convert an HH:MM:SS string column to seconds."""
    parts = pl.col(column).str.split(":")
    return (
        parts.list.get(0).cast(pl.Int64) * 3600
        + parts.list.get(1).cast(pl.Int64) * 60
        + parts.list.get(2).cast(pl.Int64)
    )


def load_segment_labels(labels_csv: Path = TRAIN_SOUNDSCAPES_LABELS_CSV) -> pl.DataFrame:
    """Load the labeled soundscape segment table with parsed labels/site/time."""
    return (
        pl.read_csv(labels_csv)
        .with_columns(
            hms_to_seconds("start").alias("start_s"),
            hms_to_seconds("end").alias("end_s"),
            pl.col("filename").str.extract(r"_(S\d{2})_", 1).alias("site"),
            pl.col("primary_label").str.split(";").alias("labels"),
        )
        .with_columns(
            pl.col("labels").list.len().alias("n_labels"),
            (pl.col("end_s") - pl.col("start_s")).alias("duration_s"),
        )
    )


def load_taxonomy(taxonomy_csv: Path = TAXONOMY_CSV) -> pl.DataFrame:
    """Load taxonomy with string labels for joins against soundscape labels."""
    return pl.read_csv(taxonomy_csv).with_columns(
        pl.col("primary_label").cast(pl.Utf8)
    )


def explode_segment_labels(segments: pl.DataFrame, taxonomy: pl.DataFrame) -> pl.DataFrame:
    """Return one row per (segment, label), enriched with taxonomy metadata."""
    return (
        segments.explode("labels")
        .rename({"labels": "label"})
        .join(
            taxonomy.rename({"primary_label": "label"}),
            on="label",
            how="left",
        )
    )


def build_summaries(segments: pl.DataFrame, labels: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Build compact summary tables for labeled soundscapes."""
    by_file = (
        segments.group_by("filename", "site")
        .agg(
            pl.len().alias("n_segments"),
            pl.col("duration_s").sum().alias("labeled_duration_s"),
            pl.col("n_labels").mean().alias("mean_labels_per_segment"),
            pl.col("n_labels").max().alias("max_labels_per_segment"),
        )
        .sort("filename")
    )

    by_species = (
        labels.group_by("label", "scientific_name", "common_name", "class_name")
        .agg(
            pl.len().alias("n_segment_occurrences"),
            pl.col("filename").n_unique().alias("n_files"),
            pl.col("site").n_unique().alias("n_sites"),
        )
        .sort("n_segment_occurrences", descending=True)
    )

    by_taxon = (
        labels.group_by("class_name")
        .agg(
            pl.len().alias("n_segment_occurrences"),
            pl.col("label").n_unique().alias("n_species"),
            pl.col("filename").n_unique().alias("n_files"),
        )
        .sort("n_segment_occurrences", descending=True)
    )

    by_site = (
        labels.group_by("site")
        .agg(
            pl.col("filename").n_unique().alias("n_files"),
            pl.len().alias("n_segment_label_occurrences"),
            pl.col("label").n_unique().alias("n_species"),
            pl.col("class_name").n_unique().alias("n_taxon_classes"),
        )
        .sort("site")
    )

    return {
        "by_file": by_file,
        "by_species": by_species,
        "by_taxon": by_taxon,
        "by_site": by_site,
    }


def render_markdown_summary(
    segments: pl.DataFrame, labels: pl.DataFrame, summaries: dict[str, pl.DataFrame]
) -> str:
    """Render a human-readable Markdown summary."""
    n_files = segments["filename"].n_unique()
    n_segments = segments.height
    n_multi = segments.filter(pl.col("n_labels") > 1).height
    n_species = labels["label"].n_unique()
    n_sites = segments["site"].n_unique()
    total_duration_min = segments["duration_s"].sum() / 60
    total_label_occurrences = labels.height
    mean_labels = segments["n_labels"].mean()

    top_species = summaries["by_species"].head(15)
    by_taxon = summaries["by_taxon"]
    by_site = summaries["by_site"]

    def table(df: pl.DataFrame) -> str:
        return df.write_csv(separator="|").strip()

    return f"""# Labeled Soundscapes Summary

## Overall

- Files: {n_files}
- Sites: {n_sites}
- 5-second labeled segments: {n_segments}
- Total labeled duration: {total_duration_min:.1f} minutes
- Segment-label occurrences after exploding multi-label rows: {total_label_occurrences}
- Unique labeled species/classes: {n_species}
- Multi-label segments: {n_multi} ({n_multi / n_segments:.1%})
- Mean labels per segment: {mean_labels:.2f}

## Taxon class coverage

```text
{table(by_taxon)}
```

## Site coverage

```text
{table(by_site)}
```

## Top labeled species by segment occurrence

```text
{table(top_species)}
```
"""


def run(output_dir: Path | None = None) -> Path:
    """Run the labeled soundscape summary and return the output directory."""
    output_dir = output_dir or artifact_dir("labeled_soundscapes")
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = load_segment_labels()
    taxonomy = load_taxonomy()
    labels = explode_segment_labels(segments, taxonomy)
    summaries = build_summaries(segments, labels)

    segments.drop("labels").write_csv(output_dir / "segments.csv")
    labels.write_csv(output_dir / "segment_labels_enriched.csv")
    for name, df in summaries.items():
        df.write_csv(output_dir / f"{name}.csv")

    summary_md = render_markdown_summary(segments, labels, summaries)
    (output_dir / "summary.md").write_text(summary_md)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/labeled_soundscapes/.",
    )
    args = parser.parse_args()
    output_dir = run(args.output_dir)
    print(f"Wrote labeled soundscape summary to {output_dir}")


if __name__ == "__main__":
    main()
