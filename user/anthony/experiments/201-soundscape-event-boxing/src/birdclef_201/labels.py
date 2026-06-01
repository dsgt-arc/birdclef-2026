"""Soundscape label + taxonomy loading and window indexing."""

from __future__ import annotations

import argparse
import functools
import logging
import re

import polars as pl

from birdclef_201.paths import (
    TAXONOMY_CSV,
    TRAIN_SOUNDSCAPES_LABELS_CSV,
    artifact_dir,
)
from birdclef_201.spectrogram import hms_to_sec

logger = logging.getLogger(__name__)

_SITE_RE = re.compile(r"_(S\d+)_")


def site_of(filename: str) -> str:
    """Extract the ``S##`` site code from a soundscape filename (``?`` if none)."""
    m = _SITE_RE.search(filename)
    return m.group(1) if m else "?"


@functools.lru_cache(maxsize=1)
def load_taxonomy() -> pl.DataFrame:
    return pl.read_csv(TAXONOMY_CSV, infer_schema_length=10000)


@functools.lru_cache(maxsize=1)
def insect_sonotypes() -> list[str]:
    """The 25 ``47158sonNN`` sonotypes — the Insecta *sonotypes* specifically
    (the prefix excludes the 3 named Insecta species, which are not v1 targets)."""
    tax = load_taxonomy()
    labels = tax.filter(pl.col("class_name") == "Insecta")["primary_label"].to_list()
    return sorted(str(x) for x in labels if str(x).startswith("47158son"))


@functools.lru_cache(maxsize=1)
def insect_sites() -> tuple[str, ...]:
    """Sites where any insect sonotype is labeled — data-derived, not pinned, so
    the cross-site split stays correct if the labeled set changes."""
    long, _ = load_windows()
    son = insect_sonotypes()
    return tuple(sorted(long.filter(pl.col("label").is_in(son))["site"].unique().to_list()))


@functools.lru_cache(maxsize=1)
def load_windows() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load + tidy the soundscape labels.

    Returns ``(long, wide)``:
    - ``long``: one row per (window, label) — filename, site, window_id,
      start_sec, end_sec, label.
    - ``wide``: one row per window — filename, site, window_id, start_sec,
      end_sec, labels(list[str]).
    ``window_id = f"{filename}#{int(start_sec)}"``.

    The published CSV duplicates every row exactly (1,478 rows = 2 x 739 real
    windows), so we dedup at load.
    """
    raw = pl.read_csv(TRAIN_SOUNDSCAPES_LABELS_CSV, infer_schema_length=10000).unique(
        maintain_order=True
    )
    wide = raw.with_columns(
        pl.col("start").map_elements(hms_to_sec, return_dtype=pl.Float64).alias("start_sec"),
        pl.col("end").map_elements(hms_to_sec, return_dtype=pl.Float64).alias("end_sec"),
        pl.col("primary_label").str.split(";").alias("labels"),
        pl.col("filename").map_elements(site_of, return_dtype=pl.Utf8).alias("site"),
    ).with_columns(
        (
            pl.col("filename")
            + "#"
            + pl.col("start_sec").cast(pl.Int64).cast(pl.Utf8)
        ).alias("window_id")
    )
    long = (
        wide.explode("labels")
        .rename({"labels": "label"})
        .select("filename", "site", "window_id", "start_sec", "end_sec", "label")
    )
    wide = wide.select("filename", "site", "window_id", "start_sec", "end_sec", "labels")
    return long, wide


def _run_summary() -> None:
    long, wide = load_windows()
    son = set(insect_sonotypes())
    tax = load_taxonomy()
    out = artifact_dir("labels")

    files = long["filename"].n_unique()
    site_counts = (
        wide.group_by("site").len().sort("len", descending=True)
    )
    insect_windows = wide.filter(
        pl.col("labels").list.eval(pl.element().is_in(list(son))).list.any()
    ).height
    per_son = (
        long.filter(pl.col("label").is_in(list(son)))
        .group_by("label")
        .len()
        .sort("len", descending=True)
        .rename({"len": "n_present_windows"})
    )
    labeled = set(long["label"].unique().to_list())
    class_brk = (
        tax.filter(pl.col("primary_label").is_in(list(labeled)))
        .group_by("class_name")
        .len()
        .sort("len", descending=True)
    )

    per_son.write_csv(out / "sonotype_presence.csv")
    site_counts.write_csv(out / "site_file_counts.csv")
    class_brk.write_csv(out / "labeled_class_breakdown.csv")

    logger.info("files=%d windows=%d labels=%d", files, long.select(pl.col("window_id").n_unique()).item(), len(labeled))
    logger.info("insect-bearing windows=%d  sonotypes=%d", insect_windows, len(son))
    logger.info("sites: %s", site_counts.to_dicts())
    logger.info("class breakdown: %s", class_brk.to_dicts())
    logger.info("wrote summaries under %s", out)
    print(per_son)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run_summary()


if __name__ == "__main__":
    main()
