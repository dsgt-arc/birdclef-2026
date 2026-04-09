"""Partition extracted feature shards by species and site.

Reads the 16 randomly-sharded parquet files once and writes:
    focal/by_species/{species_name}.parquet
    soundscape_unlabeled/by_site/{site_id}.parquet

Downstream tasks can then load exactly the data they need with a single
read_parquet() call instead of scanning all shards.

This is a one-time transformation — run once after extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from birdclef_106.analysis.base import AnalysisTask
from birdclef_106.data_loading import (
    extract_site,
    load_focal_shards,
    load_unlabeled_shards,
)

logger = logging.getLogger(__name__)


class PartitionFeatures(AnalysisTask):
    """Partition extracted features by species (focal) and site (soundscapes)."""

    def output(self):
        return self.artifact("_partition_complete")

    def run(self):
        self.ensure_parent(self.output())
        features_dir = Path(self.features_dir).expanduser()

        self._partition_focal(features_dir)
        self._partition_unlabeled(features_dir)

        Path(self.output().path).touch()

    def _partition_focal(self, features_dir: Path):
        """Partition focal shards by primary_label → by_species/{label}.parquet."""
        out_dir = features_dir / "focal" / "by_species"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("PartitionFeatures: partitioning focal by species")

        # Collect per-species DataFrames across shards
        species_dfs: dict[str, list[pl.DataFrame]] = {}

        for i, shard in enumerate(load_focal_shards(self.features_dir)):
            for label, group in shard.group_by("primary_label"):
                sp = str(label[0]).strip()
                if sp not in species_dfs:
                    species_dfs[sp] = []
                species_dfs[sp].append(group)

            if i % 4 == 0:
                logger.info("  focal shard %d done (%d species so far)", i, len(species_dfs))

        # Write per-species parquets
        for sp, dfs in species_dfs.items():
            df = pl.concat(dfs)
            df.write_parquet(out_dir / f"{sp}.parquet")

        logger.info("PartitionFeatures: wrote %d species parquets to %s",
                     len(species_dfs), out_dir)
        del species_dfs

    def _partition_unlabeled(self, features_dir: Path):
        """Partition unlabeled shards by site → by_site/{site_id}.parquet."""
        out_dir = features_dir / "soundscape_unlabeled" / "by_site"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("PartitionFeatures: partitioning unlabeled by site")

        site_dfs: dict[str, list[pl.DataFrame]] = {}

        for i, shard in enumerate(load_unlabeled_shards(self.features_dir)):
            # Add site column
            shard = shard.with_columns(
                pl.col("filename").map_elements(
                    lambda fn: extract_site(str(fn)), return_dtype=pl.Utf8
                ).alias("site")
            )

            for site, group in shard.group_by("site"):
                site_id = str(site[0])
                if site_id not in site_dfs:
                    site_dfs[site_id] = []
                site_dfs[site_id].append(group)

            if i % 4 == 0:
                logger.info("  unlabeled shard %d done (%d sites so far)", i, len(site_dfs))

        for site_id, dfs in site_dfs.items():
            df = pl.concat(dfs)
            df.write_parquet(out_dir / f"{site_id}.parquet")

        logger.info("PartitionFeatures: wrote %d site parquets to %s",
                     len(site_dfs), out_dir)
        del site_dfs
