"""Luigi tasks for BirdCLEF 2026 EDA.

Each task reads raw competition data and produces a derived parquet dataset
in the artifacts directory. Tasks declare dependencies so Luigi handles
ordering and skips completed work.
"""

import json
from multiprocessing import Pool
from pathlib import Path

import luigi
import polars as pl
import soundfile as sf

# Default paths — override via CLI or luigi.cfg
DATA_DIR = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026"
ARTIFACTS_DIR = "artifacts"


def _parse_time_to_seconds(time_str: str) -> int:
    """Parse HH:MM:SS to total seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _classify_label_format(col: pl.Expr) -> pl.Expr:
    """Classify a primary_label column into ebird|numeric|songtype."""
    return (
        pl.when(col.str.contains(r"son\d+$"))
        .then(pl.lit("songtype"))
        .when(col.str.contains(r"^\d+$"))
        .then(pl.lit("numeric"))
        .otherwise(pl.lit("ebird"))
    )


class EDATask(luigi.Task):
    """Base task with shared parameters."""

    data_dir = luigi.Parameter(default=DATA_DIR)
    artifacts_dir = luigi.Parameter(default=ARTIFACTS_DIR)

    def artifact(self, name: str) -> luigi.LocalTarget:
        path = Path(self.artifacts_dir) / name
        return luigi.LocalTarget(str(path))

    def ensure_parent(self, target: luigi.LocalTarget):
        Path(target.path).parent.mkdir(parents=True, exist_ok=True)


class LabelMapping(EDATask):
    """Build label mapping from primary_label to submission column index.

    Handles three label formats: eBird codes, iNat numeric IDs, and
    47158 song-type splits. This is blocking infrastructure for all models.

    Output: artifacts/label_mapping.json
    Format: {primary_label: column_index, ...} (234 entries)
    """

    def output(self):
        return self.artifact("label_mapping.json")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        # read submission header to get target columns in order
        submission_header = pl.read_csv(
            data / "sample_submission.csv", n_rows=0
        ).columns
        target_cols = submission_header[1:]  # skip row_id

        # build mapping: label -> index
        mapping = {label: idx for idx, label in enumerate(target_cols)}

        # verify against taxonomy
        taxonomy = pl.read_csv(data / "taxonomy.csv")
        tax_labels = set(taxonomy["primary_label"].to_list())
        sub_labels = set(target_cols)

        info = {
            "mapping": mapping,
            "n_targets": len(target_cols),
            "in_taxonomy_not_submission": sorted(tax_labels - sub_labels),
            "in_submission_not_taxonomy": sorted(sub_labels - tax_labels),
            "target_columns": target_cols,
        }

        with open(self.output().path, "w") as f:
            json.dump(info, f, indent=2)

        print(
            f"LabelMapping: {len(mapping)} targets, "
            f"{len(tax_labels - sub_labels)} in taxonomy but not submission, "
            f"{len(sub_labels - tax_labels)} in submission but not taxonomy"
        )


class TrainMetadata(EDATask):
    """Join train.csv with taxonomy.csv, add derived columns.

    Output: artifacts/train_metadata.parquet
    Columns: all train.csv cols + class_name, inat_taxon_id (from taxonomy)
             + n_secondary, has_location, label_format
    """

    def output(self):
        return self.artifact("train_metadata.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        train = pl.read_csv(data / "train.csv")
        taxonomy = pl.read_csv(data / "taxonomy.csv").select(
            "primary_label", "class_name"
        )

        # parse secondary_labels: "['label1','label2']" -> count
        df = train.with_columns(
            pl.col("secondary_labels")
            .str.strip_chars("[]")
            .str.replace_all("'", "", literal=True)
            .str.strip_chars()
            .alias("_sec_clean"),
        ).with_columns(
            pl.when(pl.col("_sec_clean") == "")
            .then(pl.lit(0))
            .otherwise(pl.col("_sec_clean").str.count_matches(",") + 1)
            .alias("n_secondary"),
            (pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()).alias(
                "has_location"
            ),
            _classify_label_format(pl.col("primary_label")).alias("label_format"),
        )
        df = df.drop("_sec_clean")

        # join taxonomy for class_name (left join — keep all train rows)
        df = df.join(taxonomy, on="primary_label", how="left")

        df.write_parquet(self.output().path)
        print(f"TrainMetadata: {df.shape[0]} rows, {df.shape[1]} cols")


class TaxonomyCoverage(EDATask):
    """Compare taxonomy entries against train_audio directories and submission.

    Output: artifacts/taxonomy_coverage.parquet
    Columns: primary_label, class_name, has_train_audio, n_train_recordings,
             label_format, in_submission, is_songtype, parent_species
    """

    def requires(self):
        return TrainMetadata(
            data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
        )

    def output(self):
        return self.artifact("taxonomy_coverage.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        taxonomy = pl.read_csv(data / "taxonomy.csv")
        train_meta = pl.read_parquet(self.input().path)

        # count recordings per species from train metadata
        rec_counts = (
            train_meta.group_by("primary_label")
            .agg(pl.len().alias("n_train_recordings"))
        )

        # check which species dirs exist
        train_dirs = {d.name for d in (data / "train_audio").iterdir() if d.is_dir()}

        # parse submission columns
        submission_header = pl.read_csv(
            data / "sample_submission.csv", n_rows=0
        ).columns
        submission_species = set(submission_header[1:])

        df = taxonomy.with_columns(
            pl.col("primary_label")
            .is_in(list(train_dirs))
            .alias("has_train_audio"),
            pl.col("primary_label")
            .is_in(list(submission_species))
            .alias("in_submission"),
            _classify_label_format(pl.col("primary_label")).alias("label_format"),
            pl.col("primary_label")
            .str.contains(r"son\d+$")
            .alias("is_songtype"),
            # for song-type entries, extract parent species ID
            pl.when(pl.col("primary_label").str.contains(r"son\d+$"))
            .then(pl.col("primary_label").str.replace(r"son\d+$", ""))
            .otherwise(pl.lit(None))
            .alias("parent_species"),
        )

        df = df.join(rec_counts, on="primary_label", how="left").with_columns(
            pl.col("n_train_recordings").fill_null(0)
        )

        df.write_parquet(self.output().path)
        missing = df.filter(~pl.col("has_train_audio") & ~pl.col("is_songtype"))
        songtype = df.filter(pl.col("is_songtype"))
        print(
            f"TaxonomyCoverage: {df.shape[0]} species, "
            f"{missing.shape[0]} genuinely missing training audio, "
            f"{songtype.shape[0]} song-type entries (share parent dir)"
        )


class ClassDistribution(EDATask):
    """Per-species and per-class recording counts.

    Output: artifacts/class_distribution.parquet
    Columns: primary_label, class_name, collection, n_recordings,
             label_format, common_name, mean_rating, etc.
    """

    def requires(self):
        return TrainMetadata(
            data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
        )

    def output(self):
        return self.artifact("class_distribution.parquet")

    def run(self):
        self.ensure_parent(self.output())
        train = pl.read_parquet(self.input().path)

        # per species x collection counts
        dist = (
            train.group_by("primary_label", "class_name", "collection", "label_format", "common_name")
            .agg(
                pl.len().alias("n_recordings"),
                pl.col("rating").mean().alias("mean_rating"),
                pl.col("rating").median().alias("median_rating"),
                pl.col("n_secondary").mean().alias("mean_n_secondary"),
                pl.col("has_location").mean().alias("frac_with_location"),
            )
            .sort("n_recordings", descending=True)
        )

        dist.write_parquet(self.output().path)
        total_species = dist.select("primary_label").n_unique()
        print(f"ClassDistribution: {total_species} species across {dist.shape[0]} species x collection groups")


class SoundscapeLabels(EDATask):
    """Parse and enrich soundscape labels.

    Explodes semicolon-separated multi-labels into one row per species per window,
    then joins with taxonomy for class info. Converts time strings to seconds.

    Output: artifacts/soundscape_labels.parquet
    Columns: filename, start_sec, end_sec, primary_label, class_name,
             window_idx, n_species_in_window, label_format
    """

    def output(self):
        return self.artifact("soundscape_labels.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        raw = pl.read_csv(data / "train_soundscapes_labels.csv")
        taxonomy = pl.read_csv(data / "taxonomy.csv").select(
            "primary_label", "class_name", "common_name"
        )

        # convert time strings to seconds
        raw = raw.with_columns(
            pl.col("start").map_elements(_parse_time_to_seconds, return_dtype=pl.Int32).alias("start_sec"),
            pl.col("end").map_elements(_parse_time_to_seconds, return_dtype=pl.Int32).alias("end_sec"),
        ).drop("start", "end")

        # count species per window before exploding
        window_counts = raw.with_columns(
            pl.col("primary_label")
            .str.split(";")
            .list.len()
            .alias("n_species_in_window"),
        )

        # explode to one row per species per window
        exploded = window_counts.with_columns(
            pl.col("primary_label").str.split(";"),
        ).explode("primary_label")

        # trim whitespace from labels after split
        exploded = exploded.with_columns(
            pl.col("primary_label").str.strip_chars(),
        )

        # add window index and label format
        exploded = exploded.with_columns(
            pl.col("start_sec").rank("ordinal").over("filename").cast(pl.Int32).alias("window_idx"),
            _classify_label_format(pl.col("primary_label")).alias("label_format"),
        )

        # join taxonomy
        exploded = exploded.join(taxonomy, on="primary_label", how="left")

        exploded.write_parquet(self.output().path)
        n_windows = window_counts.shape[0]
        n_files = window_counts.select("filename").n_unique()
        n_species = exploded.select("primary_label").n_unique()
        print(
            f"SoundscapeLabels: {exploded.shape[0]} rows from {n_windows} windows "
            f"across {n_files} files, {n_species} unique species"
        )


class SoundscapeWindowStats(EDATask):
    """Per-window summary statistics for soundscapes.

    Output: artifacts/soundscape_window_stats.parquet
    Columns: filename, start_sec, end_sec, n_species, n_aves, n_amphibia,
             n_insecta, n_mammalia, n_reptilia, has_multi_taxa
    """

    def requires(self):
        return SoundscapeLabels(
            data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
        )

    def output(self):
        return self.artifact("soundscape_window_stats.parquet")

    def run(self):
        self.ensure_parent(self.output())
        labels = pl.read_parquet(self.input().path)

        stats = (
            labels.group_by("filename", "start_sec", "end_sec")
            .agg(
                pl.len().alias("n_species"),
                (pl.col("class_name") == "Aves").sum().alias("n_aves"),
                (pl.col("class_name") == "Amphibia").sum().alias("n_amphibia"),
                (pl.col("class_name") == "Insecta").sum().alias("n_insecta"),
                (pl.col("class_name") == "Mammalia").sum().alias("n_mammalia"),
                (pl.col("class_name") == "Reptilia").sum().alias("n_reptilia"),
            )
            .with_columns(
                (
                    (pl.col("n_aves") > 0).cast(pl.Int8)
                    + (pl.col("n_amphibia") > 0).cast(pl.Int8)
                    + (pl.col("n_insecta") > 0).cast(pl.Int8)
                    + (pl.col("n_mammalia") > 0).cast(pl.Int8)
                    + (pl.col("n_reptilia") > 0).cast(pl.Int8)
                    > 1
                ).alias("has_multi_taxa")
            )
            .sort("filename", "start_sec")
        )

        stats.write_parquet(self.output().path)
        print(
            f"SoundscapeWindowStats: {stats.shape[0]} windows, "
            f"mean {stats['n_species'].mean():.1f} species/window, "
            f"{stats.filter(pl.col('has_multi_taxa')).shape[0]} multi-taxa windows"
        )


class SoundscapeCoverage(EDATask):
    """Which species appear in soundscapes vs focal-only?

    Output: artifacts/soundscape_coverage.parquet
    Columns: primary_label, class_name, n_focal, n_soundscape_windows,
             in_focal, in_soundscape
    """

    def requires(self):
        return {
            "train": TrainMetadata(
                data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
            ),
            "soundscape": SoundscapeLabels(
                data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
            ),
        }

    def output(self):
        return self.artifact("soundscape_coverage.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        train = pl.read_parquet(self.input()["train"].path)
        sc_labels = pl.read_parquet(self.input()["soundscape"].path)
        taxonomy = pl.read_csv(data / "taxonomy.csv").select(
            "primary_label", "class_name"
        )

        focal_counts = train.group_by("primary_label").agg(
            pl.len().alias("n_focal")
        )
        sc_counts = sc_labels.group_by("primary_label").agg(
            pl.len().alias("n_soundscape_windows")
        )

        df = (
            taxonomy.join(focal_counts, on="primary_label", how="left")
            .join(sc_counts, on="primary_label", how="left")
            .with_columns(
                pl.col("n_focal").fill_null(0),
                pl.col("n_soundscape_windows").fill_null(0),
            )
            .with_columns(
                (pl.col("n_focal") > 0).alias("in_focal"),
                (pl.col("n_soundscape_windows") > 0).alias("in_soundscape"),
            )
            .sort("primary_label")
        )

        df.write_parquet(self.output().path)
        both = df.filter(pl.col("in_focal") & pl.col("in_soundscape")).shape[0]
        sc_only = df.filter(~pl.col("in_focal") & pl.col("in_soundscape")).shape[0]
        focal_only = df.filter(pl.col("in_focal") & ~pl.col("in_soundscape")).shape[0]
        neither = df.filter(~pl.col("in_focal") & ~pl.col("in_soundscape")).shape[0]
        print(
            f"SoundscapeCoverage: {both} in both, {focal_only} focal-only, "
            f"{sc_only} soundscape-only, {neither} in neither"
        )


def _get_duration(path: str) -> dict:
    """Get duration of a single audio file using soundfile (no librosa overhead)."""
    try:
        info = sf.info(path)
        return {"path": path, "duration": info.duration, "sample_rate": info.samplerate, "error": None}
    except Exception as e:
        return {"path": path, "duration": None, "sample_rate": None, "error": str(e)}


class AudioDurations(EDATask):
    """Measure duration of all focal training audio files.

    Uses multiprocessing with soundfile (lightweight, no librosa/numba needed).

    Output: artifacts/audio_durations.parquet
    Columns: path (relative), primary_label, duration, sample_rate, error
    """

    n_workers = luigi.IntParameter(default=8)

    def output(self):
        return self.artifact("audio_durations.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        audio_dir = data / "train_audio"
        audio_prefix = str(audio_dir) + "/"
        files = sorted(audio_dir.glob("**/*.ogg"))
        paths = [str(f) for f in files]
        print(f"AudioDurations: measuring {len(paths)} files with {self.n_workers} workers")

        results = []
        with Pool(self.n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_get_duration, paths)):
                results.append(result)
                if (i + 1) % 5000 == 0:
                    print(f"  progress: {i + 1}/{len(paths)}")

        df = pl.DataFrame(results)
        # make path relative and extract species label
        df = df.with_columns(
            pl.col("path")
            .str.replace(audio_prefix, "", literal=True)
            .alias("path"),
        ).with_columns(
            pl.col("path").str.split("/").list.first().alias("primary_label"),
        )

        df.write_parquet(self.output().path)
        valid = df.filter(pl.col("duration").is_not_null())
        print(
            f"AudioDurations: {valid.shape[0]} files, "
            f"{valid['duration'].sum() / 3600:.1f} hours total, "
            f"mean {valid['duration'].mean():.1f}s, "
            f"median {valid['duration'].median():.1f}s"
        )
        errors = df.filter(pl.col("error").is_not_null())
        if errors.shape[0] > 0:
            print(f"  {errors.shape[0]} errors")


class AllEDA(EDATask):
    """Meta-task that runs all EDA tasks."""

    n_workers = luigi.IntParameter(default=8)

    def requires(self):
        params = dict(data_dir=self.data_dir, artifacts_dir=self.artifacts_dir)
        return [
            LabelMapping(**params),
            TrainMetadata(**params),
            TaxonomyCoverage(**params),
            ClassDistribution(**params),
            SoundscapeLabels(**params),
            SoundscapeWindowStats(**params),
            SoundscapeCoverage(**params),
            AudioDurations(n_workers=self.n_workers, **params),
        ]

    def output(self):
        return self.artifact("_eda_complete")

    def run(self):
        self.ensure_parent(self.output())
        with self.output().open("w") as f:
            f.write("done\n")
        print("All EDA tasks complete!")
