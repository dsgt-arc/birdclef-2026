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


# Schema override: primary_label is mixed numeric/alpha, force string
_STR_OVERRIDES = {"primary_label": pl.Utf8, "secondary_labels": pl.Utf8}


def _read_csv(path, **kwargs):
    """Read CSV with primary_label forced to string (Polars infers i64 from leading numeric rows)."""
    schema_overrides = {**_STR_OVERRIDES, **kwargs.pop("schema_overrides", {})}
    return pl.read_csv(path, schema_overrides=schema_overrides, **kwargs)


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
        taxonomy = _read_csv(data / "taxonomy.csv")
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

        train = _read_csv(data / "train.csv")
        taxonomy = _read_csv(data / "taxonomy.csv").select(
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

        taxonomy = _read_csv(data / "taxonomy.csv")
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

        raw = _read_csv(data / "train_soundscapes_labels.csv")
        taxonomy = _read_csv(data / "taxonomy.csv").select(
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


class SoundscapeTaxaSummary(EDATask):
    """Per-taxa summary of soundscape label presence and species counts.

    Output: artifacts/soundscape_taxa_summary.parquet
    Columns: class_name, n_unique_species, n_windows_present, frac_windows,
             mean_species_per_window (when present)
    """

    def requires(self):
        return {
            "labels": SoundscapeLabels(
                data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
            ),
            "window_stats": SoundscapeWindowStats(
                data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
            ),
        }

    def output(self):
        return self.artifact("soundscape_taxa_summary.parquet")

    def run(self):
        self.ensure_parent(self.output())
        labels = pl.read_parquet(self.input()["labels"].path)
        stats = pl.read_parquet(self.input()["window_stats"].path)

        total_windows = stats.shape[0]

        # per-taxa: unique species and window presence
        taxa_col_map = {
            "Aves": "n_aves",
            "Amphibia": "n_amphibia",
            "Insecta": "n_insecta",
            "Mammalia": "n_mammalia",
            "Reptilia": "n_reptilia",
        }

        rows = []
        for class_name, col in taxa_col_map.items():
            n_species = labels.filter(pl.col("class_name") == class_name).select("primary_label").n_unique()
            present = stats.filter(pl.col(col) > 0)
            n_windows = present.shape[0]
            mean_per_window = present[col].mean() if n_windows > 0 else 0.0
            rows.append({
                "class_name": class_name,
                "n_unique_species": n_species,
                "n_windows_present": n_windows,
                "frac_windows": n_windows / total_windows,
                "mean_count_per_window": mean_per_window,
            })

        df = pl.DataFrame(rows).sort("n_windows_present", descending=True)

        # also add overall density stats
        quantiles = {
            f"species_p{int(q*100)}": stats["n_species"].quantile(q)
            for q in [0.25, 0.5, 0.75, 0.9, 1.0]
        }

        df.write_parquet(self.output().path)
        print(f"SoundscapeTaxaSummary: {total_windows} windows, {labels.select('primary_label').n_unique()} species")
        for row in df.iter_rows(named=True):
            print(
                f"  {row['class_name']}: {row['n_unique_species']} species, "
                f"{row['n_windows_present']} windows ({row['frac_windows']*100:.1f}%)"
            )
        print(f"  Species/window: median={quantiles['species_p50']:.0f}, "
              f"p90={quantiles['species_p90']:.0f}, max={quantiles['species_p100']:.0f}")


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
        taxonomy = _read_csv(data / "taxonomy.csv").select(
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


class GeographicGap(EDATask):
    """Analyze geographic overlap between focal training data and Pantanal test site.

    Pantanal bounding box (from recording_location.txt):
      lat: -21.6 to -16.5, lon: -57.6 to -55.9

    Output: artifacts/geographic_gap.parquet
    Columns: primary_label, class_name, n_total, n_with_location,
             n_in_pantanal, frac_in_pantanal, mean_lat, mean_lon
    """

    # Pantanal bounding box from recording_location.txt
    PANTANAL_LAT_MIN = -21.6
    PANTANAL_LAT_MAX = -16.5
    PANTANAL_LON_MIN = -57.6
    PANTANAL_LON_MAX = -55.9

    def requires(self):
        return TrainMetadata(
            data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
        )

    def output(self):
        return self.artifact("geographic_gap.parquet")

    def run(self):
        self.ensure_parent(self.output())
        train = pl.read_parquet(self.input().path)

        # flag recordings within Pantanal bounding box
        train = train.with_columns(
            (
                pl.col("latitude").is_between(self.PANTANAL_LAT_MIN, self.PANTANAL_LAT_MAX)
                & pl.col("longitude").is_between(self.PANTANAL_LON_MIN, self.PANTANAL_LON_MAX)
            )
            .fill_null(False)
            .alias("in_pantanal"),
        )

        # per-species geographic summary
        geo = (
            train.group_by("primary_label", "class_name")
            .agg(
                pl.len().alias("n_total"),
                pl.col("has_location").sum().alias("n_with_location"),
                pl.col("in_pantanal").sum().alias("n_in_pantanal"),
                pl.col("latitude").mean().alias("mean_lat"),
                pl.col("longitude").mean().alias("mean_lon"),
            )
            .with_columns(
                (pl.col("n_in_pantanal") / pl.col("n_with_location").cast(pl.Float64))
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("frac_in_pantanal"),
            )
            .sort("frac_in_pantanal", descending=True)
        )

        geo.write_parquet(self.output().path)
        total_with_loc = train.filter(pl.col("has_location")).shape[0]
        total_pantanal = train.filter(pl.col("in_pantanal")).shape[0]
        species_in_pantanal = geo.filter(pl.col("n_in_pantanal") > 0).shape[0]
        print(
            f"GeographicGap: {total_pantanal}/{total_with_loc} recordings in Pantanal "
            f"({total_pantanal / max(total_with_loc, 1) * 100:.1f}%), "
            f"{species_in_pantanal} species with Pantanal recordings"
        )


class TestSetStructure(EDATask):
    """Parse sample_submission row IDs to extract test set structure.

    Row ID format: BC2026_Test_{file_id}_{site}_{date}_{time}_{offset_sec}

    Output: artifacts/test_set_structure.parquet
    Columns: row_id, file_id, site, date, time, offset_sec, hour, file_key
    """

    def output(self):
        return self.artifact("test_set_structure.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        submission = pl.read_csv(data / "sample_submission.csv", n_rows=0)
        # read just row_id column (first col) — avoid loading all 234 target columns
        sub = pl.read_csv(data / "sample_submission.csv").select("row_id")

        # parse row_id: BC2026_Test_0001_S05_20250227_010002_5
        df = sub.with_columns(
            pl.col("row_id").str.extract(r"BC2026_Test_(\d+)_", 1).alias("file_id"),
            pl.col("row_id").str.extract(r"_(S\d+)_", 1).alias("site"),
            pl.col("row_id").str.extract(r"_S\d+_(\d{8})_", 1).alias("date"),
            pl.col("row_id").str.extract(r"_(\d{6})_\d+$", 1).alias("time"),
            pl.col("row_id").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("offset_sec"),
        ).with_columns(
            pl.col("time").str.slice(0, 2).cast(pl.Int8).alias("hour"),
            # unique file key (file_id + site + date + time)
            (pl.col("file_id") + "_" + pl.col("site") + "_" + pl.col("date") + "_" + pl.col("time"))
            .alias("file_key"),
        )

        df.write_parquet(self.output().path)
        n_files = df.select("file_key").n_unique()
        n_sites = df.select("site").n_unique()
        dates = df.select("date").unique().sort("date")
        hours = df.select("hour").unique().sort("hour")
        print(
            f"TestSetStructure: {df.shape[0]} windows across {n_files} files, "
            f"{n_sites} sites, {dates.shape[0]} dates"
        )
        print(f"  Date range: {dates['date'][0]} to {dates['date'][-1]}")
        print(f"  Hours: {sorted(hours['hour'].to_list())}")


class SoundscapeNegatives(EDATask):
    """Analyze whether unlabeled windows in labeled soundscape files can serve as negatives.

    Compares the set of 5s windows that exist in labeled files against the
    windows that actually have labels. Gaps are candidate hard negatives.

    Output: artifacts/soundscape_negatives.parquet
    Columns: filename, max_labeled_sec, n_labeled_windows, n_possible_windows,
             n_unlabeled_windows, frac_labeled
    """

    def requires(self):
        return SoundscapeLabels(
            data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
        )

    def output(self):
        return self.artifact("soundscape_negatives.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)
        labels = pl.read_parquet(self.input().path)

        # get labeled window counts per file
        labeled = (
            labels.group_by("filename")
            .agg(
                pl.col("start_sec").n_unique().alias("n_labeled_windows"),
                pl.col("start_sec").max().alias("max_labeled_sec"),
            )
        )

        # estimate total possible 5s windows per file from max labeled timestamp
        # (actual file duration would be more accurate, but this gives a lower bound)
        labeled = labeled.with_columns(
            ((pl.col("max_labeled_sec") / 5) + 1).cast(pl.Int32).alias("n_possible_windows"),
        ).with_columns(
            (pl.col("n_possible_windows") - pl.col("n_labeled_windows")).alias("n_unlabeled_windows"),
            (pl.col("n_labeled_windows") / pl.col("n_possible_windows").cast(pl.Float64))
            .alias("frac_labeled"),
        ).sort("frac_labeled")

        labeled.write_parquet(self.output().path)
        total_labeled = labeled["n_labeled_windows"].sum()
        total_possible = labeled["n_possible_windows"].sum()
        total_unlabeled = labeled["n_unlabeled_windows"].sum()
        print(
            f"SoundscapeNegatives: {labeled.shape[0]} labeled files, "
            f"{total_labeled}/{total_possible} windows labeled "
            f"({total_labeled / max(total_possible, 1) * 100:.1f}%), "
            f"{total_unlabeled} candidate negative windows"
        )


class SpeciesOverlap2025(EDATask):
    """Compare BirdCLEF 2026 species with 2025 to assess transfer learning viability.

    Output: artifacts/species_overlap_2025.parquet
    Columns: primary_label, class_name, in_2025, in_2026, label_2025, label_2026
    """

    birdclef_2025_csv = luigi.Parameter(
        default="~/scratch/birdclef-2026/data/birdclef-2025/train.csv"
    )

    def output(self):
        return self.artifact("species_overlap_2025.parquet")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)

        taxonomy_2026 = _read_csv(data / "taxonomy.csv")

        csv_2025 = Path(self.birdclef_2025_csv).expanduser()
        if not csv_2025.exists():
            print(f"WARNING: BirdCLEF 2025 train.csv not found at {csv_2025}")
            print("Producing 2026-only output.")
            df = taxonomy_2026.select("primary_label", "class_name").with_columns(
                pl.lit(False).alias("in_2025"),
                pl.lit(True).alias("in_2026"),
                pl.lit(None).cast(pl.Utf8).alias("scientific_name_2025"),
            )
            df.write_parquet(self.output().path)
            return

        train_2025 = _read_csv(csv_2025)
        # 2025 uses primary_label (eBird codes or numeric IDs) and scientific_name
        species_2025 = (
            train_2025.select("primary_label", "scientific_name")
            .unique()
            .rename({"primary_label": "label_2025", "scientific_name": "sci_name_2025"})
        )

        # match by scientific_name (more robust than label which may change format)
        taxonomy_2026 = taxonomy_2026.with_columns(
            pl.lit(True).alias("in_2026"),
        )
        species_2025 = species_2025.with_columns(
            pl.lit(True).alias("in_2025"),
        )

        # full outer join on scientific_name
        merged = (
            taxonomy_2026.select("primary_label", "class_name", "scientific_name", "in_2026")
            .join(
                species_2025,
                left_on="scientific_name",
                right_on="sci_name_2025",
                how="full",
                coalesce=True,
            )
            .with_columns(
                pl.col("in_2025").fill_null(False),
                pl.col("in_2026").fill_null(False),
            )
            .sort("primary_label")
        )

        merged.write_parquet(self.output().path)
        both = merged.filter(pl.col("in_2025") & pl.col("in_2026")).shape[0]
        only_2026 = merged.filter(~pl.col("in_2025") & pl.col("in_2026")).shape[0]
        only_2025 = merged.filter(pl.col("in_2025") & ~pl.col("in_2026")).shape[0]
        print(
            f"SpeciesOverlap2025: {both} in both years, "
            f"{only_2026} new in 2026, {only_2025} dropped from 2025"
        )


class ClassWeights(EDATask):
    """Compute recommended class weights for cmAP-optimal training.

    Produces inverse-frequency weights normalized so the mean weight = 1.0.
    For cmAP (class-mean AP), rare species matter equally — these weights
    can be used directly in a weighted sampler or loss function.

    Output: artifacts/class_weights.json
    Format: {primary_label: weight, ...}
    """

    def requires(self):
        return TrainMetadata(
            data_dir=self.data_dir, artifacts_dir=self.artifacts_dir
        )

    def output(self):
        return self.artifact("class_weights.json")

    def run(self):
        self.ensure_parent(self.output())
        data = Path(self.data_dir)
        train = pl.read_parquet(self.input().path)

        # count per species
        counts = (
            train.group_by("primary_label")
            .agg(pl.len().alias("n"))
        )

        # inverse frequency, normalized so mean = 1.0
        total = counts["n"].sum()
        n_classes = counts.shape[0]
        counts = counts.with_columns(
            (total / (n_classes * pl.col("n").cast(pl.Float64))).alias("weight"),
        )

        # normalize so mean weight = 1.0
        mean_w = counts["weight"].mean()
        counts = counts.with_columns(
            (pl.col("weight") / mean_w).alias("weight"),
        )

        weights = dict(zip(
            counts["primary_label"].to_list(),
            [round(w, 4) for w in counts["weight"].to_list()],
        ))

        # include summary stats
        info = {
            "weights": weights,
            "n_classes": n_classes,
            "min_count": int(counts["n"].min()),
            "max_count": int(counts["n"].max()),
            "median_count": int(counts["n"].median()),
            "imbalance_ratio": round(int(counts["n"].max()) / max(int(counts["n"].min()), 1), 1),
            "min_weight": round(float(counts["weight"].min()), 4),
            "max_weight": round(float(counts["weight"].max()), 4),
        }

        with open(self.output().path, "w") as f:
            json.dump(info, f, indent=2)

        print(
            f"ClassWeights: {n_classes} classes, "
            f"imbalance ratio {info['imbalance_ratio']}x, "
            f"weight range [{info['min_weight']}, {info['max_weight']}]"
        )


class AllEDA(EDATask):
    """Meta-task that runs all EDA tasks."""

    n_workers = luigi.IntParameter(default=8)
    birdclef_2025_csv = luigi.Parameter(
        default="~/scratch/birdclef-2026/data/birdclef-2025/train.csv"
    )

    def requires(self):
        params = dict(data_dir=self.data_dir, artifacts_dir=self.artifacts_dir)
        return [
            LabelMapping(**params),
            TrainMetadata(**params),
            TaxonomyCoverage(**params),
            ClassDistribution(**params),
            ClassWeights(**params),
            SoundscapeLabels(**params),
            SoundscapeWindowStats(**params),
            SoundscapeTaxaSummary(**params),
            SoundscapeCoverage(**params),
            SoundscapeNegatives(**params),
            GeographicGap(**params),
            TestSetStructure(**params),
            SpeciesOverlap2025(birdclef_2025_csv=self.birdclef_2025_csv, **params),
            AudioDurations(n_workers=self.n_workers, **params),
        ]

    def output(self):
        return self.artifact("_eda_complete")

    def run(self):
        self.ensure_parent(self.output())
        with self.output().open("w") as f:
            f.write("done\n")
        print("All EDA tasks complete!")
