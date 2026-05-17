"""Resumable Luigi driver: auto-motif over the soundscape∩train species set.

Each species is one :class:`AutoMotifSpeciesTask`. Its Luigi target is a
per-species status JSON under ``artifacts/auto_motif/[<mode>/]_batch/status/``,
so a completed (or attempted) species is skipped on re-run — resumability comes
for free from Luigi's target check. Failures are captured into the status JSON
(``status="failed"`` + traceback summary) and the task still *succeeds*, so one
bad species never blocks the batch; the merged manifest records every outcome.

The default ``consensus``/``global`` mode writes to the legacy paths
(``auto_motif/<label>``, ``_batch/status/``, ``_batch/manifest.csv``) so the
validated batch is untouched. Any other ``(band_mode, cluster_mode)`` is
written under a mode-tagged subtree (matching the A/B artifact convention,
e.g. ``auto_motif/_gabor_stack/<label>``) with its own resumable status dir
and manifest — a ``gabor_stack`` batch can never clobber the ``consensus``
batch.

To re-run a single species (e.g. after a transient failure), delete its
``status/<label>.json`` and rebuild. The species set comes from
:func:`data_analysis.common.selected_species.soundscape_train_species`
(filtered by ``--min-focal``, default 8), or an explicit ``--labels`` subset.

CLI::

    python -m data_analysis.auto_motif.batch --list                  # preview set
    python -m data_analysis.auto_motif.batch --label 22967           # one species
    python -m data_analysis.auto_motif.batch                         # full consensus batch
    python -m data_analysis.auto_motif.batch \\
        --band-mode free --cluster-mode gabor_stack \\
        --labels 74113,555146,23158,...                              # mode-tagged subset
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import luigi
import polars as pl

from data_analysis.auto_motif.mining import AutoConfig
from data_analysis.auto_motif.mining import run as run_auto_motif
from data_analysis.auto_motif.mining import species_output_dir
from data_analysis.common.paths import (
    ARTIFACTS_DIR,
    TAXONOMY_CSV,
    recording_files_for_label,
)
from data_analysis.common.selected_species import soundscape_train_species

logger = logging.getLogger(__name__)

AUTO_MOTIF_DIR = ARTIFACTS_DIR / "auto_motif"

# Map (band_mode, cluster_mode) -> artifact subtree tag. The empty tag is the
# legacy validated path; every other entry matches the A/B convention exactly
# so batch and ad-hoc runs share one artifact tree. Unmapped combos raise
# rather than silently writing to a surprising path.
_MODE_TAG: dict[tuple[str, str], str] = {
    ("consensus", "global"): "",
    ("free", "global"): "_bandfree",
    ("free", "patch"): "_patch",
    ("free", "gabor_stack"): "_gabor_stack",
    ("gabor", "global"): "_gabor",
}

# Only the Marbled White-lipped Frog has hand-curated motif boxes today; pass
# them through so its run still reports curated recovery.
CURATED_PATHS = {
    "22967": ARTIFACTS_DIR
    / "marbled_frog"
    / "motif_boxes"
    / "curated"
    / "tables"
    / "curated_motif_boxes.csv",
}


def _mode_tag(band_mode: str, cluster_mode: str) -> str:
    try:
        return _MODE_TAG[(band_mode, cluster_mode)]
    except KeyError:
        raise ValueError(
            f"unmapped (band_mode={band_mode!r}, cluster_mode={cluster_mode!r}); "
            f"add it to _MODE_TAG"
        ) from None


def _batch_dir(tag: str) -> Path:
    return (AUTO_MOTIF_DIR / tag / "_batch") if tag else (AUTO_MOTIF_DIR / "_batch")


def _status_dir(tag: str) -> Path:
    return _batch_dir(tag) / "status"


def _species_dir(label: str, tag: str) -> Path:
    """Mode-aware analogue of :func:`species_output_dir` (legacy when tag='')."""
    return species_output_dir(label) if not tag else AUTO_MOTIF_DIR / tag / label


def _curated_path_for(label: str) -> Path | None:
    path = CURATED_PATHS.get(label)
    return path if path is not None and path.exists() else None


def _read_height(path: Path) -> int | None:
    if not path.exists():
        return None
    return pl.read_csv(path).height


def _collect_metrics(tables_root: Path) -> dict:
    """Pull a few headline numbers from a finished species' artifacts."""
    tables = tables_root / "tables"
    metrics: dict = {
        "n_bands": _read_height(tables / "auto_bands.csv"),
        "n_families": _read_height(tables / "auto_templates_metadata.csv"),
        "curated_recovery": None,
    }
    recovery = tables / "curated_recovery.csv"
    if recovery.exists():
        df = pl.read_csv(recovery)
        if df.height and "recovered" in df.columns:
            metrics["curated_recovery"] = float(df["recovered"].mean())
    return metrics


def _selected_labels(min_focal: int, labels: str) -> list[str]:
    """Explicit ``--labels`` subset if given, else the derived species set."""
    if labels:
        return [s for s in (x.strip() for x in labels.split(",")) if s]
    return [label for label, _ in soundscape_train_species(int(min_focal))]


class AutoMotifSpeciesTask(luigi.Task):
    """Run the auto-motif pipeline for one species; capture status as target."""

    label = luigi.Parameter()
    band_mode = luigi.Parameter(default="consensus")
    cluster_mode = luigi.Parameter(default="global")

    @property
    def _tag(self) -> str:
        return _mode_tag(str(self.band_mode), str(self.cluster_mode))

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(str(_status_dir(self._tag) / f"{self.label}.json"))

    def run(self) -> None:
        tag = self._tag
        status_dir = _status_dir(tag)
        status_dir.mkdir(parents=True, exist_ok=True)
        species_dir = _species_dir(str(self.label), tag)
        n_focal = len(recording_files_for_label(str(self.label)))
        record: dict = {
            "label": str(self.label),
            "band_mode": str(self.band_mode),
            "cluster_mode": str(self.cluster_mode),
            "n_focal": n_focal,
        }
        try:
            run_auto_motif(
                label=str(self.label),
                output_dir=None if not tag else species_dir,
                config=AutoConfig(
                    band_mode=str(self.band_mode),
                    cluster_mode=str(self.cluster_mode),
                ),
                curated_path=_curated_path_for(str(self.label)),
            )
            record["status"] = "ok"
            record["error"] = None
            record.update(_collect_metrics(species_dir))
        except Exception as exc:  # noqa: BLE001 — capture, don't abort the batch
            logger.exception("auto-motif failed for %s", self.label)
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-2000:]
            record.setdefault("n_bands", None)
            record.setdefault("n_families", None)
            record.setdefault("curated_recovery", None)
        with self.output().open("w") as fh:
            json.dump(record, fh, indent=2, default=str)


class AutoMotifBatchTask(luigi.WrapperTask):
    """Require an :class:`AutoMotifSpeciesTask` for every selected species."""

    min_focal = luigi.IntParameter(default=8)
    labels = luigi.Parameter(default="")
    band_mode = luigi.Parameter(default="consensus")
    cluster_mode = luigi.Parameter(default="global")

    def requires(self):
        return [
            AutoMotifSpeciesTask(
                label=label,
                band_mode=self.band_mode,
                cluster_mode=self.cluster_mode,
            )
            for label in _selected_labels(int(self.min_focal), str(self.labels))
        ]


class AutoMotifManifestTask(luigi.Task):
    """Merge per-species status JSONs into one manifest CSV."""

    min_focal = luigi.IntParameter(default=8)
    labels = luigi.Parameter(default="")
    band_mode = luigi.Parameter(default="consensus")
    cluster_mode = luigi.Parameter(default="global")

    def requires(self):
        return AutoMotifBatchTask(
            min_focal=self.min_focal,
            labels=self.labels,
            band_mode=self.band_mode,
            cluster_mode=self.cluster_mode,
        )

    def output(self) -> luigi.LocalTarget:
        tag = _mode_tag(str(self.band_mode), str(self.cluster_mode))
        return luigi.LocalTarget(str(_batch_dir(tag) / "manifest.csv"))

    def run(self) -> None:
        tag = _mode_tag(str(self.band_mode), str(self.cluster_mode))
        status_dir = _status_dir(tag)
        tax = (
            pl.read_csv(TAXONOMY_CSV, infer_schema_length=10000)
            .with_columns(pl.col("primary_label").cast(pl.Utf8))
            .select(["primary_label", "common_name", "class_name"])
        )
        tax_lookup = {
            r["primary_label"]: (r["common_name"], r["class_name"])
            for r in tax.iter_rows(named=True)
        }
        rows: list[dict] = []
        for label in _selected_labels(int(self.min_focal), str(self.labels)):
            status_path = status_dir / f"{label}.json"
            rec = json.loads(status_path.read_text()) if status_path.exists() else {
                "label": label,
                "status": "missing",
            }
            common_name, class_name = tax_lookup.get(label, (None, None))
            rows.append(
                {
                    "label": label,
                    "common_name": common_name,
                    "class_name": class_name,
                    "band_mode": rec.get("band_mode", str(self.band_mode)),
                    "cluster_mode": rec.get("cluster_mode", str(self.cluster_mode)),
                    "n_focal": rec.get("n_focal"),
                    "status": rec.get("status"),
                    "n_bands": rec.get("n_bands"),
                    "n_families": rec.get("n_families"),
                    "curated_recovery": rec.get("curated_recovery"),
                    "error": rec.get("error"),
                }
            )
        df = pl.DataFrame(rows)
        out = Path(self.output().path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(out)
        counts = df.group_by("status").len().sort("status").to_dicts()
        logger.info("auto-motif batch manifest: %d species %s", df.height, counts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-focal",
        type=int,
        default=8,
        help="Drop species with fewer focal recordings (default: 8)",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated label subset (overrides the derived species set)",
    )
    parser.add_argument(
        "--band-mode",
        default="consensus",
        help="AutoConfig.band_mode (default: consensus)",
    )
    parser.add_argument(
        "--cluster-mode",
        default="global",
        help="AutoConfig.cluster_mode (default: global)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Run a single species' task instead of the whole batch",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected species + focal counts and exit",
    )
    parser.add_argument(
        "--luigi-workers",
        type=int,
        default=1,
        help="Species to run concurrently (each saturates CPU internally)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        labels = _selected_labels(args.min_focal, args.labels)
        tag = _mode_tag(args.band_mode, args.cluster_mode) or "(legacy consensus/global)"
        print(f"{len(labels)} species [{tag}] (min_focal={args.min_focal}):")
        for label in labels:
            print(f"  {label:>10}  n_focal={len(recording_files_for_label(label))}")
        return

    if args.label is not None:
        task: luigi.Task = AutoMotifSpeciesTask(
            label=args.label,
            band_mode=args.band_mode,
            cluster_mode=args.cluster_mode,
        )
    else:
        task = AutoMotifManifestTask(
            min_focal=args.min_focal,
            labels=args.labels,
            band_mode=args.band_mode,
            cluster_mode=args.cluster_mode,
        )

    luigi.build(
        [task],
        local_scheduler=True,
        workers=args.luigi_workers,
        log_level="INFO",
    )


if __name__ == "__main__":
    main()
