"""Resumable Luigi driver: auto-motif over the soundscape∩train species set.

Each species is one :class:`AutoMotifSpeciesTask`. Its Luigi target is a
per-species status JSON under ``artifacts/auto_motif/_batch/status/``, so a
completed (or attempted) species is skipped on re-run — resumability comes for
free from Luigi's target check. Failures are captured into the status JSON
(``status="failed"`` + traceback summary) and the task still *succeeds*, so one
bad species never blocks the batch; the merged manifest records every outcome.

To re-run a single species (e.g. after a transient failure), delete its
``_batch/status/<label>.json`` and rebuild. The set of species comes from
:func:`data_analysis.common.selected_species.soundscape_train_species`
(derived from the data, filtered by ``--min-focal``, default 8).

CLI::

    python -m data_analysis.auto_motif.batch --list           # preview set
    python -m data_analysis.auto_motif.batch --label 22967     # one species
    python -m data_analysis.auto_motif.batch                   # full batch
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import luigi
import polars as pl

from data_analysis.auto_motif.mining import run as run_auto_motif
from data_analysis.auto_motif.mining import species_output_dir
from data_analysis.common.paths import (
    ARTIFACTS_DIR,
    TAXONOMY_CSV,
    recording_files_for_label,
)
from data_analysis.common.selected_species import soundscape_train_species

logger = logging.getLogger(__name__)

BATCH_DIR = ARTIFACTS_DIR / "auto_motif" / "_batch"
STATUS_DIR = BATCH_DIR / "status"

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


def _curated_path_for(label: str) -> Path | None:
    path = CURATED_PATHS.get(label)
    return path if path is not None and path.exists() else None


def _read_height(path: Path) -> int | None:
    if not path.exists():
        return None
    return pl.read_csv(path).height


def _collect_metrics(label: str) -> dict:
    """Pull a few headline numbers from a finished species' artifacts."""
    tables = species_output_dir(label) / "tables"
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


class AutoMotifSpeciesTask(luigi.Task):
    """Run the auto-motif pipeline for one species; capture status as target."""

    label = luigi.Parameter()

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(str(STATUS_DIR / f"{self.label}.json"))

    def run(self) -> None:
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        n_focal = len(recording_files_for_label(str(self.label)))
        record: dict = {"label": str(self.label), "n_focal": n_focal}
        try:
            run_auto_motif(
                label=str(self.label),
                curated_path=_curated_path_for(str(self.label)),
            )
            record["status"] = "ok"
            record["error"] = None
            record.update(_collect_metrics(str(self.label)))
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

    def requires(self):
        return [
            AutoMotifSpeciesTask(label=label)
            for label, _ in soundscape_train_species(int(self.min_focal))
        ]


class AutoMotifManifestTask(luigi.Task):
    """Merge per-species status JSONs into one manifest CSV."""

    min_focal = luigi.IntParameter(default=8)

    def requires(self):
        return AutoMotifBatchTask(min_focal=self.min_focal)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(str(BATCH_DIR / "manifest.csv"))

    def run(self) -> None:
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
        for label, _ in soundscape_train_species(int(self.min_focal)):
            status_path = STATUS_DIR / f"{label}.json"
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
        selected = soundscape_train_species(args.min_focal)
        print(f"{len(selected)} species (min_focal={args.min_focal}):")
        for label, n in selected:
            print(f"  {label:>10}  n_focal={n}")
        return

    if args.label is not None:
        task: luigi.Task = AutoMotifSpeciesTask(label=args.label)
    else:
        task = AutoMotifManifestTask(min_focal=args.min_focal)

    luigi.build(
        [task],
        local_scheduler=True,
        workers=args.luigi_workers,
        log_level="INFO",
    )


if __name__ == "__main__":
    main()
