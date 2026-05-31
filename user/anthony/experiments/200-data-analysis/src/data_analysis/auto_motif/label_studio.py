"""Label Studio bridge for the manual-curation anchor (export + parse).

The CSV sheets (:mod:`curation_scaffold`) split the work into a *precision*
sheet (one row per detection, judged T/F/?) and a *recall* sheet (per
recording, missed-call timestamps). Label Studio collapses both into one
audio UI: the pipeline's NMS detections are imported as **pre-annotations
(predictions)** so the annotator reviews existing time-regions (precision)
*and* drags new regions for misses (recall) in a single pass, with a
per-region True/False/Ambiguous verdict.

Two projects, by role (they need different effort):

* ``broadband_anchor`` (``74113``, ``hyamac1``, ``fusfly1``) — production
  free+global leg. Calibrate ``member_bg`` d′→recovery and adjudicate the
  ``74113`` false positive. **Precision AND recall** (draw the misses).
* ``attractors`` (``326272``, ``25092``) — strand-5 (D1) species that ran the
  *validated consensus* leg with healthy d′ yet ``auc_vs_class`` ≤ 0.47. The
  decisive signal is **precision**: do the boxes sit on real calls of this
  species, or on generic energy / a continuous noise band (25092's 4.5 kHz
  stridulation case)? Recall is secondary — a generic-energy detector's
  *misses* are uninformative.

Audio and the pipeline's own PCEN spectrogram PNG are served via Label
Studio local-files serving rooted at ``/mnt/data`` (no copying, per CLAUDE.md);
tasks reference ``/data/local-files/?d=<path-relative-to-/mnt/data>``.

CLI::

    # 1. generate both projects (config + tasks.json + per-project README)
    python -m data_analysis.auto_motif.label_studio export

    # 2. (after annotating in Label Studio and exporting JSON) score it
    python -m data_analysis.auto_motif.label_studio parse \\
        --project attractors --export ~/Downloads/attractors_export.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

from data_analysis.auto_motif.batch import AUTO_MOTIF_DIR
from data_analysis.auto_motif.curation_scaffold import (
    CURATION_DIR,
    PRODUCTION_TARGETS,
    _events,
    _stratified_recordings,
)
from data_analysis.common.paths import TRAIN_AUDIO_DIR

logger = logging.getLogger(__name__)

LS_DIR = CURATION_DIR / "label_studio"
DOC_ROOT = Path("/mnt/data")  # Label Studio LOCAL_FILES_DOCUMENT_ROOT
EXP_DIR = AUTO_MOTIF_DIR.parents[1]
MANIFEST_CSV = AUTO_MOTIF_DIR / "_production_manifest.csv"

# label -> mode tag, from the single source of truth in curation_scaffold.
_TAG = {label: tag for label, tag, _ in PRODUCTION_TARGETS}

# Project definitions, keyed by role. `require_recall` only changes the
# instructions shown to the annotator; the config supports recall either way.
PROJECTS: dict[str, dict] = {
    "broadband_anchor": {
        "labels": ["74113", "hyamac1", "fusfly1"],
        "require_recall": True,
        "title": "Auto-motif broadband anchor — precision + recall",
    },
    "attractors": {
        "labels": ["326272", "25092"],
        "require_recall": False,
        "title": "Auto-motif strand-5 attractors — precision",
    },
}


def _label_config(require_recall: bool) -> str:
    """Labeling interface: reference PCEN image + audio with per-region verdict.

    The pipeline PCEN spectrogram (``$spectrogram``) is shown read-only as
    visual context (Label Studio's own audio view is time-only and loses the
    frequency band each detection was scored in). Regions are created on the
    audio object; ``perRegion`` Choices force a verdict on every region.
    """
    recall_note = (
        "Then scan for any real call of this species with NO region and drag "
        "a new region over it, verdict = 'True call' (recall)."
        if require_recall
        else "Recall (drawing missed calls) is optional for this project."
    )
    return f"""\
<View>
  <Header value="$title"/>
  <Text name="instructions" value="Review each cyan region: set its verdict. {recall_note}"/>
  <Image name="ref" value="$spectrogram" zoom="true" zoomControl="true"
         maxWidth="100%"/>
  <Labels name="region" toName="audio">
    <Label value="call" background="#2a9d8f"/>
  </Labels>
  <Audio name="audio" value="$audio" zoom="true"/>
  <Choices name="verdict" toName="audio" perRegion="true" required="true"
           choice="single">
    <Choice value="True call"/>
    <Choice value="False"/>
    <Choice value="Ambiguous"/>
  </Choices>
</View>
"""


def _local_files_url(abs_path: Path) -> str:
    return f"/data/local-files/?d={abs_path.relative_to(DOC_ROOT)}"


def _spectrogram_abspath(label: str, filename: str) -> Path:
    """Per-recording PNG the scaffold rendered (same stem convention)."""
    return CURATION_DIR / label / f"{Path(filename).stem}.png"


def _tasks_for_label(label: str, max_render_sec: float) -> list[dict]:
    """One Label Studio task per density-stratified recording of a species.

    Detections become a single prediction whose region ids are ``det_<idx>``
    (the global row index, matching the CSV precision sheet) so the parser can
    tell pipeline regions from annotator-drawn misses.
    """
    tag = _TAG[label]
    ev = _events(tag, label).with_row_index("det_id")
    sel = _stratified_recordings(ev, max_render_sec)
    selected = set(sel["filename"].to_list())

    tasks: list[dict] = []
    for fn in sorted(selected):
        dets = ev.filter(pl.col("filename") == fn).sort("start_sec")
        regions = [
            {
                "id": f"det_{row['det_id']}",
                "type": "labels",
                "from_name": "region",
                "to_name": "audio",
                "value": {
                    "start": round(row["start_sec"], 3),
                    "end": round(row["end_sec"], 3),
                    "channel": 0,
                    "labels": ["call"],
                },
                "score": round(row["score"], 4),
            }
            for row in dets.to_dicts()
        ]
        meta = sel.filter(pl.col("filename") == fn).to_dicts()[0]
        tasks.append(
            {
                "data": {
                    "audio": _local_files_url(TRAIN_AUDIO_DIR / fn),
                    "spectrogram": _local_files_url(_spectrogram_abspath(label, fn)),
                    "title": f"{label}  {fn}  ({len(regions)} detections)",
                    "species": label,
                    "recording": fn,
                    "n_det": len(regions),
                    "duration_sec": meta["duration_sec"],
                    "density_bucket": meta["density_bucket"],
                    # frequency bands kept for reference (not shown in the UI)
                    "det_bands_hz": {
                        f"det_{row['det_id']}": [
                            round(row["fmin_hz"], 1), round(row["fmax_hz"], 1)
                        ]
                        for row in dets.to_dicts()
                    },
                },
                "predictions": [
                    {"model_version": "auto_motif_nms", "result": regions}
                ],
            }
        )
    return tasks


_README_TMPL = """\
# Label Studio project: {key}

{title}

## Species in this project
{species_lines}

## Launch (one-time, serves audio + spectrograms from /mnt/data)

```bash
export LOCAL_FILES_SERVING_ENABLED=true
export LOCAL_FILES_DOCUMENT_ROOT=/mnt/data
label-studio start
```

In the web UI:
1. **Create Project** → name it `{key}`.
2. **Settings → Labeling Interface → Code**: paste `label_config.xml` from
   this directory.
3. **Settings → Cloud Storage** is not needed — local files are served
   directly because `LOCAL_FILES_SERVING_ENABLED=true`.
4. **Import** → upload `tasks.json` (it carries the pipeline detections as
   pre-annotations, so each recording opens with its cyan regions already
   drawn).

## Annotate
Per region (cyan): pick a **verdict** — `True call` / `False` / `Ambiguous`.
{recall_line}
The read-only image at the top is the pipeline's own PCEN spectrogram for the
whole recording (with the detection boxes), as visual context.

## Score it
Export the project as **JSON** from the UI, then:

```bash
python -m data_analysis.auto_motif.label_studio parse \\
    --project {key} --export <exported.json>
```

This writes `results_summary.csv` / `.md` here: per-species precision (and
recall), joined against the manifest d′ and curated recovery to test whether
d′ tracks true recovery.
"""


def _write_project(key: str, spec: dict, max_render_sec: float) -> int:
    out = LS_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_config.xml").write_text(_label_config(spec["require_recall"]))

    tasks: list[dict] = []
    for label in spec["labels"]:
        tasks.extend(_tasks_for_label(label, max_render_sec))
    (out / "tasks.json").write_text(json.dumps(tasks, indent=2))

    species_lines = "\n".join(
        f"- `{label}` ({_TAG[label] or 'consensus'} leg)" for label in spec["labels"]
    )
    recall_line = (
        "Then drag a new region over any **missed** real call (verdict "
        "`True call`) so recall can be computed."
        if spec["require_recall"]
        else "Drawing missed calls is optional in this project (precision-focused)."
    )
    (out / "README.md").write_text(
        _README_TMPL.format(
            key=key, title=spec["title"], species_lines=species_lines,
            recall_line=recall_line,
        )
    )
    n_reg = sum(len(t["predictions"][0]["result"]) for t in tasks)
    logger.info(
        "project %s: %d tasks, %d detection-regions -> %s",
        key, len(tasks), n_reg, out,
    )
    return len(tasks)


# --------------------------------------------------------------------------- #
# parse                                                                        #
# --------------------------------------------------------------------------- #
def _region_verdicts(result: list[dict]) -> dict[str, dict]:
    """Collapse one annotation `result` list into {region_id: {...}}.

    A region contributes a `labels` item (the box) and, via `perRegion`, a
    `choices` item that shares the same region id (the verdict).
    """
    reg: dict[str, dict] = {}
    for item in result:
        rid = item.get("id")
        if rid is None:
            continue
        slot = reg.setdefault(rid, {"is_pipeline": rid.startswith("det_")})
        if item.get("type") == "labels":
            slot["start"] = item["value"].get("start")
            slot["end"] = item["value"].get("end")
        elif item.get("type") == "choices":
            choices = item["value"].get("choices", [])
            slot["verdict"] = choices[0] if choices else None
    return reg


def parse_export(project: str, export_path: Path) -> pl.DataFrame:
    """Score a Label Studio JSON export into per-species precision/recall."""
    tasks = json.loads(Path(export_path).read_text())
    rows: list[dict] = []
    for task in tasks:
        species = task.get("data", {}).get("species")
        anns = task.get("annotations") or []
        if not species or not anns:
            continue
        # use the first (or only) annotation per task
        for rid, slot in _region_verdicts(anns[0].get("result", [])).items():
            rows.append(
                {
                    "species": species,
                    "region_id": rid,
                    "is_pipeline": slot.get("is_pipeline", False),
                    "verdict": slot.get("verdict"),
                }
            )
    if not rows:
        raise SystemExit(
            f"no annotations found in {export_path} — export the project as "
            "JSON *with* completed annotations"
        )
    df = pl.DataFrame(rows)

    per_species = []
    for species, g in df.group_by("species"):
        sp = species[0] if isinstance(species, tuple) else species
        pipe = g.filter(pl.col("is_pipeline"))
        manual = g.filter(~pl.col("is_pipeline"))
        tp = pipe.filter(pl.col("verdict") == "True call").height
        fp = pipe.filter(pl.col("verdict") == "False").height
        amb = pipe.filter(pl.col("verdict") == "Ambiguous").height
        unjudged = pipe.filter(pl.col("verdict").is_null()).height
        misses = manual.filter(pl.col("verdict") == "True call").height
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + misses) if (tp + misses) else None
        per_species.append(
            {
                "species": sp,
                "n_pipeline_judged": tp + fp + amb,
                "tp": tp, "fp": fp, "ambiguous": amb, "unjudged": unjudged,
                "recall_misses": misses,
                "precision": round(precision, 3) if precision is not None else None,
                "recall": round(recall, 3) if recall is not None else None,
            }
        )
    res = pl.DataFrame(per_species)

    # join against the manifest so d′ / curated recovery sit next to truth
    if MANIFEST_CSV.exists():
        man = pl.read_csv(MANIFEST_CSV, infer_schema_length=10000).select(
            pl.col("label").alias("species"),
            "common_name", "routing_leg",
            "production_member_bg_dprime", "curated_recovery",
            "indep_perch_emb_auc_production",
        )
        res = res.join(man, on="species", how="left")
    return res


def _markdown_table(df: pl.DataFrame) -> list[str]:
    """Render a polars frame as GitHub-markdown rows (no pandas/pyarrow)."""
    cols = df.columns
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in df.iter_rows()
    ]
    return [head, sep, *body]


def _run_parse(project: str, export_path: Path) -> None:
    res = parse_export(project, export_path)
    out = LS_DIR / project
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "results_summary.csv")

    lines = [
        f"# Label Studio results — {project}",
        "",
        "Per-species precision/recall from manual annotation, joined against the",
        "production manifest. Tests whether pipeline d′ tracks true recovery.",
        "",
        *_markdown_table(res),
        "",
    ]
    (out / "results_summary.md").write_text("\n".join(lines))
    logger.info("wrote %s", out / "results_summary.md")
    print(res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("export", help="generate Label Studio projects")
    pe.add_argument("--max-render-sec", type=float, default=180.0)
    pp = sub.add_parser("parse", help="score a Label Studio JSON export")
    pp.add_argument("--project", required=True, choices=sorted(PROJECTS))
    pp.add_argument("--export", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "export":
        LS_DIR.mkdir(parents=True, exist_ok=True)
        total = 0
        for key, spec in PROJECTS.items():
            total += _write_project(key, spec, args.max_render_sec)
        logger.info(
            "exported %d Label Studio projects (%d tasks) under %s",
            len(PROJECTS), total, LS_DIR,
        )
    elif args.cmd == "parse":
        _run_parse(args.project, args.export)


if __name__ == "__main__":
    main()
