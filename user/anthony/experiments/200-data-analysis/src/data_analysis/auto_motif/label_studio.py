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

Review is **windowed**: each recording is cut into ~5s clips (one Label
Studio task per clip) so the annotator reviews a bite-size chunk instead of
scrolling a 160s recording. ``broadband_anchor`` gets *every* contiguous
window (empty ones keep recall honest); ``attractors`` gets only windows that
own a detection (precision). A detection is owned by the single window its
peak falls in, so it is judged exactly once; clips carry ±0.5s of context.

Clips + their per-clip PCEN spectrograms are written under
``/mnt/data/scratch/label-studio-clips`` and served via Label Studio
local-files serving (rooted at ``/mnt/data``; no copying of raw audio per
CLAUDE.md). Serving requires a ``LocalFilesImportStorage`` per project whose
``path`` prefixes the files — ``push`` registers the two roots automatically.

CLI::

    # 1. generate both projects (clips + config + tasks.json + README)
    python -m data_analysis.auto_motif.label_studio export        # --window-sec 5

    # 2. (re)provision the running Label Studio + import the tasks via API
    python -m data_analysis.auto_motif.label_studio push --replace

    # 3. (after annotating in Label Studio and exporting JSON) score it
    python -m data_analysis.auto_motif.label_studio parse \\
        --project attractors --export ~/Downloads/attractors_export.json
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import soundfile as sf  # noqa: E402

from data_analysis.auto_motif.batch import AUTO_MOTIF_DIR  # noqa: E402
from data_analysis.auto_motif.curation_scaffold import (  # noqa: E402
    CURATION_DIR,
    PRODUCTION_TARGETS,
    _events,
    _stratified_recordings,
)
from data_analysis.common.paths import (  # noqa: E402
    TRAIN_AUDIO_DIR,
    TRAIN_CSV,
    TRAIN_SCHEMA_OVERRIDES,
)
from data_analysis.common.spectrogram import (  # noqa: E402
    MEL_SAMPLE_RATE,
    WIDEBAND,
    wideband_pcen_spectrogram,
)

logger = logging.getLogger(__name__)

LS_DIR = CURATION_DIR / "label_studio"
DOC_ROOT = Path("/mnt/data")  # Label Studio LOCAL_FILES_DOCUMENT_ROOT
EXP_DIR = AUTO_MOTIF_DIR.parents[1]
MANIFEST_CSV = AUTO_MOTIF_DIR / "_production_manifest.csv"

# Windowed-review constants. Each task is a short clip so the annotator reviews
# ~5s at a time (Label Studio's <Audio> tag has no per-task viewport). Clips +
# their spectrograms are written under scratch (already a serving root, keeps
# the experiment dir clean, regenerable from `export`).
WINDOW_SEC = 5.0
PAD_SEC = 0.5  # context on each side of the core window (clip ≈ 6s)
CLIPS_DIR = Path("/mnt/data/scratch/label-studio-clips")

# Label Studio API defaults (the local single-user instance) + the two
# local-files serving roots that cover audio (scratch) and spectrograms.
LS_BASE = "http://localhost:8080"
LS_TOKEN = "birdclef-local-token-200"
SERVING_ROOTS = ["/mnt/data/scratch", "/mnt/data/research"]

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
  <Text name="instructions" value="~5s clip. Review each cyan region: set its verdict. {recall_note}"/>
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


@functools.lru_cache(maxsize=1)
def _recording_meta() -> dict[str, tuple]:
    """``{filename: (rating, type)}`` from train.csv — recordist context.

    ``rating`` is the Xeno-canto quality score (0.5–5; 0 = unrated, ≈ all
    iNaturalist). ``type`` is the recordist's label of *what the sound is*
    (``song``/``call``/``sniffing``/…); a non-vocal type like ``sniffing`` is a
    strong hint the pipeline's detections there are false positives.
    """
    df = pl.read_csv(
        TRAIN_CSV, infer_schema_length=10000, schema_overrides=TRAIN_SCHEMA_OVERRIDES
    )
    return {
        r["filename"]: (r["rating"], r["type"])
        for r in df.select("filename", "rating", "type").to_dicts()
    }


def _fmt_meta(filename: str) -> tuple[float, str, str]:
    """Return ``(rating, type_clean, label_str)`` for a recording's header."""
    rating, rtype = _recording_meta().get(filename, (0.0, None))
    rating = float(rating or 0.0)
    tclean = (
        (rtype or "").strip("[]").replace("'", "").strip() or "—"
    )
    rstr = f"{rating:g}" if rating > 0 else "unrated"
    return rating, tclean, f"rating {rstr} · type: {tclean}"


def _region(det_id: int, start: float, end: float, score: float) -> dict:
    """A Label Studio audio time-region for one pipeline detection.

    The id ``det_<global_idx>`` is stable across tasks (matching the CSV
    precision sheet) so the parser tells pipeline regions from annotator-drawn
    misses, and so each detection is counted once even across windowed tasks.
    """
    return {
        "id": f"det_{det_id}",
        "type": "labels",
        "from_name": "region",
        "to_name": "audio",
        "value": {
            "start": round(max(0.0, start), 3),
            "end": round(end, 3),
            "channel": 0,
            "labels": ["call"],
        },
        "score": round(score, 4),
    }


def _render_window(
    audio: np.ndarray, dets_local: list[dict], out_png: Path,
    *, core_start: float, core_end: float,
) -> None:
    """Single-strip PCEN spectrogram of one clip (clip-local x-axis).

    Mirrors the look of ``curation_scaffold._render_recording`` but one row:
    the ±``PAD_SEC`` context margins (outside the core window) are shaded so the
    annotator knows which part is "owned" by this task; cyan boxes are the
    owned detections in clip-local time.
    """
    _, s, freqs, times = wideband_pcen_spectrogram(audio, band=WIDEBAND)
    total = float(times[-1]) if times.size else 0.0
    fig, ax = plt.subplots(figsize=(12, 2.6))
    if times.size:
        vmin, vmax = np.percentile(s, [5, 99])
        ax.pcolormesh(
            times, freqs, s, vmin=vmin, vmax=vmax, shading="nearest", cmap="magma"
        )
    for x0, x1 in [(0.0, core_start), (core_end, total)]:
        if x1 > x0:
            ax.axvspan(x0, x1, color="white", alpha=0.12)
    for d in dets_local:
        ax.add_patch(
            plt.Rectangle(
                (d["start"], d["fmin_hz"]), d["end"] - d["start"],
                d["fmax_hz"] - d["fmin_hz"],
                fill=False, edgecolor="cyan", linewidth=1.4,
            )
        )
        ax.text(
            d["start"], d["fmax_hz"], f'#{d["det_id"]}',
            color="cyan", fontsize=7, va="bottom",
        )
    ax.set_xlim(0, total if total else WINDOW_SEC)
    ax.set_ylim(WIDEBAND)
    ax.set_xlabel(
        "clip time (s) — shaded = ±0.5s context; cyan = pipeline detections",
        fontsize=8,
    )
    ax.set_ylabel("Hz", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _window_tasks_for_label(
    label: str, spec: dict, window_sec: float, max_render_sec: float,
) -> list[dict]:
    """One Label Studio task per ~``window_sec`` clip of each selected recording.

    Recording selection is unchanged (density-stratified). Each recording is
    tiled into windows; ``require_recall`` projects keep every window (empty
    ones drive recall), others keep only windows that own a detection. A
    detection is owned by ``floor(peak / window_sec)`` so it is judged once.
    Clips carry ±``PAD_SEC`` context but the owned-region times are clip-local.
    """
    tag = _TAG[label]
    ev = _events(tag, label).with_row_index("det_id")
    sel = _stratified_recordings(ev, max_render_sec)

    species_dir = CLIPS_DIR / label
    shutil.rmtree(species_dir, ignore_errors=True)
    species_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for meta in sel.to_dicts():
        fn = meta["filename"]
        stem = Path(fn).stem
        det_rows = (
            ev.filter(pl.col("filename") == fn)
            .with_columns(
                (pl.col("peak_time_sec") // window_sec).cast(pl.Int64).alias("win")
            )
            .to_dicts()
        )
        owned: dict[int, list[dict]] = {}
        for r in det_rows:
            owned.setdefault(int(r["win"]), []).append(r)

        # Load the recording once; normalize every clip by the *recording's*
        # peak (not the clip's own) so a quiet call-less window stays quiet
        # instead of having its background — breathing, wind, handling noise —
        # boosted to full scale. Slicing in memory also avoids re-reading audio.
        full = librosa.load(
            TRAIN_AUDIO_DIR / fn, sr=MEL_SAMPLE_RATE, mono=True
        )[0].astype(np.float32)
        gpeak = float(np.max(np.abs(full))) or 1.0
        dur = float(len(full) / MEL_SAMPLE_RATE)

        n_win = max(1, math.ceil(dur / window_sec))
        windows = range(n_win) if spec["require_recall"] else sorted(owned)
        for w in windows:
            win_start = w * window_sec
            clip_start = max(0.0, win_start - PAD_SEC)
            clip_end = min(dur, win_start + window_sec + PAD_SEC)
            seg = full[
                int(clip_start * MEL_SAMPLE_RATE): int(clip_end * MEL_SAMPLE_RATE)
            ]
            if seg.size == 0:
                continue
            audio = np.clip((seg - float(seg.mean())) / gpeak, -1.0, 1.0)
            clip_len = clip_end - clip_start
            core_start = win_start - clip_start
            core_end = min(clip_end, win_start + window_sec) - clip_start

            dets_local, regions = [], []
            for r in owned.get(w, []):
                ls = round(r["start_sec"] - clip_start, 3)
                le = round(min(r["end_sec"] - clip_start, clip_len), 3)
                dets_local.append(
                    {"start": ls, "end": le, "fmin_hz": r["fmin_hz"],
                     "fmax_hz": r["fmax_hz"], "det_id": r["det_id"]}
                )
                regions.append(_region(int(r["det_id"]), ls, le, r["score"]))

            clip_path = species_dir / f"{stem}_w{w:03d}.ogg"
            png_path = species_dir / f"{stem}_w{w:03d}.png"
            sf.write(clip_path, audio, MEL_SAMPLE_RATE, format="OGG", subtype="VORBIS")
            _render_window(
                audio, dets_local, png_path, core_start=core_start, core_end=core_end
            )
            rating, rtype, meta_str = _fmt_meta(fn)
            tasks.append(
                {
                    "data": {
                        "audio": _local_files_url(clip_path),
                        "spectrogram": _local_files_url(png_path),
                        "title": f"{label}  {fn}  [win {w} @ {win_start:.0f}s]  "
                                 f"({len(regions)} det)  ·  {meta_str}",
                        "species": label,
                        "recording": fn,
                        "rating": rating,
                        "rec_type": rtype,
                        "window_index": w,
                        "window_start_sec": round(win_start, 1),
                        "n_det": len(regions),
                        "duration_sec": round(dur, 1),
                        "density_bucket": meta["density_bucket"],
                        "det_bands_hz": {
                            f"det_{d['det_id']}": [
                                round(d["fmin_hz"], 1), round(d["fmax_hz"], 1)
                            ]
                            for d in dets_local
                        },
                    },
                    "predictions": [
                        {"model_version": "auto_motif_nms", "result": regions}
                    ],
                }
            )
    return tasks


def _recording_tasks_for_label(label: str, max_render_sec: float) -> list[dict]:
    """Legacy whole-recording mode (one task per recording) — used when
    ``--window-sec 0`` is passed. Detections reference the recording's full
    audio and the scaffold's whole-recording spectrogram PNG.
    """
    tag = _TAG[label]
    ev = _events(tag, label).with_row_index("det_id")
    sel = _stratified_recordings(ev, max_render_sec)
    selected = set(sel["filename"].to_list())

    tasks: list[dict] = []
    for fn in sorted(selected):
        dets = ev.filter(pl.col("filename") == fn).sort("start_sec")
        regions = [
            _region(int(row["det_id"]), row["start_sec"], row["end_sec"], row["score"])
            for row in dets.to_dicts()
        ]
        meta = sel.filter(pl.col("filename") == fn).to_dicts()[0]
        rating, rtype, meta_str = _fmt_meta(fn)
        tasks.append(
            {
                "data": {
                    "audio": _local_files_url(TRAIN_AUDIO_DIR / fn),
                    "spectrogram": _local_files_url(_spectrogram_abspath(label, fn)),
                    "title": f"{label}  {fn}  ({len(regions)} detections)  ·  {meta_str}",
                    "species": label,
                    "recording": fn,
                    "rating": rating,
                    "rec_type": rtype,
                    "n_det": len(regions),
                    "duration_sec": meta["duration_sec"],
                    "density_bucket": meta["density_bucket"],
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

## Tasks are ~5s clips
Each task is a single ~5-second clip (±0.5s context) of one recording, with
the pipeline detections owned by that window pre-drawn as cyan regions. Clips +
their spectrograms live under `/mnt/data/scratch/label-studio-clips/`.

## Launch + provision (serves clips/spectrograms from /mnt/data)

```bash
# server (once): env must be in-process; data dir kept off the home quota
env LOCAL_FILES_SERVING_ENABLED=true LOCAL_FILES_DOCUMENT_ROOT=/mnt/data \\
    label-studio start --data-dir /mnt/data/scratch/label-studio \\
    --username <you> --password <pw> \\
    --user-token birdclef-local-token-200 --enable-legacy-api-token

# create/refresh projects + serving storages + import tasks (re-runnable)
python -m data_analysis.auto_motif.label_studio push --replace
```

`push` registers, per project, a local-files serving storage for
`/mnt/data/scratch` and `/mnt/data/research` (required for the
`/data/local-files/?d=...` URLs to resolve). **Do not click "Sync"** on those
storages — they are serving whitelists only, not import scans.

## Annotate
Per region (cyan): pick a **verdict** — `True call` / `False` / `Ambiguous`.
{recall_line}
The read-only image at the top is the clip's PCEN spectrogram (shaded margins =
±0.5s context outside this window).

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


def _write_project(
    key: str, spec: dict, window_sec: float, max_render_sec: float
) -> int:
    out = LS_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_config.xml").write_text(_label_config(spec["require_recall"]))

    tasks: list[dict] = []
    for label in spec["labels"]:
        if window_sec and window_sec > 0:
            tasks.extend(
                _window_tasks_for_label(label, spec, window_sec, max_render_sec)
            )
        else:
            tasks.extend(_recording_tasks_for_label(label, max_render_sec))
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


# --------------------------------------------------------------------------- #
# push (provision the running Label Studio via REST API; stdlib only)          #
# --------------------------------------------------------------------------- #
def _api(base: str, token: str, method: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach Label Studio at {base}: {e}") from e


def _push(base: str, token: str, replace: bool) -> None:
    """(Re)create each project, ensure serving storages, import its tasks.json."""
    st, projs = _api(base, token, "GET", "/api/projects?page_size=200")
    if st != 200:
        raise SystemExit(f"GET /api/projects failed ({st}): {projs}")
    byname = {p["title"]: p["id"] for p in projs.get("results", [])}

    for key, spec in PROJECTS.items():
        title = f"auto_motif {key}"
        cfg = (LS_DIR / key / "label_config.xml").read_text()
        tasks = json.loads((LS_DIR / key / "tasks.json").read_text())

        if replace and title in byname:  # clean slate: delete + recreate
            _api(base, token, "DELETE", f"/api/projects/{byname.pop(title)}")
        if title in byname:
            pid = byname[title]
            _api(base, token, "PATCH", f"/api/projects/{pid}", {"label_config": cfg})
        else:
            st, p = _api(base, token, "POST", "/api/projects",
                         {"title": title, "label_config": cfg})
            if not isinstance(p, dict) or "id" not in p:
                logger.error("create %s failed (%s): %s", title, st, p)
                continue
            pid = p["id"]

        st, existing = _api(base, token, "GET", f"/api/storages/localfiles?project={pid}")
        have = {s["path"].rstrip("/") for s in (existing or [])}
        for root in SERVING_ROOTS:
            if root not in have:
                _api(base, token, "POST", "/api/storages/localfiles",
                     {"project": pid, "path": root,
                      "title": f"serving {root} (do not sync)",
                      "use_blob_urls": False, "regex_filter": ""})

        st, imp = _api(base, token, "POST", f"/api/projects/{pid}/import", tasks)
        n = imp.get("task_count") if isinstance(imp, dict) else "?"
        logger.info("pushed %s (id %s): import HTTP %s, %s tasks", title, pid, st, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("export", help="generate Label Studio projects (clips)")
    pe.add_argument("--window-sec", type=float, default=WINDOW_SEC,
                    help="clip length; 0 = legacy whole-recording tasks")
    pe.add_argument("--max-render-sec", type=float, default=180.0)
    pu = sub.add_parser("push", help="provision projects + import tasks via API")
    pu.add_argument("--base", default=LS_BASE)
    pu.add_argument("--token", default=LS_TOKEN)
    pu.add_argument("--replace", action="store_true",
                    help="delete + recreate each project (clears old tasks)")
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
            total += _write_project(key, spec, args.window_sec, args.max_render_sec)
        logger.info(
            "exported %d Label Studio projects (%d tasks) under %s",
            len(PROJECTS), total, LS_DIR,
        )
    elif args.cmd == "push":
        _push(args.base, args.token, args.replace)
    elif args.cmd == "parse":
        _run_parse(args.project, args.export)


if __name__ == "__main__":
    main()
