"""Label Studio bridge for 2-D time-frequency box curation (standalone).

Unlike exp 200's 1-D audio time-regions, boxes here are 2-D: the annotator draws
``RectangleLabels`` on a spectrogram ``<Image>`` (with a read-only ``<Audio>`` to
listen). Auto-proposed boxes are imported as **predictions** so the annotator
reviews/corrects them and adds missed ones. The pixel<->(time,freq) mapping lives
in :mod:`birdclef_201.geom` (single source of truth, exact round-trip).

Two projects:
- ``insects_v1`` — label set = the 25 ``47158sonNN`` sonotypes (+ ``unattributed``);
  tasks = windows owning an insect-attributed or ambiguous box.
- ``all_taxa`` — label set = all 75 labeled species (+ ``unattributed``);
  tasks = every window owning a box.

Serving: clips/PNGs under ``/mnt/data/scratch/birdclef-201-clips`` are served via
Label Studio local-files; ``push`` registers a ``LocalFilesImportStorage`` per
serving root (``/mnt/data/scratch`` + ``/mnt/data/research``) — each path must be a
strict subdir of ``LOCAL_FILES_DOCUMENT_ROOT=/mnt/data`` and prefix the file.

CLI::

    python -m birdclef_201.label_studio export
    python -m birdclef_201.label_studio push --replace
    python -m birdclef_201.label_studio parse --project insects_v1 --export <export.json>
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import soundfile as sf  # noqa: E402

from birdclef_201.geom import (  # noqa: E402
    SpecExtent,
    box_to_ls_percent,
    ls_percent_to_box,
    norm01,
)
from birdclef_201.labels import insect_sonotypes, load_windows  # noqa: E402
from birdclef_201.paths import SCRATCH_CLIPS_DIR, artifact_dir  # noqa: E402
from birdclef_201.spectrogram import (  # noqa: E402
    MEL_N_FFT,
    MEL_SAMPLE_RATE,
    read_soundscape_window,
    wideband_pcen_spectrogram,
)

logger = logging.getLogger(__name__)

WINDOW_SEC = 5.0
PAD_SEC = 0.5
DOC_ROOT = Path("/mnt/data")
LS_BASE = "http://localhost:8080"
LS_TOKEN = "birdclef-local-token-201"
SERVING_ROOTS = ["/mnt/data/scratch", "/mnt/data/research"]

BOXES_DIR = artifact_dir("boxes")
LS_DIR = artifact_dir("label_studio")

_PALETTE = [
    "#e63946", "#2a9d8f", "#e9c46a", "#f4a261", "#264653", "#8ecae6",
    "#ff006e", "#3a86ff", "#fb5607", "#8338ec", "#06d6a0", "#ef476f",
]


def _project_specs() -> dict[str, dict]:
    long, _ = load_windows()
    son = insect_sonotypes()
    all_species = sorted(long["label"].unique().to_list())
    return {
        "insects_v1": {
            "labels": son + ["unattributed"],
            "box_filter": pl.col("is_insect") | pl.col("ambiguous"),
            "title": "201 insects v1 — sonotype boxes",
        },
        "all_taxa": {
            "labels": all_species + ["unattributed"],
            "box_filter": pl.lit(True),
            "title": "201 all taxa — event boxes",
        },
    }


def _label_config(label_values: list[str]) -> str:
    labels_xml = "\n".join(
        f'    <Label value="{v}" background="{_PALETTE[i % len(_PALETTE)]}"/>'
        for i, v in enumerate(label_values)
    )
    return f"""\
<View>
  <Header value="$title"/>
  <Text name="instructions" value="Draw/adjust a box per sound event; pick its species. Frequency is vertical (high = top)."/>
  <Image name="spec" value="$spectrogram" zoom="true" zoomControl="true" maxWidth="100%"/>
  <RectangleLabels name="box" toName="spec">
{labels_xml}
  </RectangleLabels>
  <Audio name="audio" value="$audio" zoom="true"/>
</View>
"""


def _local_files_url(abs_path: Path) -> str:
    return f"/data/local-files/?d={abs_path.relative_to(DOC_ROOT)}"


def _render_spec_png(pcen: np.ndarray, freqs: np.ndarray, times: np.ndarray, out_png: Path) -> SpecExtent:
    """1px==1cell PCEN render (origin lower via flipud) -> exact SpecExtent."""
    import matplotlib.pyplot as plt

    plt.imsave(out_png, np.flipud(norm01(pcen)), cmap="magma")
    # 1px==1cell render -> image edges are cell edges (±½ cell), so RectangleLabels
    # percent (relative to image edges) maps to time/freq exactly.
    dt = float(times[1] - times[0]) if times.size > 1 else 0.0
    dfq = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    return SpecExtent(
        t0=float(times[0]) - dt / 2, t1=float(times[-1]) + dt / 2,
        f0=float(freqs[0]) - dfq / 2, f1=float(freqs[-1]) + dfq / 2,
        width_px=int(times.size), height_px=int(freqs.size),
    )


def _box_prediction(box: dict, ext: SpecExtent, clip_start: float, label: str, gid: int) -> dict:
    x, y, w, h = box_to_ls_percent(
        box["start_sec"] - clip_start, box["end_sec"] - clip_start,
        box["fmin_hz"], box["fmax_hz"], ext,
    )
    return {
        "id": f"box_{gid}",
        "type": "rectanglelabels",
        "from_name": "box",
        "to_name": "spec",
        "original_width": ext.width_px,
        "original_height": ext.height_px,
        "image_rotation": 0,
        "value": {
            "x": round(x, 3), "y": round(y, 3),
            "width": round(w, 3), "height": round(h, 3),
            "rotation": 0, "rectanglelabels": [label],
        },
    }


def _export_project(key: str, spec: dict, boxes: pl.DataFrame, wide: pl.DataFrame) -> int:
    label_set = set(spec["labels"])
    sel = boxes.filter(spec["box_filter"])
    win_ids = sel["window_id"].unique().to_list()
    win_meta = {w["window_id"]: w for w in wide.to_dicts()}

    clips_dir = SCRATCH_CLIPS_DIR / key
    clips_dir.mkdir(parents=True, exist_ok=True)
    out = LS_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_config.xml").write_text(_label_config(spec["labels"]))

    tasks: list[dict] = []
    gid = 0
    for wid in sorted(win_ids):
        w = win_meta[wid]
        audio, clip_start, _ = read_soundscape_window(
            w["filename"], w["start_sec"], w["end_sec"], pad_sec=PAD_SEC
        )
        if audio.size < MEL_N_FFT:
            continue
        _, pcen, freqs, times = wideband_pcen_spectrogram(audio)
        stem = f"{Path(w['filename']).stem}_{int(w['start_sec']):03d}"
        png_path = clips_dir / f"{stem}.png"
        ogg_path = clips_dir / f"{stem}.ogg"
        ext = _render_spec_png(pcen, freqs, times, png_path)
        sf.write(ogg_path, audio, MEL_SAMPLE_RATE, format="OGG", subtype="VORBIS")

        preds = []
        for b in sel.filter(pl.col("window_id") == wid).to_dicts():
            lab = b["attributed_label"] if b["attributed_label"] in label_set else "unattributed"
            preds.append(_box_prediction(b, ext, clip_start, lab, gid))
            gid += 1
        tasks.append(
            {
                "data": {
                    "spectrogram": _local_files_url(png_path),
                    "audio": _local_files_url(ogg_path),
                    "title": f"{key}  {w['filename']}  @{int(w['start_sec'])}s  [{w['site']}]  ({len(preds)} box)",
                    "window_id": wid,
                    "filename": w["filename"],
                    "site": w["site"],
                    "window_start_sec": float(w["start_sec"]),
                    "clip_start_sec": round(clip_start, 4),
                    "labels": list(w["labels"]),
                    "spec_extent": ext.as_dict(),
                },
                "predictions": [{"model_version": "box_proposer_v1", "result": preds}],
            }
        )
    (out / "tasks.json").write_text(json.dumps(tasks, indent=2))
    logger.info("project %s: %d tasks, %d predicted boxes -> %s", key, len(tasks), gid, out)
    return len(tasks)


def _run_export() -> None:
    boxes_path = BOXES_DIR / "attributed_boxes.parquet"
    if not boxes_path.exists():
        raise SystemExit(f"missing {boxes_path}; run `python -m birdclef_201.boxes` first")
    boxes = pl.read_parquet(boxes_path)
    _, wide = load_windows()
    for key, spec in _project_specs().items():
        _export_project(key, spec, boxes, wide)


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
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:400]}
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach Label Studio at {base}: {e}") from e


def _push(base: str, token: str, replace: bool) -> None:
    st, projs = _api(base, token, "GET", "/api/projects?page_size=200")
    if st != 200:
        raise SystemExit(f"GET /api/projects failed ({st}): {projs}")
    byname = {p["title"]: p["id"] for p in projs.get("results", [])}

    for key, spec in _project_specs().items():
        title = f"birdclef_201 {key}"
        cfg = (LS_DIR / key / "label_config.xml").read_text()
        tasks = json.loads((LS_DIR / key / "tasks.json").read_text())

        if replace and title in byname:
            _api(base, token, "DELETE", f"/api/projects/{byname.pop(title)}")
        if title in byname:
            pid = byname[title]
            _api(base, token, "PATCH", f"/api/projects/{pid}", {"label_config": cfg})
        else:
            st, p = _api(base, token, "POST", "/api/projects", {"title": title, "label_config": cfg})
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


# --------------------------------------------------------------------------- #
# parse                                                                        #
# --------------------------------------------------------------------------- #
def parse_export(project: str, export_path: Path) -> pl.DataFrame:
    """Read corrected boxes back into a tidy gold-box frame (recording time)."""
    tasks = json.loads(Path(export_path).read_text())
    rows: list[dict] = []
    for task in tasks:
        data = task.get("data", {})
        ext_d = data.get("spec_extent")
        anns = task.get("annotations") or []
        if not ext_d or not anns:
            continue
        ext = SpecExtent.from_dict(ext_d)
        clip_start = float(data.get("clip_start_sec", 0.0))
        for item in anns[0].get("result", []):
            if item.get("type") != "rectanglelabels":
                continue
            v = item["value"]
            t0, t1, f0, f1 = ls_percent_to_box(v["x"], v["y"], v["width"], v["height"], ext)
            labels = v.get("rectanglelabels", [])
            rid = item.get("id", "")
            rows.append(
                {
                    "filename": data.get("filename"),
                    "window_id": data.get("window_id"),
                    "site": data.get("site"),
                    "t0": round(clip_start + t0, 4),
                    "t1": round(clip_start + t1, 4),
                    "f0": round(f0, 2),
                    "f1": round(f1, 2),
                    "label": labels[0] if labels else None,
                    "source": "pipeline" if str(rid).startswith("box_") else "human",
                }
            )
    if not rows:
        raise SystemExit(f"no boxed annotations in {export_path}")
    df = pl.DataFrame(rows)
    out = LS_DIR / project
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "gold_boxes.parquet")
    df.write_csv(out / "gold_boxes.csv")
    logger.info("parsed %d gold boxes -> %s", df.height, out / "gold_boxes.parquet")
    print(df.group_by("label").len().sort("len", descending=True).head(20))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export", help="render clips/PNGs + write tasks.json per project")
    pu = sub.add_parser("push", help="provision projects + import tasks via API")
    pu.add_argument("--base", default=LS_BASE)
    pu.add_argument("--token", default=LS_TOKEN)
    pu.add_argument("--replace", action="store_true")
    pp = sub.add_parser("parse", help="read corrected boxes from a JSON export")
    pp.add_argument("--project", required=True)
    pp.add_argument("--export", required=True, type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "export":
        _run_export()
    elif args.cmd == "push":
        _push(args.base, args.token, args.replace)
    elif args.cmd == "parse":
        parse_export(args.project, args.export)


if __name__ == "__main__":
    main()
