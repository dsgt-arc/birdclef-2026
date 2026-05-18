"""Extrinsic efficacy of auto-motif templates against labeled soundscapes.

This is the label-based validation the concept doc flagged as missing: the
Perch-embedding screen is a *proxy* and the manual-curation anchor is still
unlabeled, so neither tells us whether the mined templates actually fire on
the species they were mined for in real soundscape recordings. The BirdCLEF
``train_soundscapes_labels.csv`` set (66 files, 739 weakly-labeled 5 s windows,
2 annotator passes/window) is exactly that ground truth, and all 41 production
species are present in it.

Method (no re-mining, no recompute of the templates):

* Resolve each species' **production** templates via
  :func:`production_manifest.build_production_manifest` (conservative routing),
  load the persisted ``family_*.npz`` patches and their **committed**
  ``threshold`` from ``auto_templates_metadata.csv``.
* For every labeled soundscape: one ``wideband_pcen_spectrogram`` per file,
  then the *exact* production detection arithmetic — :func:`mining._score_track`
  → ``find_peaks(height=committed_threshold, distance, prominence=0.02)`` with
  the same ``min_spacing`` rule → :func:`mining._refine_anchor_via_envelope`
  → per-species :func:`nms_events` + :func:`consolidate_events` at
  ``AutoConfig`` defaults. The *only* substitution vs. the in-miner detector
  is the threshold source: on a held-out soundscape the EM member/background
  scores do not exist, so the committed per-family threshold (the operating
  point the pipeline froze) is used — which is precisely the thing under test.
* Per (5 s window, species) we keep both the **threshold-free** max score
  (template *discriminability*, independent of the committed cut) and whether
  a consolidated detection fired (the committed **operating point**).

Two metrics, reported side by side because they answer different questions
and conflating them is the trap:

* ``roc_auc`` / ``pr_auc`` — threshold-free: *is the mined motif a real,
  separable acoustic pattern on soundscapes?* This is the honest efficacy
  signal and the one to trust.
* ``precision`` / ``recall`` / ``f1`` at the committed threshold — does the
  pipeline's self-calibrated operating point (set on focal audio) transfer
  to soundscapes? Focal→soundscape domain shift is expected to hurt this;
  a low number here with a high AUC means "good template, miscalibrated
  threshold", not "bad template".

Caveats (surfaced, not buried): soundscape labels are weak presence over 5 s
and multi-label; templates cover *recurring motifs*, not a species' whole
vocal repertoire, so recall is upper-bounded by motif coverage of the
repertoire — a species calling with a non-mined call type is a legitimate
miss, not a template failure. Positives use the **union** of the two
annotator passes (inclusive presence); the intersection is also reported.

Driver: a ``spawn``-context :class:`ProcessPoolExecutor`. 66 files is far
below the scale where a Luigi sharded fan-out earns its weight, so this is
deliberately *not* Luigi (unlike the long mining batch in
:mod:`data_analysis.auto_motif.batch`); resumability is just a per-file
status JSON + parquet cache (a file with a status JSON is skipped on re-run,
a failure is captured and never blocks the rest). The expensive template
bank is identical for every soundscape, so a pool ``initializer`` loads it
**once per worker** instead of once per file; ``spawn`` makes the
fork-after-threadpool deadlock impossible by construction.

CLI::

    python -m data_analysis.auto_motif.soundscape_eval --list
    python -m data_analysis.auto_motif.soundscape_eval --files BC2026_Train_0039_S22_20211231_201500.ogg
    python -m data_analysis.auto_motif.soundscape_eval                       # full production bank
    python -m data_analysis.auto_motif.soundscape_eval --bank gabor_stack    # the unvalidated selector's alt
    python -m data_analysis.auto_motif.soundscape_eval --rescore cfar        # window-level CFAR operating point

The ``--rescore cfar`` run is the operating-point fix. A probe showed the
committed cut is dominated by the ``[0.45, 0.90]`` focal clamp (67/87
thresholds pinned at the 0.45 floor), so the failure is the floor plus a
frame-vs-window granularity mismatch — not a score shift the focal recipe
can be re-pointed at. ``cfar`` therefore calibrates at the **5 s-window**
unit (the competition/label grid) with **no focal floor**: per species, fire
a window iff its window-max score clears ``Q99.5`` of that species'
corpus-pooled scored-window distribution (label-free background estimate;
0.995 is the pipeline's own committed false-alarm design, re-expressed at
window granularity). ``max_score`` is untouched so ROC/PR-AUC are
byte-identical to the baseline; only ``fired`` (hence precision/recall)
moves. Writes to a parallel ``<bank>__cfar/`` tree, leaving the
committed-cut baseline intact for a clean A/B.
"""

from __future__ import annotations

import os

# Cap math-library threadpools to one thread *before* numpy/librosa/sklearn
# import. This is NOT the fork-safety fix (``main()`` sets the multiprocessing
# start method to ``spawn``, which makes the fork-after-threadpool deadlock
# impossible by construction). It is an independent, principled perf setting:
# the work is Python-loop bound (``_score_track``), not BLAS bound, so 1
# thread/worker removes ``workers × cores`` oversubscription on a shared box
# without giving up any real parallelism. Set as defaults so an explicit
# override still wins; placed before numpy import so the cap is read at
# OpenBLAS init in both the parent and every spawned worker.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, roc_auc_score

from data_analysis.auto_motif.batch import (
    AUTO_MOTIF_DIR,
    _mode_tag,
    _species_dir,
)
from data_analysis.auto_motif.mining import (
    AutoConfig,
    _refine_anchor_via_envelope,
    _score_track,
)
from data_analysis.auto_motif.production_manifest import build_production_manifest
from data_analysis.common.paths import TRAIN_SOUNDSCAPES_DIR
from data_analysis.common.spectrogram import (
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    WIDEBAND,
    wideband_pcen_spectrogram,
)
from data_analysis.marbled_frog.motif_template_calibration import (
    consolidate_events,
    nms_events,
)
from data_analysis.soundscapes.labeled_soundscapes import load_segment_labels

logger = logging.getLogger(__name__)

EVAL_DIR = AUTO_MOTIF_DIR / "_soundscape_eval"
WINDOW_SEC = 5.0  # the soundscape label grid is a fixed 5 s grid


# -----------------------------------------------------------------------------
# Template bank (persisted production templates + committed thresholds)
# -----------------------------------------------------------------------------


def _load_family_bank(species_dir: Path) -> list[dict]:
    """Persisted templates for one species dir + their committed thresholds.

    ``auto_templates_metadata.csv`` holds the frozen per-family ``threshold``
    (the production operating point); ``templates/family_*.npz`` holds the
    template patch and its support scalars.
    """
    meta_path = species_dir / "tables" / "auto_templates_metadata.csv"
    if not meta_path.exists():
        return []
    meta = pl.read_csv(meta_path)
    bank: list[dict] = []
    for row in meta.iter_rows(named=True):
        mc = int(row["motif_cluster"])
        npz_path = species_dir / "templates" / f"family_{mc:04d}_template.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        support = {
            k: float(npz[k][0]) for k in npz.files if k != "template"
        }
        bank.append(
            {
                "motif_cluster": mc,
                "template": npz["template"].astype(np.float32),
                "support": support,
                "threshold": float(row["threshold"]),
            }
        )
    return bank


def load_bank(bank: str, min_focal: int) -> dict[str, list[dict]]:
    """Map species label -> list of family dicts for the chosen routing.

    ``production`` uses the conservative routing from the production manifest
    (the thing actually shipped). ``gabor_stack`` forces every species onto
    its ``free``/``gabor_stack`` artifacts where that batch produced
    templates — i.e. the alternative the *unvalidated* per-species mode
    selector would pick — so this stream can settle that open question with
    real labels instead of the Perch-embedding proxy.
    """
    manifest = build_production_manifest(min_focal=min_focal)
    out: dict[str, list[dict]] = {}
    for row in manifest.iter_rows(named=True):
        label = row["label"]
        if bank == "production":
            if row["production_status"] != "ok":
                continue
            tag = _mode_tag(
                row["production_band_mode"], row["production_cluster_mode"]
            )
        elif bank == "gabor_stack":
            if row["alt_gabor_stack_status"] != "ok":
                continue
            tag = _mode_tag("free", "gabor_stack")
        else:
            raise ValueError(f"unknown bank {bank!r}")
        families = _load_family_bank(_species_dir(label, tag))
        if families:
            out[label] = families
    return out


# -----------------------------------------------------------------------------
# Per-file scoring (exact production detection arithmetic, frozen threshold)
# -----------------------------------------------------------------------------


# Window-level CFAR design point. The committed cut is dominated by the
# ``[threshold_floor, threshold_ceiling]`` clamp (a probe showed 67/87 focal
# thresholds pinned at the 0.45 floor), so the precision collapse is not a
# score domain-shift the in-miner recipe can be re-pointed at — it is the
# *floor* plus a frame-vs-window granularity mismatch (a per-frame quantile,
# max-pooled to 5 s windows over a whole recording, floods the window-level
# false-alarm rate). ``cfar`` therefore calibrates at the **5 s-window**
# unit — exactly the competition/label grid — with **no focal floor**: fire a
# (window, species) iff its window-max matched-filter score exceeds the
# pipeline's own design false-alarm tail of the per-species *background*
# window-score distribution. Background is estimated label-free by pooling
# every scored window across the corpus per species: a 60 s soundscape has
# only ~12 windows, so a *per-recording* Q99.5 is degenerate (≈ the max),
# whereas the pooled window-score distribution is overwhelmingly
# target-absent (most species 1–5 % prevalent) and statistically stable. The
# quantile is not a tuned constant: 0.995 is the pipeline's own committed
# false-alarm design (``q995_bg`` in ``mining._member_vs_background_threshold``),
# re-expressed at the window granularity instead of per frame.
CFAR_BACKGROUND_QUANTILE = 0.995


def _detect_family(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    fam: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(center_times, scores, consolidatable raw-event rows) for one family.

    Replicates :func:`mining.detect_with_auto_templates` peak-picking exactly
    (same ``min_spacing``/``distance``, ``prominence=0.02``, envelope anchor),
    at the committed persisted threshold. Only the ``none`` baseline consumes
    the event rows; ``cfar`` decides ``fired`` at window granularity in
    :func:`_aggregate` and never reaches this peak/NMS path.
    """
    support, template = fam["support"], fam["template"]
    ct, sc = _score_track(z, freqs, times, template, support)
    if ct.size == 0:
        return ct, sc, []
    threshold = fam["threshold"]
    dt = MEL_HOP_LENGTH / MEL_SAMPLE_RATE
    min_spacing = max(0.12, 0.60 * float(support["template_duration_sec"]))
    distance = max(1, int(round(min_spacing / dt)))
    peaks, props = find_peaks(
        sc, height=threshold, distance=distance, prominence=0.02
    )
    rows: list[dict] = []
    for p in peaks:
        raw_t = float(ct[p])
        peak_t = _refine_anchor_via_envelope(
            z, freqs, times, raw_t, support, onset_fraction=0.30
        )
        rows.append(
            {
                "motif_cluster": fam["motif_cluster"],
                "peak_time_sec": peak_t,
                "start_sec": max(0.0, peak_t - support["template_pre_sec"]),
                "end_sec": peak_t + support["template_post_sec"],
                "fmin_hz": support["template_fmin_hz"],
                "fmax_hz": support["template_fmax_hz"],
                "score": float(sc[p]),
                "threshold": threshold,
                "prominence": float(
                    props["prominences"][np.where(peaks == p)[0][0]]
                ),
                "best_template_index": 0,
            }
        )
    return ct, sc, rows


def score_file(
    filename: str,
    bank: dict[str, list[dict]],
    config: AutoConfig,
    rescore: str,
) -> pl.DataFrame:
    """Score one soundscape against every species' bank.

    Returns one row per (5 s window, species) with the threshold-free
    ``max_score`` and the operating-point ``fired`` flag (committed cut for
    ``rescore="none"``, per-recording CFAR cut for ``rescore="cfar"``).
    """
    audio, _ = librosa.load(
        str(TRAIN_SOUNDSCAPES_DIR / filename),
        sr=MEL_SAMPLE_RATE,
        mono=True,
    )
    duration = len(audio) / MEL_SAMPLE_RATE
    n_windows = int(np.ceil(duration / WINDOW_SEC))
    _db, z, freqs, times = wideband_pcen_spectrogram(audio, band=WIDEBAND)

    rows: list[dict] = []
    for label, families in bank.items():
        # threshold-free: per-window max score over all of this species' families
        win_max = np.zeros(n_windows, dtype=np.float64)
        win_seen = np.zeros(n_windows, dtype=bool)
        raw_rows: list[dict] = []
        for fam in families:
            ct, sc, ev = _detect_family(z, freqs, times, fam)
            if ct.size:
                widx = np.clip(
                    (ct // WINDOW_SEC).astype(int), 0, n_windows - 1
                )
                np.maximum.at(win_max, widx, sc)
                win_seen[np.unique(widx)] = True
            for r in ev:
                raw_rows.append({"filename": filename, **r})

        # ``cfar`` decides ``fired`` at window granularity, corpus-pooled, in
        # :func:`_aggregate`; the focal-cut event/NMS pipeline below is the
        # ``none`` baseline only (and is irrelevant at the 5 s-window unit).
        fired = np.zeros(n_windows, dtype=bool)
        if rescore == "none" and raw_rows:
            nms = nms_events(
                pl.DataFrame(raw_rows),
                time_iou_threshold=config.nms_time_iou,
                peak_distance_sec=config.nms_peak_distance_sec,
            )
            cons = consolidate_events(
                nms,
                merge_peak_distance_sec=config.consolidate_peak_distance_sec,
                min_time_iou=config.consolidate_min_time_iou,
            )
            if cons.height:
                widx = np.clip(
                    (cons["peak_time_sec"].to_numpy() // WINDOW_SEC).astype(
                        int
                    ),
                    0,
                    n_windows - 1,
                )
                fired[np.unique(widx)] = True

        for w in range(n_windows):
            rows.append(
                {
                    "filename": filename,
                    "win_start_s": float(w * WINDOW_SEC),
                    "species": label,
                    "max_score": float(win_max[w]) if win_seen[w] else 0.0,
                    "seen": bool(win_seen[w]),
                    "fired": bool(fired[w]),
                }
            )
    return pl.DataFrame(rows)


# -----------------------------------------------------------------------------
# Labels + metrics
# -----------------------------------------------------------------------------


def reconciled_labels() -> pl.DataFrame:
    """One row per labeled (filename, win_start_s) with union/intersection sets.

    The CSV carries two annotator passes per window; ``pos_union`` (species
    flagged by *either* pass) is the inclusive presence used for the headline
    metrics, ``pos_inter`` (flagged by *both*) is reported alongside.
    """
    seg = load_segment_labels()
    per_pass = (
        seg.select(["filename", "start_s", "labels"])
        .with_columns(pl.col("start_s").cast(pl.Float64).alias("win_start_s"))
        .group_by(["filename", "win_start_s"])
        .agg(pl.col("labels").alias("passes"))
    )
    rows = []
    for r in per_pass.iter_rows(named=True):
        passes = [set(p) for p in r["passes"]]
        union = set().union(*passes) if passes else set()
        inter = set(passes[0]).intersection(*passes[1:]) if passes else set()
        rows.append(
            {
                "filename": r["filename"],
                "win_start_s": r["win_start_s"],
                "pos_union": sorted(union),
                "pos_inter": sorted(inter),
            }
        )
    return pl.DataFrame(rows)


def compute_metrics(
    window_scores: pl.DataFrame, labels: pl.DataFrame
) -> pl.DataFrame:
    """Per-species efficacy over the labeled-window universe.

    Evaluation universe = the labeled 5 s windows only (windows absent from
    the labels CSV are not ground-truthed and cannot score precision/recall).
    """
    lab = labels.explode("pos_union").rename({"pos_union": "lab_species"})
    universe = labels.select(["filename", "win_start_s"]).unique()
    scored = window_scores.join(
        universe, on=["filename", "win_start_s"], how="inner"
    )

    out = []
    for species in sorted(scored["species"].unique().to_list()):
        sp = scored.filter(pl.col("species") == species)
        pos = (
            lab.filter(pl.col("lab_species") == species)
            .select(["filename", "win_start_s"])
            .with_columns(pl.lit(1).alias("y"))
        )
        ev = sp.join(
            pos, on=["filename", "win_start_s"], how="left"
        ).with_columns(pl.col("y").fill_null(0))
        y = ev["y"].to_numpy()
        s = ev["max_score"].to_numpy()
        f = ev["fired"].to_numpy().astype(int)
        n_pos = int(y.sum())
        tp = int(((f == 1) & (y == 1)).sum())
        fp = int(((f == 1) & (y == 0)).sum())
        fn = int(((f == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * prec * rec / (prec + rec)
            if prec == prec and rec == rec and (prec + rec) > 0
            else float("nan")
        )
        has_both = 0 < n_pos < len(y)
        out.append(
            {
                "species": species,
                "n_windows": len(y),
                "n_pos": n_pos,
                "prevalence": round(n_pos / len(y), 4) if len(y) else 0.0,
                "roc_auc": round(roc_auc_score(y, s), 4)
                if has_both
                else float("nan"),
                "pr_auc": round(average_precision_score(y, s), 4)
                if has_both
                else float("nan"),
                "precision": round(prec, 4) if prec == prec else float("nan"),
                "recall": round(rec, 4) if rec == rec else float("nan"),
                "f1": round(f1, 4) if f1 == f1 else float("nan"),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return pl.DataFrame(out).sort("roc_auc", descending=True, nulls_last=True)


# Minimum positive 5 s windows for a per-species ROC/PR-AUC to be
# interpretable. With only a handful of positives the AUC is dominated by
# sampling noise (a single positive makes it nearly meaningless), so the
# headline is computed over adequately-supported species and the rest are
# reported separately as underpowered — not as evidence either way. This is
# a small-sample-statistics floor, not a tuned cutoff.
MIN_POS_SUPPORT = 10


def render_summary(metrics: pl.DataFrame, bank: str, rescore: str) -> str:
    valid = metrics.filter(pl.col("roc_auc").is_not_null())
    sup = valid.filter(pl.col("n_pos") >= MIN_POS_SUPPORT)
    underpowered = valid.filter(pl.col("n_pos") < MIN_POS_SUPPORT)
    ns = sup.height
    med_auc = float(sup["roc_auc"].median()) if ns else float("nan")
    med_pr = float(sup["pr_auc"].median()) if ns else float("nan")
    strong = sup.filter(pl.col("roc_auc") >= 0.70)["species"].to_list()
    chance = sup.filter(pl.col("roc_auc") < 0.55)["species"].to_list()
    up = sorted(
        zip(
            underpowered["species"].to_list(),
            underpowered["n_pos"].to_list(),
        )
    )
    with pl.Config(tbl_rows=-1, fmt_str_lengths=40, tbl_width_chars=200):
        table = str(metrics)
    if rescore == "cfar":
        title = (
            f"Auto-motif soundscape efficacy ({bank} bank, "
            "window-level CFAR)"
        )
        op_para = (
            "`roc_auc`/`pr_auc` are threshold-free and **byte-identical to "
            "the committed-cut baseline** (CFAR moves only the operating "
            "point, never the score). `precision`/`recall`/`f1` are at the "
            "**window-level CFAR cut**: per species, fire a 5 s window iff "
            "its window-max score clears `Q99.5` of that species' "
            "corpus-pooled scored-window distribution — no focal floor, "
            "label-free, the pipeline's own 0.5 % false-alarm design "
            "re-expressed at the competition's window grid. Compare to the `"
            + bank
            + "` baseline: precision recovering at unchanged AUC = the floor/"
            "granularity fix landing. Read with the support floor and the "
            "self-masking caveat (high-prevalence species violate "
            "pool≈background, so their cut is conservative by construction — "
            "an asymmetry to report, not a defect)."
        )
    else:
        title = f"Auto-motif soundscape efficacy ({bank} bank)"
        op_para = (
            "`roc_auc`/`pr_auc` are threshold-free (template discriminability "
            "— trust these, within the support caveat). "
            "`precision`/`recall`/`f1` are at the **committed** per-family "
            "threshold (focal→soundscape operating-point transfer; low here "
            "with high AUC = good template, miscalibrated cut, not a template "
            "failure)."
        )
    return f"""# {title}

Label-based extrinsic validation against `train_soundscapes_labels.csv`
(66 files, labeled 5 s windows, annotator-pass **union** as positive).

**Headline is computed only over the {ns} species with ≥ {MIN_POS_SUPPORT}
positive windows** — for the rest the per-species AUC is sampling noise and
is reported, but is *not* evidence for or against the template.

- Species scored: {metrics.height} ({valid.height} with both classes
  present; {ns} adequately supported, {underpowered.height} underpowered)
- Median threshold-free **ROC-AUC** (supported): {med_auc:.3f}
- Median **PR-AUC** (supported): {med_pr:.3f}
- ROC-AUC ≥ 0.70, n_pos ≥ {MIN_POS_SUPPORT} (real, separable on
  soundscapes): {len(strong)} — {strong}
- ROC-AUC < 0.55, n_pos ≥ {MIN_POS_SUPPORT} (≈ chance, does not
  transfer): {len(chance)} — {chance}
- Underpowered (n_pos < {MIN_POS_SUPPORT}, AUC not interpretable):
  {up}

{op_para}

```text
{table}
```
"""


# -----------------------------------------------------------------------------
# Driver: spawn-context process pool, bank loaded once per worker
# -----------------------------------------------------------------------------


def _bank_dir(bank: str, rescore: str) -> Path:
    """Output root. ``rescore="none"`` keeps the original ``<bank>/`` path
    (the committed-cut baseline stays byte-for-byte); ``cfar`` writes to a
    parallel ``<bank>__cfar/`` tree so status/cache/summary never collide and
    the two runs are a clean A/B.
    """
    return EVAL_DIR / (bank if rescore == "none" else f"{bank}__{rescore}")


def labeled_filenames() -> list[str]:
    return sorted(load_segment_labels()["filename"].unique().to_list())


# Per-worker template bank, populated once by the pool initializer and reused
# for every file that worker scores. The bank is identical across soundscapes;
# the only reason the first cut rebuilt it per file was the per-task Luigi
# model, which is exactly the waste this driver removes.
_WORKER_BANK: dict[str, list[dict]] | None = None
_WORKER_CONFIG: AutoConfig | None = None


def _worker_init(bank: str, min_focal: int) -> None:
    global _WORKER_BANK, _WORKER_CONFIG
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _WORKER_BANK = load_bank(bank, min_focal)
    _WORKER_CONFIG = AutoConfig()


def _score_one(filename: str, bank: str, rescore: str) -> dict:
    """Score one soundscape with the worker-resident bank; persist the cache.

    Returns (and writes) the per-file status record; one bad file is captured
    here and never aborts the pool.
    """
    bdir = _bank_dir(bank, rescore)
    (bdir / "status").mkdir(parents=True, exist_ok=True)
    (bdir / "per_file").mkdir(parents=True, exist_ok=True)
    rec: dict = {"filename": filename, "bank": bank, "rescore": rescore}
    try:
        assert _WORKER_BANK is not None and _WORKER_CONFIG is not None
        df = score_file(filename, _WORKER_BANK, _WORKER_CONFIG, rescore)
        out_parquet = bdir / "per_file" / f"{filename}.parquet"
        df.write_parquet(out_parquet)
        rec.update(
            status="ok",
            n_species=len(_WORKER_BANK),
            n_rows=df.height,
            parquet=str(out_parquet.relative_to(AUTO_MOTIF_DIR)),
        )
    except Exception as exc:  # noqa: BLE001 - one bad file never blocks the run
        rec.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-2000:],
        )
        logger.exception("scoring failed for %s", filename)
    (bdir / "status" / f"{filename}.json").write_text(
        json.dumps(rec, indent=2)
    )
    return rec


def _cfar_window_fired(window_scores: pl.DataFrame) -> pl.DataFrame:
    """Window-level CFAR operating point (label-free, no focal floor).

    Per species, the background window-score level is ``Q99.5`` of that
    species' *corpus-pooled scored-window* distribution (sparse targets ⇒ the
    pool is overwhelmingly target-absent); a (window, species) fires iff its
    window-max score clears that level. ``max_score`` is untouched, so the
    threshold-free ROC/PR are byte-identical to the baseline — this changes
    only the operating point. Caveat surfaced, not buried: high-prevalence
    species violate the "pool ≈ background" assumption (CFAR self-masking),
    so their cut is conservative — that asymmetry is a result, not a bug.
    """
    if window_scores.is_empty():
        return window_scores
    thr = (
        window_scores.filter(pl.col("seen"))
        .group_by("species")
        .agg(
            pl.col("max_score")
            .quantile(CFAR_BACKGROUND_QUANTILE, interpolation="linear")
            .alias("_cfar_thr")
        )
    )
    return (
        window_scores.join(thr, on="species", how="left")
        .with_columns(
            (
                pl.col("seen")
                & pl.col("_cfar_thr").is_not_null()
                & (pl.col("max_score") >= pl.col("_cfar_thr"))
            ).alias("fired")
        )
        .drop("_cfar_thr")
    )


def _aggregate(bank: str, rescore: str) -> Path:
    """Join every cached per-file score against the labels; write metrics."""
    bdir = _bank_dir(bank, rescore)
    statuses = [
        json.loads(p.read_text())
        for p in sorted((bdir / "status").glob("*.json"))
    ]
    pl.DataFrame(statuses).write_csv(bdir / "manifest.csv")
    ok = [s for s in statuses if s.get("status") == "ok"]
    frames = [pl.read_parquet(AUTO_MOTIF_DIR / s["parquet"]) for s in ok]
    window_scores = pl.concat(frames) if frames else pl.DataFrame()
    if rescore == "cfar":
        # The cfar fired flag is a corpus-level decision (per-recording window
        # counts are too small for a stable Q99.5); the per-file parquet
        # carried a placeholder. Decide it here, once, over the full pool.
        window_scores = _cfar_window_fired(window_scores)
    window_scores.write_parquet(bdir / "_window_scores.parquet")
    metrics = compute_metrics(window_scores, reconciled_labels())
    out = bdir / "_metrics_per_species.csv"
    metrics.write_csv(out)
    (bdir / "_summary.md").write_text(render_summary(metrics, bank, rescore))
    logger.info("%d/%d files ok; wrote %s", len(ok), len(statuses), out)
    return out


def run_eval(
    bank: str,
    min_focal: int,
    files: list[str],
    workers: int,
    rescore: str,
) -> None:
    """Score `files` (skipping any already cached), then aggregate.

    Resumable: a file whose status JSON exists is not re-scored — delete
    ``<bank[__rescore]>/status/<file>.json`` to force one file to re-run.
    Aggregation always recomputes over *every* cached file (cheap), so it
    reflects the full corpus even after a targeted ``--files`` re-score. The
    ``cfar`` run is a separate tree, so it never invalidates the baseline.
    """
    bdir = _bank_dir(bank, rescore)
    (bdir / "status").mkdir(parents=True, exist_ok=True)
    (bdir / "per_file").mkdir(parents=True, exist_ok=True)
    todo = [
        fn for fn in files if not (bdir / "status" / f"{fn}.json").exists()
    ]
    logger.info(
        "%d/%d files to score (%d cached), %d workers",
        len(todo),
        len(files),
        len(files) - len(todo),
        workers,
    )
    if todo:
        # spawn: a fresh interpreter per worker cannot inherit the parent's
        # locked OpenBLAS/OMP threadpool mutex, so the fork-after-threadpool
        # deadlock is impossible by construction (the thread caps at module
        # top are now only the anti-oversubscription perf setting).
        with ProcessPoolExecutor(
            max_workers=max(1, workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(bank, min_focal),
        ) as ex:
            futs = {
                ex.submit(_score_one, fn, bank, rescore): fn for fn in todo
            }
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                logger.info(
                    "[%d/%d] %s %s", i, len(todo), rec["filename"], rec["status"]
                )
    _aggregate(bank, rescore)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        default="production",
        choices=["production", "gabor_stack"],
        help="template routing under test (default: production)",
    )
    parser.add_argument("--min-focal", type=int, default=8)
    parser.add_argument(
        "--files",
        default=None,
        help="comma-separated soundscape filenames; default = all 66 labeled",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--rescore",
        default="none",
        choices=["none", "cfar"],
        help=(
            "operating point. 'none' = frozen committed focal cut (the "
            "baseline under test). 'cfar' = window-level CFAR: per species, "
            "Q99.5 of the corpus-pooled scored-window score distribution, no "
            "focal floor; writes to a parallel <bank>__cfar/ tree."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the labeled soundscape set and exit",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        for fn in labeled_filenames():
            print(fn)
        return

    files = (
        [s.strip() for s in args.files.split(",") if s.strip()]
        if args.files
        else labeled_filenames()
    )
    run_eval(args.bank, args.min_focal, files, args.workers, args.rescore)


if __name__ == "__main__":
    main()
