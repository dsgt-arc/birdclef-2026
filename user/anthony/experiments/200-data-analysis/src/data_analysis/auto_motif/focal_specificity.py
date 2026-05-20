"""Cross-species focal specificity — the D1 probe (concept doc §v4, strand 5).

Strand 3 found a population of production species whose mined templates are
**at chance on labeled soundscapes** (ROC ≈ 0.5) even though every label-free
screen — the pipeline ``member_bg`` d′ *and* the independent Perch-embedding
AUC (strand 1) — rates them indistinguishable from the separable ones. So the
failure is, by construction, invisible to every label-free metric tried so
far and only shows up against truth. That left two mechanisms unresolved and
conflated:

* **H_guild** — the focal set ``train_audio/<sp>/`` is species-*curated*, so
  a template can look specific *there* while actually keying on a motif that
  guild-mates / congeners also produce (5 frogs in one swamp; 3 *Amazona*
  parrots). A normalized-cross-correlation matcher then fires on all of them;
  focal specificity ⇏ soundscape specificity. This is a **mining** failure
  (the mined object is not species-specific) — no threshold or calibration
  recovers it; the fix is discriminative mining or a learned embedding.
* **H_domain / H_template** — the template *is* species-specific on clean
  focal audio but a single EM-averaged rigid template does not survive the
  focal→soundscape domain shift (far-field, low SNR, masking). This is a
  **representation/adaptation** problem (domain-adaptive refinement, an
  exemplar bank), *not* a mining problem.

This module separates them with the one ground truth that needs **no
labels**: the focal *folder* is curation truth. It is the exact focal analog
of the strand-3 soundscape ROC — the *identical* production matched-filter
(:func:`soundscape_eval.load_bank` → :func:`mining._score_track`,
OR-over-families = per-recording max, the same threshold-free score strand 3
trusts) run over focal recordings of the target species vs. other species,
scored as a ROC AUC of "is this the target's own focal recording":

* ``auc_vs_class`` — target's focal vs. **same-class** other species' focal
  (other frogs for a frog). This *is* the H_guild test: ≈ 0.5 means the
  template cannot tell its own species' *clean* focal audio from a
  guild-mate's *clean* focal audio, so it certainly cannot in a noisy mixed
  soundscape — the failure is upstream, in mining.
* ``auc_vs_all`` — target's focal vs. **all** other species' focal (broad
  specificity, the looser bound).

**The asymmetry is stated, not buried (this is what D1 can and cannot
conclude).** A *low* ``auc_vs_class`` is **decisive for H_guild** — clean
focal self-vs-congener separability is a *necessary* condition for soundscape
recovery, and failing the necessary condition is conclusive. A *high*
``auc_vs_class`` is **not sufficient**: it only clears the mining charge; the
species can still fail in-field by H_domain. So D1 *confirms* H_guild and
*triages the remainder* into "specific on clean focal, lost in the field"
(→ the focal-vs-soundscape gap analysis, the strand-3 follow-up), it does not
by itself prove recovery.

Method (no re-mining, no recompute of templates):

* Targets: species under test (default = the strand-3 at-chance set + the
  separable + originally-validated controls so the AUC scale is anchored —
  ``22967`` the validated frog is the instrument sanity check).
* Scoring pool: focal recordings across **every** production species (so each
  target has same-class negatives). Per species the contribution is capped at
  the **corpus median ``n_focal``** and evenly stride-subsampled from the
  sorted file list — a data-derived cap (no tuned ``K``) so no large corpus
  dominates a class stratum, and a deterministic prefix-stride (no RNG seed)
  so the run is reproducible and resumable. AUC is a rank statistic so the
  absolute pool size does not bias it; only per-stratum coverage matters,
  which the cap equalises. (Genericity tradeoff, surfaced: the stride
  subsample assumes a species' first-vs-rest focal files are acoustically
  exchangeable for a *specificity* screen — true for repertoire coverage,
  not a claim about site/recency structure.)
* Per pooled recording, one ``wideband_pcen_spectrogram`` then, for every
  target species' bank, the *exact* production threshold-free score: max over
  the species' families of :func:`mining._score_track`'s peak score (the same
  OR-over-families = ``max_score`` strand 3 reports, at whole-recording
  granularity — a focal clip is the natural single-context unit, the analog
  of strand 3's 5 s window).

The readout joins ``_soundscape_eval/production/_metrics_per_species.csv`` so
each target lands in a four-quadrant verdict directly (interpretation bands
mirror :mod:`soundscape_eval`'s own ``0.70`` strong / ``0.55`` chance bands —
deliberately the *same* bands for cross-strand comparability, not new tuned
constants).

Driver: a ``spawn``-context :class:`ProcessPoolExecutor` with the bank loaded
once per worker and a per-file status/parquet cache — a deliberate mirror of
:mod:`soundscape_eval` (same scale class: with the median cap the pool stays
far below the >~1k-file threshold where a Luigi sharded fan-out earns its
weight; ``spawn`` makes the fork-after-threadpool BLAS deadlock impossible by
construction).

CLI::

    python -m data_analysis.auto_motif.focal_specificity                 # default targets
    python -m data_analysis.auto_motif.focal_specificity --targets all    # every production species
    python -m data_analysis.auto_motif.focal_specificity --targets 23158,chacha1,orwpar
"""

from __future__ import annotations

import os

# Cap math-library threadpools before numpy import — same rationale as
# :mod:`soundscape_eval` (the work is Python-loop bound in ``_score_track``,
# not BLAS bound; ``spawn`` is the actual fork-safety fix, this is only the
# anti-oversubscription perf setting). Defaults so an explicit override wins.
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
from sklearn.metrics import roc_auc_score

from data_analysis.auto_motif.batch import AUTO_MOTIF_DIR
from data_analysis.auto_motif.mining import AutoConfig, _score_track
from data_analysis.auto_motif.production_manifest import build_production_manifest
from data_analysis.auto_motif.soundscape_eval import EVAL_DIR, load_bank
from data_analysis.common.paths import (
    TRAIN_AUDIO_DIR,
    recording_files_for_label,
)
from data_analysis.common.spectrogram import (
    MEL_SAMPLE_RATE,
    WIDEBAND,
    wideband_pcen_spectrogram,
)

logger = logging.getLogger(__name__)

OUT_DIR = AUTO_MOTIF_DIR / "_focal_specificity"
SOUNDSCAPE_METRICS = EVAL_DIR / "production" / "_metrics_per_species.csv"

# Strand-3 at-chance set (ROC < 0.55 on labeled soundscapes) + the separable /
# originally-validated controls. ``22967`` (the hand-validated Marbled Frog)
# is the instrument sanity check: its focal specificity *must* be high.
AT_CHANCE = ["23158", "25092", "undtin1", "chacha1", "43435", "326272", "orwpar"]
CONTROLS = ["22967", "hyamac1", "65380"]
DEFAULT_TARGETS = AT_CHANCE + CONTROLS

# Interpretation bands — *the same* ones :mod:`soundscape_eval` uses in its
# summary (0.70 "real/separable", 0.55 "≈ chance"). Reused verbatim for
# cross-strand comparability; these are narrative bands for a diagnostic, not
# tuned pipeline parameters (cf. ``soundscape_eval.render_summary``).
STRONG = 0.70
CHANCE = 0.55


# -----------------------------------------------------------------------------
# Scoring pool: every production species' focal audio, median-capped
# -----------------------------------------------------------------------------


def build_pool(min_focal: int) -> tuple[list[dict], int, dict[str, str]]:
    """Pooled focal recordings across every ``ok`` production species.

    Returns ``(rows, cap, class_by_label)`` where ``rows`` is one dict per
    sampled recording ``{rel, source_label, source_class}``, ``cap`` is the
    data-derived per-species file cap (the corpus median ``n_focal``), and
    ``class_by_label`` maps every pooled species to its taxonomic class.

    The cap is the species set's own median focal-count (no tuned ``K``); each
    species is evenly stride-subsampled from its sorted file list down to the
    cap (deterministic, no RNG) so no large corpus dominates a class stratum.
    """
    manifest = build_production_manifest(min_focal=min_focal)
    ok = manifest.filter(pl.col("production_status") == "ok")
    class_by_label = dict(
        zip(ok["label"].to_list(), ok["class_name"].to_list())
    )
    cap = int(round(float(ok["n_focal"].median())))

    rows: list[dict] = []
    for label, cls in class_by_label.items():
        files = recording_files_for_label(label)  # sorted "<label>/<name>"
        if not files:
            continue
        if len(files) > cap:
            step = len(files) / cap
            files = [files[int(i * step)] for i in range(cap)]
        for rel in files:
            rows.append(
                {"rel": rel, "source_label": label, "source_class": cls}
            )
    return rows, cap, class_by_label


# -----------------------------------------------------------------------------
# Per-recording threshold-free score (exact production OR-over-families = max)
# -----------------------------------------------------------------------------


def _recording_max_score(
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    families: list[dict],
) -> float:
    """Production threshold-free score for one recording, OR over families.

    Identical arithmetic to :func:`soundscape_eval.score_file`'s ``win_max``
    (``_score_track`` peak score, max over the species' families) but at
    whole-recording granularity: a focal clip is the natural single-context
    unit, the analog of strand 3's 5 s soundscape window.
    """
    best = 0.0
    for fam in families:
        _ct, sc = _score_track(
            z, freqs, times, fam["template"], fam["support"]
        )
        if sc.size:
            best = max(best, float(sc.max()))
    return best


def score_recording(
    rel: str,
    targets_bank: dict[str, list[dict]],
) -> list[dict]:
    """One PCEN per recording; score it against every target species' bank."""
    audio, _ = librosa.load(
        str(TRAIN_AUDIO_DIR / rel), sr=MEL_SAMPLE_RATE, mono=True
    )
    _db, z, freqs, times = wideband_pcen_spectrogram(audio, band=WIDEBAND)
    out: list[dict] = []
    for target, families in targets_bank.items():
        out.append(
            {
                "rel": rel,
                "target_species": target,
                "max_score": _recording_max_score(z, freqs, times, families),
            }
        )
    return out


# -----------------------------------------------------------------------------
# Driver: spawn pool, production bank loaded once per worker (mirror of
# soundscape_eval); per-recording status JSON + parquet cache for resumability
# -----------------------------------------------------------------------------

_WORKER_TARGETS_BANK: dict[str, list[dict]] | None = None


def _worker_init(min_focal: int, targets: tuple[str, ...]) -> None:
    global _WORKER_TARGETS_BANK
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    full = load_bank("production", min_focal)
    _WORKER_TARGETS_BANK = {
        sp: full[sp] for sp in targets if sp in full and full[sp]
    }


def _safe_name(rel: str) -> str:
    return rel.replace("/", "__")


def _score_one(row: dict) -> dict:
    """Score one pooled recording with the worker-resident target banks."""
    (OUT_DIR / "status").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_file").mkdir(parents=True, exist_ok=True)
    rec: dict = {**row}
    try:
        assert _WORKER_TARGETS_BANK is not None
        scores = score_recording(row["rel"], _WORKER_TARGETS_BANK)
        df = pl.DataFrame(
            [
                {
                    "rel": row["rel"],
                    "source_label": row["source_label"],
                    "source_class": row["source_class"],
                    **s,
                }
                for s in scores
            ]
        )
        out_parquet = OUT_DIR / "per_file" / f"{_safe_name(row['rel'])}.parquet"
        df.write_parquet(out_parquet)
        rec.update(
            status="ok",
            n_targets=len(_WORKER_TARGETS_BANK),
            parquet=str(out_parquet.relative_to(AUTO_MOTIF_DIR)),
        )
    except Exception as exc:  # noqa: BLE001 - one bad file never blocks the run
        rec.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-2000:],
        )
        logger.exception("scoring failed for %s", row["rel"])
    (OUT_DIR / "status" / f"{_safe_name(row['rel'])}.json").write_text(
        json.dumps(rec, indent=2)
    )
    return rec


# -----------------------------------------------------------------------------
# Aggregation: per-target focal AUCs + four-quadrant verdict vs strand-3 ROC
# -----------------------------------------------------------------------------


def _soundscape_roc() -> dict[str, dict]:
    """Strand-3 per-species ROC/n_pos, for the four-quadrant join (or {})."""
    if not SOUNDSCAPE_METRICS.exists():
        logger.warning(
            "strand-3 metrics absent (%s); verdict left null", SOUNDSCAPE_METRICS
        )
        return {}
    m = pl.read_csv(SOUNDSCAPE_METRICS)
    return {
        r["species"]: {"soundscape_roc": r["roc_auc"], "soundscape_n_pos": r["n_pos"]}
        for r in m.iter_rows(named=True)
    }


def _verdict(auc_vs_class: float, ss_roc: float | None, ss_n_pos: int | None) -> str:
    """Four-quadrant mechanism call (see module docstring for the asymmetry)."""
    if ss_roc is not None and ss_n_pos is not None and ss_n_pos >= 10 and ss_roc >= STRONG:
        return "transfers (control: separable on soundscapes)"
    if auc_vs_class != auc_vs_class:  # NaN: no same-class negatives
        return "indeterminate (no same-class negatives)"
    if auc_vs_class < CHANCE:
        return "H_guild — guild-generic, mining failure (→ discriminative mining / learned embedding)"
    if auc_vs_class < STRONG:
        return "weak focal specificity (partial H_guild)"
    if ss_roc is not None and ss_n_pos is not None and ss_n_pos >= 10 and ss_roc < CHANCE:
        return "H_domain/H_template — focal-specific, lost in field (→ domain adapt / exemplar bank)"
    return "focal-specific (soundscape support insufficient to localise the field failure)"


def aggregate(class_by_label: dict[str, str], cap: int) -> Path:
    """Join cached per-recording scores into per-target focal-specificity."""
    statuses = [
        json.loads(p.read_text())
        for p in sorted((OUT_DIR / "status").glob("*.json"))
    ]
    pl.DataFrame(statuses).write_csv(OUT_DIR / "manifest.csv")
    ok = [s for s in statuses if s.get("status") == "ok"]
    frames = [pl.read_parquet(AUTO_MOTIF_DIR / s["parquet"]) for s in ok]
    scores = pl.concat(frames) if frames else pl.DataFrame()
    scores.write_parquet(OUT_DIR / "_scores.parquet")

    ss = _soundscape_roc()
    rows = []
    for target in sorted(scores["target_species"].unique().to_list()):
        sp = scores.filter(pl.col("target_species") == target)
        cls = class_by_label.get(target)
        y_in = (sp["source_label"] == target).to_numpy()
        s = sp["max_score"].to_numpy()

        same_class = (sp["source_class"] == cls).to_numpy()
        # vs same-class others (H_guild test) and vs all others
        m_class = y_in | (same_class & ~y_in)
        y_c, s_c = y_in[m_class], s[m_class]
        y_a, s_a = y_in, s

        def _auc(y: np.ndarray, sc: np.ndarray) -> float:
            return (
                round(float(roc_auc_score(y.astype(int), sc)), 4)
                if 0 < int(y.sum()) < len(y)
                else float("nan")
            )

        auc_class = _auc(y_c, s_c)
        ssr = ss.get(target, {})
        rows.append(
            {
                "target_species": target,
                "class_name": cls,
                "n_in": int(y_in.sum()),
                "n_class_neg": int((same_class & ~y_in).sum()),
                "n_all_neg": int((~y_in).sum()),
                "auc_vs_class": auc_class,
                "auc_vs_all": _auc(y_a, s_a),
                "soundscape_roc": ssr.get("soundscape_roc"),
                "soundscape_n_pos": ssr.get("soundscape_n_pos"),
                "verdict": _verdict(
                    auc_class,
                    ssr.get("soundscape_roc"),
                    ssr.get("soundscape_n_pos"),
                ),
            }
        )
    metrics = pl.DataFrame(rows).sort("auc_vs_class", nulls_last=True)
    out = OUT_DIR / "_metrics.csv"
    metrics.write_csv(out)
    (OUT_DIR / "_summary.md").write_text(_render_summary(metrics, cap))
    logger.info("%d/%d recordings ok; wrote %s", len(ok), len(statuses), out)
    return out


def _render_summary(metrics: pl.DataFrame, cap: int) -> str:
    with pl.Config(tbl_rows=-1, fmt_str_lengths=70, tbl_width_chars=200):
        table = str(metrics)
    return f"""# Cross-species focal specificity (D1 probe — concept §v4 strand 5)

The focal analog of the strand-3 soundscape ROC: the **exact production
matched-filter** (same `load_bank` → `_score_track` OR-over-families = max,
the threshold-free score strand 3 trusts) over focal recordings, scored as a
ROC AUC of "is this the target species' own focal recording". The focal
*folder* is curation ground truth — fully **label-free**.

Per-species scoring pool capped at the corpus median `n_focal` = **{cap}**
(data-derived, no tuned K), evenly stride-subsampled (deterministic, no RNG).

- `auc_vs_class` — target focal vs **same-class** other species' focal. This
  is the H_guild test. **≈ 0.5 ⇒ the template cannot separate its own
  species' *clean* focal audio from a guild-mate's *clean* focal audio**, so
  it certainly cannot in a noisy mixed soundscape: the failure is upstream in
  mining, not a calibration or domain problem.
- `auc_vs_all` — vs all other species' focal (the looser specificity bound).
- `verdict` joins strand-3 soundscape ROC for the four-quadrant call.

**Asymmetry (what D1 can and cannot conclude).** A *low* `auc_vs_class` is
**decisive for H_guild** — clean self-vs-congener separability is a
*necessary* condition for soundscape recovery; failing it is conclusive and
no threshold/calibration fix applies (the constructive next step is
discriminative mining — the concept's deferred v2 — or routing to a learned
embedding). A *high* `auc_vs_class` is **necessary, not sufficient**: it only
clears the mining charge; the species can still fail in-field by H_domain
(focal-specific, lost to far-field SNR/masking), which is the strand-3
follow-up (focal-vs-soundscape gap), not a mining fix.

Bands `0.70`/`0.55` mirror `soundscape_eval`'s own strong/chance bands
(cross-strand comparability, not new tuned constants).

```text
{table}
```
"""


def run(min_focal: int, targets: list[str], workers: int) -> None:
    """Score the pool (skipping cached recordings), then aggregate.

    Resumable: a recording whose status JSON exists is not re-scored — delete
    ``_focal_specificity/status/<rel-with-__>.json`` to force a re-run.
    """
    pool, cap, class_by_label = build_pool(min_focal)
    if targets == ["all"]:
        targets = sorted(load_bank("production", min_focal).keys())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "status").mkdir(parents=True, exist_ok=True)
    todo = [
        r
        for r in pool
        if not (OUT_DIR / "status" / f"{_safe_name(r['rel'])}.json").exists()
    ]
    logger.info(
        "pool=%d recordings (cap=%d/species), %d targets, %d to score "
        "(%d cached), %d workers",
        len(pool),
        cap,
        len(targets),
        len(todo),
        len(pool) - len(todo),
        workers,
    )
    if todo:
        with ProcessPoolExecutor(
            max_workers=max(1, workers),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(min_focal, tuple(targets)),
        ) as ex:
            futs = {ex.submit(_score_one, r): r["rel"] for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                logger.info(
                    "[%d/%d] %s %s", i, len(todo), rec["rel"], rec["status"]
                )
    aggregate(class_by_label, cap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-focal", type=int, default=8)
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help=(
            "comma-separated species under test, or 'all'. Default = the "
            "strand-3 at-chance set + separable/validated controls."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    run(args.min_focal, targets, args.workers)


if __name__ == "__main__":
    main()
