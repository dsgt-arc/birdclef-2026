"""Build the manual-curation anchor scaffold for the broadband recovery check.

The Perch-embedding validation left a split verdict: the mined motifs are
genuine recurrent objects (qualitative win) but the pipeline `member_bg` d′
and the per-species mode pick are NOT independently validated, and `74113` is
a concrete suspected false positive. The *only* instrument that can calibrate
the d′→recovery link and adjudicate `74113` is a small **exhaustive human
annotation** — and it must yield *recall*, not just precision (a
detection-only review can never tell you what the pipeline missed).

Production-leg detection counts are far too large to label one-by-one
(74113≈589, hyamac1≈1344, fusfly1≈3917). So the protocol is **recording-
sampled exhaustive annotation**: per species, pick a small density-stratified
set of recordings; within each, the annotator marks every pipeline detection
true/false AND notes any *missed* true calls (→ recall). "Exhaustive" is
within a bounded recording sample — that bound is what makes 3 species
tractable while still giving an honest precision *and* recall.

For `fusfly1` a secondary, lighter sheet probes the *opportunistic*
`gabor_stack` leg (nfam≈11): is that family richness real, or fragmentation?
— sample a few detections per motif cluster and ask coherence per cluster.

Spectrograms are rendered with the pipeline's own
:func:`wideband_pcen_spectrogram`, so the annotator sees exactly what the
miner consumed. Audio is referenced at its scratch path (never copied into
the experiment dir, per CLAUDE.md).

CLI::

    python -m data_analysis.auto_motif.curation_scaffold            # default 3
    python -m data_analysis.auto_motif.curation_scaffold --max-render-sec 240
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from data_analysis.auto_motif.batch import AUTO_MOTIF_DIR, _species_dir  # noqa: E402
from data_analysis.auto_motif.cofire_fragmentation import (  # noqa: E402
    cofire_collapse,
)
from data_analysis.common.paths import TRAIN_AUDIO_DIR  # noqa: E402
from data_analysis.common.spectrogram import (  # noqa: E402
    WIDEBAND,
    wideband_pcen_spectrogram,
)

logger = logging.getLogger(__name__)

CURATION_DIR = AUTO_MOTIF_DIR / "_curation"

# (label, mode_tag, leg-name). The first three are the production (free+global)
# anchor; the last is the fusfly1 gabor_stack opportunistic probe.
PRODUCTION_TARGETS = [
    ("74113", "_bandfree", "production"),
    ("hyamac1", "_bandfree", "production"),
    ("fusfly1", "_bandfree", "production"),
]
GABOR_PROBE = ("fusfly1", "_gabor_stack")

N_RECORDINGS = 4          # density-stratified recordings per species
STRIP_SEC = 12.0          # seconds per spectrogram strip row


def _events(tag: str, label: str) -> pl.DataFrame:
    p = _species_dir(label, tag) / "tables" / "template_match_events_nms.csv"
    return pl.read_csv(p)


def _duration(filename: str) -> float:
    try:
        return float(librosa.get_duration(path=str(TRAIN_AUDIO_DIR / filename)))
    except Exception:  # noqa: BLE001 — unreadable file -> treat as long
        return float("inf")


def _stratified_recordings(ev: pl.DataFrame, max_render_sec: float) -> pl.DataFrame:
    """Pick N_RECORDINGS spanning the detection-density range.

    Honest recall needs the *whole* recording annotated, so candidates are
    restricted to recordings short enough to render in full
    (``duration <= max_render_sec``); only if too few such exist do we fall
    back to (flagged) truncated recordings. Without this, an ultra-dense
    species' densest recording is also its longest — exactly the case where a
    truncated recall would be most misleading (e.g. the 74113 false positive).
    """
    per = (
        ev.group_by("filename")
        .agg(
            pl.len().alias("n_det"),
            pl.col("score").mean().round(3).alias("mean_score"),
        )
        .sort("n_det", descending=True)
        .to_dicts()
    )
    for r in per:  # duration is read once here, reused downstream
        r["duration_sec"] = round(_duration(r["filename"]), 1)
        r["fully_rendered"] = bool(r["duration_sec"] <= max_render_sec)

    pool = [r for r in per if r["fully_rendered"]] or per
    h = len(pool)
    if h <= N_RECORDINGS:
        idx = list(range(h))
    else:  # quantile-spaced ranks within the pool: densest ... sparsest
        idx = sorted({round(q * (h - 1)) for q in np.linspace(0, 1, N_RECORDINGS)})
    picks = [
        {**pool[rank], "rank": rank,
         "density_bucket": f"{rank + 1}/{h} (by n_det desc, "
         f"{'renderable pool' if pool is not per else 'all — none renderable'})"}
        for rank in idx
    ]
    return pl.DataFrame(picks)


def _render_recording(
    label: str, filename: str, dets: pl.DataFrame, out_png: Path,
    max_render_sec: float,
) -> None:
    audio, _ = librosa.load(
        TRAIN_AUDIO_DIR / filename, sr=32_000, mono=True,
        duration=max_render_sec,
    )
    _, s, freqs, times = wideband_pcen_spectrogram(audio, band=WIDEBAND)
    total = float(times[-1]) if times.size else 0.0
    n_strips = max(1, int(np.ceil(total / STRIP_SEC)))
    fig, axes = plt.subplots(
        n_strips, 1, figsize=(14, 1.7 * n_strips + 0.5), squeeze=False
    )
    vmin, vmax = np.percentile(s, [5, 99])
    d = dets.to_dicts()
    for k in range(n_strips):
        ax = axes[k][0]
        t0, t1 = k * STRIP_SEC, (k + 1) * STRIP_SEC
        m = (times >= t0) & (times < t1)
        if m.any():
            ax.pcolormesh(
                times[m], freqs, s[:, m], vmin=vmin, vmax=vmax,
                shading="nearest", cmap="magma",
            )
        for row in d:
            if row["start_sec"] >= t1 or row["end_sec"] <= t0:
                continue
            ax.add_patch(
                plt.Rectangle(
                    (row["start_sec"], row["fmin_hz"]),
                    row["end_sec"] - row["start_sec"],
                    row["fmax_hz"] - row["fmin_hz"],
                    fill=False, edgecolor="cyan", linewidth=1.2,
                )
            )
            ax.text(
                row["start_sec"], row["fmax_hz"], f'#{row["det_id"]}',
                color="cyan", fontsize=7, va="bottom",
            )
        ax.set_xlim(t0, min(t1, total) if total else t1)
        ax.set_ylim(WIDEBAND)
        ax.set_ylabel("Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(np.arange(t0, t1 + 0.01, 1.0))
    axes[-1][0].set_xlabel("time (s) — cyan boxes = pipeline detections", fontsize=9)
    fig.suptitle(
        f"{label}  {filename}   ({len(d)} detections, "
        f"{total:.0f}s rendered)", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _build_production(label: str, tag: str, max_render_sec: float) -> tuple:
    ev = _events(tag, label).with_row_index("det_id")
    sel = _stratified_recordings(ev, max_render_sec)
    sp_dir = CURATION_DIR / label
    sp_dir.mkdir(parents=True, exist_ok=True)

    prec, recall = [], []
    for r in sel.to_dicts():
        fn = r["filename"]
        dets = ev.filter(pl.col("filename") == fn).sort("start_sec")
        stem = Path(fn).stem
        png = sp_dir / f"{stem}.png"
        _render_recording(label, fn, dets, png, max_render_sec)
        for row in dets.to_dicts():
            prec.append(
                dict(
                    species=label, recording=fn, det_id=row["det_id"],
                    peak_time_sec=round(row["peak_time_sec"], 3),
                    start_sec=round(row["start_sec"], 3),
                    end_sec=round(row["end_sec"], 3),
                    score=round(row["score"], 4),
                    spectrogram=str(png.relative_to(AUTO_MOTIF_DIR.parents[1])),
                    is_true_call="",  # annotator: T / F / ?
                    notes="",
                )
            )
        recall.append(
            dict(
                species=label, recording=fn,
                audio_abspath=str(TRAIN_AUDIO_DIR / fn),
                spectrogram=str(png.relative_to(AUTO_MOTIF_DIR.parents[1])),
                duration_sec=r["duration_sec"],
                fully_rendered=r["fully_rendered"],
                density_bucket=r["density_bucket"],
                n_detections=r["n_det"], mean_score=r["mean_score"],
                missed_call_times_sec="",  # annotator: comma-sep timestamps
                n_missed="",
                notes="" if r["fully_rendered"]
                else f"render truncated at {max_render_sec:.0f}s — "
                     f"recall is within first {max_render_sec:.0f}s only",
            )
        )
    return sel, pl.DataFrame(prec), pl.DataFrame(recall)


def _build_gabor_probe(label: str, tag: str) -> pl.DataFrame:
    """Collapsed-effective-group view of the gabor_stack opportunistic leg.

    The label-free co-firing test (:mod:`cofire_fragmentation`) already
    answers "is the family richness real?" structurally for most species; the
    residual human question on a *flagged* (collapsing) species is only
    whether the few co-firing groups are distinct coherent calls. So instead
    of N hard-to-compare raw-cluster montages, render ONE montage: rows =
    effective groups, columns = one top-score exemplar per member family
    (within-row consistency ⇒ the fragments are the same call; across-row
    difference ⇒ the groups are genuinely distinct).
    """
    ev = _events(tag, label)
    r = cofire_collapse(ev)
    counts = dict(ev.group_by("motif_cluster").len().iter_rows())
    sp_dir = CURATION_DIR / f"{label}_gabor_stack_probe"
    sp_dir.mkdir(parents=True, exist_ok=True)
    groups = r["groups"]
    max_cols = max(min(len(g), 6) for g in groups)
    png = sp_dir / "effective_groups.png"
    rel_png = str(png.relative_to(AUTO_MOTIF_DIR.parents[1]))

    fig, axes = plt.subplots(
        len(groups), max_cols,
        figsize=(2.4 * max_cols, 2.4 * len(groups)), squeeze=False,
    )
    rows = []
    for gi, members in enumerate(groups):
        for j in range(max_cols):
            ax = axes[gi][j]
            ax.set_ylim(WIDEBAND)
            ax.tick_params(labelsize=6)
            if j >= len(members):
                ax.axis("off")
                continue
            c = members[j]
            top = (
                ev.filter(pl.col("motif_cluster") == c)
                .sort("score", descending=True)
                .head(1)
                .to_dicts()
            )
            if not top:
                ax.axis("off")
                continue
            row = top[0]
            pad = 0.25
            a, _ = librosa.load(
                TRAIN_AUDIO_DIR / row["filename"], sr=32_000, mono=True,
                offset=max(0.0, row["start_sec"] - pad),
                duration=(row["end_sec"] - row["start_sec"]) + 2 * pad,
            )
            if a.size:
                _, s, freqs, times = wideband_pcen_spectrogram(a, band=WIDEBAND)
                ax.pcolormesh(
                    times, freqs, s,
                    vmin=np.percentile(s, 5), vmax=np.percentile(s, 99),
                    shading="nearest", cmap="magma",
                )
            ax.set_title(f"fam {c}\n{row['score']:.2f}", fontsize=7)
        axes[gi][0].set_ylabel(
            f"group {gi + 1}\n{len(members)} fam", fontsize=8
        )
        rows.append(
            dict(
                species=label, effective_group=gi + 1,
                n_member_families=len(members),
                member_families=",".join(str(c) for c in members),
                n_det=sum(counts.get(c, 0) for c in members),
                frac_distinct=r["frac_distinct"],
                montage=rel_png,
                coherent="",                 # annotator: Y / P / N
                distinct_from_other_groups="",  # annotator: Y / N
                notes="",
            )
        )
    fig.suptitle(
        f"{label} gabor_stack: {r['n_families']} families → "
        f"{r['effective_groups']} co-firing groups "
        f"(frac_distinct={r['frac_distinct']}, eps={r['eps_sec']}s). "
        f"rows=groups, cols=one exemplar per member family",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return pl.DataFrame(rows)


_README = """\
# Manual-curation anchor — annotation protocol

**Why:** Perch-embedding validation confirmed the mined motifs are genuine
recurrent objects *qualitatively*, but could NOT validate the `member_bg` d′
magnitude or the per-species mode pick, and flagged `74113` as a suspected
false positive. This exhaustive human pass is the only instrument that
calibrates d′→true-recovery and adjudicates `74113`.

**Scope:** 3 species, a small density-stratified recording sample each
(`recordings_selected.csv`). Exhaustive *within those recordings*.

## Per recording (do both)

Open `<species>/<recording>.png` and the audio at `audio_abspath`
(from `recall_sheet.csv`; play it from the scratch path, do not copy it).

1. **Precision** — `precision_sheet.csv`, one row per pipeline detection
   (cyan box, numbered `#det_id`). Set `is_true_call` to:
   - `T` the box is on a real call of this species
   - `F` it is noise / another species / silence
   - `?` genuinely ambiguous (use sparingly)
2. **Recall** — `recall_sheet.csv`, one row per recording. Listen/scan the
   whole recording; for every real call of this species that has **no** box,
   add its approximate peak time to `missed_call_times_sec` (comma-separated)
   and set `n_missed`. If `fully_rendered` is false, recall counts only
   within the rendered window (noted per row).

## fusfly1 gabor_stack probe (secondary, mostly answered structurally)

"Is gabor_stack's family richness real or fragmentation?" is answered
label-free by the co-firing test (`cofire_fragmentation`): 6/12 rich species
show *zero* collapse (genuine richness); fusfly1 is the worst case (11
families → 3 co-firing groups, frac_distinct 0.27). So the only residual
human check is on that flagged species: in
`fusfly1_gabor_stack_probe/effective_groups.png` (rows = co-firing groups,
one exemplar per member family) + `gabor_stack_group_sheet.csv`, per group
set `coherent` = `Y`/`P`/`N` (is the group one repeated call type — i.e. were
the fragments correctly merged?) and `distinct_from_other_groups` = `Y`/`N`.

## After annotation

Re-run the validation join to compute precision/recall per species and test
whether pipeline d′ / the mode pick tracks true recovery.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-render-sec", type=float, default=180.0)
    parser.add_argument(
        "--no-gabor-probe", action="store_true",
        help="skip the fusfly1 gabor_stack secondary sheet",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    CURATION_DIR.mkdir(parents=True, exist_ok=True)
    sel_all, prec_all, recall_all = [], [], []
    for label, tag, leg in PRODUCTION_TARGETS:
        logger.info("scaffolding %s (%s, %s)", label, tag, leg)
        sel, prec, recall = _build_production(label, tag, args.max_render_sec)
        sel_all.append(sel.with_columns(pl.lit(label).alias("species")))
        prec_all.append(prec)
        recall_all.append(recall)

    pl.concat(sel_all, how="diagonal").write_csv(
        CURATION_DIR / "recordings_selected.csv"
    )
    pl.concat(prec_all).write_csv(CURATION_DIR / "precision_sheet.csv")
    pl.concat(recall_all).write_csv(CURATION_DIR / "recall_sheet.csv")

    if not args.no_gabor_probe:
        label, tag = GABOR_PROBE
        logger.info("scaffolding gabor_stack probe %s (%s)", label, tag)
        g = _build_gabor_probe(label, tag)
        g.write_csv(CURATION_DIR / "gabor_stack_group_sheet.csv")

    (CURATION_DIR / "README.md").write_text(_README)

    n_prec = sum(p.height for p in prec_all)
    logger.info(
        "scaffold ready in %s : %d recordings, %d detections to judge "
        "(+ recall scan)", CURATION_DIR,
        sum(s.height for s in sel_all), n_prec,
    )
    print(textwrap.indent(_README, "  "))


if __name__ == "__main__":
    main()
