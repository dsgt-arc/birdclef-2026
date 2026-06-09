"""Perch-prototype retrieval: the feasibility gate (Stage A) + ranked pre-boxed
batch (Stage B) of the semi-supervised box-annotation loop.

The loop's premise (``docs/concepts/semi-supervised-box-annotation-loop.md``):
a contrastive Perch prototype **routes** windows, the DSP front-end **pre-draws**
boxes, the human adjudicates only the margin. Perch is window-level (one 1536-d
vector per 5s window, frequency collapsed) so it cannot box — it ranks.

Stage A is the gate that decides whether the rest is worth building: can a
weak-label-seeded contrastive prototype (``mean(present) - mean(absent)`` in
embedding space) rank a sonotype's windows cross-site, beating trivial baselines
under leave-one-insect-site-out, without firing on the insect-free site? If not,
stop.

CLI::

    python -m birdclef_201.retrieve gate
    python -m birdclef_201.retrieve export --cap 300   # Stage B (built separately)
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl

from birdclef_201.boxes import analyze_labeled_windows, discriminative_band
from birdclef_201.labels import insect_sonotypes, insect_sites, load_windows
from birdclef_201.paths import PERCH_SOUNDSCAPE_EMBEDDINGS, artifact_dir

logger = logging.getLogger(__name__)

MIN_PRESENT = 8  # n_present gate (matches boxes.MIN_PRESENT)
RECALL_OP = 0.8  # operating point for the specificity (false-positive) probe
N_RANDOM_SEEDS = 20  # random-within-site baseline averaging

# Perfectly co-occurring sonotype pairs the weak labels cannot separate
# (identical present-window sets) -> merge each into one canonical label.
_PAIRS = {"47158son16": "47158son15", "47158son23": "47158son22"}


def canon(label: str) -> str:
    """Map a raw sonotype to its canonical (merged) label."""
    return _PAIRS.get(label, label)


# --------------------------------------------------------------------------- #
# embeddings                                                                   #
# --------------------------------------------------------------------------- #
def load_embeddings(filenames: set[str] | None = None) -> tuple[pl.DataFrame, list[str]]:
    """Load Perch window embeddings -> (df with filename/start_sec + emb_*, emb_cols).

    ``filenames`` restricts the scan (predicate pushdown) to e.g. the 66 labeled
    files for the gate. ``start_sec`` is cast to Float64 to match window starts.
    """
    lf = pl.scan_parquet(PERCH_SOUNDSCAPE_EMBEDDINGS)
    if filenames is not None:
        lf = lf.filter(pl.col("filename").is_in(list(filenames)))
    df = lf.collect()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df = df.with_columns(pl.col("start_sec").cast(pl.Float64))
    return df, emb_cols


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def build_prototype(emb: np.ndarray, present: np.ndarray, absent: np.ndarray) -> np.ndarray:
    """L2-normalized contrastive prototype = normalize(mean(present) - mean(absent))."""
    proto = emb[present].mean(axis=0) - emb[absent].mean(axis=0)
    nrm = float(np.linalg.norm(proto))
    return proto / nrm if nrm > 0 else proto


# --------------------------------------------------------------------------- #
# metrics (hand-rolled; no sklearn dependency)                                 #
# --------------------------------------------------------------------------- #
def average_precision(scores: np.ndarray, y: np.ndarray) -> float:
    """Step-wise AP matching sklearn.average_precision_score, with **tie grouping**
    (equal scores share one PR threshold, so a constant-score ranker scores exactly
    prevalence). NaN if no positives."""
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    yy = y[order].astype(float)
    tp = np.cumsum(yy)
    fp = np.cumsum(1.0 - yy)
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]  # last idx of each tied group
    precision = tp[ends] / (tp[ends] + fp[ends])
    recall = tp[ends] / yy.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def precision_at_k(scores: np.ndarray, y: np.ndarray, k: int = 20) -> float:
    order = np.argsort(-scores, kind="stable")
    kk = min(k, len(y))
    return float(y[order][:kk].sum() / kk) if kk else float("nan")


def fp_at_recall(scores: np.ndarray, y: np.ndarray, spec: np.ndarray, recall: float = RECALL_OP) -> float:
    """False-positive rate on the insect-free probe at the threshold that retains
    ``recall`` of the held-out-site positives. NaN if no positives / no probe."""
    pos = scores[y]
    sp = scores[spec]
    if len(pos) == 0 or len(sp) == 0:
        return float("nan")
    thr = float(np.quantile(pos, 1.0 - recall, method="lower"))
    return float((sp >= thr).mean())


# --------------------------------------------------------------------------- #
# Stage A: feasibility gate                                                    #
# --------------------------------------------------------------------------- #
def _load_gate_data():
    """Join labeled windows to their Perch embeddings; return aligned arrays.

    Returns both raw-normalized embeddings ``emb`` and per-recording-centered
    ``emb_c`` (each window minus its file's mean over all 12 chunks, then
    renormalized) — the centering strips the recording/site background offset
    that confounds cross-site transfer. Centering is label-free (no leakage).
    """
    _, wide = load_windows()
    fnames = set(wide["filename"].to_list())
    emb_df, emb_cols = load_embeddings(fnames)
    full = emb_df.select(emb_cols).to_numpy().astype(np.float32)
    ff = np.array(emb_df["filename"].to_list())
    file_mean = {f: full[ff == f].mean(axis=0) for f in np.unique(ff)}
    joined = wide.join(
        emb_df.select(["filename", "start_sec", *emb_cols]),
        on=["filename", "start_sec"],
        how="inner",
    )
    assert joined.height == wide.height, (
        f"embedding join lost rows: {joined.height} != {wide.height}"
    )
    raw = joined.select(emb_cols).to_numpy().astype(np.float32)
    centered = raw - np.array([file_mean[f] for f in joined["filename"].to_list()], dtype=np.float32)
    emb = _normalize_rows(raw)
    emb_c = _normalize_rows(centered)
    sites = np.array(joined["site"].to_list())
    wids = joined["window_id"].to_list()
    label_lists = joined["labels"].to_list()
    return wide, joined, emb, emb_c, sites, wids, label_lists


def _present_masks(label_lists: list, n: int) -> dict[str, np.ndarray]:
    """Canonical-label -> boolean present-mask over windows (merges co-occurring pairs)."""
    masks: dict[str, np.ndarray] = {}
    for i, labs in enumerate(label_lists):
        for raw in labs:
            c = canon(raw)
            masks.setdefault(c, np.zeros(n, dtype=bool))[i] = True
    return masks


def _band_energy_scores(
    label: str,
    train_wide: pl.DataFrame,
    spectra: dict,
    eval_idx: np.ndarray,
    wids: list[str],
) -> np.ndarray:
    """'Where energy sits' baseline: in-band mean of each eval window's spectrum,
    using a discriminative band built on TRAINING sites only (no leakage)."""
    train_spectra = {w: spectra[w] for w in train_wide["window_id"].to_list() if w in spectra}
    info = discriminative_band(label, train_wide, train_spectra)
    any_freqs = next(iter(spectra.values()))[1]
    if info["status"] == "ok" and info["bands"]:
        lo, hi = info["bands"][0]
    else:  # no discriminative band -> fall back to the full wideband mean
        lo, hi = float(any_freqs[0]), float(any_freqs[-1])
    out = np.zeros(len(eval_idx), dtype=np.float64)
    for j, i in enumerate(eval_idx):
        spec, freqs = spectra[wids[i]]
        m = (freqs >= lo) & (freqs <= hi)
        out[j] = float(spec[m].mean()) if np.any(m) else 0.0
    return out


def loso_gate() -> None:
    wide, _, emb, emb_c, sites, wids, label_lists = _load_gate_data()
    n = len(wids)
    isites = set(insect_sites())
    spec_mask = np.array([s not in isites for s in sites])  # insect-free probe (S22-dominated)
    present = _present_masks(label_lists, n)

    ins_labels = sorted({canon(s) for s in insect_sonotypes()})
    # cross-site-testable = present in >=2 insect sites AND clears the n>=8 gate
    testable = []
    for lab in ins_labels:
        if lab not in present:
            continue
        p = present[lab]
        p_sites = {s for s, ok in zip(sites, p) if ok and s in isites}
        if len(p_sites) >= 2 and int(p.sum()) >= MIN_PRESENT:
            testable.append(lab)
    logger.info("insect canonical labels=%d  cross-site-testable=%d %s",
                len(ins_labels), len(testable), testable)
    assert spec_mask.sum() > 0, "no insect-free probe windows found"

    # band-energy baseline needs per-window spectra (recompute; cache was removed)
    logger.info("computing per-window spectra for the band-energy baseline ...")
    _, spectra = analyze_labeled_windows(wide)

    rng_scores = {seed: np.random.default_rng(seed).random(n) for seed in range(N_RANDOM_SEEDS)}
    rows: list[dict] = []  # one per (sonotype, fold, ranker)
    debug: list[dict] = []

    for lab in testable:
        p = present[lab]
        p_sites = sorted({s for s, ok in zip(sites, p) if ok and s in isites})
        for h in p_sites:  # hold out each site where the sonotype is present
            train = np.array([(s in isites) and (s != h) for s in sites])
            present_train = train & p
            absent_train = train & ~p
            if present_train.sum() == 0 or absent_train.sum() == 0:
                continue
            eval_mask = (sites == h) | spec_mask
            eval_idx = np.where(eval_mask)[0]
            y = p[eval_idx]
            spec_e = spec_mask[eval_idx]
            assert y[spec_e].sum() == 0, "insect-free probe must contain no positives"
            if y.sum() == 0:
                continue

            # AP / P@20 measure WITHIN-SITE routing (held-out-site windows only, so the
            # huge insect-free probe can't hand a trivial site-detector free credit);
            # the probe is used only for the specificity FP rate.
            in_h = ~spec_e

            def _proto_scores(E):
                pr = build_prototype(E, present_train, absent_train)
                ct = _normalize_rows(E[present_train].mean(axis=0, keepdims=True))[0]
                return E[eval_idx] @ pr, E[eval_idx] @ ct

            proto_s, centroid_s = _proto_scores(emb)
            proto_cs, centroid_cs = _proto_scores(emb_c)  # per-recording centered
            ranker_scores = {
                "prototype": proto_s,                  # contrastive, raw
                "centroid": centroid_s,                # mean(present), raw
                "prototype_centered": proto_cs,        # contrastive, per-recording centered
                "centroid_centered": centroid_cs,      # mean(present), per-recording centered
                "band_energy": _band_energy_scores(
                    lab, wide.filter(pl.col("site").is_in([s for s in isites if s != h])),
                    spectra, eval_idx, wids,
                ),
                # trivial floor: constant within a site -> AP == within-site base rate
                "site_prior": np.full(len(eval_idx), 1.0),
            }
            for name, sc in ranker_scores.items():
                rows.append({
                    "sonotype": lab, "fold": h, "ranker": name,
                    "ap": average_precision(sc[in_h], y[in_h]),
                    "p_at_20": precision_at_k(sc[in_h], y[in_h]),
                    "probe_fp": fp_at_recall(sc, y, spec_e), "n_pos": int(y[in_h].sum()),
                    "n_eval": int(in_h.sum()),
                })
                if name == "prototype":
                    for jj, i in enumerate(eval_idx):
                        debug.append({"sonotype": lab, "fold": h, "window_id": wids[i],
                                      "site": sites[i], "score": float(sc[jj]),
                                      "is_present": bool(y[jj]), "is_probe": bool(spec_e[jj])})
            # random-within-site: average over seeds
            aps, pks, fps = [], [], []
            for seed in range(N_RANDOM_SEEDS):
                sc = rng_scores[seed][eval_idx]
                aps.append(average_precision(sc[in_h], y[in_h]))
                pks.append(precision_at_k(sc[in_h], y[in_h]))
                fps.append(fp_at_recall(sc, y, spec_e))
            rows.append({"sonotype": lab, "fold": h, "ranker": "random",
                         "ap": float(np.nanmean(aps)), "p_at_20": float(np.nanmean(pks)),
                         "probe_fp": float(np.nanmean(fps)), "n_pos": int(y[in_h].sum()),
                         "n_eval": int(in_h.sum())})

    per = pl.DataFrame(rows)
    out = artifact_dir("retrieve/gate")
    per.write_csv(out / "per_sonotype.csv")
    pl.DataFrame(debug).write_parquet(out / "scores_debug.parquet")

    # macro over (sonotype, fold) per ranker
    table = (
        per.group_by("ranker")
        .agg(pl.col("ap").mean().alias("within_site_AP"),
             pl.col("p_at_20").mean().alias("within_site_P@20"),
             pl.col("probe_fp").mean().alias("insect_free_FP@recall0.8"),
             pl.len().alias("n_folds"))
        .sort("within_site_AP", descending=True)
    )
    table.write_csv(out / "gate_table.csv")

    def _val(ranker, col):
        r = table.filter(pl.col("ranker") == ranker)
        return float(r[col][0]) if r.height else float("nan")

    # PASS iff ANY embedding ranker clears all three gates (beats every baseline on
    # within-site AP & P@20, and insect-free FP <= 0.05).
    emb_rankers = ("prototype", "centroid", "prototype_centered", "centroid_centered")
    baselines = ("band_energy", "site_prior", "random")

    def _passes(r):
        ap_pk = all(
            _val(r, "within_site_AP") > _val(b, "within_site_AP")
            and _val(r, "within_site_P@20") > _val(b, "within_site_P@20")
            for b in baselines
        )
        return ap_pk and _val(r, "insect_free_FP@recall0.8") <= 0.05

    passers = [r for r in emb_rankers if _passes(r)]
    passed = bool(passers)
    best = max(passers or emb_rankers, key=lambda r: _val(r, "within_site_AP"))
    best_fp = _val(best, "insect_free_FP@recall0.8")
    beats = all(
        _val(best, "within_site_AP") > _val(b, "within_site_AP")
        and _val(best, "within_site_P@20") > _val(b, "within_site_P@20")
        for b in baselines
    )

    n_single = len(ins_labels) - len(testable)
    lines = [
        "# Stage A — Perch-prototype feasibility gate",
        "",
        f"**Verdict: {'PASS' if passed else 'FAIL'}** — best embedding ranker "
        f"(`{best}`) {'beats' if beats else 'does NOT beat'} every baseline on within-site "
        f"AP & P@20; its insect-free FP@recall{RECALL_OP} = {best_fp:.3f} "
        f"({'≤' if best_fp <= 0.05 else '>'} 0.05).",
        "",
        f"Leave-one-insect-site-out over the **{len(testable)}** testable sonotypes "
        f"{testable}, macro over (sonotype, held-out-site) folds. {n_single} canonical "
        "insect labels are single-site (cross-site-untestable), excluded. **within_site_AP / "
        "P@20** rank only the held-out site's windows (pure routing, site confound removed); "
        "**insect_free_FP** is the false-positive rate on the insect-free probe (sites "
        f"outside S08/S15/S19/S23) at the threshold retaining recall {RECALL_OP} of held-out "
        "positives. Rankers: `prototype` (contrastive mean(present)-mean(absent)), `centroid` "
        "(mean(present) only), `band_energy` (discriminative band, trained on training sites "
        "only), `site_prior` (constant within a site = the within-site base-rate floor), "
        "`random`. The `*_centered` rankers subtract each recording's mean embedding "
        "(over all 12 chunks) before prototype/scoring, to strip the site/background offset.",
        "",
        "## Macro summary",
        "",
        _md_table(table),
        "",
    ]
    (out / "gate.md").write_text("\n".join(lines))

    # production prototypes (train on ALL insect sites) for Stage B
    prod = {}
    for lab in ins_labels:
        if lab not in present or int(present[lab].sum()) < MIN_PRESENT:
            continue
        p = present[lab]
        ins_mask = np.array([s in isites for s in sites])
        pr, ab = ins_mask & p, ins_mask & ~p
        if pr.sum() and ab.sum():
            prod[lab] = build_prototype(emb, pr, ab)
    np.savez(artifact_dir("retrieve") / "prototypes.npz", **prod)

    logger.info("wrote %s", out / "gate.md")
    print(table)
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}  (best ranker=`{best}` beats "
          f"baselines={beats}, insect-free FP@{RECALL_OP}={best_fp:.3f})")


def _md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in row
        ) + " |"
        for row in df.iter_rows()
    ]
    return "\n".join([head, sep, *body])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate", help="Stage A: leave-one-insect-site-out feasibility gate")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "gate":
        loso_gate()


if __name__ == "__main__":
    main()
