"""Propagate insect signatures + validate on held-out labeled windows.

Structural fact that drives the protocol: insect labels occur in only **4 sites**
(S08, S15, S19, S23) — *zero* in S22 (the dominant site). And only 9 of 25
sonotypes span >=2 sites, so cross-site transfer is only testable for those.

Honest protocol:
- **Leave-one-insect-site-out (LOSO)**: for each insect site held out, build
  signatures + calibrate thresholds on the other insect sites, evaluate on the
  held-out site. Pooled across folds -> the real cross-site number (only
  multi-site sonotypes get one).
- **in_site ceiling**: full-insect-site signatures evaluated on their own fit
  windows (optimistic).
- **S22 specificity**: do the (insect) signatures wrongly fire on S22, which has
  no insects at all? Pure false-positive test.

Scoring a window against a sonotype's signature uses two cheap, label-free
quantities in the signature's band: the band's elevation over the window
background (``band_ratio``) and the cosine between the window's in-band spectral
shape and the signature shape. Thresholds are calibrated, never hand-tuned.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import polars as pl

from birdclef_201.boxes import load_window_spectra
from birdclef_201.labels import insect_sites, load_windows, site_of
from birdclef_201.paths import TRAIN_SOUNDSCAPES_DIR, artifact_dir
from birdclef_201.signature import InsectSignature, _l2, build_signatures, save_signatures
from birdclef_201.spectrogram import (
    MEL_N_FFT,
    MEL_SAMPLE_RATE,
    read_soundscape_window,
    wideband_pcen_spectrogram,
)
from birdclef_201.band import per_recording_mean_spectrum

logger = logging.getLogger(__name__)

RATIO_GRID_Q = (0.5, 0.7, 0.8, 0.9)
COS_GRID = (0.5, 0.6, 0.7, 0.8)


def _score(spectrum: np.ndarray, freqs: np.ndarray, sig: InsectSignature) -> tuple[float, float]:
    mask = sig.band_mask(freqs)
    if not np.any(mask):
        return 0.0, 0.0
    band = spectrum[mask]
    band_ratio = float(np.mean(band))
    cosine = float(np.dot(_l2(np.clip(band, 0.0, None)), sig.band_spectrum))
    return band_ratio, cosine


_SCORE_SCHEMA = {
    "label": pl.Utf8, "window_id": pl.Utf8, "site": pl.Utf8,
    "band_ratio": pl.Float64, "cosine": pl.Float64, "present": pl.Boolean,
}


def score_frame(
    sigs: dict[str, InsectSignature],
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    wide: pl.DataFrame,
) -> pl.DataFrame:
    """Long frame: (label, window_id, site, band_ratio, cosine, present) over all windows."""
    win = {w["window_id"]: w for w in wide.to_dicts()}
    rows: list[dict] = []
    for wid, (spec, freqs) in spectra.items():
        w = win.get(wid)
        if w is None:
            continue
        labset = set(w["labels"])
        for label, sig in sigs.items():
            br, cos = _score(spec, freqs, sig)
            rows.append(
                {"label": label, "window_id": wid, "site": w["site"],
                 "band_ratio": br, "cosine": cos, "present": label in labset}
            )
    return pl.DataFrame(rows, schema=_SCORE_SCHEMA) if rows else pl.DataFrame(schema=_SCORE_SCHEMA)


def calibrate(score_df: pl.DataFrame, train_ids: set[str]) -> dict[str, dict]:
    tr = score_df.filter(pl.col("window_id").is_in(list(train_ids)))
    out: dict[str, dict] = {}
    for label, g in tr.group_by("label"):
        lab = label[0] if isinstance(label, tuple) else label
        n_present = g.filter("present").height
        if n_present == 0:
            continue
        cand_r = [float(np.quantile(g["band_ratio"].to_numpy(), q)) for q in RATIO_GRID_Q]
        best = (-1.0, cand_r[0], COS_GRID[0])
        for rthr in cand_r:
            for cthr in COS_GRID:
                fire = g.filter((pl.col("band_ratio") >= rthr) & (pl.col("cosine") >= cthr))
                tp = fire.filter("present").height
                fp = fire.filter(~pl.col("present")).height
                fn = n_present - tp
                denom = 2 * tp + fp + fn
                f1 = (2 * tp / denom) if denom > 0 else 0.0
                if f1 > best[0]:
                    best = (f1, rthr, cthr)
        out[lab] = {"f1_train": round(best[0], 3), "ratio_thr": round(best[1], 4),
                    "cos_thr": best[2], "n_present_train": n_present}
    return out


def evaluate(score_df: pl.DataFrame, thresholds: dict[str, dict], eval_ids: set[str], tag: str) -> pl.DataFrame:
    ev = score_df.filter(pl.col("window_id").is_in(list(eval_ids)))
    rows: list[dict] = []
    for label, th in thresholds.items():
        g = ev.filter(pl.col("label") == label)
        if g.height == 0:
            continue
        fire = g.filter((pl.col("band_ratio") >= th["ratio_thr"]) & (pl.col("cosine") >= th["cos_thr"]))
        tp = fire.filter("present").height
        fp = fire.filter(~pl.col("present")).height
        n_present = g.filter("present").height
        fn = n_present - tp
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        rows.append(
            {"split": tag, "label": label, "n_present_eval": n_present,
             "tp": tp, "fp": fp, "fn": fn,
             "precision": round(prec, 3) if prec is not None else None,
             "recall": round(rec, 3) if rec is not None else None}
        )
    return pl.DataFrame(rows)


def _markdown_table(df: pl.DataFrame) -> str:
    cols = df.columns
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |"
        for row in df.iter_rows()
    ]
    return "\n".join([head, sep, *body])


def _pooled(rows: pl.DataFrame, split: str) -> pl.DataFrame:
    """Pool tp/fp/fn per sonotype across folds -> precision/recall for ``split``."""
    if rows.is_empty():
        return rows
    g = (
        rows.group_by("label")
        .agg(pl.col("tp").sum(), pl.col("fp").sum(), pl.col("fn").sum(),
             pl.col("n_present_eval").sum())
        .with_columns(
            split=pl.lit(split),
            precision=pl.when(pl.col("tp") + pl.col("fp") > 0)
            .then((pl.col("tp") / (pl.col("tp") + pl.col("fp"))).round(3))
            .otherwise(None),
            recall=pl.when(pl.col("tp") + pl.col("fn") > 0)
            .then((pl.col("tp") / (pl.col("tp") + pl.col("fn"))).round(3))
            .otherwise(None),
        )
    )
    return g.select("split", "label", "n_present_eval", "tp", "fp", "fn", "precision", "recall")


def _run_validate() -> None:
    spectra, wide, _ = load_window_spectra()
    sites = list(insect_sites())
    insect_ids = set(wide.filter(pl.col("site").is_in(sites))["window_id"].to_list())
    # The insect-free dominant site(s) = everything labeled that has no insects;
    # used as a pure false-positive (specificity) probe. S22 in the current data.
    spec_ids = set(wide.filter(~pl.col("site").is_in(sites))["window_id"].to_list())
    logger.info("insect sites=%s (%d windows)  specificity windows=%d",
                sites, len(insect_ids), len(spec_ids))
    out = artifact_dir("propagate")

    # --- LOSO cross-site over the insect sites ------------------------------ #
    fold_rows = []
    for test_site in sites:
        train_ids = {w for w in insect_ids if w not in set(
            wide.filter(pl.col("site") == test_site)["window_id"].to_list()
        )}
        test_ids = set(wide.filter(pl.col("site") == test_site)["window_id"].to_list())
        sigs = build_signatures(window_subset=train_ids)
        if not sigs:
            continue
        score_df = score_frame(sigs, spectra, wide)
        thresholds = calibrate(score_df, train_ids)
        # Keep ALL evaluated labels (not just those present in the held-out
        # site): a signature firing in a site where its sonotype is absent is a
        # real cross-site false positive and must count against precision.
        fold_rows.append(evaluate(score_df, thresholds, test_ids, f"fold_{test_site}"))
    loso = pl.concat(fold_rows) if fold_rows else pl.DataFrame()
    cross = _pooled(loso, "cross_site(LOSO)")
    # drop labels that were never present anywhere and never fired (no signal)
    if not cross.is_empty():
        cross = cross.filter((pl.col("n_present_eval") > 0) | (pl.col("tp") + pl.col("fp") > 0))

    # --- full-insect-site signatures: in-site ceiling + S22 specificity ----- #
    full_sigs = build_signatures(window_subset=insect_ids)
    save_signatures(full_sigs, "signatures_insect")
    full_scores = score_frame(full_sigs, spectra, wide)
    full_th = calibrate(full_scores, insect_ids)
    pl.DataFrame([{"label": k, **v} for k, v in full_th.items()]).sort("label").write_csv(
        out / "thresholds.csv"
    )
    in_site = evaluate(full_scores, full_th, insect_ids, "in_site(fit ceiling)")

    # Specificity: every fire on an insect-free site is a false positive.
    spec_rows = []
    spec_scores = full_scores.filter(pl.col("window_id").is_in(list(spec_ids)))
    for label, th in full_th.items():
        g = spec_scores.filter(pl.col("label") == label)
        fires = g.filter((pl.col("band_ratio") >= th["ratio_thr"]) & (pl.col("cosine") >= th["cos_thr"])).height
        spec_rows.append({"label": label, "insect_free_windows": g.height, "false_fires": fires,
                          "fp_rate": round(fires / g.height, 3) if g.height else None})
    spec = pl.DataFrame(spec_rows).sort("fp_rate", descending=True)

    res = pl.concat([in_site.select(cross.columns) if not in_site.is_empty() else in_site, cross]) \
        if not cross.is_empty() else in_site
    res.write_csv(out / "validation.csv")
    spec.write_csv(out / "specificity.csv")

    def _macro(df, split):
        prec = df.filter(pl.col("precision").is_not_null())["precision"]
        rec = df.filter(pl.col("recall").is_not_null())["recall"]
        return {"split": split, "n_fired": prec.len(), "n_recall_testable": rec.len(),
                "macro_precision": round(float(prec.mean()), 3) if prec.len() else None,
                "macro_recall": round(float(rec.mean()), 3) if rec.len() else None}

    summ = pl.DataFrame([_macro(in_site, "in_site(fit ceiling)"), _macro(cross, "cross_site(LOSO)")])

    lines = [
        "# Propagation validation (insects v1)",
        "",
        "Insect labels occur only in S08/S15/S19/S23 (none in S22). cross_site = "
        "leave-one-insect-site-out, pooled; only the 9 multi-site sonotypes are testable. "
        "in_site is the full-insect-site fit set (optimistic ceiling).",
        "",
        "## Macro summary", "", _markdown_table(summ), "",
        "## Specificity (insect signatures vs insect-free sites e.g. S22; fires = false positives)",
        "", _markdown_table(spec), "",
        "## Per-sonotype", "", _markdown_table(res.sort(["split", "label"])), "",
    ]
    (out / "validation.md").write_text("\n".join(lines))
    logger.info("wrote %s", out / "validation.md")
    print(summ)
    print(cross.sort("label"))
    print(spec)


def _pool_files(n: int, seed: int = 0) -> list[str]:
    import random

    labeled = set(load_windows()[1]["filename"].to_list())
    allf = sorted(p.name for p in TRAIN_SOUNDSCAPES_DIR.glob("*.ogg"))
    pool = [f for f in allf if f not in labeled]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def _run_pool(n_files: int) -> None:
    """Yield report: scan a sample of unlabeled recordings with S22 signatures."""
    import json
    from pathlib import Path

    sig_path = artifact_dir("signatures") / "signatures_insect.npz"
    th_path = artifact_dir("propagate") / "thresholds.csv"
    if not sig_path.exists() or not Path(th_path).exists():
        raise SystemExit("run `python -m birdclef_201.propagate` (validate) first")
    npz = np.load(sig_path)
    th = {r["label"]: r for r in pl.read_csv(th_path).to_dicts()}
    sigs = {
        lab: InsectSignature(lab, float(npz[f"{lab}::band"][0]), float(npz[f"{lab}::band"][1]), npz[lab], 0)
        for lab in th
        if lab in npz.files
    }

    files = _pool_files(n_files)
    logger.info("scanning %d unlabeled recordings with %d signatures", len(files), len(sigs))
    fired: dict[str, dict] = {lab: {"windows": 0, "files": set(), "sites": set()} for lab in sigs}
    n_windows = 0
    for fi, fn in enumerate(files):
        try:
            audio, _, _ = read_soundscape_window(fn, 0.0, None)  # whole file to EOF
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s: %s", fn, e)
            continue
        if audio.size < MEL_N_FFT:
            continue
        # PCEN the whole file once, then slice per 5s window (vs one STFT/window).
        _, pcen, freqs, times = wideband_pcen_spectrogram(audio)
        total = audio.size / MEL_SAMPLE_RATE
        site = site_of(fn)
        for w0 in np.arange(0.0, total, 5.0):
            cols = (times >= w0) & (times < w0 + 5.0)
            if int(cols.sum()) < 2:
                continue
            spec = per_recording_mean_spectrum(pcen[:, cols])
            n_windows += 1
            for lab, sig in sigs.items():
                br, cos = _score(spec, freqs, sig)
                if br >= float(th[lab]["ratio_thr"]) and cos >= float(th[lab]["cos_thr"]):
                    fired[lab]["windows"] += 1
                    fired[lab]["files"].add(fn)
                    fired[lab]["sites"].add(site)
        if (fi + 1) % 25 == 0:
            logger.info("  scanned %d/%d files (%d windows)", fi + 1, len(files), n_windows)

    rows = [
        {"label": lab, "windows_fired": d["windows"], "n_files": len(d["files"]),
         "n_sites": len(d["sites"]),
         "frac_windows": round(d["windows"] / n_windows, 4) if n_windows else 0.0,
         "background_flag": (d["windows"] / n_windows > 0.20) if n_windows else False}
        for lab, d in sorted(fired.items())
    ]
    out = artifact_dir("propagate")
    yld = pl.DataFrame(rows).sort("windows_fired", descending=True)
    yld.write_csv(out / "pool_yield.csv")
    (out / "pool_meta.json").write_text(json.dumps({"n_files": len(files), "n_windows": n_windows}, indent=2))
    logger.info("scanned %d windows; wrote %s", n_windows, out / "pool_yield.csv")
    print(yld)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", action="store_true", help="yield report on a sample of unlabeled recordings")
    p.add_argument("--pool-files", type=int, default=50)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.pool:
        _run_pool(args.pool_files)
    else:
        _run_validate()


if __name__ == "__main__":
    main()
