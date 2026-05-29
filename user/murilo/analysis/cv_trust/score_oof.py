#!/usr/bin/env python3
"""Phase 1b — score the reconstructed OOF (build_oof.py) with the competition
skip-rule macro-AUC under several variants, and test whether a full-corpus OOF
metric un-breaks the LB ranking that the single-fold metric got backwards.

Variants:
  V0  single-fold (the CURRENT broken metric) — one fold's held-out windows only
  V1  full-corpus, all 234 classes, all 739 windows  (closest LB analog)
  V2  75 present species only
  V3  non-S22 windows only (drop the 40-file S22 site)
  V4  >=5-positive classes only (drop tiny-sample AUCs)
  V5  per-site macro (average the per-site V1 scores)

Outputs: cv_scores.csv, cv_lb_correlation.csv, bootstrap_ci.csv
Run: ~/clef/birdclef-2026/.venv/bin/python user/murilo/analysis/cv_trust/score_oof.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

OUT = Path(
    "/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/analysis/data/cv_trust"
)
MODELS = ["v5j", "distilled", "perch"]

# LB anchors (verified this session). Perch anchor is APPROX — perch_234_probs is
# raw Perch mapped to 234, whose clean solo LB we don't have; the Perch-probe family
# is ~0.911 and the Perch+SED ensemble is 0.929. Flagged; do not let it dominate.
LB = {"v5j": 0.913, "distilled": 0.915, "perch": 0.911}
LB_NOTE = {"v5j": "v5j-4fold raw", "distilled": "distilled-5fold", "perch": "~probe (approx)"}


def macro_auc(Y: np.ndarray, P: np.ndarray, cols: np.ndarray | None = None) -> tuple[float, int]:
    idx = range(Y.shape[1]) if cols is None else cols
    aucs = []
    for c in idx:
        col = Y[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, P[:, c]))
        except ValueError:
            continue
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def main() -> None:
    Y = np.load(OUT / "gt_Y.npy")
    windows = pd.read_parquet(OUT / "gt_windows.parquet")
    species = json.loads((OUT / "species_cols.json").read_text())
    fold_maps = json.loads((OUT / "oof_fold_maps.json").read_text())
    P = {m: pd.read_parquet(OUT / f"oof_{m}.parquet")[species].to_numpy(np.float32) for m in MODELS}

    site = windows["site"].to_numpy()
    is_s22 = site == "S22"
    present = np.where(Y.sum(0) > 0)[0]
    ge5 = np.where(Y.sum(0) >= 5)[0]
    print(
        f"windows={len(Y)} present_species={len(present)} ge5_species={len(ge5)} "
        f"S22_windows={is_s22.sum()} sites={sorted(set(site))}"
    )

    rows = []
    for m in MODELS:
        Pm = P[m]
        # V0 single-fold: one fold's held-out windows (the broken metric). Use fold 0
        # for distilled (14 files); for v5j fold 0 holds 55 files. perch has no folds → skip.
        v0 = float("nan")
        if m in fold_maps:
            f0_files = [f for f, fd in fold_maps[m].items() if fd == 0]
            mask = windows["filename"].isin(f0_files).to_numpy()
            v0, _ = macro_auc(Y[mask], Pm[mask])
        v1, n1 = macro_auc(Y, Pm)
        v2, _ = macro_auc(Y, Pm, present)
        v3, _ = macro_auc(Y[~is_s22], Pm[~is_s22])
        v4, _ = macro_auc(Y, Pm, ge5)
        # V5 per-site macro
        site_scores = []
        for s in sorted(set(site)):
            sm = site == s
            sc, nc = macro_auc(Y[sm], Pm[sm])
            if not np.isnan(sc):
                site_scores.append(sc)
        v5 = float(np.mean(site_scores)) if site_scores else float("nan")
        rows.append(
            {
                "model": m,
                "lb": LB[m],
                "lb_note": LB_NOTE[m],
                "V0_singlefold": v0,
                "V1_full": v1,
                "V1_nclasses": n1,
                "V2_present": v2,
                "V3_nonS22": v3,
                "V4_ge5": v4,
                "V5_persite": v5,
            }
        )
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "cv_scores.csv", index=False)
    print("\n=== cv_scores ===")
    print(scores.to_string(index=False))

    # ---- CV/LB rank correlation per variant (n=3; sign-flip test, not calibration) ----
    lb_vec = np.array([LB[m] for m in MODELS])
    corr_rows = []
    for v in ["V0_singlefold", "V1_full", "V2_present", "V3_nonS22", "V4_ge5", "V5_persite"]:
        vals = scores.set_index("model").loc[MODELS, v].to_numpy()
        ok = ~np.isnan(vals)
        if ok.sum() < 3:
            corr_rows.append(
                {
                    "variant": v,
                    "spearman": float("nan"),
                    "kendall": float("nan"),
                    "n": int(ok.sum()),
                }
            )
            continue
        sp = spearmanr(vals[ok], lb_vec[ok]).statistic
        kt = kendalltau(vals[ok], lb_vec[ok]).statistic
        corr_rows.append(
            {
                "variant": v,
                "spearman": round(float(sp), 3),
                "kendall": round(float(kt), 3),
                "n": int(ok.sum()),
            }
        )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "cv_lb_correlation.csv", index=False)
    print("\n=== cv_lb_correlation (n=3, sign-flip test) ===")
    print(corr.to_string(index=False))

    # ---- bootstrap 95% CI for V1 (resample files w/ replacement) ----
    rng = np.random.default_rng(42)
    files = windows["filename"].to_numpy()
    uniq = np.unique(files)
    boot = {m: [] for m in MODELS}
    for _ in range(1000):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ridx = np.concatenate([np.where(files == f)[0] for f in samp])
        Yb = Y[ridx]
        for m in MODELS:
            a, _ = macro_auc(Yb, P[m][ridx])
            boot[m].append(a)
    ci_rows = []
    for m in MODELS:
        arr = np.array(boot[m])
        ci_rows.append(
            {
                "model": m,
                "V1_mean": round(float(arr.mean()), 4),
                "ci_lo": round(float(np.percentile(arr, 2.5)), 4),
                "ci_hi": round(float(np.percentile(arr, 97.5)), 4),
            }
        )
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "bootstrap_ci.csv", index=False)
    print("\n=== bootstrap 95% CI (V1, file-resampled, 1000x) ===")
    print(ci.to_string(index=False))
    # do v5j and distilled CIs overlap?
    lo_d, hi_v = (
        ci.set_index("model").loc["distilled", "ci_lo"],
        ci.set_index("model").loc["v5j", "ci_hi"],
    )
    print(
        f"\nv5j vs distilled resolvable? v5j_hi={hi_v} distilled_lo={lo_d} -> "
        f"{'OVERLAP (not resolvable)' if lo_d <= hi_v else 'separated'}"
    )
    print(f"\nWrote cv_scores.csv, cv_lb_correlation.csv, bootstrap_ci.csv to {OUT}")


if __name__ == "__main__":
    main()
