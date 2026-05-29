#!/usr/bin/env python3
"""Phase 2 — failure-diagnosis battery on the trustworthy OOF (build_oof.py).

Localizes where the SED->Perch (0.915->0.929) gap lives, using only on-disk OOF
predictions + GT (no model loading, no GPU):

  1. Per-class AUC on the 75 present species (v5j / distilled / perch + n_pos)
  2. SED-vs-Perch divergence: per-class (perch - distilled) ranked; by taxonomic class
  3. Long-tail correlation: per-class AUC vs focal training-recording count (Spearman)
  4. Per-site breakdown: macro-AUC per site for each model

Outputs: per_class_auc.csv, per_site_auc.csv  (+ printed summaries)
Run: ~/clef/birdclef-2026/.venv/bin/python user/murilo/analysis/cv_trust/diagnose_oof.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DATA = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
OUT = Path(
    "/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026/user/murilo/analysis/data/cv_trust"
)
MODELS = ["v5j", "distilled", "perch"]


def per_class_auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def macro(Y, P, mask_rows=None):
    if mask_rows is not None:
        Y, P = Y[mask_rows], P[mask_rows]
    aucs = [per_class_auc(Y[:, c], P[:, c]) for c in range(Y.shape[1])]
    aucs = [a for a in aucs if not np.isnan(a)]
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def main() -> None:
    Y = np.load(OUT / "gt_Y.npy")
    windows = pd.read_parquet(OUT / "gt_windows.parquet")
    species = json.loads((OUT / "species_cols.json").read_text())
    P = {m: pd.read_parquet(OUT / f"oof_{m}.parquet")[species].to_numpy(np.float32) for m in MODELS}
    tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")
    train_counts = pd.read_csv(DATA / "train.csv")["primary_label"].value_counts().to_dict()

    # ---- 1+2+3: per-class table over present species ----
    present = [c for c in range(234) if Y[:, c].sum() > 0]
    rows = []
    for c in present:
        sp = species[c]
        rows.append(
            {
                "species": sp,
                "common_name": tax.loc[sp, "common_name"] if sp in tax.index else "?",
                "class_name": tax.loc[sp, "class_name"] if sp in tax.index else "?",
                "n_pos": int(Y[:, c].sum()),
                "train_count": int(train_counts.get(sp, 0)),
                "auc_v5j": per_class_auc(Y[:, c], P["v5j"][:, c]),
                "auc_distilled": per_class_auc(Y[:, c], P["distilled"][:, c]),
                "auc_perch": per_class_auc(Y[:, c], P["perch"][:, c]),
            }
        )
    pc = pd.DataFrame(rows)
    pc["perch_minus_distilled"] = pc["auc_perch"] - pc["auc_distilled"]
    pc = pc.sort_values("perch_minus_distilled", ascending=False)
    pc.to_csv(OUT / "per_class_auc.csv", index=False)

    print(f"=== per-class AUC over {len(pc)} present species ===")
    print(
        f"macro: v5j={pc.auc_v5j.mean():.4f} distilled={pc.auc_distilled.mean():.4f} perch={pc.auc_perch.mean():.4f}"
    )
    print("\n--- top 12 species where Perch beats our distilled SED (the ensemble lift source) ---")
    print(
        pc.head(12)[
            [
                "species",
                "common_name",
                "class_name",
                "n_pos",
                "train_count",
                "auc_distilled",
                "auc_perch",
                "perch_minus_distilled",
            ]
        ].to_string(index=False)
    )
    print("\n--- 8 species where distilled already matches/beats Perch ---")
    print(
        pc.tail(8)[
            [
                "species",
                "common_name",
                "class_name",
                "n_pos",
                "train_count",
                "auc_distilled",
                "auc_perch",
                "perch_minus_distilled",
            ]
        ].to_string(index=False)
    )

    # taxonomic-class breakdown of the gap
    print("\n--- gap by taxonomic class (mean perch - distilled) ---")
    by_cls = (
        pc.groupby("class_name")
        .agg(
            n=("species", "size"),
            distilled=("auc_distilled", "mean"),
            perch=("auc_perch", "mean"),
            gap=("perch_minus_distilled", "mean"),
        )
        .round(4)
    )
    print(by_cls.to_string())

    # ---- 3: long-tail correlation (focal-trained species only; count>0) ----
    has_ct = pc[pc.train_count > 0]
    print(
        f"\n=== long-tail: AUC vs focal train_count (n={len(has_ct)} focal species; "
        f"{len(pc) - len(has_ct)} soundscape-only species have 0 focal recordings) ==="
    )
    for m in ["distilled", "perch"]:
        r = spearmanr(has_ct["train_count"], has_ct[f"auc_{m}"])
        print(f"  spearman(train_count, auc_{m}) = {r.statistic:.3f} (p={r.pvalue:.3f})")
    # bucket
    for lo, hi, lbl in [(0, 50, "<50"), (50, 150, "50-150"), (150, 10**9, ">=150")]:
        b = has_ct[(has_ct.train_count >= lo) & (has_ct.train_count < hi)]
        if len(b):
            print(
                f"  train_count {lbl:>7}: n={len(b):2d} distilled={b.auc_distilled.mean():.3f} perch={b.auc_perch.mean():.3f}"
            )

    # ---- 4: per-site breakdown ----
    site = windows["site"].to_numpy()
    srows = []
    for s in sorted(set(site)):
        sm = site == s
        rec = {"site": s, "n_windows": int(sm.sum())}
        for m in MODELS:
            a, n = macro(Y, P[m], sm)
            rec[f"auc_{m}"] = round(a, 4)
            rec["n_eval"] = n
        srows.append(rec)
    ps = pd.DataFrame(srows).sort_values("n_windows", ascending=False)
    ps.to_csv(OUT / "per_site_auc.csv", index=False)
    print("\n=== per-site macro-AUC ===")
    print(ps.to_string(index=False))
    print(f"\nWrote per_class_auc.csv, per_site_auc.csv to {OUT}")


if __name__ == "__main__":
    main()
