"""Exp 037 — leakage-free labeled-soundscape macro-AUC (Gate A, rough screen only).

Reuses the exp 035 `analysis/data/cv_trust/` scaffold (739 expert-labeled windows over 66
files, gt_Y for 75 classes-with-positives, per-file OOF fold maps). Computes the competition
metric (macro ROC-AUC, skipping classes with no positives) for:
  - reference bars from cached OOF parquets (distilled baseline, v5j, perch, probe)
  - the 0.929 ENSEMBLE teacher blend on labeled files (context; partly leaked via Perch probe)
  - any student/teacher predictions parquet passed via --pred

**Gate A bar = the BASELINE distilled-sed-kd OOF (V1_full ~0.763), NOT the teacher.** The
teacher's labeled-SS AUC is leaked (its Perch probe was fit on these labels → ~0.995) so it is
not a fair bar. A round-1 PL student should beat the no-PL baseline's OOF here.

**CAVEAT (load-bearing):** labeled-SS OOF ANTI-correlates with LB even between our own models
(distilled 0.763 OOF -> 0.915 LB vs v5j 0.791 OOF -> 0.913 LB). Treat this as a sanity check
that the student isn't broken — NOT a precise ranker. **LB is the arbiter.**

Usage:
    python scripts/eval_labeled_ss.py                       # reference bars + teacher blend
    python scripts/eval_labeled_ss.py --pred <student_oof.parquet>   # + score a student
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026")
SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
CV = REPO / "user/murilo/analysis/data/cv_trust"
PERCH_PROBS = (
    SHARED / "inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet"
)
SED_DIR = SHARED / "inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes"
KEY = ["filename", "chunk_idx"]


def macro_auc_skip_empty(Y: np.ndarray, P: np.ndarray, cols=None) -> tuple[float, int]:
    """Competition metric: per-class ROC-AUC, mean over classes with >=1 positive (and not all-pos)."""
    aucs = []
    rng = range(Y.shape[1]) if cols is None else cols
    for c in rng:
        col = Y[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, P[:, c]))
        except ValueError:
            continue
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def align_preds(pred_df: pd.DataFrame, windows: pd.DataFrame, species: list[str]) -> np.ndarray:
    """Align a (filename, chunk_idx, +species) preds frame to the gt windows order."""
    pred_df = pred_df.copy()
    pred_df["chunk_idx"] = pred_df["chunk_idx"].astype(int)
    missing = [s for s in species if s not in pred_df.columns]
    if missing:
        raise ValueError(f"pred parquet missing {len(missing)} species cols (first: {missing[:3]})")
    merged = windows[KEY].merge(pred_df[KEY + species], on=KEY, how="left")
    P = merged[species].to_numpy(np.float32)
    if np.isnan(P).any():
        n_missing = int(np.isnan(P).any(axis=1).sum())
        raise ValueError(f"{n_missing}/{len(P)} gt windows missing from preds parquet")
    return P


def teacher_blend_on_labeled(
    windows: pd.DataFrame, species: list[str], w_perch: float = 0.5
) -> np.ndarray:
    """0.929 ensemble blend restricted to the 66 labeled files (context bar)."""
    perch = pd.read_parquet(PERCH_PROBS)
    perch["chunk_idx"] = perch["chunk_idx"].astype(int)
    Pp = align_preds(perch, windows, species)
    sed_sum = np.zeros_like(Pp)
    for fold in range(4):
        sed = pd.read_parquet(SED_DIR / f"fold{fold}" / "logits.parquet")
        sed["chunk_idx"] = sed["chunk_idx"].astype(int)
        sed_sum += _sigmoid(align_preds(sed, windows, species))
    Ps = sed_sum / 4.0
    return w_perch * Pp + (1.0 - w_perch) * Ps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Student/teacher preds parquet (filename, chunk_idx, +234).",
    )
    ap.add_argument(
        "--baseline", type=float, default=0.7632, help="Gate-A bar = baseline distilled-sed-kd OOF."
    )
    args = ap.parse_args()

    windows = pd.read_parquet(CV / "gt_windows.parquet").reset_index(drop=True)
    windows["chunk_idx"] = windows["chunk_idx"].astype(int)
    Y = np.load(CV / "gt_Y.npy")
    species = json.loads((CV / "species_cols.json").read_text())
    site = windows["site"].to_numpy()
    non_s22 = np.where(site != "S22")[0]
    print(
        f"scaffold: windows={len(windows)} files={windows['filename'].nunique()} "
        f"classes-with-pos={int((Y.sum(0) > 0).sum())}"
    )

    def report(name: str, P: np.ndarray):
        full, n = macro_auc_skip_empty(Y, P)
        # nonS22: restrict ROWS (S22 dominates 40/66 files) — recompute per-class on the subset
        ns_full, _ = macro_auc_skip_empty(Y[non_s22], P[non_s22])
        print(f"  {name:<34} macro-AUC(full)={full:.4f} ({n} cls) | nonS22={ns_full:.4f}")

    print("\n=== reference bars (cached OOF parquets) ===")
    for tag, fn in [
        ("distilled (BASELINE bar)", "oof_distilled.parquet"),
        ("v5j", "oof_v5j.parquet"),
        ("perch (LEAKED probe)", "oof_perch.parquet"),
        ("oof_probe (leakage-free)", "oof_probe.parquet"),
    ]:
        p = CV / fn
        if p.exists():
            report(tag, align_preds(pd.read_parquet(p), windows, species))

    print("\n=== 0.929 ensemble teacher blend on labeled files (context; Perch half is leaked) ===")
    report("teacher = 0.5*perch + 0.5*sed", teacher_blend_on_labeled(windows, species))

    if args.pred is not None:
        print(f"\n=== student preds: {args.pred} ===")
        P = align_preds(pd.read_parquet(args.pred), windows, species)
        full, n = macro_auc_skip_empty(Y, P)
        report(args.pred.stem, P)
        verdict = "PASS (> baseline)" if full > args.baseline else "below baseline"
        print(f"  Gate-A bar (baseline distilled OOF) = {args.baseline:.4f}  ->  {verdict}")
        print("  NOTE: labeled-SS OOF anti-correlates with LB — rough screen only. LB decides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
