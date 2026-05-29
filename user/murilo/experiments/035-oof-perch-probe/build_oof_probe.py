#!/usr/bin/env python3
"""Phase 1 — leakage-free OOF Perch probe + honest macro-AUC (claim 1).

The deployed Perch probe (`perch_234_probs`, exp 018) is fit on ALL labeled-soundscape
labels and evaluated on the same windows → 0.995 is leakage (see
analysis/notes/2026-05-29-cv-trust-and-failure-localization.md). This rebuilds the probe
with proper per-fold OOF (fit scaler/PCA/probes on train folds, predict held-out fold),
using the SAME by-file fold map as the CV-trust SED OOF so probe and SED OOF align
window-for-window (needed for honest ensemble tuning later).

Inputs (all leakage-free, on disk):
  - analysis/data/cv_trust/{gt_windows.parquet, gt_Y.npy, species_cols.json, oof_fold_maps.json}
  - perch-v2-tf/train_soundscapes/perch_234_raw_probs.parquet  (taxonomic base map; insects=0)
  - perch-v2-tf/train_soundscapes/embeddings.parquet           (1536-d Perch embeddings)

Recipe (exp 018 / v18 lineage; Phase-2 tunable): StandardScaler + PCA(128) per fold;
per-species LogReg on [PCA, base, prev, next, file_mean, file_max]; alpha-blend with base.

Outputs: analysis/data/cv_trust/oof_probe.parquet + printed honest metrics + leak-check.
Run: ~/clef/birdclef-2026/.venv/bin/python user/murilo/experiments/035-oof-perch-probe/build_oof_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026")
PERCH = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes"
)
CV = REPO / "user" / "murilo" / "analysis" / "data" / "cv_trust"

PCA_DIM = 128
MIN_POS = 8  # exp 018 min_positive_samples
ALPHA = 0.45  # exp 018 / v18 base/probe blend; Phase-2 tunable
LOGREG_C = 1.0


def macro_auc(Y: np.ndarray, P: np.ndarray, cols=None) -> tuple[float, int]:
    aucs = []
    for c in range(Y.shape[1]) if cols is None else cols:
        col = Y[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, P[:, c]))
        except ValueError:
            continue
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def species_context(base_c: np.ndarray, files: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    """prev / next / file_mean / file_max of the base score for one species (by file, chunk order)."""
    n = len(base_c)
    prev = np.zeros(n, np.float32)
    nxt = np.zeros(n, np.float32)
    fmean = np.zeros(n, np.float32)
    fmax = np.zeros(n, np.float32)
    for f in np.unique(files):
        idx = np.where(files == f)[0]
        idx = idx[np.argsort(chunks[idx])]
        s = base_c[idx]
        fmean[idx] = s.mean()
        fmax[idx] = s.max()
        for j, i in enumerate(idx):
            prev[i] = s[j - 1] if j > 0 else 0.0
            nxt[i] = s[j + 1] if j < len(idx) - 1 else 0.0
    return np.column_stack([prev, nxt, fmean, fmax])


def main() -> None:
    # ---- load trusted eval scaffold ----
    windows = pd.read_parquet(CV / "gt_windows.parquet").reset_index(drop=True)
    Y = np.load(CV / "gt_Y.npy")
    species = json.loads((CV / "species_cols.json").read_text())
    fold_map = json.loads((CV / "oof_fold_maps.json").read_text())["distilled"]
    windows["fold"] = windows["filename"].map(fold_map).astype(int)
    files = windows["filename"].to_numpy()
    chunks = windows["chunk_idx"].to_numpy()
    folds = windows["fold"].to_numpy()
    n_folds = int(folds.max()) + 1
    print(f"windows={len(windows)} folds={n_folds} species={len(species)}")

    # ---- load leakage-free base map + embeddings, aligned to windows ----
    key = windows[["filename", "chunk_idx"]].copy()
    base_df = pd.read_parquet(PERCH / "perch_234_raw_probs.parquet")
    base_df["chunk_idx"] = base_df["chunk_idx"].astype(int)
    B = key.merge(base_df, on=["filename", "chunk_idx"], how="left")[species].to_numpy(np.float32)
    assert not np.isnan(B).any(), "base map missing windows"

    emb_df = pd.read_parquet(PERCH / "embeddings.parquet")
    emb_df["chunk_idx"] = emb_df["chunk_idx"].astype(int)
    emb_cols = [c for c in emb_df.columns if str(c).startswith("emb_")]
    E = key.merge(
        emb_df[["filename", "chunk_idx"] + emb_cols], on=["filename", "chunk_idx"], how="left"
    )[emb_cols].to_numpy(np.float32)
    assert not np.isnan(E).any(), "embeddings missing windows"
    print(f"base {B.shape} emb {E.shape}")

    def fit_predict(train_mask: np.ndarray, pred_mask: np.ndarray, out: np.ndarray) -> int:
        """Fit scaler+PCA+per-species LogReg on train_mask, write alpha-blended preds into out[pred_mask]."""
        scaler = StandardScaler().fit(E[train_mask])
        pca = PCA(n_components=PCA_DIM, random_state=42).fit(scaler.transform(E[train_mask]))
        pemb_tr = pca.transform(scaler.transform(E[train_mask]))
        pemb_pr = pca.transform(scaler.transform(E[pred_mask]))
        n_probes = 0
        for c in range(len(species)):
            ytr = Y[train_mask, c]
            if ytr.sum() < MIN_POS:
                continue  # base fallback (out already = base)
            ctx = species_context(B[:, c], files, chunks)
            Xtr = np.column_stack([pemb_tr, B[train_mask, c], ctx[train_mask]])
            Xpr = np.column_stack([pemb_pr, B[pred_mask, c], ctx[pred_mask]])
            lr = LogisticRegression(
                C=LOGREG_C, class_weight="balanced", max_iter=1000, random_state=42
            )
            lr.fit(Xtr, ytr)
            probe = lr.predict_proba(Xpr)[:, 1]
            out[pred_mask, c] = (1 - ALPHA) * B[pred_mask, c] + ALPHA * probe
            n_probes += 1
        return n_probes

    # ---- OOF: per fold, fit on others, predict held-out ----
    oof = B.copy()  # base fallback everywhere; probes overwrite where trained
    for k in range(n_folds):
        tr, pr = folds != k, folds == k
        np_k = fit_predict(tr, pr, oof)
        print(f"  fold {k}: train={tr.sum()} pred={pr.sum()} probes={np_k}")

    # ---- leaked variant (fit on ALL, predict ALL) — the leakage the deployed probe has ----
    leaked = B.copy()
    all_mask = np.ones(len(windows), bool)
    fit_predict(all_mask, all_mask, leaked)

    # ---- metrics ----
    a_base, _ = macro_auc(Y, B)
    a_oof, n_oof = macro_auc(Y, oof)
    a_leak, _ = macro_auc(Y, leaked)
    print("\n=== honest macro-AUC (739 windows, skip-rule) ===")
    print(f"  base-only (leakage-free floor) : {a_base:.4f}")
    print(f"  OOF probe (leakage-free)       : {a_oof:.4f}  ({n_oof} eval classes)")
    print(
        f"  leaked probe (fit-all/pred-all): {a_leak:.4f}   <- should ~reproduce the 0.995 inflation"
    )
    print(
        "  reference: SED OOF distilled=0.763 v5j=0.791 ; deployed perch_234_probs=0.995 (leaked)"
    )
    flag = "OK (OOF << leaked → leakage-free)" if a_oof < a_leak - 0.02 else "!! WARN: OOF ~ leaked"
    print(f"  leak-check: {flag}")
    print(f"  probe vs base (honest lift): {a_oof - a_base:+.4f}")

    # soundscape-only species (insects/amphibians, base=0)
    so = [c for c in range(len(species)) if Y[:, c].sum() > 0 and B[:, c].max() == 0.0]
    if so:
        sa_base, _ = macro_auc(Y, B, so)
        sa_oof, _ = macro_auc(Y, oof, so)
        print(
            f"\n  soundscape-only species (n={len(so)}, base=0): base={sa_base:.4f} OOF-probe={sa_oof:.4f}"
        )

    # ---- save ----
    out_df = windows[["filename", "chunk_idx", "start_sec", "site", "fold"]].copy()
    out_df = pd.concat([out_df, pd.DataFrame(oof, columns=species)], axis=1)
    out_df.to_parquet(CV / "oof_probe.parquet")
    print(f"\nWrote {CV / 'oof_probe.parquet'}")


if __name__ == "__main__":
    main()
