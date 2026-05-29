#!/usr/bin/env python3
"""Phase 1a — reconstruct TRUE out-of-fold (OOF) predictions over the labeled
soundscapes, so a competition-faithful macro-AUC can be computed (see score_oof.py).

Why this exists: the materialized fold parquets each contain predictions for ALL
10,658 train_soundscape files (each fold's checkpoint was inferenced over the whole
corpus), so they are NOT disjoint OOF slices. For an honest offline metric we must
pick, for each labeled window, the prediction from the fold whose checkpoint did NOT
train on that window's file (the fold where the file is held out).

Fold maps (verified against training code, 2026-05-29):
  - distilled (exp 032): train.py:490 uses `sorted(filenames)` + GroupKFold(5),
    deterministic. We replicate via build_sc_cache_meta + build_sc_folds.
  - v5j (exp 021): read directly from the on-disk fold_assignments.json
    (MultiLabelStratifiedGroupKFold seed=1086) — no risky replay. Heavily skewed
    (fold 0 = 55 of 66 labeled files), so a leak-check is printed.
  - Perch: frozen extractor, never trained on soundscapes → one prediction per window.

Outputs (user/murilo/analysis/data/cv_trust/):
  oof_{v5j,distilled,perch}.parquet, gt_Y.npy, gt_windows.parquet,
  species_cols.json, oof_fold_maps.json

Run: ~/clef/birdclef-2026/.venv/bin/python user/murilo/analysis/cv_trust/build_oof.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026")
SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA = SHARED / "data" / "birdclef-2026"
INF = SHARED / "inference" / "birdclef-2026"
OUT = REPO / "user" / "murilo" / "analysis" / "data" / "cv_trust"
V5J_FOLD_JSON = (
    REPO / "user" / "murilo" / "experiments" / "021-sed-training" / "fold_assignments.json"
)

sys.path.insert(0, str(REPO / "user" / "murilo" / "src"))  # for birdclef.* imports

MODELS = {
    "v5j": {"dir": INF / "hgnet-b0-sed-v5j-4fold", "n_folds": 4, "kind": "sed"},
    "distilled": {"dir": INF / "distilled-sed-kd-5fold", "n_folds": 5, "kind": "sed"},
    "perch": {"dir": INF / "perch-v2-tf", "n_folds": 0, "kind": "perch"},
}


def _macro_auc(Y: np.ndarray, P: np.ndarray) -> float:
    """Competition skip-rule macro-AUC (byte-identical to tucker_macro / vendored notebook)."""
    aucs = []
    for c in range(Y.shape[1]):
        col = Y[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, P[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- species columns (taxonomy order) ----
    tax = pd.read_csv(DATA / "taxonomy.csv")
    species = tax["primary_label"].astype(str).tolist()
    assert len(species) == 234, len(species)
    label2idx = {s: i for i, s in enumerate(species)}

    # ---- ground-truth windows + multi-hot Y ----
    gt = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    gt["start_sec"] = pd.to_timedelta(gt["start"]).dt.total_seconds().astype(int)
    gt["chunk_idx"] = (gt["start_sec"] // 5).astype(int)
    windows = gt[["filename", "chunk_idx", "start_sec"]].drop_duplicates().reset_index(drop=True)
    windows["site"] = windows["filename"].str.extract(r"_(S\d+)_")
    key2row = {
        (f, c): i for i, (f, c) in enumerate(zip(windows.filename, windows.chunk_idx, strict=False))
    }
    Y = np.zeros((len(windows), 234), np.float32)
    for _, r in gt.iterrows():
        i = key2row[(r["filename"], int(r["chunk_idx"]))]
        for sp in str(r["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in label2idx:
                Y[i, label2idx[sp]] = 1.0
    n_win = len(windows)
    labeled_files = sorted(gt["filename"].unique().tolist())
    print(
        f"GT: {n_win} windows, {len(labeled_files)} files, "
        f"{int((Y.sum(0) > 0).sum())} species present, "
        f"mean {Y.sum(1).mean():.2f} labels/window"
    )
    assert n_win == 739, n_win

    # ---- fold maps ----
    from birdclef.data.distilled_sed_datamodule import build_sc_cache_meta, build_sc_folds

    sc_df = build_sc_folds(build_sc_cache_meta(labeled_files), n_folds=5)
    distilled_fold = {f: int(v) for f, v in sc_df.groupby("filename")["fold"].first().items()}
    v5j_json = json.loads(V5J_FOLD_JSON.read_text())["soundscape"]
    v5j_fold = {f: int(v5j_json[f]) for f in labeled_files}
    fold_maps = {"v5j": v5j_fold, "distilled": distilled_fold}
    for name, fm in fold_maps.items():
        dist = pd.Series(list(fm.values())).value_counts().sort_index().to_dict()
        print(f"{name} fold sizes (files/fold): {dist}")

    # ---- assemble OOF per model ----
    def load_fold(model_dir: Path, k: int) -> pd.DataFrame:
        df = pd.read_parquet(model_dir / "train_soundscapes" / f"fold{k}" / "logits.parquet")
        keep = ["filename", "chunk_idx"] + species
        df = df[[c for c in keep if c in df.columns]].copy()
        df["chunk_idx"] = df["chunk_idx"].astype(int)
        return df

    def assemble(name: str, *, leaked: bool = False) -> np.ndarray:
        cfg = MODELS[name]
        fm = fold_maps[name]
        n_folds = cfg["n_folds"]
        parts = []
        for k in range(n_folds):
            files_k = [f for f, fd in fm.items() if (fd != k if leaked else fd == k)]
            # For "leaked", a file appears in every fold != its OOF fold; pick one
            # deterministically (the first such fold) so each window appears once.
            if leaked:
                files_k = [f for f, fd in fm.items() if (k == (fd + 1) % n_folds)]
            if not files_k:
                continue
            df = load_fold(cfg["dir"], k)
            df = df[df.filename.isin(files_k)]
            parts.append(df)
        allp = pd.concat(parts, ignore_index=True)
        merged = windows[["filename", "chunk_idx"]].merge(
            allp, on=["filename", "chunk_idx"], how="left"
        )
        miss = merged[species].isna().any(axis=1).sum()
        assert miss == 0, f"{name} (leaked={leaked}): {miss} windows missing predictions"
        assert len(merged) == n_win, f"{name}: row count {len(merged)} != {n_win}"
        return merged[species].to_numpy(np.float32)

    oof = {}
    for name in ("v5j", "distilled"):
        oof[name] = assemble(name)
        leaked = assemble(name, leaked=True)
        a_oof, a_leak = _macro_auc(Y, oof[name]), _macro_auc(Y, leaked)
        flag = (
            "OK (OOF < leaked)"
            if a_oof < a_leak
            else "!! WARN: OOF >= leaked (fold map/convention suspect)"
        )
        print(f"{name} leak-check: macroAUC OOF={a_oof:.4f}  leaked={a_leak:.4f}  -> {flag}")

    # Perch: single prediction per window (no folds)
    perch_df = pd.read_parquet(
        MODELS["perch"]["dir"] / "train_soundscapes" / "perch_234_probs.parquet"
    )
    perch_df = perch_df[["filename", "chunk_idx"] + species].copy()
    perch_df["chunk_idx"] = perch_df["chunk_idx"].astype(int)
    pm = windows[["filename", "chunk_idx"]].merge(
        perch_df, on=["filename", "chunk_idx"], how="left"
    )
    assert pm[species].isna().any(axis=1).sum() == 0, "perch missing windows"
    oof["perch"] = pm[species].to_numpy(np.float32)
    print(f"perch: macroAUC={_macro_auc(Y, oof['perch']):.4f}")

    # ---- save ----
    np.save(OUT / "gt_Y.npy", Y)
    windows.to_parquet(OUT / "gt_windows.parquet")
    (OUT / "species_cols.json").write_text(json.dumps(species))
    (OUT / "oof_fold_maps.json").write_text(json.dumps(fold_maps, indent=2))
    for name, P in oof.items():
        df = windows[["filename", "chunk_idx", "start_sec", "site"]].copy()
        df[species] = P
        df.to_parquet(OUT / f"oof_{name}.parquet")
    print(f"\nWrote OOF artifacts to {OUT}")


if __name__ == "__main__":
    main()
