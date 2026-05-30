"""Exp 037 — teacher-blend correctness + PLDS-compatible output schema.

Builds tiny synthetic Perch-probs and SED-fold caches, monkeypatches the generator's cache
paths, and verifies:
  - blend == w*perch + (1-w)*mean_folds(sigmoid(sed_logits))  (prob-space, matches 0.929)
  - alignment is by (filename, chunk_idx), not row order
  - output schema = (filename, chunk_idx, <234 species cols ∈ [0,1]>) — what PLDS requires
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).parent.parent.parent / "experiments/037-noisy-student-pl/scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load("generate_ensemble_pl")

SP = [f"sp{i}" for i in range(234)]
FILES = ["BC2026_Train_0001_S01_20240101_000000.ogg", "BC2026_Train_0002_S02_20240101_000000.ogg"]


def _make_caches(tmp: Path, rng):
    rows = [(f, c) for f in FILES for c in range(12)]  # 24 rows
    n = len(rows)
    fn = [r[0] for r in rows]
    ck = [r[1] for r in rows]

    perch_P = rng.uniform(0, 1, size=(n, 234)).astype(np.float32)
    meta = pd.DataFrame(
        {"filename": fn, "chunk_idx": ck, "site": "S01", "hour_utc": 0, "source": "perch"}
    )
    perch = pd.concat([meta, pd.DataFrame(perch_P, columns=SP)], axis=1)
    perch_path = tmp / "perch_234_probs.parquet"
    perch.to_parquet(perch_path)

    sed_dir = tmp / "sed"
    sed_logits = []
    for fold in range(4):
        L = rng.normal(0, 3, size=(n, 234)).astype(np.float32)
        sed_logits.append(L)
        d = sed_dir / f"fold{fold}"
        d.mkdir(parents=True, exist_ok=True)
        # shuffle row order to prove alignment is key-based, not positional
        perm = rng.permutation(n)
        smeta = pd.DataFrame(
            {
                "filename": [fn[i] for i in perm],
                "source": "sed",
                "chunk_idx": [ck[i] for i in perm],
                "start_sec": 0.0,
                "end_sec": 5.0,
            }
        )
        sed = pd.concat([smeta, pd.DataFrame(L[perm], columns=SP)], axis=1)
        sed.to_parquet(d / "logits.parquet")
    return perch_P, sed_logits, perch_path, sed_dir


def test_blend_and_schema(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    perch_P, sed_logits, perch_path, sed_dir = _make_caches(tmp_path, rng)
    monkeypatch.setattr(gen, "PERCH_PROBS", perch_path)
    monkeypatch.setattr(gen, "SED_DIR", sed_dir)

    out, species = gen.load_teacher_blend(w_perch=0.5, limit=None)

    assert species == SP
    assert list(out.columns[:2]) == ["filename", "chunk_idx"]
    assert len(out) == 24
    vals = out[species].to_numpy()
    assert vals.min() >= 0.0 and vals.max() <= 1.0

    # expected blend in the perch key order
    sed_prob = np.mean([gen.sigmoid(L) for L in sed_logits], axis=0)
    expected = 0.5 * perch_P + 0.5 * sed_prob
    np.testing.assert_allclose(vals, expected, atol=1e-5)


def test_limit_subsets_files(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    _, _, perch_path, sed_dir = _make_caches(tmp_path, rng)
    monkeypatch.setattr(gen, "PERCH_PROBS", perch_path)
    monkeypatch.setattr(gen, "SED_DIR", sed_dir)
    out, _ = gen.load_teacher_blend(w_perch=0.5, limit=1)
    assert out["filename"].nunique() == 1
    assert len(out) == 12
