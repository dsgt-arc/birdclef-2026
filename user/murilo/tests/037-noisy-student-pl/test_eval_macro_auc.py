"""Exp 037 — labeled-SS eval metric tests (the competition metric).

macro_auc_skip_empty must: skip classes with no positives (or all-positive), average per-class
ROC-AUC, and reward correct ranking. align_preds must fail loudly on missing windows/species.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parent.parent.parent / "experiments/037-noisy-student-pl/scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_labeled_ss")


def test_macro_auc_perfect_and_skip():
    # class0: perfect (auc 1); class1: all-zero (skip); class2: fully reversed (auc 0)
    Y = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    # col2: negatives (rows 0,2) score HIGH, positives (rows 1,3) score LOW -> auc 0
    P = np.array(
        [[0.9, 0.1, 0.9], [0.1, 0.2, 0.1], [0.8, 0.3, 0.8], [0.2, 0.4, 0.2]], dtype=np.float32
    )
    auc, n = ev.macro_auc_skip_empty(Y, P)
    assert n == 2  # class1 skipped (no positives)
    assert abs(auc - 0.5) < 1e-9  # mean(class0=1.0, class2=0.0)


def test_macro_auc_skips_all_positive():
    Y = np.array([[1, 1], [0, 1], [1, 1]], dtype=np.float32)  # class1 all-positive → skip
    P = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5]], dtype=np.float32)
    auc, n = ev.macro_auc_skip_empty(Y, P)
    assert n == 1 and abs(auc - 1.0) < 1e-9


def test_align_preds_missing_species_raises():
    windows = pd.DataFrame({"filename": ["a.ogg"], "chunk_idx": [0]})
    pred = pd.DataFrame({"filename": ["a.ogg"], "chunk_idx": [0], "spX": [0.5]})
    with pytest.raises(ValueError, match="missing"):
        ev.align_preds(pred, windows, ["spX", "spY"])


def test_align_preds_missing_window_raises():
    windows = pd.DataFrame({"filename": ["a.ogg", "b.ogg"], "chunk_idx": [0, 0]})
    pred = pd.DataFrame({"filename": ["a.ogg"], "chunk_idx": [0], "spX": [0.5]})
    with pytest.raises(ValueError, match="missing from preds"):
        ev.align_preds(pred, windows, ["spX"])


def test_align_preds_is_key_based():
    windows = pd.DataFrame({"filename": ["a.ogg", "a.ogg"], "chunk_idx": [0, 1]})
    pred = pd.DataFrame({"filename": ["a.ogg", "a.ogg"], "chunk_idx": [1, 0], "spX": [0.7, 0.2]})
    P = ev.align_preds(pred, windows, ["spX"])
    assert P[0, 0] == pytest.approx(0.2) and P[1, 0] == pytest.approx(0.7)
