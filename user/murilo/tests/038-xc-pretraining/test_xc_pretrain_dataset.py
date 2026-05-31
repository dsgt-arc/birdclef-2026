"""Exp 038 — XCPretrainDataset tests (label construction, secondaries, crop/pad, mixup, guards)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.data.xc_pretrain_dataset import XCPretrainDataset, _iter_secondary_codes  # noqa: E402

VOCAB = {"a": 0, "b": 1, "c": 2}
WIN = 100


def _const_loader(value: float):
    """Audio loader stub: returns a full-length constant waveform (no filesystem I/O)."""

    def loader(path, n_samples, start_sample=0, sample_rate=32000):
        return np.full(n_samples, value, dtype=np.float32)

    return loader


def _ds(df, **kw):
    kw.setdefault("audio_loader", _const_loader(0.5))
    return XCPretrainDataset(df, VOCAB, audio_dir="/nonexistent", window_samples=WIN, **kw)


def test_iter_secondary_codes_handles_all_cell_types():
    assert _iter_secondary_codes(np.array(["b", "z"])) == ["b", "z"]
    assert _iter_secondary_codes(["b", "c"]) == ["b", "c"]
    assert _iter_secondary_codes(None) == []
    assert _iter_secondary_codes(float("nan")) == []
    assert _iter_secondary_codes(np.array([], dtype=object)) == []
    assert _iter_secondary_codes("b") == ["b"]


def test_build_label_primary_plus_invocab_secondary_drops_oov():
    df = pd.DataFrame({"filepath": ["XC1.ogg"], "ebird_code": ["a"], "length": [10]})
    df["ebird_code_secondary"] = [np.array(["b", "z"])]  # 'b' in vocab, 'z' out-of-vocab
    lb = _ds(df)._build_label(df.iloc[0])
    assert lb.tolist() == [1.0, 1.0, 0.0]  # primary a + secondary b; z dropped


def test_build_label_no_secondary():
    df = pd.DataFrame(
        {
            "filepath": ["XC1.ogg"],
            "ebird_code": ["c"],
            "length": [10],
            "ebird_code_secondary": [None],
        }
    )
    lb = _ds(df)._build_label(df.iloc[0])
    assert lb.tolist() == [0.0, 0.0, 1.0]


def test_getitem_shape_and_padding():
    # loader returns a SHORT array → dataset must left-pad to window_samples.
    def short_loader(path, n_samples, start_sample=0, sample_rate=32000):
        return np.ones(n_samples - 10, dtype=np.float32)

    df = pd.DataFrame(
        {
            "filepath": ["XC1.ogg"],
            "ebird_code": ["a"],
            "length": [10],
            "ebird_code_secondary": [None],
        }
    )
    wav, lb = _ds(df, audio_loader=short_loader, aug=False)[0]
    assert tuple(wav.shape) == (1, WIN)
    assert tuple(lb.shape) == (3,)
    assert lb.tolist() == [1.0, 0.0, 0.0]


def test_zero_length_row_does_not_crash():
    df = pd.DataFrame(
        {
            "filepath": ["XC1.ogg"],
            "ebird_code": ["a"],
            "length": [0],
            "ebird_code_secondary": [None],
        }
    )
    wav, lb = _ds(df, aug=True, mixup=False)[0]  # length=0 → start=0, no random crop
    assert tuple(wav.shape) == (1, WIN)


def test_mixup_hard_max_unions_labels():
    # 2 rows, distinct single labels; mixup_prob=1 + len>1 → partner is the other row (skip self).
    df = pd.DataFrame(
        {
            "filepath": ["XC1.ogg", "XC2.ogg"],
            "ebird_code": ["a", "b"],
            "length": [10, 10],
            "ebird_code_secondary": [None, None],
        }
    )
    ds = _ds(df, aug=True, mixup=True, mixup_prob=1.0, seed=0)
    _, lb = ds[0]
    assert lb.tolist() == [1.0, 1.0, 0.0]  # max(a, b) = union


def test_no_mixup_when_disabled():
    df = pd.DataFrame(
        {
            "filepath": ["XC1.ogg", "XC2.ogg"],
            "ebird_code": ["a", "b"],
            "length": [10, 10],
            "ebird_code_secondary": [None, None],
        }
    )
    ds = _ds(df, aug=True, mixup=False, mixup_prob=1.0, seed=0)
    _, lb = ds[0]
    assert lb.tolist() == [1.0, 0.0, 0.0]  # only the primary, no union


def test_unloadable_file_returns_silent_zero_labels():
    def boom(path, n_samples, start_sample=0, sample_rate=32000):
        raise OSError("simulated missing file")

    df = pd.DataFrame(
        {
            "filepath": ["XC1.ogg"],
            "ebird_code": ["a"],
            "length": [10],
            "ebird_code_secondary": [None],
        }
    )
    wav, lb = _ds(df, audio_loader=boom)[0]
    assert tuple(wav.shape) == (1, WIN)
    assert wav.abs().sum().item() == 0.0
    assert lb.sum().item() == 0.0
