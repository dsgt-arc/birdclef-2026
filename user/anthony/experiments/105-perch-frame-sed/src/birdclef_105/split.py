"""Soundscape file-level train/val split.

Replicates `_split_soundscape_files(seed=42)` from exp 102
(`user/anthony/experiments/102-efficientnet-sed/src/birdclef_102/data.py:438`).

Kept as a local copy so exp 105 is self-contained. The test
`tests/test_split_matches_exp102.py` asserts this function produces the same
train/val file lists as exp 102 on the real soundscape labels CSV — that is
the guarantee that downstream cmAP comparisons are apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_soundscape_files(
    soundscape_labels_csv: str,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Deterministic file-level 50/50 split of the soundscape labels CSV.

    Matches exp 102's `_split_soundscape_files(seed=42)` byte-for-byte when
    given the same CSV and seed.

    Args:
        soundscape_labels_csv: Path to `train_soundscapes_labels.csv`.
        seed: RNG seed for the shuffle. Must be 42 to match exp 102.

    Returns:
        `(train_files, val_files)` — sets of filename strings.
    """
    labels = pd.read_csv(soundscape_labels_csv)
    files = sorted(labels["filename"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    mid = len(files) // 2
    return set(files[:mid]), set(files[mid:])


def assign_split(filename: str, train_files: set[str], val_files: set[str]) -> str:
    """Return 'train' or 'val' for a given soundscape filename."""
    if filename in train_files:
        return "train"
    if filename in val_files:
        return "val"
    raise KeyError(f"Filename {filename!r} not in train or val set")
