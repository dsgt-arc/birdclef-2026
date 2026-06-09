"""Species selection for cross-species analyses.

``REPRESENTATIVE_LABELS`` is a small hand-curated cross-taxon set used by the
per-species PaCMAP visualization. ``soundscape_train_species`` is the derived
selector: every species that appears in the labeled training soundscapes *and*
has focal ``train_audio`` recordings, computed from the data files (not a
frozen list) so it tracks dataset refreshes.
"""

from __future__ import annotations

import polars as pl

from data_analysis.common.paths import (
    TRAIN_AUDIO_DIR,
    TRAIN_SOUNDSCAPES_LABELS_CSV,
)

REPRESENTATIVE_LABELS = [
    "65380",  # Dwarf Tree Frog
    "22973",  # Whistling Grass Frog
    "555146",  # Chaco Tree Frog
    "23158",  # Pale-legged Weeping Frog
    "24279",  # Lesser Snouted Tree Frog
    "22967",  # Marbled White-lipped Frog
    "chacha1",  # Chaco Chachalaca
    "whtdov",  # White-tipped Dove
    "undtin1",  # Undulated Tinamou
    "compau",  # Pauraque
]


def soundscape_labels() -> set[str]:
    """Distinct primary labels appearing in the labeled training soundscapes.

    ``train_soundscapes_labels.csv`` carries a ``;``-joined multilabel
    ``primary_label`` per 5 s window; this returns the flattened label set.
    """
    df = pl.read_csv(TRAIN_SOUNDSCAPES_LABELS_CSV)
    labels: set[str] = set()
    for value in df["primary_label"].drop_nulls().to_list():
        labels.update(part for part in str(value).split(";") if part)
    return labels


def focal_recording_count(label: str) -> int:
    """Number of focal ``train_audio/<label>/*.ogg`` recordings for a label."""
    species_dir = TRAIN_AUDIO_DIR / label
    if not species_dir.is_dir():
        return 0
    return sum(1 for _ in species_dir.glob("*.ogg"))


def soundscape_train_species(min_focal: int = 1) -> list[tuple[str, int]]:
    """Species in both the labeled soundscapes and focal train, with counts.

    Returns ``(label, n_focal)`` for every label in
    :func:`soundscape_labels` that also has at least ``min_focal`` focal
    ``train_audio`` recordings, sorted by ``n_focal`` ascending. Auto-motif's
    band discovery is a cross-recording median consensus, so a handful of
    recordings can't form a stable one — ``min_focal=8`` is the lowest count
    we have validated end to end (the Marbled White-lipped Frog). Ascending
    order front-loads the cheapest species, so a batch surfaces failures fast
    and runs the validated n=8 reference first as a smoke test.
    """
    counted = [
        (label, focal_recording_count(label))
        for label in sorted(soundscape_labels())
    ]
    selected = [(label, n) for label, n in counted if n >= min_focal]
    selected.sort(key=lambda item: item[1])
    return selected
