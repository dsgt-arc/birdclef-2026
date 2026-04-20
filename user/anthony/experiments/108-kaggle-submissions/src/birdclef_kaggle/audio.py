"""60-second soundscape → 12 × 5-second window loader.

Perch v2 expects 5s windows at 32 kHz (160,000 samples). BirdCLEF 2026
test soundscapes are 1 minute each, so every file produces 12 windows
with `end_seconds ∈ {5, 10, ..., 60}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 32_000
WINDOW_SECONDS = 5
WINDOWS_PER_CLIP = 12
SAMPLES_PER_WINDOW = SAMPLE_RATE * WINDOW_SECONDS  # 160_000
SAMPLES_PER_CLIP = SAMPLES_PER_WINDOW * WINDOWS_PER_CLIP  # 1_920_000


@dataclass(frozen=True)
class Clip:
    path: Path
    windows: np.ndarray  # (12, 160000) float32
    end_seconds: tuple[int, ...]  # (5, 10, ..., 60)


def load_clip(path: str | Path) -> Clip:
    """Load a 60s soundscape and return 12 stacked 5s windows.

    Shorter clips are zero-padded to 60s; longer clips are truncated.
    Mono is enforced by averaging channels.
    """
    import soundfile as sf

    path = Path(path)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32, copy=False)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected sr={SAMPLE_RATE}, got sr={sr} for {path}. "
            "Resample upstream if the test set ships at a different rate."
        )

    if audio.shape[0] < SAMPLES_PER_CLIP:
        pad = SAMPLES_PER_CLIP - audio.shape[0]
        audio = np.pad(audio, (0, pad), mode="constant")
    elif audio.shape[0] > SAMPLES_PER_CLIP:
        audio = audio[:SAMPLES_PER_CLIP]

    windows = audio.reshape(WINDOWS_PER_CLIP, SAMPLES_PER_WINDOW)
    end_seconds = tuple((i + 1) * WINDOW_SECONDS for i in range(WINDOWS_PER_CLIP))
    return Clip(path=path, windows=windows, end_seconds=end_seconds)


def stack_windows(clips: list[Clip]) -> np.ndarray:
    """Stack multiple clips' windows into a single (N*12, 160000) batch."""
    if not clips:
        return np.zeros((0, SAMPLES_PER_WINDOW), dtype=np.float32)
    return np.concatenate([c.windows for c in clips], axis=0)
