"""Datasets for encoder adaptation.

FocalDataset    — supervised multi-label focal windows (train.csv) for LoRA.
SoundscapeDataset — unlabeled soundscape windows for DAPT (continued SSL).

Both return 16 kHz mono 5 s waveforms shaped (1, N), matching the encoder
wrappers' `extract_features` input. Audio I/O reuses exp 300's 16 kHz loader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from birdclef_300.audio import ENCODER_SR, ENCODER_WINDOW_SAMPLES, fit_window

try:  # soundfile is only needed at runtime, not import-time for smoke tests
    import soundfile as sf
except Exception:  # noqa: BLE001
    sf = None


def _random_window(path: str, window_samples: int, rng: np.random.RandomState) -> np.ndarray:
    """Load a random `window_samples` crop at 16 kHz mono (zero-padded if short)."""
    import torchaudio.functional as taF

    info = sf.info(path)
    sr = info.samplerate
    win_src = int(window_samples * sr / ENCODER_SR)
    total = info.frames
    start = 0 if total <= win_src else int(rng.randint(0, total - win_src))
    audio, sr = sf.read(path, start=start, stop=start + win_src, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != ENCODER_SR:
        audio = taF.resample(torch.from_numpy(audio).float(), sr, ENCODER_SR).numpy()
    return fit_window(audio.astype(np.float32), window_samples)


class FocalDataset(Dataset):
    """Focal recordings with multi-hot (primary + secondary) species labels."""

    def __init__(self, train_csv: str, audio_dir: str, window_samples: int = ENCODER_WINDOW_SAMPLES,
                 seed: int = 42):
        self.audio_dir = Path(audio_dir)
        self.window_samples = window_samples
        self._rng = np.random.RandomState(seed)
        df = pl.read_csv(train_csv, infer_schema_length=10000)
        self.species_list = sorted(df["primary_label"].unique().to_list())
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}
        self.rows = df.select(["filename", "primary_label", "secondary_labels"]).to_dicts()

    @property
    def num_classes(self) -> int:
        return len(self.species_list)

    def _labels(self, row: dict) -> torch.Tensor:
        y = torch.zeros(self.num_classes)
        y[self.species_to_idx[row["primary_label"]]] = 1.0
        sec = row.get("secondary_labels")
        if sec and str(sec) not in ("", "None", "nan", "null", "[]"):
            for sp in str(sec).replace("[", "").replace("]", "").replace("'", "").split(","):
                sp = sp.strip()
                if sp in self.species_to_idx:
                    y[self.species_to_idx[sp]] = 1.0
        return y

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        wav = _random_window(str(self.audio_dir / row["filename"]), self.window_samples, self._rng)
        return torch.from_numpy(wav).unsqueeze(0), self._labels(row)


class SoundscapeDataset(Dataset):
    """Unlabeled soundscape windows (the 176.5 h Pantanal pool) for DAPT."""

    def __init__(self, soundscape_dir: str, window_samples: int = ENCODER_WINDOW_SAMPLES,
                 pattern: str = "*.ogg", seed: int = 42):
        self.files = sorted(str(p) for p in Path(soundscape_dir).glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No audio matching {pattern} under {soundscape_dir}")
        self.window_samples = window_samples
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        wav = _random_window(self.files[idx], self.window_samples, self._rng)
        return torch.from_numpy(wav).unsqueeze(0)
