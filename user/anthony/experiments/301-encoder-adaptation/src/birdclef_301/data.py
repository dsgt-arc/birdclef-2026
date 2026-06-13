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
    """Unlabeled soundscape windows (the 176.5 h Pantanal pool) for DAPT.

    `split` carves a deterministic held-out fraction by FILE (disjoint train/val,
    so no soundscape leaks across the split), for measuring a held-out SSL objective
    each epoch — the train/val gap is the overfitting detector for the epoch ladder:
        train = SoundscapeDataset(dir, split="train", val_frac=0.05)
        val   = SoundscapeDataset(dir, split="val",   val_frac=0.05)
    `val` is `deterministic` by default: each file's crop is fixed across epochs
    (seeded by index) so the val loss reflects model drift, not crop/mask noise.
    """

    def __init__(self, soundscape_dir: str, window_samples: int = ENCODER_WINDOW_SAMPLES,
                 pattern: str = "*.ogg", seed: int = 42, split: str = "all",
                 val_frac: float = 0.0, split_seed: int = 1234,
                 deterministic: bool | None = None):
        files = sorted(str(p) for p in Path(soundscape_dir).glob(pattern))
        if not files:
            raise FileNotFoundError(f"No audio matching {pattern} under {soundscape_dir}")
        if split != "all" and val_frac > 0:
            perm = np.random.RandomState(split_seed).permutation(len(files))
            n_val = max(1, int(round(len(files) * val_frac)))
            val_idx = set(perm[:n_val].tolist())
            keep = (lambda i: i in val_idx) if split == "val" else (lambda i: i not in val_idx)
            files = [f for i, f in enumerate(files) if keep(i)]
            if not files:
                raise ValueError(f"{split!r} split is empty (val_frac={val_frac}); "
                                 "a drop_last train loader would silently yield no batches")
        self.files = files
        self.window_samples = window_samples
        self._rng = np.random.RandomState(seed)
        # val defaults to fixed crops; train to fresh random crops per epoch.
        self.deterministic = (split == "val") if deterministic is None else deterministic
        self._det_base = split_seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rng = np.random.RandomState(self._det_base + idx) if self.deterministic else self._rng
        wav = _random_window(self.files[idx], self.window_samples, rng)
        return torch.from_numpy(wav).unsqueeze(0)
