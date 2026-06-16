"""Exp 038 — XC supervised-pretraining dataset + datamodule (broad Neotropical bird backbone).

Trains a multi-label classifier over a fixed N-species vocabulary on the curated
BirdSet-Neotropical XC corpus, under the SAME input/pooling regime as the downstream SED stack
(5 s @ 32 kHz waveforms → the LightningModule's GPU Tucker-mel → `AttHeadLogitSpace`) so the
backbone transfers. Emits the `(waveform (1, N), label (C,))` 2-tuple that
`DistilledSEDFinetuneModule` consumes directly (its training/validation steps read batch[0]/[1]).

Label = multi-hot over `species_to_idx`: primary `ebird_code` ∪ in-vocab `ebird_code_secondary`
(out-of-vocab secondaries — e.g. non-Neotropical background birds — are dropped). Audio is a
random 5 s crop (the crop offset is the time augmentation); `apply_aug` (gain/noise) and an
optional focal×focal waveform mixup (Beta(α), hard-max labels) mirror the Tucker `FocalDS`
recipe — those building blocks are imported from `distilled_sed_dataset` so they stay
bit-identical to the fine-tune path.

Label smoothing is deliberately NOT applied here: it lives in the LightningModule loss
(`label_smoothing` hparam) so that the module's `labels.int()`-based metrics keep seeing hard
targets. See user/murilo/experiments/038-xc-pretraining/ (proposal + plan).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .distilled_sed_dataset import _default_audio_loader, _extract_chunk, apply_aug

logger = logging.getLogger(__name__)


def _iter_secondary_codes(val) -> list[str]:
    """Normalize a manifest `ebird_code_secondary` cell to a list of code strings.

    The curated manifest stores secondaries as a numpy array of eBird codes (34.6% of rows),
    or NaN/None when absent. Handles ndarray/list/tuple of strings, a lone string, and the
    NaN/None empty cases (without calling `pd.isna` on an array, which would raise)."""
    if val is None:
        return []
    if isinstance(val, list | tuple | np.ndarray):
        return [str(s) for s in val if isinstance(s, str) and s]
    if isinstance(val, float):  # scalar NaN
        return []
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


class XCPretrainDataset(Dataset):
    """Curated-XC focal recordings → (waveform (1, N), multi-hot label (C,)).

    Args:
        df: manifest rows with `filepath` (bare `<XCid>.ogg`), `ebird_code` (primary),
            `ebird_code_secondary` (array|NaN), `length` (int seconds).
        species_to_idx: fixed species → label-index map (defines C = vocabulary size).
        audio_dir: root of `audio/<ebird_code>/<filepath>`.
        window_samples: samples per crop (32000 × 5 = 160000).
        aug: enable random-offset crop + `apply_aug` (gain/noise). False for val (start=0,
            no aug → deterministic, stable metric).
        mixup: enable focal×focal waveform mixup (only when `aug` too).
        mixup_prob/mixup_alpha/mixup_hard: Beta(α) mixup; hard-max labels (Tucker default).
        aug_prob: per-augmentation firing probability inside `apply_aug`.
        audio_loader: `(path, n_samples, start_sample, sample_rate) -> np.ndarray`. Injected
            so tests can stub it (no filesystem I/O). Defaults to soundfile-backed loader.
        sample_rate: 32000.
        seed: per-instance RNG seed (advanced per `__getitem__`).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        species_to_idx: dict[str, int],
        audio_dir: str | Path,
        window_samples: int = 160_000,
        aug: bool = False,
        mixup: bool = False,
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.4,
        mixup_hard: bool = True,
        aug_prob: float = 0.5,
        audio_loader: Callable[..., np.ndarray] = _default_audio_loader,
        sample_rate: int = 32000,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.num_classes = len(species_to_idx)
        self.audio_dir = Path(audio_dir)
        self.window_samples = window_samples
        self.aug = aug
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.mixup_hard = mixup_hard
        self.aug_prob = aug_prob
        self.audio_loader = audio_loader
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, row: pd.Series) -> str:
        return str(self.audio_dir / str(row["ebird_code"]) / str(row["filepath"]))

    def _build_label(self, row: pd.Series) -> np.ndarray:
        """Multi-hot: primary ∪ in-vocab secondaries (out-of-vocab dropped)."""
        lb = np.zeros(self.num_classes, dtype=np.float32)
        idx = self.species_to_idx.get(str(row["ebird_code"]))
        if idx is not None:
            lb[idx] = 1.0
        for s in _iter_secondary_codes(row.get("ebird_code_secondary")):
            j = self.species_to_idx.get(s)
            if j is not None:
                lb[j] = 1.0
        return lb

    def _load_chunk(self, row: pd.Series) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load a (random-offset, training) 5 s crop. Returns (waveform, label) or (None, None)."""
        path = self._resolve_path(row)
        length_sec = int(row.get("length", 0) or 0)
        total = max(length_sec * self.sample_rate, 0)
        if self.aug and total > self.window_samples:
            start = int(self.rng.integers(0, total - self.window_samples + 1))
        else:
            start = 0
        try:
            wav = self.audio_loader(path, self.window_samples, start, self.sample_rate)
        except Exception as e:  # noqa: BLE001 — guard a single bad file, keep the epoch going
            logger.warning(f"XCPretrainDataset: failed to load {path}: {e}")
            return None, None
        if wav is None or wav.shape[0] < self.window_samples:
            wav = _extract_chunk(wav if wav is not None else np.zeros(0), 0, self.window_samples)
        return wav.astype(np.float32), self._build_label(row)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        ch1, lb1 = self._load_chunk(row)
        if ch1 is None:
            # Unloadable file (rare; guarded) → silent clip + zero labels.
            return (
                torch.zeros(1, self.window_samples, dtype=torch.float32),
                torch.zeros(self.num_classes, dtype=torch.float32),
            )

        # --- focal × focal mixup ---
        if self.aug and self.mixup and len(self.df) > 1 and self.rng.random() < self.mixup_prob:
            ch2 = lb2 = None
            for _ in range(3):
                j = int(self.rng.integers(0, len(self.df)))
                if j == i:
                    continue
                ch2, lb2 = self._load_chunk(self.df.iloc[j])
                if ch2 is not None:
                    break
            if ch2 is not None:
                lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
                ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                ch_mix = apply_aug(ch_mix, self.rng, aug_prob=self.aug_prob)
                lb = np.maximum(lb1, lb2) if self.mixup_hard else (lam * lb1 + (1 - lam) * lb2)
                return (
                    torch.from_numpy(ch_mix).unsqueeze(0),
                    torch.from_numpy(lb.astype(np.float32)),
                )

        # --- no mixup ---
        if self.aug:
            ch1 = apply_aug(ch1, self.rng, aug_prob=self.aug_prob)
        return torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0), torch.from_numpy(lb1)


class XCPretrainDataModule(pl.LightningDataModule):
    """Loads the curated manifest + a fixed species vocabulary, makes a stratified per-species
    train/val holdout, and serves the 2-tuple batches for `DistilledSEDFinetuneModule`.

    Args:
        manifest_path: curated `manifest.parquet`.
        species_list_path: `species_list.json` (ordered → defines the label index).
        audio_dir: root of `audio/<ebird_code>/<XCid>.ogg`.
        batch_size, num_workers: loader knobs (num_workers=8 default per project convention).
        window_samples, sample_rate: 160000 / 32000.
        val_fraction: per-species holdout fraction for the XC macro-AUC pretrain signal.
        aug/mixup/mixup_prob/mixup_alpha/aug_prob: train-time augmentation (val is always clean).
        seed: split + dataset RNG seed.
        require_audio: drop manifest rows whose audio is not on disk (True → supports a partial
            download / smoke run; set False once the full corpus is present to skip the scan).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        species_list_path: str | Path,
        audio_dir: str | Path,
        batch_size: int = 128,
        num_workers: int = 8,
        window_samples: int = 160_000,
        val_fraction: float = 0.02,
        aug: bool = True,
        mixup: bool = True,
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.4,
        aug_prob: float = 0.5,
        sample_rate: int = 32000,
        seed: int = 42,
        require_audio: bool = True,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.species_list_path = Path(species_list_path)
        self.audio_dir = Path(audio_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_samples = window_samples
        self.val_fraction = val_fraction
        self.aug = aug
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.aug_prob = aug_prob
        self.sample_rate = sample_rate
        self.seed = seed
        self.require_audio = require_audio
        self.species_to_idx: dict[str, int] = {}
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None

    def _load_species_to_idx(self) -> dict[str, int]:
        species = json.loads(self.species_list_path.read_text())
        return {sp: i for i, sp in enumerate(species)}

    def _present_audio(self) -> set[tuple[str, str]]:
        """One directory walk → {(ebird_code, filename)} present on disk (fast vs per-row stat)."""
        present: set[tuple[str, str]] = set()
        if not self.audio_dir.exists():
            return present
        for code_dir in self.audio_dir.iterdir():
            if code_dir.is_dir():
                for f in code_dir.iterdir():
                    present.add((code_dir.name, f.name))
        return present

    @staticmethod
    def stratified_split(
        df: pd.DataFrame, val_fraction: float, seed: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Per-species holdout: round(val_fraction × n) val rows per species (rest train).
        Rare species (round → 0) contribute 0 val rows (those classes are simply skipped by
        TuckerMacroAUC). No species leaks fully into val — train always keeps the bulk."""
        rng = np.random.default_rng(seed)
        val_idx: list[int] = []
        for _code, grp in df.groupby("ebird_code", sort=False):
            idx = grp.index.to_numpy()
            n_val = int(round(val_fraction * len(idx)))
            if n_val > 0:
                val_idx.extend(rng.choice(idx, size=n_val, replace=False).tolist())
        val_mask = df.index.isin(val_idx)
        return (
            df[~val_mask].reset_index(drop=True),
            df[val_mask].reset_index(drop=True),
        )

    def setup(self, stage: str | None = None) -> None:
        self.species_to_idx = self._load_species_to_idx()
        df = pd.read_parquet(self.manifest_path)
        if self.require_audio:
            present = self._present_audio()
            keep = [
                (str(c), str(f)) in present
                for c, f in zip(df["ebird_code"], df["filepath"], strict=False)
            ]
            n_missing = len(df) - int(np.sum(keep))
            if n_missing:
                logger.warning(
                    f"XCPretrainDataModule: {n_missing:,}/{len(df):,} manifest rows have no audio "
                    "on disk — dropping (partial download / smoke)."
                )
            df = df[keep].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(
                f"XCPretrainDataModule: no usable rows (manifest={self.manifest_path}, "
                f"audio_dir={self.audio_dir}, require_audio={self.require_audio})."
            )
        self.train_df, self.val_df = self.stratified_split(df, self.val_fraction, self.seed)
        logger.info(
            f"XCPretrain split: {len(self.train_df):,} train / {len(self.val_df):,} val "
            f"| {df['ebird_code'].nunique()} species present | vocab {len(self.species_to_idx)}"
        )

    @property
    def num_classes(self) -> int:
        return len(self.species_to_idx)

    def _make_dataset(self, df: pd.DataFrame, *, aug: bool, mixup: bool, seed: int):
        return XCPretrainDataset(
            df=df,
            species_to_idx=self.species_to_idx,
            audio_dir=self.audio_dir,
            window_samples=self.window_samples,
            aug=aug,
            mixup=mixup,
            mixup_prob=self.mixup_prob,
            mixup_alpha=self.mixup_alpha,
            aug_prob=self.aug_prob,
            sample_rate=self.sample_rate,
            seed=seed,
        )

    def train_dataloader(self) -> DataLoader:
        ds = self._make_dataset(self.train_df, aug=self.aug, mixup=self.mixup, seed=self.seed)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        # Val: deterministic (start=0), no aug, no mixup, no smoothing → clean metric targets.
        ds = self._make_dataset(self.val_df, aug=False, mixup=False, seed=self.seed)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
