"""Datasets for exp 032 (Tucker distilled-SED replication).

Verbatim port of `tuckerarrants/bc2026-distilled-sed` cells 12 + 7:

- `FocalDS`: focal recordings with random crop + per-sample augmentation. Each
  call samples one of three paths:
  - Focal × Focal mixup (waveform, Beta(0.4, 0.4) lam, MIXUP_HARD=True →
    `max(lb1, lb2)`): prob `mixup_prob` (default 0.5).
  - Focal × labeled-SS mixup (waveform, Beta(0.4, 0.4), MIXUP_HARD=True):
    prob `focal_sc_mixup_prob` (default 0.5) — only reached when focal-focal
    didn't fire.
  - No-mixup: just the focal sample.
  In all three branches, `apply_aug` (gain jitter + Gaussian noise, no
  time-shift) is applied to the final waveform.

- `ScDS`: labeled soundscape windows. `apply_aug` only (no mixup) — Tucker
  doesn't apply mixup on SS samples in this position; SS gets mixed INTO focal
  via FocalDS instead.

- `apply_aug`: gain jitter (±6 dB, p=0.5) + SNR-controlled Gaussian noise
  (10–30 dB, p=0.5). Time-shift comes from the random crop offset at load
  time, NOT from this function.

- `MixSamp`: batch sampler that composes batches from multiple data sources
  with per-source shares. For Tucker's `{"focal": 0.9, "sc": 0.1}, bs=64`
  defaults: 58 focal + 6 SC samples per batch. Used with `ConcatDataset` of
  `[FocalDS, ScDS]` in that order.

Kept separate from `data/sed.py` (per the reproducibility argument for the
LightningModule: bit-identical preservation of exp 030/031 paths).

References:
- analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md
- analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md (§§ 1, 2, 3)
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cells 7, 12)

Design notes:
- The audio loader is injected (`audio_loader: callable(path, n_samples, start)`)
  so tests can stub it with synthetic waveforms — no real filesystem I/O needed
  for unit tests. Production uses `_default_audio_loader`.
- A per-instance `numpy.random.Generator` is used for all RNG so tests can
  fix seeds for deterministic outcomes.
- The instance attribute `mixup_counts` is a `Counter` over the three paths;
  tests inspect it to verify rates. It is not used by training.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


# ============================================================================
# apply_aug — gain jitter + Gaussian noise (Tucker cell 12)
# ============================================================================


def apply_aug(
    waveform: np.ndarray,
    rng: np.random.Generator,
    aug_prob: float = 0.5,
    gain_db_range: tuple[float, float] = (-6.0, 6.0),
    noise_snr_db_range: tuple[float, float] = (10.0, 30.0),
) -> np.ndarray:
    """Simple waveform augmentation: gain jitter + additive Gaussian noise.

    Each augmentation fires independently with probability `aug_prob`.

    Args:
        waveform: 1-D float32 numpy array (the audio chunk).
        rng: `numpy.random.Generator` for reproducible RNG.
        aug_prob: Per-augmentation firing probability (default 0.5).
        gain_db_range: Uniform range for gain jitter (default ±6 dB).
        noise_snr_db_range: Uniform range for SNR-controlled noise (10–30 dB).

    Returns:
        Augmented waveform (same shape/dtype).

    Notes:
        Tucker's cell 12 implementation:
            if np.random.random() < AUG_PROB:
                w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
            if np.random.random() < AUG_PROB:
                sp = (w ** 2).mean()
                if sp > 1e-10:
                    w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                        sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))

        Time-shift is NOT applied here (the docstring of Tucker's notebook
        mentions three augmentations but the code only does two; time-shift
        comes from the random crop offset at load time).
    """
    w = waveform
    # Gain jitter
    if rng.random() < aug_prob:
        gain_db = rng.uniform(*gain_db_range)
        w = w * (10 ** (gain_db / 20))
    # SNR-controlled Gaussian noise
    if rng.random() < aug_prob:
        sp = (w**2).mean()
        if sp > 1e-10:
            snr_db = rng.uniform(*noise_snr_db_range)
            noise_var = sp / (10 ** (snr_db / 10))
            w = w + rng.standard_normal(w.shape).astype(w.dtype) * np.sqrt(noise_var)
    return w


# ============================================================================
# Audio loading helpers
# ============================================================================


def _default_audio_loader(
    path: Path | str, n_samples: int, start_sample: int = 0, sample_rate: int = 32000
) -> np.ndarray:
    """Default loader. Reuses `data.dataset._load_audio_chunk` (soundfile + torchaudio
    resampling). Returns a numpy float32 array of exactly `n_samples` samples
    (zero-padded if the source is shorter)."""
    from .dataset import _load_audio_chunk

    audio = _load_audio_chunk(
        path, target_sr=sample_rate, start_sample=start_sample, num_samples=n_samples
    )
    arr = audio.numpy().astype(np.float32)
    if arr.shape[0] < n_samples:
        arr = np.pad(arr, (0, n_samples - arr.shape[0]))
    return arr[:n_samples]


def _extract_chunk(waveform: np.ndarray, start_sample: int, n_samples: int) -> np.ndarray:
    """Slice a chunk from a full waveform; left-pad with zeros if total < n_samples
    (matches Tucker's `extract_chunk_np` in cell 12)."""
    total = len(waveform)
    if total <= n_samples:
        return np.pad(waveform, (n_samples - total, 0))
    end = start_sample + n_samples
    if end > total:
        start_sample = max(0, total - n_samples)
    return waveform[start_sample : start_sample + n_samples]


# ============================================================================
# FocalDS — focal recordings with Tucker mixup
# ============================================================================


class FocalDS(Dataset):
    """Focal recordings dataset (verbatim Tucker port of cell 12).

    Returns 5-tuple `(waveform_tensor, label_tensor, wt, mk, source_tag)`
    where:
    - `waveform_tensor`: (1, n_samples) float32 — Tucker uses a channel dim
    - `label_tensor`: (num_classes,) float32 multi-hot
    - `wt`: (num_classes,) all ones (placeholder for class weighting — unused
      in Tucker's notebook per 2026-05-23 audit § 4)
    - `mk`: (num_classes,) all ones (placeholder for class masking — unused)
    - `source_tag`: "focal" or "focal_missing"

    The 3-way path (focal-focal mixup / focal-SS mixup / no-mixup) is
    transparent to the consumer — `source_tag` always returns "focal" unless
    the audio file failed to load. The `mixup_counts` attribute on the
    instance tracks which path was chosen, for test introspection.

    Args:
        focal_df: DataFrame with at least `filename`, `primary_label`,
            `original_idx`, and optionally `secondary_labels` columns. Each
            row is a focal recording. Path resolution is via `audio_loader`.
        species_to_idx: Map species name → label index.
        sc_pool: Optional. List of `(sc_df, Y_SC)` tuples — labeled SS windows
            available for focal × SS mixup. Set `None` to disable that path.
            `sc_df` rows must have `filename`, `start_sec`, `cache_file`,
            `fold` columns. `Y_SC` is the corresponding (n_sc_rows, num_classes)
            multi-hot label matrix.
        fold_k: Fold index used to filter the SS mixup pool — rows with
            `fold == fold_k` are excluded (held out for validation).
        window_samples: Number of samples per window. Tucker: 32000 × 5 = 160000.
        num_classes: Output class count (234 for BirdCLEF 2026).
        aug: Whether to apply augmentation (True for training, False for eval).
        mixup_prob: Focal-focal mixup firing probability (default 0.5).
        mixup_alpha: Beta(α, α) for mixup λ (default 0.4 — Tucker).
        mixup_hard: When True, use `max(lb1, lb2)` (default True — Tucker).
            When False, use λ-weighted blend.
        focal_sc_mixup_prob: Focal × labeled-SS mixup firing probability
            when focal-focal mixup did NOT fire (default 0.5).
        focal_sc_mixup_alpha: Beta(α, α) for focal-SS mixup λ (default 0.4).
        aug_prob: Per-augmentation firing probability for `apply_aug`.
        secondary_labels_lookup: Optional dict `{original_idx: [secondary
            species strings]}` for multi-hot labels.
        audio_loader: Callable `(path, n_samples, start_sample, sample_rate) →
            np.ndarray` that returns the loaded waveform. Defaults to
            `_default_audio_loader` (soundfile + resampling).
        sc_audio_loader: Callable for loading SS waveforms in the mixup pool.
            Defaults to `_default_audio_loader`.
        sample_rate: Sample rate (default 32000).
        seed: RNG seed. Each `__getitem__` advances the generator.
    """

    def __init__(
        self,
        focal_df: pd.DataFrame,
        species_to_idx: dict[str, int],
        sc_pool: list[tuple[pd.DataFrame, np.ndarray]] | None = None,
        fold_k: int | None = None,
        window_samples: int = 160_000,
        num_classes: int = 234,
        aug: bool = False,
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.4,
        mixup_hard: bool = True,
        focal_sc_mixup_prob: float = 0.5,
        focal_sc_mixup_alpha: float = 0.4,
        aug_prob: float = 0.5,
        secondary_labels_lookup: dict[int, list[str]] | None = None,
        audio_loader: Callable[..., np.ndarray] = _default_audio_loader,
        sc_audio_loader: Callable[..., np.ndarray] = _default_audio_loader,
        sample_rate: int = 32000,
        seed: int = 42,
    ):
        self.df = focal_df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.sc_pool = sc_pool
        self.fold_k = fold_k
        self.window_samples = window_samples
        self.num_classes = num_classes
        self.aug = aug
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.mixup_hard = mixup_hard
        self.focal_sc_mixup_prob = focal_sc_mixup_prob
        self.focal_sc_mixup_alpha = focal_sc_mixup_alpha
        self.aug_prob = aug_prob
        self.secondary_labels_lookup = secondary_labels_lookup or {}
        self.audio_loader = audio_loader
        self.sc_audio_loader = sc_audio_loader
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        # Test introspection: count how many times each path fired
        self.mixup_counts = Counter({"focal_focal": 0, "focal_sc": 0, "none": 0, "missing": 0})

    def __len__(self) -> int:
        return len(self.df)

    def _build_label(self, row: pd.Series) -> np.ndarray:
        """Multi-hot from primary + secondary labels."""
        lb = np.zeros(self.num_classes, dtype=np.float32)
        primary = str(row.get("primary_label", ""))
        if primary in self.species_to_idx:
            lb[self.species_to_idx[primary]] = 1.0
        # Secondary labels — pulled from precomputed lookup keyed by original_idx
        if "original_idx" in row.index:
            oidx = int(row["original_idx"])
            for s in self.secondary_labels_lookup.get(oidx, []):
                if s in self.species_to_idx:
                    lb[self.species_to_idx[s]] = 1.0
        return lb

    def _load_focal_chunk(self, row: pd.Series) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load + (training) random-crop a focal chunk. Returns (waveform, label)
        or (None, None) if the file failed to load."""
        try:
            path = row.get("cache_file") or row.get("filename")
            # If we have total_samples in the row, use it for crop offset; else
            # let the loader read the requested chunk.
            total = int(row.get("n_samples", self.window_samples))
            if self.aug and total > self.window_samples:
                start = int(self.rng.integers(0, total - self.window_samples + 1))
            else:
                start = int(row.get("start_sec", 0)) * self.sample_rate
            wav = self.audio_loader(path, self.window_samples, start, self.sample_rate)
        except Exception as e:
            logger.warning(f"FocalDS: failed to load {row.get('filename')}: {e}")
            return None, None
        # Defensive: enforce shape
        if wav is None or wav.shape[0] < self.window_samples:
            wav = _extract_chunk(wav if wav is not None else np.zeros(0), 0, self.window_samples)
        lb = self._build_label(row)
        return wav.astype(np.float32), lb

    def _load_sc_chunk(self, sc_row: pd.Series, sc_labels: np.ndarray, sc_idx: int):
        """Load a labeled SS window for mixup. Returns (waveform, label) or None."""
        try:
            path = sc_row.get("cache_file") or sc_row.get("filename")
            start = int(sc_row["start_sec"]) * self.sample_rate
            wav = self.sc_audio_loader(path, self.window_samples, start, self.sample_rate)
        except Exception as e:
            logger.warning(f"FocalDS sc-mixup: failed to load {sc_row.get('filename')}: {e}")
            return None
        if wav is None or wav.shape[0] < self.window_samples:
            return None
        lb = sc_labels[sc_idx].astype(np.float32)
        return wav.astype(np.float32), lb

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        ch1, lb1 = self._load_focal_chunk(row)
        if ch1 is None:
            # Missing audio — return zero waveform + zero labels with focal_missing tag.
            self.mixup_counts["missing"] += 1
            return (
                torch.zeros(1, self.window_samples, dtype=torch.float32),
                torch.zeros(self.num_classes, dtype=torch.float32),
                torch.ones(self.num_classes, dtype=torch.float32),
                torch.ones(self.num_classes, dtype=torch.float32),
                "focal_missing",
            )

        # --- Path A: Focal × Focal mixup ---
        if self.aug and self.rng.random() < self.mixup_prob:
            for _ in range(3):
                j = int(self.rng.integers(0, len(self.df)))
                ch2, lb2 = self._load_focal_chunk(self.df.iloc[j])
                if ch2 is not None:
                    break
            else:
                ch2 = None
            if ch2 is not None:
                lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
                ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                ch_mix = apply_aug(ch_mix, self.rng, aug_prob=self.aug_prob)
                lb = np.maximum(lb1, lb2) if self.mixup_hard else lam * lb1 + (1 - lam) * lb2
                self.mixup_counts["focal_focal"] += 1
                return (
                    torch.from_numpy(ch_mix).unsqueeze(0),
                    torch.from_numpy(lb),
                    torch.ones(self.num_classes, dtype=torch.float32),
                    torch.ones(self.num_classes, dtype=torch.float32),
                    "focal",
                )

        # --- Path B: Focal × labeled-SS mixup ---
        # Only reached when Path A didn't fire (the `mixup_prob < r` branch).
        if self.aug and self.sc_pool and self.rng.random() < self.focal_sc_mixup_prob:
            pool_idx = int(self.rng.integers(0, len(self.sc_pool)))
            sc_df, sc_labels = self.sc_pool[pool_idx]
            # Exclude the validation fold from the SS mixup pool
            if self.fold_k is not None and "fold" in sc_df.columns:
                eligible = sc_df[sc_df["fold"] != self.fold_k]
            else:
                eligible = sc_df
            if len(eligible) > 0:
                pick = int(self.rng.integers(0, len(eligible)))
                sc_row = eligible.iloc[pick]
                # The label-row index in Y_SC corresponds to the row's
                # position in the ORIGINAL sc_df, not the filtered subset.
                sc_idx = int(sc_row.get("label_idx", eligible.index[pick]))
                sc_loaded = self._load_sc_chunk(sc_row, sc_labels, sc_idx)
                if sc_loaded is not None:
                    sc_chunk, sc_lb = sc_loaded
                    lam = float(self.rng.beta(self.focal_sc_mixup_alpha, self.focal_sc_mixup_alpha))
                    ch_mix = (lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32)
                    ch_mix = apply_aug(ch_mix, self.rng, aug_prob=self.aug_prob)
                    lb = (
                        np.maximum(lb1, sc_lb) if self.mixup_hard else lam * lb1 + (1 - lam) * sc_lb
                    )
                    self.mixup_counts["focal_sc"] += 1
                    return (
                        torch.from_numpy(ch_mix).unsqueeze(0),
                        torch.from_numpy(lb),
                        torch.ones(self.num_classes, dtype=torch.float32),
                        torch.ones(self.num_classes, dtype=torch.float32),
                        "focal",
                    )

        # --- Path C: No mixup ---
        if self.aug:
            ch1 = apply_aug(ch1, self.rng, aug_prob=self.aug_prob)
        self.mixup_counts["none"] += 1
        return (
            torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(lb1),
            torch.ones(self.num_classes, dtype=torch.float32),
            torch.ones(self.num_classes, dtype=torch.float32),
            "focal",
        )


# ============================================================================
# ScDS — labeled soundscape windows (no mixup)
# ============================================================================


class ScDS(Dataset):
    """Labeled soundscape windows (verbatim Tucker port of cell 12).

    Returns 5-tuple `(waveform_tensor, label_tensor, wt, mk, source_tag)`
    with `source_tag = "sc"`. No mixup branch — SS samples get mixed INTO
    focal via FocalDS's focal-SS mixup path instead. Only applies
    `apply_aug` (gain + noise) when `aug=True`.

    Args:
        sc_df: DataFrame with `filename`, `start_sec`, `cache_file` columns.
            One row per labeled SS window.
        Y_SC: (n_rows, num_classes) multi-hot label matrix indexed by sc_df row.
        window_samples: Samples per window (default 160000).
        num_classes: Class count (default 234).
        aug: Whether to apply augmentation.
        aug_prob: Per-augmentation firing probability.
        audio_loader: Same callable convention as FocalDS.
        sample_rate: Sample rate (default 32000).
        seed: RNG seed.
    """

    def __init__(
        self,
        sc_df: pd.DataFrame,
        Y_SC: np.ndarray,
        window_samples: int = 160_000,
        num_classes: int = 234,
        aug: bool = False,
        aug_prob: float = 0.5,
        audio_loader: Callable[..., np.ndarray] = _default_audio_loader,
        sample_rate: int = 32000,
        seed: int = 42,
    ):
        if len(sc_df) != len(Y_SC):
            raise ValueError(
                f"sc_df has {len(sc_df)} rows but Y_SC has {len(Y_SC)} rows — must match."
            )
        self.df = sc_df.reset_index(drop=True)
        self.Y_SC = Y_SC
        self.window_samples = window_samples
        self.num_classes = num_classes
        self.aug = aug
        self.aug_prob = aug_prob
        self.audio_loader = audio_loader
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        try:
            path = row.get("cache_file") or row.get("filename")
            start = int(row["start_sec"]) * self.sample_rate
            wav = self.audio_loader(path, self.window_samples, start, self.sample_rate)
        except Exception as e:
            logger.warning(f"ScDS: failed to load {row.get('filename')}: {e}")
            wav = np.zeros(self.window_samples, dtype=np.float32)

        if wav is None or wav.shape[0] < self.window_samples:
            wav = _extract_chunk(wav if wav is not None else np.zeros(0), 0, self.window_samples)
        wav = wav.astype(np.float32)

        if self.aug:
            wav = apply_aug(wav, self.rng, aug_prob=self.aug_prob)

        lb = self.Y_SC[i].astype(np.float32)
        return (
            torch.from_numpy(wav).unsqueeze(0),
            torch.from_numpy(lb),
            torch.ones(self.num_classes, dtype=torch.float32),
            torch.ones(self.num_classes, dtype=torch.float32),
            "sc",
        )


# ============================================================================
# Pseudo-label datasets (exp 033 — own-teacher PL on Tucker recipe)
# ============================================================================


class PLDS(Dataset):
    """Pseudo-labeled soundscape windows (exp 033 — own-teacher PL).

    Returns 6-tuple `(waveform_tensor, soft_label_tensor, wt, mk, source_tag, loss_weight)`
    with `source_tag = "pl"` and `loss_weight` carrying the per-sample PL weight
    (Antoine 1:1 protocol → 1.0; exp 025-style down-weighting → 0.5).

    Identical audio-loading semantics to `ScDS` — but labels are SOFT
    probabilities (in [0, 1]) read from a precomputed PL parquet, and the
    6th tuple element carries the per-sample loss weight that the training
    loop applies to the BCE branch (NOT the KD MSE branch).

    Args:
        pl_df: DataFrame with at least `filename`, `chunk_idx` columns + 234
            species columns (soft probabilities). Mirrors v1/sed_pseudo_labels.parquet
            after filtering for val-fold leakage prevention.
        species_list: Ordered list of 234 species labels — used to extract the
            label matrix from pl_df in a consistent column order. MUST match
            the model's output ordering.
        soundscape_dir: Directory containing the `train_soundscapes/*.ogg` files
            referenced by `pl_df.filename`.
        window_samples: Samples per window (default 160000).
        loss_weight: Per-sample loss weight (default 1.0 = Antoine 1:1 protocol).
        audio_loader: Same callable convention as ScDS.
        sample_rate: Sample rate (default 32000).
        aug: Whether to apply augmentation (default False — match ScDS).
        aug_prob: Per-augmentation firing probability.
        seed: RNG seed for augmentation.
    """

    def __init__(
        self,
        pl_df: pd.DataFrame,
        species_list: list[str],
        soundscape_dir: str | Path,
        window_samples: int = 160_000,
        loss_weight: float = 1.0,
        audio_loader: Callable[..., np.ndarray] = _default_audio_loader,
        sample_rate: int = 32000,
        aug: bool = False,
        aug_prob: float = 0.5,
        seed: int = 42,
    ):
        required_cols = {"filename", "chunk_idx"}
        missing_cols = required_cols - set(pl_df.columns)
        if missing_cols:
            raise ValueError(f"pl_df missing required cols: {missing_cols}")
        missing_species = [sp for sp in species_list if sp not in pl_df.columns]
        if missing_species:
            raise ValueError(
                f"pl_df missing {len(missing_species)} species cols (first 5: {missing_species[:5]})"
            )
        self.df = pl_df.reset_index(drop=True)
        self.species_list = list(species_list)
        self.num_classes = len(self.species_list)
        # Pre-extract labels matrix (fast __getitem__)
        self.labels_matrix = self.df[self.species_list].values.astype(np.float32)
        self.soundscape_dir = Path(soundscape_dir)
        self.window_samples = window_samples
        self.loss_weight = float(loss_weight)
        self.audio_loader = audio_loader
        self.sample_rate = sample_rate
        self.aug = aug
        self.aug_prob = aug_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        audio_path = str(self.soundscape_dir / row["filename"])
        start = int(row["chunk_idx"]) * 5 * self.sample_rate  # 5-sec windows
        try:
            wav = self.audio_loader(audio_path, self.window_samples, start, self.sample_rate)
        except Exception as e:
            logger.warning(f"PLDS: failed to load {row['filename']}: {e}")
            wav = np.zeros(self.window_samples, dtype=np.float32)

        if wav is None or wav.shape[0] < self.window_samples:
            wav = _extract_chunk(wav if wav is not None else np.zeros(0), 0, self.window_samples)
        wav = wav.astype(np.float32)

        if self.aug:
            wav = apply_aug(wav, self.rng, aug_prob=self.aug_prob)

        lb = self.labels_matrix[i]
        return (
            torch.from_numpy(wav).unsqueeze(0),
            torch.from_numpy(lb),
            torch.ones(
                self.num_classes, dtype=torch.float32
            ),  # wt: per-class (unused, kept for collate compat)
            torch.ones(
                self.num_classes, dtype=torch.float32
            ),  # mk: per-class (unused, kept for collate compat)
            "pl",
            self.loss_weight,  # NEW: per-sample loss weight (Antoine 1:1 → 1.0)
        )


class GTDatasetWrapper(Dataset):
    """Wraps a 5-tuple dataset (FocalDS/ScDS) to add `loss_weight=1.0` as the
    6th tuple element, matching the 6-tuple shape PLDS returns.

    Required when ConcatDataset combines GT + PL datasets — collate fn needs
    uniform tuple length across samples.

    Converts (wav, lb, wt, mk, tag) → (wav, lb, wt, mk, tag, loss_weight=1.0).
    """

    def __init__(self, dataset: Dataset, loss_weight: float = 1.0):
        self.dataset = dataset
        self.loss_weight = float(loss_weight)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int):
        item = self.dataset[i]
        # Defensive: handle both 5-tuple (FocalDS/ScDS) and 6-tuple (already wrapped)
        if len(item) == 6:
            return item
        wav, lb, wt, mk, tag = item
        return wav, lb, wt, mk, tag, self.loss_weight


# ============================================================================
# MixSamp — multi-source batch sampler (Tucker cell 12)
# ============================================================================


class MixSamp(Sampler[list[int]]):
    """Batch sampler that composes each batch from multiple data sources with
    configurable per-source shares. Verbatim port of `tuckerarrants/bc2026-
    distilled-sed` cell 12.

    Designed to be used with `torch.utils.data.ConcatDataset` containing N
    underlying datasets concatenated in order. Each batch is built by sampling
    `per_src[i]` indices from source `i` (with replacement, uniform over its
    range), translating them by the source's offset within the concat, and
    shuffling the combined batch.

    For Tucker's defaults (`shares={"focal": 0.9, "sc": 0.1}, bs=64`), this
    yields **58 focal + 6 SC** indices per batch (per audit § 3 verification).

    Args:
        sizes: List of per-source dataset sizes, in the order they appear in
            the concat dataset. e.g. `[len(focal_ds), len(sc_ds)]`.
        names: List of per-source names, same order as `sizes`. Used to look up
            shares.
        shares: Per-source share dict (`name → fraction`). Fractions that don't
            match a key default to 0.0. The shares need not sum to 1.0; the
            sampler rounds each to an integer count, then distributes any
            rounding remainder to the largest-count source.
        bs: Batch size (total indices per batch).
        nst: Number of steps per epoch (batches yielded by `__iter__`).
        seed: RNG seed. **Subsequent epochs should rebuild the sampler with a
            shifted seed** (Tucker's training loop uses `seed=42 + ep`) so the
            batch composition varies — this Sampler doesn't auto-shift.

    Notes:
        - Sampling is with-replacement per source. Each draw is uniform on
          `[0, sizes[i])`, then translated by `offsets[i] = sum(sizes[:i])`
          so the index points into the concat dataset.
        - `__len__` returns `nst` (number of batches). The DataLoader uses this
          to size the epoch.
        - Tucker's `nst = max(100, sum(sizes) // bs)`. We don't impose that
          floor here — the caller sets `nst` explicitly (typically in the
          DataModule's `train_dataloader` setup).
        - Sources with `n=0` or `size=0` are skipped without raising — useful
          when disabling a source via `shares` (e.g. `shares["sc"]=0.0`).

    Reference:
    - analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md (§ 3)
    - vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 12)
    """

    def __init__(
        self,
        sizes: list[int],
        names: list[str],
        shares: dict[str, float],
        bs: int,
        nst: int,
        seed: int = 0,
    ):
        if len(sizes) != len(names):
            raise ValueError(f"sizes ({len(sizes)}) and names ({len(names)}) must have same length")
        if bs <= 0:
            raise ValueError(f"bs must be > 0, got {bs}")
        if nst < 0:
            raise ValueError(f"nst must be >= 0, got {nst}")

        self.sizes = list(sizes)
        self.names = list(names)
        self.shares = dict(shares)
        self.bs = int(bs)
        self.nst = int(nst)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        # Per-source batch counts. Round each share × bs to nearest int (with
        # a floor of 1 for any source that has ANY share > 0). Then distribute
        # any rounding remainder to the largest source.
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        # If a share is exactly 0 the user is opting that source out — keep 0
        # (override the max(1, ...) floor for the explicit-zero case).
        per_src = [
            0 if shares.get(n, 0.0) == 0.0 else c for n, c in zip(names, per_src, strict=False)
        ]

        total = sum(per_src)
        if total != bs:
            # Distribute the rounding remainder to the source with the largest
            # current count. With `0 == 0`, this also handles the case where
            # all shares rounded to 0 (rare; caller responsibility).
            idx = int(np.argmax(per_src)) if per_src else 0
            per_src[idx] += bs - total
        self.per_src = per_src

        # Offsets into the concat dataset's linear index space:
        # `concat[offset_i + j]` is the j-th sample of source i.
        self.offsets = [0]
        for s in self.sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)

    def __len__(self) -> int:
        return self.nst

    def __iter__(self) -> Iterator[list[int]]:
        for _ in range(self.nst):
            batch: list[int] = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src, strict=False):
                if n <= 0 or size <= 0:
                    continue
                idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch

    def extra_repr(self) -> str:
        per_src_str = ", ".join(f"{n}={c}" for n, c in zip(self.names, self.per_src, strict=False))
        return (
            f"sizes={self.sizes}, shares={self.shares}, bs={self.bs}, "
            f"nst={self.nst}, per_src=[{per_src_str}], seed={self.seed}"
        )
