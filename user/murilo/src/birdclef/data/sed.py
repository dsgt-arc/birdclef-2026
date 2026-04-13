"""SED data module with configurable windows for Sound Event Detection training.

Provides datasets that load audio windows (5-10s). Focal recordings use random
crops; soundscape segments use context-padded windows centered on the labeled
5s segment.

Supports both fraction-based splitting (90/10) and fold-based splitting
(MultiLabelStratifiedGroupKFold for 4-fold cross-validation).

Used by Experiments 016, 017, 019 (SED architecture).
"""

import json
import logging
import random as py_random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .dataset import _load_audio_chunk, build_species_list
from .soundscape import (
    FocalDataset234,
    build_full_species_list,
    load_soundscape_labels,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Split helpers (same logic as MixedDataModule / StudentDataModule)
# ============================================================================


def _split_focal_dataframe(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split of focal data by primary_label.

    Supports three modes:
    - 80/10/10 (test_fraction>0, val_fraction>0): train/val/test split
    - 90/10 (test_fraction=0, val_fraction>0): train/val only
    - 100/0/0 (val_fraction=0): fulldata training, all samples in train
    """
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    has_test = test_fraction > 0
    has_val = val_fraction > 0

    # Fulldata mode: everything goes to train
    if not has_val:
        return df, df.iloc[:0], df.iloc[:0]

    for _species, group in df.groupby("primary_label"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n = len(indices)

        if has_test:
            # Three-way split (original 80/10/10 behavior)
            if n >= 3:
                n_test = max(1, int(n * test_fraction))
                n_val = max(1, int(n * val_fraction))
                if n_test + n_val >= n:
                    n_test = 1
                    n_val = 1
                test_indices.extend(indices[:n_test])
                val_indices.extend(indices[n_test : n_test + n_val])
                train_indices.extend(indices[n_test + n_val :])
            elif n == 2:
                train_indices.append(indices[0])
                val_indices.append(indices[1])
                test_indices.append(indices[0])
            else:
                train_indices.append(indices[0])
                val_indices.append(indices[0])
                test_indices.append(indices[0])
        else:
            # Two-way split (90/10 train/val only)
            if n >= 2:
                n_val = max(1, int(n * val_fraction))
                if n_val >= n:
                    n_val = 1
                val_indices.extend(indices[:n_val])
                train_indices.extend(indices[n_val:])
            else:
                # Single sample: must appear in both for species coverage
                train_indices.append(indices[0])
                val_indices.append(indices[0])

    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[val_indices].reset_index(drop=True),
        df.loc[test_indices].reset_index(drop=True)
        if test_indices
        else df.iloc[:0].reset_index(drop=True),
    )


def _split_soundscape_by_file(
    labels_df: pd.DataFrame,
    soundscape_val_files: int = 8,
    soundscape_test_files: int = 8,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split soundscape labels by file into train/val/test.

    When soundscape_test_files=0, performs a two-way split (train/val only).
    Returns an empty test DataFrame to preserve the 3-tuple interface.
    """
    rng = np.random.RandomState(seed)

    unique_files = sorted(labels_df["filename"].unique())
    rng.shuffle(unique_files)

    n_files = len(unique_files)
    n_test = min(soundscape_test_files, n_files // 4)
    n_val = min(soundscape_val_files, n_files // 4)

    test_files = set(unique_files[:n_test])
    val_files = set(unique_files[n_test : n_test + n_val])
    train_files = set(unique_files[n_test + n_val :])

    train_df = labels_df[labels_df["filename"].isin(train_files)].reset_index(drop=True)
    val_df = labels_df[labels_df["filename"].isin(val_files)].reset_index(drop=True)
    test_df = labels_df[labels_df["filename"].isin(test_files)].reset_index(drop=True)

    logger.info(
        f"Soundscape split: {len(train_files)} train files ({len(train_df)} segments), "
        f"{len(val_files)} val files ({len(val_df)} segments), "
        f"{len(test_files)} test files ({len(test_df)} segments)"
    )

    return train_df, val_df, test_df


def _split_by_fold(
    focal_df: pd.DataFrame,
    soundscape_df: pd.DataFrame,
    fold_id: int,
    fold_assignments_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split focal and soundscape data by precomputed fold assignments.

    Rows where fold == fold_id go to val; all others go to train.
    No test set in fold mode.

    Args:
        focal_df: Focal recordings DataFrame (has 'filename' column).
        soundscape_df: Soundscape labels DataFrame (has 'filename' column).
        fold_id: Which fold to use as validation (0-3).
        fold_assignments_path: Path to fold_assignments.json.

    Returns:
        (focal_train, focal_val, sc_train, sc_val) DataFrames.
    """
    with open(fold_assignments_path) as f:
        assignments = json.load(f)

    focal_assignments = assignments["focal"]
    sc_assignments = assignments["soundscape"]

    # Focal: look up each row's filename
    focal_folds = focal_df["filename"].map(focal_assignments).astype(int)
    focal_val_mask = focal_folds == fold_id
    focal_train = focal_df[~focal_val_mask].reset_index(drop=True)
    focal_val = focal_df[focal_val_mask].reset_index(drop=True)

    # Soundscape: look up by filename (all segments from same file share fold)
    sc_folds = soundscape_df["filename"].map(sc_assignments).astype(int)
    sc_val_mask = sc_folds == fold_id
    sc_train = soundscape_df[~sc_val_mask].reset_index(drop=True)
    sc_val = soundscape_df[sc_val_mask].reset_index(drop=True)

    n_sc_train_files = sc_train["filename"].nunique() if len(sc_train) > 0 else 0
    n_sc_val_files = sc_val["filename"].nunique() if len(sc_val) > 0 else 0
    logger.info(
        f"Fold {fold_id} split: focal {len(focal_train)} train / {len(focal_val)} val, "
        f"soundscape {n_sc_train_files} train files ({len(sc_train)} segments) / "
        f"{n_sc_val_files} val files ({len(sc_val)} segments)"
    )

    return focal_train, focal_val, sc_train, sc_val


# ============================================================================
# Dataset classes
# ============================================================================


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS timestamp to seconds."""
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


class SEDFocalDataset(Dataset):
    """Focal recordings with random 10s crops for SED training.

    Unlike BirdCLEFDataset which pre-chunks into fixed 5s windows, this dataset
    stores per-recording entries and randomly crops a window at training time.
    Recordings shorter than window_duration are zero-padded.

    Returns:
        (waveform_or_features, labels) — labels is (234,) multi-hot float tensor.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        metadata_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        window_duration: float = 10.0,
        transform=None,
        rms_sampling: bool = False,
    ):
        self.audio_dir = Path(audio_dir).expanduser()
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_samples = int(window_duration * sample_rate)
        self.transform = transform
        self.rms_sampling = rms_sampling

        # Store per-recording metadata (not pre-chunked)
        self.recordings = []
        skipped = 0

        for _, row in metadata_df.iterrows():
            audio_path = self.audio_dir / row["filename"]
            if not audio_path.exists():
                skipped += 1
                continue

            try:
                info = sf.info(str(audio_path))
                total_samples = int(info.frames * sample_rate / info.samplerate)
            except Exception:
                skipped += 1
                continue

            label_vec = self._build_label_vec(row)
            primary = row.get("primary_label", "")
            primary_idx = self.species_to_idx.get(primary, 0)
            self.recordings.append((str(audio_path), total_samples, label_vec, primary_idx))

        if skipped > 0:
            logger.warning(f"Skipped {skipped} recordings (file not found or unreadable)")
        logger.info(
            f"SEDFocalDataset: {len(self.recordings)} recordings "
            f"({self.num_classes} classes, {window_duration}s windows)"
        )

    def _build_label_vec(self, row):
        """Build multi-hot label vector from metadata row."""
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        primary = row.get("primary_label", "")
        if primary in self.species_to_idx:
            label_vec[self.species_to_idx[primary]] = 1.0

        secondary = row.get("secondary_labels", "")
        if isinstance(secondary, str) and secondary.strip() and secondary != "[]":
            secondary = secondary.strip("[]' ")
            for s in secondary.replace("'", "").replace('"', "").split(","):
                s = s.strip()
                if s in self.species_to_idx:
                    label_vec[self.species_to_idx[s]] = 1.0

        return label_vec

    def __len__(self):
        return len(self.recordings)

    def _rms_start(self, audio_path: str, total_samples: int) -> int:
        """Find the start position of the loudest window via RMS energy.

        Scans with 1-second stride (matching Salman's implementation).
        Falls back to random start if the file is shorter than the window.
        """
        if total_samples <= self.window_samples:
            return 0

        # Load the full recording for RMS scanning
        full_audio = _load_audio_chunk(audio_path, self.sample_rate)
        if full_audio.shape[0] <= self.window_samples:
            return 0

        stride = self.sample_rate  # 1-second stride
        best_rms = 0.0
        best_start = 0
        audio_np = full_audio.numpy()

        for start in range(0, len(audio_np) - self.window_samples + 1, stride):
            window = audio_np[start : start + self.window_samples]
            rms = float(np.sqrt(np.mean(window**2)))
            if rms > best_rms:
                best_rms = rms
                best_start = start

        return best_start

    def __getitem__(self, idx):
        audio_path, total_samples, label_vec, primary_idx = self.recordings[idx]

        if self.rms_sampling:
            start_sample = self._rms_start(audio_path, total_samples)
        elif total_samples > self.window_samples:
            start_sample = py_random.randint(0, total_samples - self.window_samples)
        else:
            start_sample = 0

        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=start_sample,
            num_samples=self.window_samples,
        )

        # Pad if recording is shorter than window
        if waveform.shape[0] < self.window_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.window_samples - waveform.shape[0])
            )

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label_vec.clone(), primary_idx


class SEDSoundscapeDataset(Dataset):
    """Soundscape segments with 10s context window for SED training.

    Loads 10s of audio centered on each labeled 5s segment (2.5s context
    before and after). Edge segments are zero-padded when insufficient
    audio context is available.

    Returns:
        (waveform_or_features, labels) — labels is (234,) multi-hot float tensor.
    """

    def __init__(
        self,
        soundscape_dir: str | Path,
        labels_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        target_duration: float = 5.0,
        context_duration: float = 10.0,
        transform=None,
    ):
        self.soundscape_dir = Path(soundscape_dir)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.context_duration = context_duration
        self.context_samples = int(context_duration * sample_rate)
        self.extra_context_samples = int((context_duration - target_duration) / 2 * sample_rate)
        self.transform = transform

        # Build chunk index
        self.chunks = []
        skipped = 0

        for _, row in labels_df.iterrows():
            audio_path = self.soundscape_dir / row["filename"]
            if not audio_path.exists():
                skipped += 1
                continue

            start_sec = _parse_timestamp(row["start"])
            start_sample = int(start_sec * sample_rate)

            # Build multi-hot label vector
            label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
            primary_idx = 0  # default fallback
            for i, species_id in enumerate(str(row["primary_label"]).split(";")):
                species_id = species_id.strip()
                if species_id in self.species_to_idx:
                    label_vec[self.species_to_idx[species_id]] = 1.0
                    if i == 0:  # first species is primary
                        primary_idx = self.species_to_idx[species_id]

            self.chunks.append((str(audio_path), start_sample, label_vec, primary_idx))

        if skipped > 0:
            logger.warning(f"Skipped {skipped} segments (audio file not found)")
        logger.info(
            f"SEDSoundscapeDataset: {len(self.chunks)} segments from "
            f"{labels_df['filename'].nunique()} files "
            f"({self.num_classes} classes, {context_duration}s context)"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, segment_start_sample, label_vec, primary_idx = self.chunks[idx]

        # Load 10s centered on the 5s segment (2.5s before + 2.5s after)
        load_start = max(0, segment_start_sample - self.extra_context_samples)

        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=load_start,
            num_samples=self.context_samples,
        )

        # Zero-pad if at file boundary
        if waveform.shape[0] < self.context_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.context_samples - waveform.shape[0])
            )

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label_vec.clone(), primary_idx


# ============================================================================
# Pseudo-label support (exp 025)
# ============================================================================


class PseudoLabelSoundscapeDataset(Dataset):
    """Dataset for pseudo-labeled soundscape segments.

    Loads 5-second soundscape audio windows with soft pseudo-label probabilities
    from a parquet file (generated by exp 025 generate_pseudo_labels.py).

    Returns 4-tuple: (waveform, soft_labels, primary_idx, loss_weight)
    """

    def __init__(
        self,
        soundscape_dir: str | Path,
        pseudo_labels_df: pd.DataFrame,
        species_list: list[str],
        sample_rate: int = 32000,
        target_duration: float = 5.0,
        context_duration: float = 10.0,
        loss_weight: float = 0.5,
        transform=None,
    ):
        super().__init__()
        self.soundscape_dir = Path(soundscape_dir)
        self.species_list = species_list
        self.num_classes = len(species_list)
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.context_duration = context_duration
        self.context_samples = int(context_duration * sample_rate)
        self.extra_context_samples = int((context_duration - target_duration) / 2 * sample_rate)
        self.loss_weight = loss_weight
        self.transform = transform

        # Pre-extract labels matrix for fast __getitem__
        self.labels_matrix = pseudo_labels_df[species_list].values.astype(np.float32)

        # Build segment index: (audio_path, start_sample) per row
        self.segments = []
        for _, row in pseudo_labels_df.iterrows():
            audio_path = str(self.soundscape_dir / row["filename"])
            start_sec = int(row["chunk_idx"]) * target_duration
            start_sample = int(start_sec * sample_rate)
            self.segments.append((audio_path, start_sample))

        logger.info(
            f"PseudoLabelSoundscapeDataset: {len(self.segments)} segments, "
            f"loss_weight={loss_weight}"
        )

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        audio_path, segment_start_sample = self.segments[idx]

        # Load context window centered on target (same as SEDSoundscapeDataset)
        load_start = max(0, segment_start_sample - self.extra_context_samples)
        waveform = _load_audio_chunk(
            audio_path,
            self.sample_rate,
            start_sample=load_start,
            num_samples=self.context_samples,
        )

        # Zero-pad if at file boundary
        if waveform.shape[0] < self.context_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.context_samples - waveform.shape[0])
            )

        if self.transform is not None:
            waveform = self.transform(waveform)

        # Soft labels from pre-extracted matrix
        label_vec = torch.from_numpy(self.labels_matrix[idx].copy())
        primary_idx = int(label_vec.argmax())

        return waveform, label_vec, primary_idx, self.loss_weight


class GTDatasetWrapper(Dataset):
    """Wraps a 3-tuple dataset to add loss_weight=1.0 as 4th element.

    Converts (waveform, labels, primary_idx) → (waveform, labels, primary_idx, 1.0)
    for compatibility with the pseudo-label training pipeline.
    """

    def __init__(self, dataset: Dataset, loss_weight: float = 1.0):
        self.dataset = dataset
        self.loss_weight = loss_weight

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, labels, primary_idx = self.dataset[idx]
        return waveform, labels, primary_idx, self.loss_weight


class StratifiedBatchSampler(torch.utils.data.Sampler):
    """Yields batches with a fixed ratio of GT to pseudo-labeled samples.

    Given a ConcatDataset([gt_dataset, pl_dataset]) where gt_dataset
    occupies indices [0, n_gt) and pl_dataset occupies [n_gt, n_gt+n_pl),
    each batch contains half_batch samples from each pool.
    """

    def __init__(self, n_gt: int, n_pl: int, batch_size: int, drop_last: bool = True):
        self.n_gt = n_gt
        self.n_pl = n_pl
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.drop_last = drop_last

    def __iter__(self):
        gt_indices = torch.randperm(self.n_gt).tolist()
        pl_indices = (torch.randperm(self.n_pl) + self.n_gt).tolist()

        # Number of complete batches from each pool
        n_batches = min(
            len(gt_indices) // self.half_batch,
            len(pl_indices) // self.half_batch,
        )

        for i in range(n_batches):
            start = i * self.half_batch
            batch = (
                gt_indices[start : start + self.half_batch]
                + pl_indices[start : start + self.half_batch]
            )
            yield batch

    def __len__(self):
        return min(
            self.n_gt // self.half_batch,
            self.n_pl // self.half_batch,
        )


# ============================================================================
# DataModule
# ============================================================================


class SEDDataModule(pl.LightningDataModule):
    """DataModule for SED training with 10-second windows.

    Combines focal recordings (random 10s crops) and labeled soundscape
    segments (10s context windows centered on 5s targets).

    Training set: focal (234-dim) + soundscape (234-dim)
    Val/test sets: focal (234-dim) + soundscape (234-dim)
    """

    def __init__(
        self,
        data_dir: str | Path,
        taxonomy_csv: str | Path,
        sample_rate: int = 32000,
        window_duration: float = 10.0,
        batch_size: int = 64,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        soundscape_val_files: int = 8,
        soundscape_test_files: int = 8,
        seed: int = 42,
        transform=None,
        eval_transform=None,
        fold_id: int | None = None,
        fold_assignments_path: str | Path | None = None,
        # Pseudo-label support (exp 025)
        pseudo_label_path: str | Path | None = None,
        pseudo_label_weight: float = 0.5,
        pseudo_label_batch_fraction: float = 0.5,
        # Dual validation (exp 026): separate focal and soundscape val dataloaders
        dual_validation: bool = False,
        # RMS sampling (exp 027 Phase 4): select loudest window instead of random
        rms_sampling: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform", "eval_transform"])
        self.data_dir = Path(data_dir)
        self.taxonomy_csv = Path(taxonomy_csv)
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.soundscape_val_files = soundscape_val_files
        self.soundscape_test_files = soundscape_test_files
        self.seed = seed
        self.transform = transform
        # eval_transform: separate transform for val/test datasets.
        # When None (default), uses the same transform for all splits.
        # Use this when training on raw waveforms but val/test need mel spectrograms.
        self.eval_transform = eval_transform if eval_transform is not None else transform
        self.fold_id = fold_id
        self.fold_assignments_path = fold_assignments_path
        # Pseudo-label support (exp 025)
        self.pseudo_label_path = Path(pseudo_label_path) if pseudo_label_path else None
        self.pseudo_label_weight = pseudo_label_weight
        self.pseudo_label_batch_fraction = pseudo_label_batch_fraction
        self.dual_validation = dual_validation
        self.rms_sampling = rms_sampling

        # Build species lists
        self.species_list = build_full_species_list(taxonomy_csv)
        self.num_classes = len(self.species_list)
        self.species_206 = build_species_list(self.data_dir / "train.csv")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # Dual validation: separate datasets for focal and soundscape
        self.focal_val_dataset = None
        self.soundscape_val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        # --- Focal data (206 species, remapped to 234) ---
        focal_csv = self.data_dir / "train.csv"
        focal_audio_dir = self.data_dir / "train_audio"
        focal_df = pd.read_csv(focal_csv)

        # --- Soundscape labels ---
        soundscape_dir = self.data_dir / "train_soundscapes"
        soundscape_labels_csv = self.data_dir / "train_soundscapes_labels.csv"
        soundscape_df = load_soundscape_labels(soundscape_labels_csv)

        # --- Split strategy ---
        use_fold = self.fold_id is not None and self.fold_assignments_path is not None

        if use_fold:
            # Fold-based splitting (4-fold CV)
            has_test = False
            has_val = True
            fulldata = False
            focal_train_df, focal_val_df, sc_train_df, sc_val_df = _split_by_fold(
                focal_df, soundscape_df, self.fold_id, self.fold_assignments_path
            )
            focal_test_df = focal_df.iloc[:0]
            sc_test_df = soundscape_df.iloc[:0]
        else:
            # Fraction-based splitting (legacy)
            has_test = self.test_fraction > 0
            has_val = self.val_fraction > 0
            fulldata = not has_val

            focal_train_df, focal_val_df, focal_test_df = _split_focal_dataframe(
                focal_df, self.val_fraction, self.test_fraction, self.seed
            )

            soundscape_val_files = 0 if fulldata else self.soundscape_val_files
            soundscape_test_files = self.soundscape_test_files if has_test else 0
            sc_train_df, sc_val_df, sc_test_df = _split_soundscape_by_file(
                soundscape_df,
                soundscape_val_files,
                soundscape_test_files,
                self.seed + 1,
            )

            if fulldata:
                logger.info(f"Focal fulldata: {len(focal_train_df)} train (100%, no val/test)")
            else:
                logger.info(
                    f"Focal split: {len(focal_train_df)} train, {len(focal_val_df)} val"
                    + (f", {len(focal_test_df)} test" if has_test else " (no test holdout)")
                )

        focal_train_206 = SEDFocalDataset(
            audio_dir=str(focal_audio_dir),
            metadata_df=focal_train_df,
            species_list=self.species_206,
            sample_rate=self.sample_rate,
            window_duration=self.window_duration,
            transform=self.transform,
            rms_sampling=self.rms_sampling,
        )
        # Remap 206→234
        focal_train = FocalDataset234(focal_train_206, self.species_206, self.species_list)

        focal_val = None
        if not fulldata:
            focal_val_206 = SEDFocalDataset(
                audio_dir=str(focal_audio_dir),
                metadata_df=focal_val_df,
                species_list=self.species_206,
                sample_rate=self.sample_rate,
                window_duration=self.window_duration,
                transform=self.eval_transform,
            )
            focal_val = FocalDataset234(focal_val_206, self.species_206, self.species_list)

        if has_test:
            focal_test_206 = SEDFocalDataset(
                audio_dir=str(focal_audio_dir),
                metadata_df=focal_test_df,
                species_list=self.species_206,
                sample_rate=self.sample_rate,
                window_duration=self.window_duration,
                transform=self.eval_transform,
            )
            focal_test = FocalDataset234(focal_test_206, self.species_206, self.species_list)

        # --- Labeled soundscape data (234 species, context windows) ---
        sc_train = SEDSoundscapeDataset(
            soundscape_dir=str(soundscape_dir),
            labels_df=sc_train_df,
            species_list=self.species_list,
            sample_rate=self.sample_rate,
            target_duration=5.0,
            context_duration=self.window_duration,
            transform=self.transform,
        )

        # --- Combine ---
        gt_train = ConcatDataset([focal_train, sc_train])

        if not fulldata:
            sc_val = SEDSoundscapeDataset(
                soundscape_dir=str(soundscape_dir),
                labels_df=sc_val_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                target_duration=5.0,
                context_duration=self.window_duration,
                transform=self.eval_transform,
            )
            if self.dual_validation:
                # Dual validation: keep focal and soundscape val separate
                self.focal_val_dataset = focal_val
                self.soundscape_val_dataset = sc_val
                # Combined dataset for backward compat (used as fallback)
                self.val_dataset = ConcatDataset([focal_val, sc_val])
            else:
                self.val_dataset = ConcatDataset([focal_val, sc_val])
        else:
            self.val_dataset = None

        if has_test:
            sc_test = SEDSoundscapeDataset(
                soundscape_dir=str(soundscape_dir),
                labels_df=sc_test_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                target_duration=5.0,
                context_duration=self.window_duration,
                transform=self.eval_transform,
            )
            self.test_dataset = ConcatDataset([focal_test, sc_test])
        else:
            self.test_dataset = None

        # --- Pseudo-label integration (exp 025) ---
        self._use_stratified = False
        if self.pseudo_label_path is not None:
            pl_df = pd.read_parquet(self.pseudo_label_path)
            logger.info(
                f"Loaded pseudo-labels: {len(pl_df)} segments from {self.pseudo_label_path}"
            )

            # Fold leakage prevention: exclude validation soundscape files
            if use_fold and has_val:
                val_files = set(sc_val_df["filename"].unique())
                before = len(pl_df)
                pl_df = pl_df[~pl_df["filename"].isin(val_files)].reset_index(drop=True)
                logger.info(
                    f"  Excluded {before - len(pl_df)} val segments "
                    f"(fold {self.fold_id}), {len(pl_df)} remain"
                )

            pl_dataset = PseudoLabelSoundscapeDataset(
                soundscape_dir=str(soundscape_dir),
                pseudo_labels_df=pl_df,
                species_list=self.species_list,
                sample_rate=self.sample_rate,
                target_duration=5.0,
                context_duration=self.window_duration,
                loss_weight=self.pseudo_label_weight,
                transform=self.transform,
            )

            # Wrap GT with loss_weight=1.0, combine with PL
            gt_wrapped = GTDatasetWrapper(gt_train, loss_weight=1.0)
            self.train_dataset = ConcatDataset([gt_wrapped, pl_dataset])
            self._n_gt = len(gt_wrapped)
            self._n_pl = len(pl_dataset)
            self._use_stratified = True

            logger.info(
                f"  Pseudo-label training: {self._n_gt} GT + {self._n_pl} PL "
                f"= {len(self.train_dataset)} total"
            )
            logger.info(
                f"  Batch composition: {self.pseudo_label_batch_fraction:.0%} PL, "
                f"loss_weight={self.pseudo_label_weight}"
            )
        else:
            self.train_dataset = gt_train

        # --- Logging ---
        if fulldata:
            logger.info(f"SED datasets: {len(self.train_dataset)} train (fulldata, no val/test)")
            logger.info(f"  Focal: {len(focal_train)}")
            logger.info(f"  Soundscape: {len(sc_train)}")
        else:
            logger.info(
                f"SED datasets: {len(self.train_dataset)} train, "
                f"{len(self.val_dataset)} val"
                + (f", {len(self.test_dataset)} test" if self.test_dataset else " (no test)")
            )
            logger.info(
                f"  Focal: {len(focal_train)}+{len(focal_val)}"
                + (f"+{len(focal_test)}" if has_test else "")
            )
            logger.info(
                f"  Soundscape: {len(sc_train)}+{len(sc_val)}"
                + (f"+{len(sc_test)}" if has_test else "")
            )

    def train_dataloader(self):
        if self._use_stratified:
            sampler = StratifiedBatchSampler(
                n_gt=self._n_gt,
                n_pl=self._n_pl,
                batch_size=self.batch_size,
                drop_last=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
                prefetch_factor=4 if self.num_workers > 0 else None,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
        )

    def _val_loader(self, dataset):
        """Create a validation DataLoader for a given dataset."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        if self.dual_validation and self.focal_val_dataset is not None:
            # Return [focal_loader, soundscape_loader] for separate metric tracking
            return [
                self._val_loader(self.focal_val_dataset),
                self._val_loader(self.soundscape_val_dataset),
            ]
        return self._val_loader(self.val_dataset)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )
