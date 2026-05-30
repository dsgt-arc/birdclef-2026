"""LightningDataModule + fold-construction helpers for exp 032 (Tucker
distilled-SED replication).

This module owns the fold construction (StratifiedKFold focal + GroupKFold SS),
rare-species upsampling, and SS label-matrix assembly described in the
2026-05-23 audit § 1. It wraps the dataset classes from
`distilled_sed_dataset` (FocalDS / ScDS / MixSamp) into a LightningDataModule
that train.py drives.

Key behaviors (per audit § 1):
- **Focal fold**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
  on `primary_label`, deduplicating by `original_idx` first so any pre-existing
  duplication doesn't bias the split.
- **SS fold**: `GroupKFold(n_splits=5)` on `filename`. All 66 labeled SS files
  participate (S22 included); the `non_s22_mask` is computed for the
  inference-time metric only, NOT for the split.
- **Rare-species upsampling**: focal species with fewer than `MIN_SAMPLE=20`
  rows are duplicated until they have ~20 exemplars. **Applied AFTER fold
  assignment** so each upsampled copy inherits its source row's fold (no
  cross-fold leak).
- **SS label assembly**: enumerate 12 × 5s windows per labeled SS file (66 ×
  12 = 792 windows). Build `Y_SC` by matching `(filename, start_sec)` tuples
  against `train_soundscapes_labels.csv`. Windows that don't match → all-zero
  label rows (53 such windows per audit § 1.4 — included in training as
  effectively all-negative, matching Tucker's default).

Reference:
- analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md (§§ 1, 4)
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 7)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader

from .distilled_sed_dataset import (
    PLDS,
    ClassBalancedMixSamp,
    FocalDS,
    GTDatasetWrapper,
    MixSamp,
    ScDS,
    _default_audio_loader,
)
from .xcpretrain_sampler import compute_class_weights

logger = logging.getLogger(__name__)


# ============================================================================
# Focal fold construction + rare-species upsampling
# ============================================================================


def build_focal_folds(
    focal_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    min_sample: int = 20,
) -> pd.DataFrame:
    """Assign 5-fold stratified folds on focal `primary_label` + upsample rare
    species AFTER fold assignment.

    Audit § 1.1 + § 1.3.

    Args:
        focal_df: DataFrame with at least `primary_label` and `filename` columns.
            Must NOT contain a `fold` or `original_idx` column (we add them).
        n_folds: Number of stratified folds (default 5).
        seed: `random_state` for StratifiedKFold (default 42, Tucker's).
        min_sample: Minimum exemplar count per species; species with fewer rows
            get duplicated until they have ~min_sample (default 20, Tucker's).

    Returns:
        A new DataFrame with `fold` (int 0..n_folds-1) and `original_idx`
        (int 0..N_original-1) columns added. Length may exceed original
        because rare species are duplicated. Upsampled duplicates share the
        same `fold` as their source row (no cross-fold leak).

    Notes:
        - StratifiedKFold runs on `drop_duplicates("original_idx")` first to
          ensure each original recording is in exactly one fold. Then folds
          are merged back via `original_idx`.
        - Upsampling appends new rows to the end. The returned DataFrame has
          contiguous index but the `original_idx` column tracks provenance.
    """
    if "fold" in focal_df.columns:
        raise ValueError("focal_df already has a 'fold' column.")
    if "original_idx" in focal_df.columns:
        raise ValueError("focal_df already has an 'original_idx' column.")
    if "primary_label" not in focal_df.columns:
        raise ValueError("focal_df missing required 'primary_label' column.")

    df = focal_df.copy().reset_index(drop=True)
    df["original_idx"] = df.index

    # --- StratifiedKFold on unique original_idx rows ---
    # Tucker's pattern: drop_duplicates by original_idx first, split, merge back
    unique_df = df.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    unique_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(unique_df, unique_df["primary_label"])):
        unique_df.loc[val_idx, "fold"] = fold_idx
    df = df.merge(unique_df[["original_idx", "fold"]], on="original_idx", how="left")

    # --- Rare-species upsampling AFTER fold assignment ---
    counts = df["primary_label"].value_counts()
    rare_species = counts[counts < min_sample].index
    extras = []
    for sp in rare_species:
        sp_rows = df[df["primary_label"] == sp]
        n_copies = int(np.ceil(min_sample / len(sp_rows))) - 1
        for _ in range(n_copies):
            extras.append(sp_rows)
    if extras:
        df = pd.concat([df] + extras, ignore_index=True)

    logger.info(
        f"build_focal_folds: {len(unique_df)} unique recordings → {len(df)} "
        f"after upsampling (n_folds={n_folds}, min_sample={min_sample}, "
        f"rare_species={len(rare_species)})"
    )
    return df


# ============================================================================
# SS cache construction + fold + label matrix
# ============================================================================


def build_sc_cache_meta(
    labeled_files: list[str],
    file_duration_sec: int = 60,
    window_duration_sec: int = 5,
) -> pd.DataFrame:
    """Enumerate (filename, start_sec) 5s windows for all labeled SS files.

    For BirdCLEF 2026: 60-second files split into 12 × 5s windows.

    Args:
        labeled_files: List of SS filenames that appear in
            `train_soundscapes_labels.csv` (i.e. have ANY label entries).
        file_duration_sec: Total file duration (default 60).
        window_duration_sec: Window length (default 5).

    Returns:
        DataFrame with columns `filename`, `start_sec`. Row count =
        `len(labeled_files) × (file_duration_sec // window_duration_sec)`.
        For 66 files × 12 windows → 792 rows (audit § 1.4 verified count).
    """
    n_windows_per_file = file_duration_sec // window_duration_sec
    start_secs = list(range(0, file_duration_sec, window_duration_sec))
    rows = []
    for fn in labeled_files:
        for ss in start_secs:
            rows.append({"filename": fn, "start_sec": ss})
    df = pd.DataFrame(rows)
    logger.info(
        f"build_sc_cache_meta: {len(labeled_files)} files × {n_windows_per_file} "
        f"windows = {len(df)} SS windows"
    )
    return df


def build_sc_label_matrix(
    sc_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    label2idx: dict[str, int],
    num_classes: int,
) -> np.ndarray:
    """Construct the per-window multi-hot label matrix Y_SC.

    Audit § 1.4. For each row in `sc_df`, look up `(filename, start_sec)` in
    `labels_df`. If matched, set the corresponding label indices to 1.
    Unmatched rows produce all-zero labels (the 53 "all-zero" windows per
    audit § 1.4 — included in training as effectively all-negative).

    Args:
        sc_df: SS window DataFrame with `filename`, `start_sec` columns.
        labels_df: Loaded `train_soundscapes_labels.csv`. Expected columns:
            `filename`, `start` (timedelta string like "00:00:05"), `end`,
            `primary_label` (semicolon-separated species string).
        label2idx: Species name → label index map.
        num_classes: Number of output classes (e.g. 234).

    Returns:
        Y_SC of shape (len(sc_df), num_classes), dtype float32, with
        multi-hot labels for matched rows and zeros elsewhere.
    """
    labels_df = labels_df.drop_duplicates().copy()
    labels_df["start_sec"] = pd.to_timedelta(labels_df["start"]).dt.total_seconds().astype(int)

    Y_SC = np.zeros((len(sc_df), num_classes), dtype=np.float32)
    # Build a dict keyed by (filename, start_sec) → list of species strings
    label_key: dict[tuple[str, int], list[str]] = {}
    for _, row in labels_df.iterrows():
        key = (row["filename"], int(row["start_sec"]))
        species_str = str(row["primary_label"])
        species = [s.strip() for s in species_str.split(";")]
        label_key.setdefault(key, []).extend(species)

    n_matched = 0
    for i, row in sc_df.reset_index(drop=True).iterrows():
        key = (row["filename"], int(row["start_sec"]))
        species_list = label_key.get(key)
        if species_list is None:
            continue
        any_matched = False
        for sp in species_list:
            if sp in label2idx:
                Y_SC[i, label2idx[sp]] = 1.0
                any_matched = True
        if any_matched:
            n_matched += 1

    logger.info(
        f"build_sc_label_matrix: {n_matched}/{len(sc_df)} SS windows matched "
        f"to label entries; {len(sc_df) - n_matched} all-zero windows."
    )
    return Y_SC


def build_sc_folds(
    sc_df: pd.DataFrame,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Assign GroupKFold folds to SS windows by `filename` (file-level split).

    Audit § 1.2. All 12 windows of a given file share a fold. All labeled
    files participate (Tucker's choice — no S22 holdout at split time).

    Args:
        sc_df: SS window DataFrame with `filename` column.
        n_folds: Number of folds (default 5).

    Returns:
        sc_df with `fold` (int 0..n_folds-1) column added.
    """
    df = sc_df.copy().reset_index(drop=True)
    unique_files = df["filename"].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=n_folds)
    file_to_fold: dict[str, int] = {}
    # GroupKFold requires X, y, groups — but we don't have a target; just use
    # the filename itself as group (each file is its own group).
    for fold_idx, (_, val_idx) in enumerate(gkf.split(unique_files, groups=unique_files)):
        for j in val_idx:
            file_to_fold[unique_files.iloc[j]] = fold_idx
    df["fold"] = df["filename"].map(file_to_fold).fillna(-1).astype(int)
    if (df["fold"] == -1).any():
        n_unmapped = (df["fold"] == -1).sum()
        raise ValueError(
            f"{n_unmapped} SS rows could not be assigned a fold "
            "(filename missing from GroupKFold output)."
        )
    return df


# ============================================================================
# LightningDataModule
# ============================================================================


class DistilledSEDDataModule(pl.LightningDataModule):
    """LightningDataModule for the Tucker distilled-SED recipe (exp 032).

    Train DataLoader:
        - `ConcatDataset([FocalDS(non_val_fold), ScDS(non_val_fold)])`
        - `MixSamp(sizes=[len(focal), len(sc)], shares=...,  bs=..., nst=...)`
        - num_workers=0 (Tucker's default; revisit if I/O-bound)

    Val DataLoader:
        - `ScDS(val_fold)` only — focal validation isn't part of Tucker's
          metric (he validates on labeled SS only).

    Args:
        focal_df: Focal DataFrame. Must have at least `primary_label` and
            `filename` columns. `original_idx` + `fold` are computed here.
        sc_df: SS window DataFrame (output of `build_sc_cache_meta`). Must
            have `filename` and `start_sec` columns. `fold` is added here.
        sc_labels_df: `train_soundscapes_labels.csv` loaded as DataFrame.
        species_to_idx: Species name → label index map.
        num_classes: Output class count (234 for BirdCLEF 2026).
        fold_k: Which fold to hold out as validation (0..n_folds-1).
        n_folds: Total fold count (default 5).
        focal_seed: RandomState for focal StratifiedKFold (default 42).
        min_sample: Rare-species upsampling threshold (default 20).
        window_samples: Samples per window (default 160_000 = 5s @ 32 kHz).
        sample_rate: Sample rate (default 32000).
        batch_size: Batch size (default 64, Tucker's).
        focal_share: Per-batch focal fraction (default 0.9, Tucker's).
        sc_share: Per-batch SC fraction (default 0.1, Tucker's).
        steps_per_epoch: Number of training batches per epoch. None → use
            `max(100, total_samples // batch_size)` (Tucker's formula).
        num_workers: DataLoader workers (default 0, Tucker's).
        focal_audio_loader: Callable for focal waveform loading. Defaults to
            `_default_audio_loader` (soundfile + resampling). Tests inject stubs.
        sc_audio_loader: Same for SS waveforms.
        focal_ds_kwargs: Additional kwargs forwarded to FocalDS (mixup config,
            augmentation flags, etc.).
        sc_ds_kwargs: Additional kwargs forwarded to ScDS.
        seed_epoch_shift: Per-epoch seed shift for MixSamp (matches Tucker's
            `seed=42 + ep`). Set to None to disable (deterministic batches).
        base_seed: Base seed for MixSamp (default 42).

    Notes:
        - Phase 1 trains fold 0 only (`fold_k=0` in configs). Phase 2 sweeps
          {0, 1, 2, 3, 4}.
        - `nst = max(100, total_samples / batch_size)` is computed once at
          setup. Re-instantiate the DataModule to change.
        - `_sc_mixup_pool` is passed to FocalDS so the focal × SS mixup path
          can sample labeled SS windows. The pool is filtered to non-val rows
          inside FocalDS via `fold_k` exclusion.
    """

    def __init__(
        self,
        focal_df: pd.DataFrame,
        sc_df: pd.DataFrame,
        sc_labels_df: pd.DataFrame,
        species_to_idx: dict[str, int],
        num_classes: int = 234,
        fold_k: int = 0,
        n_folds: int = 5,
        focal_seed: int = 42,
        min_sample: int = 20,
        window_samples: int = 160_000,
        sample_rate: int = 32000,
        batch_size: int = 64,
        focal_share: float = 0.9,
        sc_share: float = 0.1,
        steps_per_epoch: int | None = None,
        num_workers: int = 0,
        focal_audio_loader=_default_audio_loader,
        sc_audio_loader=_default_audio_loader,
        focal_ds_kwargs: dict | None = None,
        sc_ds_kwargs: dict | None = None,
        seed_epoch_shift: bool = True,
        base_seed: int = 42,
        # --- Exp 033: own-teacher pseudo-label integration ---
        # When pseudo_label_path is None (default), the DataModule behaves
        # exactly as exp 032 (no PL integration). When set, loads the PL
        # parquet, builds a PLDS, wraps focal+sc with GTDatasetWrapper, and
        # combines via ConcatDataset for training.
        pseudo_label_path: str | Path | None = None,
        pseudo_label_weight: float = 1.0,
        pseudo_label_sampler: str = "uniform",  # "uniform" or "mixsamp"
        pseudo_label_focal_share: float = 0.5,
        pseudo_label_sc_share: float = 0.0,
        pseudo_label_pl_share: float = 0.5,
        pseudo_label_exclude_labeled_sc: bool = True,
        pseudo_label_aug: bool = False,
        soundscape_dir: str | Path | None = None,
        # --- Exp 036: Sydorskyi-faithful class-balanced focal sampling (Phase B) ---
        # When False (default), behaves exactly as exp 032/034 (MixSamp uniform draws).
        # When True (non-PL path only), focal indices are drawn with class-balanced
        # weights (sc stays uniform), preserving the focal/sc batch composition.
        use_class_balanced_sampler: bool = False,
        class_balance_mode: str = "balanced",  # "balanced" (1/n_c) | "sqrt" ((n_c/N)^power)
        class_balance_power: float = -0.5,
    ):
        super().__init__()
        self._focal_df_in = focal_df
        self._sc_df_in = sc_df
        self._sc_labels_df = sc_labels_df
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.fold_k = fold_k
        self.n_folds = n_folds
        self.focal_seed = focal_seed
        self.min_sample = min_sample
        self.window_samples = window_samples
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.focal_share = focal_share
        self.sc_share = sc_share
        self._steps_per_epoch_arg = steps_per_epoch
        self.num_workers = num_workers
        self.focal_audio_loader = focal_audio_loader
        self.sc_audio_loader = sc_audio_loader
        self.focal_ds_kwargs = focal_ds_kwargs or {}
        self.sc_ds_kwargs = sc_ds_kwargs or {}
        self.seed_epoch_shift = seed_epoch_shift
        self.base_seed = base_seed

        # PL config
        self.pseudo_label_path = Path(pseudo_label_path) if pseudo_label_path else None
        self.pseudo_label_weight = pseudo_label_weight
        self.pseudo_label_sampler = pseudo_label_sampler
        self.pseudo_label_focal_share = pseudo_label_focal_share
        self.pseudo_label_sc_share = pseudo_label_sc_share
        self.pseudo_label_pl_share = pseudo_label_pl_share
        self.pseudo_label_exclude_labeled_sc = pseudo_label_exclude_labeled_sc
        self.pseudo_label_aug = pseudo_label_aug
        self.soundscape_dir = Path(soundscape_dir) if soundscape_dir else None
        if self.pseudo_label_path is not None and self.soundscape_dir is None:
            raise ValueError(
                "soundscape_dir is required when pseudo_label_path is set "
                "(PLDS needs it to construct audio file paths)."
            )

        # Exp 036 Phase B: class-balanced focal sampling (defaults off → exp 032 MixSamp)
        self.use_class_balanced_sampler = use_class_balanced_sampler
        self.class_balance_mode = class_balance_mode
        self.class_balance_power = class_balance_power

        # Computed in setup
        self.focal_df: pd.DataFrame | None = None
        self.sc_df: pd.DataFrame | None = None
        self.Y_SC: np.ndarray | None = None
        self.steps_per_epoch: int | None = None
        self.pl_df: pd.DataFrame | None = None  # Filtered PL parquet (post-leakage-filter)

    def setup(self, stage: str | None = None):
        """Build folds + label matrix. Idempotent (safe to call multiple times)."""
        if self.focal_df is None:
            self.focal_df = build_focal_folds(
                self._focal_df_in,
                n_folds=self.n_folds,
                seed=self.focal_seed,
                min_sample=self.min_sample,
            )
        if self.sc_df is None:
            self.sc_df = build_sc_folds(self._sc_df_in, n_folds=self.n_folds)
        if self.Y_SC is None:
            self.Y_SC = build_sc_label_matrix(
                self.sc_df,
                self._sc_labels_df,
                self.species_to_idx,
                self.num_classes,
            )

        # --- Exp 033: load + filter PL parquet ---
        if self.pseudo_label_path is not None and self.pl_df is None:
            pl_df_raw = pd.read_parquet(self.pseudo_label_path)
            n_in = len(pl_df_raw)

            # Filter 1: exclude val-fold SS files (prevent val leak)
            val_sc_files = set(self.sc_df[self.sc_df["fold"] == self.fold_k]["filename"].unique())
            pl_df_filtered = pl_df_raw[~pl_df_raw["filename"].isin(val_sc_files)]
            n_after_val_filter = len(pl_df_filtered)

            # Filter 2 (optional): exclude labeled-SS train files
            # (Antoine protocol: labeled SS uses GT labels via ScDS; PL only for unlabeled)
            if self.pseudo_label_exclude_labeled_sc:
                labeled_sc_files = set(self.sc_df["filename"].unique())
                pl_df_filtered = pl_df_filtered[~pl_df_filtered["filename"].isin(labeled_sc_files)]
            n_after_labeled_filter = len(pl_df_filtered)

            self.pl_df = pl_df_filtered.reset_index(drop=True)
            logger.info(
                f"DistilledSEDDataModule PL filter: {n_in:,} → {n_after_val_filter:,} "
                f"(excl val-fold) → {n_after_labeled_filter:,} (excl labeled-sc) segments"
            )

        # Steps-per-epoch (Tucker: max(100, total_samples / batch_size))
        if self._steps_per_epoch_arg is None:
            non_val_focal = (self.focal_df["fold"] != self.fold_k).sum()
            non_val_sc = (self.sc_df["fold"] != self.fold_k).sum()
            total = non_val_focal + non_val_sc
            if self.pl_df is not None:
                total += len(self.pl_df)
            self.steps_per_epoch = max(100, total // self.batch_size)
        else:
            self.steps_per_epoch = self._steps_per_epoch_arg

        logger.info(
            f"DistilledSEDDataModule setup: fold_k={self.fold_k}, "
            f"focal={len(self.focal_df)} ({(self.focal_df['fold'] != self.fold_k).sum()} train, "
            f"{(self.focal_df['fold'] == self.fold_k).sum()} val), "
            f"sc={len(self.sc_df)} ({(self.sc_df['fold'] != self.fold_k).sum()} train, "
            f"{(self.sc_df['fold'] == self.fold_k).sum()} val), "
            f"pl={'NONE' if self.pl_df is None else len(self.pl_df)}, "
            f"steps/epoch={self.steps_per_epoch}"
        )

    def _make_focal_ds(self, training: bool) -> FocalDS:
        """Build the FocalDS for the training fold (non-val rows)."""
        train_df = self.focal_df[self.focal_df["fold"] != self.fold_k].reset_index(drop=True)
        # sc_mixup_pool: pass the entire labeled SS df + Y_SC; FocalDS filters
        # by `fold_k` internally so the val fold is excluded from the mixup pool.
        sc_pool = [(self.sc_df, self.Y_SC)] if training else None
        return FocalDS(
            focal_df=train_df,
            species_to_idx=self.species_to_idx,
            sc_pool=sc_pool,
            fold_k=self.fold_k,
            window_samples=self.window_samples,
            num_classes=self.num_classes,
            aug=training,
            audio_loader=self.focal_audio_loader,
            sc_audio_loader=self.sc_audio_loader,
            sample_rate=self.sample_rate,
            seed=self.base_seed,
            **self.focal_ds_kwargs,
        )

    def _make_sc_ds(self, fold_mask: np.ndarray, training: bool) -> ScDS:
        """Build a ScDS over rows where `fold_mask` is True."""
        sc_subset = self.sc_df[fold_mask].reset_index(drop=True)
        Y_subset = self.Y_SC[fold_mask]
        return ScDS(
            sc_df=sc_subset,
            Y_SC=Y_subset,
            window_samples=self.window_samples,
            num_classes=self.num_classes,
            aug=training,
            audio_loader=self.sc_audio_loader,
            sample_rate=self.sample_rate,
            seed=self.base_seed,
            **self.sc_ds_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        if self.focal_df is None:
            self.setup("fit")
        focal_ds = self._make_focal_ds(training=True)
        non_val_mask = (self.sc_df["fold"] != self.fold_k).values
        sc_ds = self._make_sc_ds(non_val_mask, training=True)

        # Per-epoch seed shift: Tucker uses `seed=42 + ep`. We approximate
        # with `base_seed + current_epoch` — but pl.LightningDataModule doesn't
        # have current_epoch at construction time. Lightning will call
        # `train_dataloader()` once per epoch when `reload_dataloaders_every_n_epochs=1`.
        # If that's not set, the sampler keeps the same seed (only acceptable
        # for fast-dev-run; production configs should enable reload).
        ep = getattr(self.trainer, "current_epoch", 0) if self.trainer else 0
        seed = self.base_seed + (ep if self.seed_epoch_shift else 0)

        # --- Exp 033 PL path: build PLDS + wrap focal/sc with GTDatasetWrapper ---
        if self.pl_df is not None:
            pl_ds = PLDS(
                pl_df=self.pl_df,
                species_list=sorted(self.species_to_idx, key=lambda s: self.species_to_idx[s]),
                soundscape_dir=self.soundscape_dir,
                window_samples=self.window_samples,
                loss_weight=self.pseudo_label_weight,
                audio_loader=self.sc_audio_loader,
                sample_rate=self.sample_rate,
                aug=self.pseudo_label_aug,  # exp 037: noise-the-student control (False/True)
                seed=seed,
            )
            # Wrap focal + sc so their 5-tuples become 6-tuples with loss_weight=1.0
            gt_focal = GTDatasetWrapper(focal_ds, loss_weight=1.0)
            gt_sc = GTDatasetWrapper(sc_ds, loss_weight=1.0)
            concat = ConcatDataset([gt_focal, gt_sc, pl_ds])

            if self.pseudo_label_sampler == "uniform":
                # Pure Antoine protocol: shuffle the ConcatDataset uniformly.
                logger.info(
                    f"train_dataloader (PL/uniform): focal={len(gt_focal)}, "
                    f"sc={len(gt_sc)}, pl={len(pl_ds)}, total={len(concat):,}"
                )
                return DataLoader(
                    concat,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=_collate_drop_extras,
                    drop_last=True,
                )
            elif self.pseudo_label_sampler == "mixsamp":
                # 3-source MixSamp with explicit shares
                sampler = MixSamp(
                    sizes=[len(gt_focal), len(gt_sc), len(pl_ds)],
                    names=["focal", "sc", "pl"],
                    shares={
                        "focal": self.pseudo_label_focal_share,
                        "sc": self.pseudo_label_sc_share,
                        "pl": self.pseudo_label_pl_share,
                    },
                    bs=self.batch_size,
                    nst=self.steps_per_epoch,
                    seed=seed,
                )
                logger.info(
                    f"train_dataloader (PL/mixsamp): focal={len(gt_focal)}, "
                    f"sc={len(gt_sc)}, pl={len(pl_ds)}, sampler={sampler.extra_repr()}"
                )
                return DataLoader(
                    concat,
                    batch_sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=_collate_drop_extras,
                )
            else:
                raise ValueError(
                    f"pseudo_label_sampler must be 'uniform' or 'mixsamp', "
                    f"got {self.pseudo_label_sampler!r}"
                )

        # --- Exp 032 default path: focal + sc only (no PL) ---
        concat = ConcatDataset([focal_ds, sc_ds])
        if self.use_class_balanced_sampler:
            # Exp 036 Phase B: class-balance the FOCAL draws by primary-label frequency
            # (Sydorskyi); sc stays uniform. Targets are aligned to FocalDS row order
            # (FocalDS.df is the reset-indexed train-fold df → iloc(i) matches getitem(i)).
            focal_targets = focal_ds.df["primary_label"].astype(str).tolist()
            assert len(focal_targets) == len(focal_ds), "focal targets misaligned with FocalDS"
            cw = compute_class_weights(
                focal_targets, mode=self.class_balance_mode, power_value=self.class_balance_power
            )
            focal_weights = np.array([cw[t] for t in focal_targets], dtype=np.float64)
            sampler = ClassBalancedMixSamp(
                sizes=[len(focal_ds), len(sc_ds)],
                names=["focal", "sc"],
                shares={"focal": self.focal_share, "sc": self.sc_share},
                bs=self.batch_size,
                nst=self.steps_per_epoch,
                seed=seed,
                source_weights={"focal": focal_weights, "sc": None},
            )
        else:
            sampler = MixSamp(
                sizes=[len(focal_ds), len(sc_ds)],
                names=["focal", "sc"],
                shares={"focal": self.focal_share, "sc": self.sc_share},
                bs=self.batch_size,
                nst=self.steps_per_epoch,
                seed=seed,
            )
        logger.info(
            f"train_dataloader: focal={len(focal_ds)}, sc={len(sc_ds)}, "
            f"class_balanced={self.use_class_balanced_sampler}, sampler={sampler.extra_repr()}"
        )
        return DataLoader(
            concat,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_drop_extras,
        )

    def val_dataloader(self) -> DataLoader:
        if self.focal_df is None:
            self.setup("validate")
        val_mask = (self.sc_df["fold"] == self.fold_k).values
        sc_val = self._make_sc_ds(val_mask, training=False)
        return DataLoader(
            sc_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_drop_extras,
        )


def _collate_drop_extras(batch):
    """Collate fn supporting two dataset tuple shapes (backward-compat):

    - **5-tuple** `(wav, lb, wt, mk, tag)` from FocalDS/ScDS (Tucker default):
      stacks first 4 into `(wavs, labels, wts, mks)`, drops `tag` → returns 4-tuple.
      The LightningModule reads only `batch[0]` and `batch[1]` (per audit § 4
      `wt`, `mk` are always all-ones in Tucker's recipe).

    - **6-tuple** `(wav, lb, wt, mk, tag, loss_weight)` from PLDS or
      GTDatasetWrapper (exp 033 own-teacher PL):
      stacks first 4 as above, drops `tag`, additionally stacks `loss_weight`
      (per-sample scalar) into a (B,) tensor → returns 5-tuple
      `(wavs, labels, wts, mks, loss_weights)`. The LightningModule
      `training_step` detects the 5-tuple shape and threads `loss_weight`
      into `_dual_supervision_loss` for the per-sample weighted BCE branch.

    Detection: if the first sample is a 6-tuple, the whole batch is assumed
    to be 6-tuples. Mixed-length batches are unsupported (ConcatDataset
    ensures uniform tuple length by construction).
    """
    wavs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    wts = torch.stack([b[2] for b in batch])
    mks = torch.stack([b[3] for b in batch])
    if len(batch) > 0 and len(batch[0]) > 5:
        loss_weights = torch.stack(
            [torch.as_tensor(float(b[5]), dtype=torch.float32) for b in batch]
        )
        return wavs, labels, wts, mks, loss_weights
    return wavs, labels, wts, mks
