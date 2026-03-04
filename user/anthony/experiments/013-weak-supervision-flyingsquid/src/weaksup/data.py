"""Data loading for linear probes.

Loads pre-computed embeddings from experiment 010 parquets and builds
multi-label targets from BirdCLEF 2025 train.csv + taxonomy.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def load_metadata(data_dir: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Load species and taxa metadata from BirdCLEF CSV files.

    Returns:
        file_to_species: mapping from filename to primary_label.
        species_to_taxa: mapping from primary_label to class_name.
        species_list: sorted list of all 206 species labels.
    """
    train_csv = data_dir / "train.csv"
    taxonomy_csv = data_dir / "taxonomy.csv"

    file_to_species: dict[str, str] = {}
    with open(train_csv) as f:
        for row in csv.DictReader(f):
            file_to_species[row["filename"]] = row["primary_label"]

    species_to_taxa: dict[str, str] = {}
    with open(taxonomy_csv) as f:
        for row in csv.DictReader(f):
            species_to_taxa[row["primary_label"]] = row["class_name"]

    species_list = sorted(set(file_to_species.values()))
    return file_to_species, species_to_taxa, species_list


def _extract_relative_path(abs_path: str) -> str:
    """Extract 'species/recording.ogg' from an absolute path.

    BirdNET embeds store full paths like
    /storage/.../train_audio/species/recording.ogg.
    train.csv uses relative paths like species/recording.ogg.
    """
    parts = Path(abs_path).parts
    return f"{parts[-2]}/{parts[-1]}"


def load_embeddings(
    embed_dir: Path,
    file_to_species: dict[str, str],
    species_list: list[str],
    species_to_taxa: dict[str, str],
    taxa_list: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all embedding parquets and build label matrices.

    Args:
        embed_dir: directory containing part_NNNN.parquet files.
        file_to_species: filename -> primary_label mapping.
        species_list: ordered species labels (defines column order).
        species_to_taxa: species -> taxa class mapping.
        taxa_list: ordered taxa labels. None = auto from species_to_taxa.

    Returns:
        X: (N, D) embedding matrix.
        y_species: (N, n_species) multi-hot species labels.
        y_taxa: (N,) integer taxa labels.
        recording_ids: (N,) integer recording group IDs for CV splitting.
    """
    if taxa_list is None:
        taxa_list = sorted(set(species_to_taxa.values()))

    species_to_idx = {s: i for i, s in enumerate(species_list)}
    taxa_to_idx = {t: i for i, t in enumerate(taxa_list)}

    parquet_files = sorted(embed_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {embed_dir}")

    # Read all parquets with polars (fast, memory-efficient)
    df = pl.concat([pl.read_parquet(p) for p in parquet_files])

    # Extract file column and embedding columns
    file_col = df["file"].to_list()
    meta_cols = {"file", "start_time", "end_time"}
    embed_cols = [c for c in df.columns if c not in meta_cols]
    X = df.select(embed_cols).to_numpy().astype(np.float32)

    n_species = len(species_list)
    n_samples = len(file_col)
    y_species = np.zeros((n_samples, n_species), dtype=np.float32)
    y_taxa = np.zeros(n_samples, dtype=np.int64)

    # Build recording group IDs for GroupKFold
    recording_map: dict[str, int] = {}
    recording_ids = np.zeros(n_samples, dtype=np.int64)

    skipped = 0
    for i, abs_path in enumerate(file_col):
        rel_path = _extract_relative_path(abs_path)
        species = file_to_species.get(rel_path)
        if species is None:
            skipped += 1
            continue

        if species in species_to_idx:
            y_species[i, species_to_idx[species]] = 1.0

        taxa = species_to_taxa.get(species)
        if taxa and taxa in taxa_to_idx:
            y_taxa[i] = taxa_to_idx[taxa]

        # Group by recording (species/stem without extension variations)
        if rel_path not in recording_map:
            recording_map[rel_path] = len(recording_map)
        recording_ids[i] = recording_map[rel_path]

    if skipped > 0:
        print(f"  Warning: {skipped}/{n_samples} clips had no matching species in train.csv")

    # Filter clips with no species label (prevents argmax-on-zeros bug in balanced sampling)
    labeled = y_species.sum(axis=1) > 0
    n_unlabeled = n_samples - labeled.sum()
    if n_unlabeled > 0:
        print(f"  Filtering {n_unlabeled} unlabeled clips")
        X = X[labeled]
        y_species = y_species[labeled]
        y_taxa = y_taxa[labeled]
        recording_ids = recording_ids[labeled]

    return X, y_species, y_taxa, recording_ids


INDICES_FEATURES = ["ndsi", "spectral_contrast", "spectral_flatness", "bioacoustic_index"]


def load_indices(
    indices_dir: Path,
    file_to_species: dict[str, str],
    species_list: list[str],
    species_to_taxa: dict[str, str],
    taxa_list: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load acoustic index parquets and build label matrices.

    Differences from load_embeddings():
    - Schema uses named columns (ndsi, spectral_contrast, etc.) not numbered
    - Time column is window_start (not start_time)
    - File paths are already relative (species_id/recording.ogg for focal,
      bare filename.ogg for soundscape) — no _extract_relative_path() needed
    - Log-transforms spectral_flatness (values span 1e-9 to 0.1, extremely skewed)
    - Z-score normalizes all 4 features, returns (mean, std) for inference

    Returns:
        X: (N, 4) feature matrix (z-score normalized).
        y_species: (N, n_species) multi-hot species labels.
        y_taxa: (N,) integer taxa labels.
        recording_ids: (N,) integer recording group IDs for CV splitting.
        feature_mean: (4,) per-feature mean before normalization.
        feature_std: (4,) per-feature std before normalization.
    """
    if taxa_list is None:
        taxa_list = sorted(set(species_to_taxa.values()))

    species_to_idx = {s: i for i, s in enumerate(species_list)}
    taxa_to_idx = {t: i for i, t in enumerate(taxa_list)}

    parquet_files = sorted(indices_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {indices_dir}")

    df = pl.concat([pl.read_parquet(p) for p in parquet_files])

    file_col = df["file"].to_list()
    X = df.select(INDICES_FEATURES).to_numpy().astype(np.float64)

    # Log-transform spectral_flatness (column index 2) — extremely skewed
    sf_idx = INDICES_FEATURES.index("spectral_flatness")
    X[:, sf_idx] = np.log1p(np.maximum(X[:, sf_idx], 0))

    # Z-score normalize
    feature_mean = X.mean(axis=0).astype(np.float32)
    feature_std = X.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-8] = 1.0  # avoid div-by-zero
    X = ((X - feature_mean) / feature_std).astype(np.float32)

    n_species = len(species_list)
    n_samples = len(file_col)
    y_species = np.zeros((n_samples, n_species), dtype=np.float32)
    y_taxa = np.zeros(n_samples, dtype=np.int64)

    recording_map: dict[str, int] = {}
    recording_ids = np.zeros(n_samples, dtype=np.int64)

    skipped = 0
    for i, rel_path in enumerate(file_col):
        species = file_to_species.get(rel_path)
        if species is None:
            skipped += 1
            continue

        if species in species_to_idx:
            y_species[i, species_to_idx[species]] = 1.0

        taxa = species_to_taxa.get(species)
        if taxa and taxa in taxa_to_idx:
            y_taxa[i] = taxa_to_idx[taxa]

        if rel_path not in recording_map:
            recording_map[rel_path] = len(recording_map)
        recording_ids[i] = recording_map[rel_path]

    if skipped > 0:
        print(f"  Warning: {skipped}/{n_samples} clips had no matching species in train.csv")

    # Filter clips with no species label
    labeled = y_species.sum(axis=1) > 0
    n_unlabeled = n_samples - labeled.sum()
    if n_unlabeled > 0:
        print(f"  Filtering {n_unlabeled} unlabeled clips")
        X = X[labeled]
        y_species = y_species[labeled]
        y_taxa = y_taxa[labeled]
        recording_ids = recording_ids[labeled]

    return X, y_species, y_taxa, recording_ids, feature_mean, feature_std


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_embeddings_with_energy(
    embed_dir: Path,
    indices_dir: Path,
    file_to_species: dict[str, str],
    species_list: list[str],
    species_to_taxa: dict[str, str],
    taxa_list: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings and attenuate species labels by per-clip acoustic energy.

    Instead of y[clip] = 1.0 for all clips of a recording, computes
    y[clip] = energy_weight(clip) where energy_weight is derived from
    bioacoustic_index via per-recording normalization.

    Clips in quiet portions of a recording get weight near 0, while
    clips with vocalization activity get weight near 1.

    Args:
        embed_dir: directory with embedding parquets (BirdNET/Perch).
        indices_dir: directory with acoustic index parquets (exp 010).
        file_to_species: filename -> primary_label mapping.
        species_list: ordered species labels.
        species_to_taxa: species -> taxa class mapping.
        taxa_list: ordered taxa labels.

    Returns:
        X: (N, D) embedding matrix.
        y_species: (N, n_species) energy-attenuated species labels in [0, 1].
        y_taxa: (N,) integer taxa labels.
        recording_ids: (N,) integer recording group IDs.
    """
    if taxa_list is None:
        taxa_list = sorted(set(species_to_taxa.values()))

    species_to_idx = {s: i for i, s in enumerate(species_list)}
    taxa_to_idx = {t: i for i, t in enumerate(taxa_list)}

    # --- Load embeddings ---
    parquet_files = sorted(embed_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {embed_dir}")

    df = pl.concat([pl.read_parquet(p) for p in parquet_files])
    file_col = df["file"].to_list()
    start_times = df["start_time"].to_numpy().astype(np.float32)
    meta_cols = {"file", "start_time", "end_time"}
    embed_cols = [c for c in df.columns if c not in meta_cols]
    X = df.select(embed_cols).to_numpy().astype(np.float32)

    # --- Load acoustic indices for energy weights ---
    idx_files = sorted(indices_dir.glob("part_*.parquet"))
    if not idx_files:
        raise FileNotFoundError(f"No parquet files found in {indices_dir}")
    idx_df = pl.concat([pl.read_parquet(p) for p in idx_files])

    # Build lookup: (relative_file, window_start) -> bioacoustic_index
    bi_lookup: dict[tuple[str, int], float] = {}
    idx_file_col = idx_df["file"].to_list()
    idx_start_col = idx_df["window_start"].to_numpy()
    idx_bi_col = idx_df["bioacoustic_index"].to_numpy()
    for j in range(len(idx_file_col)):
        bi_lookup[(idx_file_col[j], int(idx_start_col[j]))] = float(idx_bi_col[j])

    # --- Compute per-clip energy weights ---
    n_samples = len(file_col)
    n_species = len(species_list)

    # Extract relative paths and get BI values
    rel_paths = [_extract_relative_path(p) for p in file_col]
    bi_values = np.zeros(n_samples, dtype=np.float64)
    bi_matched = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        key = (rel_paths[i], int(start_times[i]))
        if key in bi_lookup:
            bi_values[i] = bi_lookup[key]
            bi_matched[i] = True

    print(f"  Energy: matched {bi_matched.sum()}/{n_samples} clips to indices "
          f"({100 * bi_matched.mean():.1f}%)")

    # Per-recording energy normalization
    rec_to_clips: dict[str, list[int]] = {}
    for i, rp in enumerate(rel_paths):
        rec_to_clips.setdefault(rp, []).append(i)

    energy = np.full(n_samples, 0.5, dtype=np.float64)  # default for unmatched
    for rec, clip_indices in rec_to_clips.items():
        mask = np.array(clip_indices)
        matched_mask = bi_matched[mask]
        if matched_mask.sum() < 2:
            # Not enough data for normalization, use uniform weight
            energy[mask] = 0.5
            continue
        bi_rec = bi_values[mask[matched_mask]]
        bi_median = np.median(bi_rec)
        bi_iqr = np.percentile(bi_rec, 75) - np.percentile(bi_rec, 25)
        scale = max(bi_iqr, 1.0)
        # Apply to all clips in this recording (matched get normalized, unmatched get 0.5)
        for idx in mask[matched_mask]:
            energy[idx] = _sigmoid((bi_values[idx] - bi_median) / scale)

    # --- Build labels ---
    y_species = np.zeros((n_samples, n_species), dtype=np.float32)
    y_taxa = np.zeros(n_samples, dtype=np.int64)
    recording_map: dict[str, int] = {}
    recording_ids = np.zeros(n_samples, dtype=np.int64)

    skipped = 0
    for i, rel_path in enumerate(rel_paths):
        species = file_to_species.get(rel_path)
        if species is None:
            skipped += 1
            continue

        if species in species_to_idx:
            y_species[i, species_to_idx[species]] = float(energy[i])

        taxa = species_to_taxa.get(species)
        if taxa and taxa in taxa_to_idx:
            y_taxa[i] = taxa_to_idx[taxa]

        if rel_path not in recording_map:
            recording_map[rel_path] = len(recording_map)
        recording_ids[i] = recording_map[rel_path]

    if skipped > 0:
        print(f"  Warning: {skipped}/{n_samples} clips had no matching species in train.csv")

    # Filter clips with no species label
    labeled = y_species.sum(axis=1) > 0
    n_unlabeled = n_samples - labeled.sum()
    if n_unlabeled > 0:
        print(f"  Filtering {n_unlabeled} unlabeled clips")
        X = X[labeled]
        y_species = y_species[labeled]
        y_taxa = y_taxa[labeled]
        recording_ids = recording_ids[labeled]

    # Report energy distribution
    pos_labels = y_species[y_species > 0]
    print(f"  Energy-attenuated labels: mean={pos_labels.mean():.3f}, "
          f"median={np.median(pos_labels):.3f}, "
          f"near-zero (<0.1)={100 * (pos_labels < 0.1).mean():.1f}%, "
          f"near-one (>0.9)={100 * (pos_labels > 0.9).mean():.1f}%")

    return X, y_species, y_taxa, recording_ids


class IndicesDataModule(L.LightningDataModule):
    """DataModule for acoustic index features + labels.

    Like EmbeddingDataModule but loads 4 spectral features instead of
    neural embeddings. Stores feature_mean/std for z-score normalization
    at inference time.
    """

    def __init__(
        self,
        indices_dir: Path,
        data_dir: Path,
        val_fraction: float = 0.1,
        batch_size: int = 4096,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.indices_dir = Path(indices_dir)
        self.data_dir = Path(data_dir)
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.species_list: list[str] = []
        self.taxa_list: list[str] = []
        self.n_features: int = 0
        self.n_species: int = 0
        self.n_taxa: int = 0
        self.species_pos_weight: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            return

        file_to_species, species_to_taxa, self.species_list = load_metadata(
            self.data_dir
        )
        self.taxa_list = sorted(set(species_to_taxa.values()))
        self.n_species = len(self.species_list)
        self.n_taxa = len(self.taxa_list)

        print(f"Loading acoustic indices from {self.indices_dir}...")
        X, y_species, y_taxa, recording_ids, self.feature_mean, self.feature_std = (
            load_indices(
                self.indices_dir,
                file_to_species,
                self.species_list,
                species_to_taxa,
                self.taxa_list,
            )
        )
        self.n_features = X.shape[1]
        print(f"  Loaded {X.shape[0]} clips × {X.shape[1]} features")
        print(f"  Species: {self.n_species}, Taxa: {self.n_taxa}")

        # Compute per-species positive weight (alpha) for focal loss
        pos_counts = y_species.sum(axis=0)
        neg_counts = len(y_species) - pos_counts
        self.species_pos_weight = (neg_counts / len(y_species)).astype(np.float32)
        self.species_counts = pos_counts.astype(np.float32)

        # Compute per-taxa inverse frequency weight
        taxa_counts = np.bincount(y_taxa, minlength=self.n_taxa).astype(np.float32)
        inv_freq = 1.0 / np.maximum(taxa_counts, 1)
        self.taxa_class_weight = (inv_freq / inv_freq.sum() * self.n_taxa).astype(
            np.float32
        )

        # Split by recording
        rng = np.random.RandomState(self.seed)
        unique_recordings = np.unique(recording_ids)
        n_val_recs = max(1, int(len(unique_recordings) * self.val_fraction))
        val_rec_set = set(
            rng.choice(unique_recordings, size=n_val_recs, replace=False)
        )

        val_mask = np.array([r in val_rec_set for r in recording_ids])
        train_mask = ~val_mask

        self.train_dataset = TensorDataset(
            torch.from_numpy(X[train_mask]),
            torch.from_numpy(y_species[train_mask]),
            torch.from_numpy(y_taxa[train_mask]),
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(X[val_mask]),
            torch.from_numpy(y_species[val_mask]),
            torch.from_numpy(y_taxa[val_mask]),
        )
        print(
            f"  Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)} "
            f"({len(val_rec_set)} recordings held out)"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )


class EmbeddingDataModule(L.LightningDataModule):
    """DataModule for loading pre-computed embeddings + labels.

    Loads everything into RAM (BirdNET: ~4 GB, Perch: ~1 GB) and creates
    train/val splits stratified by recording.
    """

    def __init__(
        self,
        embed_dir: Path,
        data_dir: Path,
        val_fraction: float = 0.1,
        batch_size: int = 4096,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.embed_dir = Path(embed_dir)
        self.data_dir = Path(data_dir)
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.species_list: list[str] = []
        self.taxa_list: list[str] = []
        self.n_features: int = 0
        self.n_species: int = 0
        self.n_taxa: int = 0
        self.species_pos_weight: np.ndarray | None = None

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            return  # already setup, skip re-loading

        file_to_species, species_to_taxa, self.species_list = load_metadata(
            self.data_dir
        )
        self.taxa_list = sorted(set(species_to_taxa.values()))
        self.n_species = len(self.species_list)
        self.n_taxa = len(self.taxa_list)

        print(f"Loading embeddings from {self.embed_dir}...")
        X, y_species, y_taxa, recording_ids = load_embeddings(
            self.embed_dir,
            file_to_species,
            self.species_list,
            species_to_taxa,
            self.taxa_list,
        )
        self.n_features = X.shape[1]
        print(f"  Loaded {X.shape[0]} clips × {X.shape[1]} dims")
        print(f"  Species: {self.n_species}, Taxa: {self.n_taxa}")

        # Compute per-species positive weight (alpha) for focal loss
        # alpha_s = n_neg_s / n_total (fraction of negatives)
        pos_counts = y_species.sum(axis=0)
        neg_counts = len(y_species) - pos_counts
        self.species_pos_weight = (neg_counts / len(y_species)).astype(np.float32)
        self.species_counts = pos_counts.astype(np.float32)

        # Compute per-taxa inverse frequency weight
        taxa_counts = np.bincount(y_taxa, minlength=self.n_taxa).astype(np.float32)
        inv_freq = 1.0 / np.maximum(taxa_counts, 1)
        self.taxa_class_weight = (inv_freq / inv_freq.sum() * self.n_taxa).astype(
            np.float32
        )

        # Split by recording: assign each recording to train or val
        rng = np.random.RandomState(self.seed)
        unique_recordings = np.unique(recording_ids)
        n_val_recs = max(1, int(len(unique_recordings) * self.val_fraction))
        val_rec_set = set(
            rng.choice(unique_recordings, size=n_val_recs, replace=False)
        )

        val_mask = np.array([r in val_rec_set for r in recording_ids])
        train_mask = ~val_mask

        self.train_dataset = TensorDataset(
            torch.from_numpy(X[train_mask]),
            torch.from_numpy(y_species[train_mask]),
            torch.from_numpy(y_taxa[train_mask]),
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(X[val_mask]),
            torch.from_numpy(y_species[val_mask]),
            torch.from_numpy(y_taxa[val_mask]),
        )
        print(
            f"  Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)} "
            f"({len(val_rec_set)} recordings held out)"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def species_balanced_train_dataloader(self, mode: str = "sqrt") -> DataLoader:
        """Train dataloader with WeightedRandomSampler for species balance.

        Args:
            mode: "inverse" for 1/count (aggressive), "sqrt" for 1/sqrt(count)
                  (gentler, reduces overfitting on rare species).
        """
        y_species = self.train_dataset.tensors[1]  # (N, 206) multi-hot
        primary = y_species.argmax(dim=1)  # (N,) primary species index
        class_counts = torch.bincount(primary, minlength=self.n_species).float()
        class_counts = class_counts.clamp(min=1)  # avoid div-by-zero
        if mode == "sqrt":
            sample_weights = 1.0 / class_counts[primary].sqrt()
        elif mode == "inverse":
            sample_weights = 1.0 / class_counts[primary]
        else:
            raise ValueError(f"Unknown balance mode: {mode!r} (expected 'sqrt' or 'inverse')")
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def taxa_balanced_train_dataloader(self) -> DataLoader:
        """Train dataloader with WeightedRandomSampler for taxa balance."""
        y_taxa = self.train_dataset.tensors[2]  # (N,) integer taxa labels
        class_counts = torch.bincount(y_taxa, minlength=self.n_taxa).float()
        sample_weights = 1.0 / class_counts[y_taxa]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
