"""Phase 2: Label fusion via FlyingSquid triplet method.

Builds a label matrix per species at 1s resolution, runs the triplet method
to estimate LF accuracies without ground truth, then applies Bayes rule to
produce calibrated soft labels for all tokens x 206 species.

Supports two splits:
  - focal:      4 species-specific LFs (BirdNET, Perch, indices probe, recording label)
  - soundscape: 3 species-specific LFs (BirdNET, Perch, indices probe)

Activity modulation uses acoustic indices (NDSI, spectral contrast, bioacoustic
index) as negative-only evidence. BamBird was removed — its ROI detection is
too coarse (77% of files have events, near-binary at 1s resolution) and
redundant with the per-token acoustic indices.

CLI:
    weaksup-fuse --split focal \\
        --probe-dir ~/scratch/birdclef-2026/outputs/exp013/probes \\
        --data-dir ~/scratch/birdclef-2026/data/birdclef-2025 \\
        --exp010-dir ~/scratch/birdclef-2026/outputs/exp010/labels \\
        --output-dir ~/scratch/birdclef-2026/outputs/exp013/fused
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.progress import track

from weaksup.data import INDICES_FEATURES

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

# --- Default paths ---
DEFAULT_PROBE_DIR = Path("~/scratch/birdclef-2026/outputs/exp013/probes").expanduser()
DEFAULT_DATA_DIR = Path("~/scratch/birdclef-2026/data/birdclef-2025").expanduser()
DEFAULT_EXP010_DIR = Path("~/scratch/birdclef-2026/outputs/exp010/labels").expanduser()
DEFAULT_OUTPUT_DIR = Path("~/scratch/birdclef-2026/outputs/exp013/fused").expanduser()

# Species-specific LF names per split
FOCAL_LF_NAMES = [
    "lf0_birdnet",
    "lf1_perch",
    "lf2_indices",
    "lf3_rec_label",
]
SOUNDSCAPE_LF_NAMES = [
    "lf0_birdnet",
    "lf1_perch",
    "lf2_indices",
]

# Species-agnostic LFs for activity modulation (negative-only)
AGNOSTIC_LF_NAMES = [
    "ndsi",
    "spectral_contrast",
    "bioacoustic_index",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _normalize_file_key(path: str, split: str) -> str:
    """Normalize file path to a consistent join key.

    Data sources are inconsistent:
    - BirdNET/Perch: absolute paths for both focal and soundscape
    - Indices: relative paths — focal is 'species_id/recording.ogg',
      soundscape is bare 'filename.ogg'

    Focal:      extract 'species_id/recording.ogg' from absolute paths,
                passthrough for relative paths that already match train.csv
    Soundscape: extract basename 'filename.ogg'
    """
    p = Path(path)
    if split == "soundscape":
        return p.name
    # Focal: if absolute, extract last two components; if relative, passthrough
    if p.is_absolute():
        return f"{p.parts[-2]}/{p.parts[-1]}"
    return str(p)


def _data_paths(exp010_dir: Path, split: str) -> dict[str, Path]:
    """Return per-split data source paths."""
    audio_dir = "train_audio" if split == "focal" else "train_soundscapes"
    return {
        "birdnet_embed": exp010_dir / "birdnet" / audio_dir / "parts" / "embed",
        "perch_embed": exp010_dir / "perch" / audio_dir / "parts" / "embed",
        "indices": exp010_dir / "indices" / audio_dir / "parts",
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_probe(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load probe weights from .npz -> (coef, intercept, species_list)."""
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["coef"].astype(np.float32),
        data["intercept"].astype(np.float32),
        list(data["species_list"]),
    )


def load_indices_probe(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Load indices probe with feature normalization stats.

    Returns:
        coef: (206, 4) weights
        intercept: (206,) bias
        species_list: list of species codes
        feature_mean: (4,) z-score mean
        feature_std: (4,) z-score std
    """
    data = np.load(npz_path, allow_pickle=True)
    return (
        data["coef"].astype(np.float32),
        data["intercept"].astype(np.float32),
        list(data["species_list"]),
        data["feature_mean"].astype(np.float32),
        data["feature_std"].astype(np.float32),
    )


def score_embeddings(
    embed_dir: Path,
    coef: np.ndarray,
    intercept: np.ndarray,
    split: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Score all embedding parquets with probe weights.

    Returns:
        logits: (N, 206) raw logits
        rel_files: list of N normalized file keys
        start_times: (N,) start times
    """
    parquet_files = sorted(embed_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquets in {embed_dir}")

    meta_cols = {"file", "start_time", "end_time"}
    all_logits: list[np.ndarray] = []
    all_files: list[str] = []
    all_starts: list[np.ndarray] = []

    label = embed_dir.parts[-4]  # "birdnet" or "perch"
    for pf in track(parquet_files, description=f"Scoring {label} embeddings"):
        df = pl.read_parquet(pf)
        files_raw = df["file"].to_list()
        starts = df["start_time"].to_numpy().astype(np.float32)

        embed_cols = [c for c in df.columns if c not in meta_cols]
        X = df.select(embed_cols).to_numpy().astype(np.float32)

        logits = X @ coef.T + intercept  # (N_chunk, 206)
        all_logits.append(logits)
        all_files.extend(_normalize_file_key(f, split) for f in files_raw)
        all_starts.append(starts)

    return (
        np.concatenate(all_logits, axis=0),
        all_files,
        np.concatenate(all_starts, axis=0),
    )


def score_indices_probe(
    indices_dir: Path,
    coef: np.ndarray,
    intercept: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    split: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Score acoustic index parquets with indices probe weights.

    Returns:
        logits: (N, 206) raw logits
        file_keys: list of N normalized file keys
        start_times: (N,) start times (window_start)
    """
    parquet_files = sorted(indices_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquets in {indices_dir}")

    all_logits: list[np.ndarray] = []
    all_files: list[str] = []
    all_starts: list[np.ndarray] = []

    for pf in track(parquet_files, description="Scoring indices probe"):
        df = pl.read_parquet(pf)
        files_raw = df["file"].to_list()
        starts = df["window_start"].to_numpy().astype(np.float32)

        X = df.select(INDICES_FEATURES).to_numpy().astype(np.float64)

        # Log-transform spectral_flatness (same as training)
        sf_idx = INDICES_FEATURES.index("spectral_flatness")
        X[:, sf_idx] = np.log1p(np.maximum(X[:, sf_idx], 0))

        # Z-score normalize using saved stats from training
        X = ((X - feature_mean) / feature_std).astype(np.float32)

        logits = X @ coef.T + intercept
        all_logits.append(logits)
        all_files.extend(_normalize_file_key(f, split) for f in files_raw)
        all_starts.append(starts)

    return (
        np.concatenate(all_logits, axis=0),
        all_files,
        np.concatenate(all_starts, axis=0),
    )


def load_acoustic_indices(indices_dir: Path, split: str) -> pl.DataFrame:
    """Load all acoustic index parquets (already at 3s/1s step)."""
    parquet_files = sorted(indices_dir.glob("part_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquets in {indices_dir}")
    console.print(f"Loading {len(parquet_files)} acoustic index shards...")
    df = pl.concat([pl.read_parquet(p) for p in parquet_files])
    # Normalize file keys
    df = df.with_columns(
        pl.col("file").map_elements(lambda f: _normalize_file_key(f, split), return_dtype=pl.Utf8).alias("file")
    )
    return df


def load_file_to_species(data_dir: Path) -> dict[str, str]:
    """Load train.csv -> {filename: primary_label}."""
    import csv

    mapping: dict[str, str] = {}
    with open(data_dir / "train.csv") as f:
        for row in csv.DictReader(f):
            mapping[row["filename"]] = row["primary_label"]
    return mapping


# ---------------------------------------------------------------------------
# Label matrix construction
# ---------------------------------------------------------------------------


def threshold_votes_percentile(
    logits: np.ndarray,
    pos_pct: float = 95.0,
    neg_pct: float = 50.0,
) -> np.ndarray:
    """Per-species percentile thresholding on raw logits -> {-1, 0, +1}.

    Probe logits have tiny dynamic range (std ~0.035) but real discriminative
    signal (d' = 2-12). A fixed sigmoid threshold is meaningless. Instead, use
    per-species percentiles:

        logit > p(pos_pct)  -> +1  (top ~5% — enriched for true positives)
        logit < p(neg_pct)  -> -1  (bottom 50% — depleted of true positives)
        otherwise           ->  0  (abstain)
    """
    votes = np.zeros(logits.shape, dtype=np.int8)
    pos_thresholds = np.percentile(logits, pos_pct, axis=0)  # (n_species,)
    neg_thresholds = np.percentile(logits, neg_pct, axis=0)  # (n_species,)
    votes[logits > pos_thresholds] = 1
    votes[logits < neg_thresholds] = -1
    return votes


def build_species_agnostic_lfs(
    rel_files: list[str],
    start_times: np.ndarray,
    indices_df: pl.DataFrame,
    ndsi_neg: float,
    bi_neg: float,
) -> np.ndarray:
    """Build species-agnostic LF columns: NDSI, spectral_contrast, BI.

    IMPORTANT: Species-agnostic LFs provide NEGATIVE EVIDENCE ONLY.
    They vote -1 (silence/anthropophony -> species absent) or abstain (0).

    Returns:
        agnostic_lfs: (N, 3) array of {-1, 0} for NDSI, SC, BI
    """
    n = len(rel_files)
    lfs = np.zeros((n, 3), dtype=np.int8)

    console.print("Building acoustic index lookup...")
    idx_file = indices_df["file"].to_list()
    idx_start = indices_df["window_start"].to_numpy()
    idx_ndsi = indices_df["ndsi"].to_numpy()
    idx_sc = indices_df["spectral_contrast"].to_numpy()
    idx_bi = indices_df["bioacoustic_index"].to_numpy()

    idx_lookup: dict[tuple[str, int], int] = {}
    for j in range(len(idx_file)):
        idx_lookup[(idx_file[j], int(idx_start[j]))] = j

    matched = 0
    for i in range(n):
        key = (rel_files[i], int(start_times[i]))
        j = idx_lookup.get(key)
        if j is None:
            continue
        matched += 1

        # NDSI — negative when anthropophony dominates
        if idx_ndsi[j] < ndsi_neg:
            lfs[i, 0] = -1
        # Spectral contrast — low = flat/noisy spectrum, no harmonic content
        if idx_sc[j] < 20.0:
            lfs[i, 1] = -1
        # Bioacoustic index — low = very quiet in 2-15kHz
        if idx_bi[j] < bi_neg:
            lfs[i, 2] = -1

    console.print(f"  Acoustic indices matched: {matched}/{n} ({100*matched/n:.1f}%)")
    for name, col in [("NDSI", 0), ("SpectralContrast", 1), ("BioIndex", 2)]:
        neg = (lfs[:, col] == -1).sum()
        console.print(f"  {name}: -1={neg} ({100*neg/n:.1f}%)")

    return lfs


def build_perch_lookup(
    pc_logits: np.ndarray,
    pc_files: list[str],
    pc_starts: np.ndarray,
    pos_pct: float = 95.0,
    neg_pct: float = 50.0,
) -> dict[tuple[str, int], np.ndarray]:
    """Build Perch vote lookup: (file, 1s_bin) -> (206,) votes.

    Perch has 5s windows. Forward-fill to 1s bins within each window.
    """
    pc_votes = threshold_votes_percentile(pc_logits, pos_pct, neg_pct)
    lookup: dict[tuple[str, int], np.ndarray] = {}

    for i in range(len(pc_files)):
        rel_file = pc_files[i]
        t_start = int(pc_starts[i])
        votes = pc_votes[i]
        for t in range(t_start, t_start + 5):
            lookup[(rel_file, t)] = votes

    return lookup


def build_indices_lookup(
    idx_logits: np.ndarray,
    idx_files: list[str],
    idx_starts: np.ndarray,
    pos_pct: float = 95.0,
    neg_pct: float = 50.0,
) -> dict[tuple[str, int], np.ndarray]:
    """Build indices probe vote lookup: (file, 1s_bin) -> (206,) votes.

    Indices have 3s windows at 1s step — direct 1:1 mapping.
    """
    idx_votes = threshold_votes_percentile(idx_logits, pos_pct, neg_pct)
    lookup: dict[tuple[str, int], np.ndarray] = {}

    for i in range(len(idx_files)):
        key = (idx_files[i], int(idx_starts[i]))
        lookup[key] = idx_votes[i]

    return lookup


# ---------------------------------------------------------------------------
# Triplet method (FlyingSquid)
# ---------------------------------------------------------------------------


def flying_squid(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triplet method for label fusion.

    Args:
        L: (n_tokens, m_lfs) array of {-1, 0, +1}

    Returns:
        soft_labels: (n_tokens,) probabilities in [0, 1]
        accuracies: (m_lfs,) estimated accuracy per LF
    """
    n, m = L.shape
    accuracies = np.zeros(m)

    for k in range(m):
        estimates = []
        others = [j for j in range(m) if j != k]
        for i, j in combinations(others, 2):
            mask = (L[:, i] != 0) & (L[:, j] != 0) & (L[:, k] != 0)
            if mask.sum() < 10:
                continue
            Lm = L[mask]
            mu_ij = (Lm[:, i] * Lm[:, j]).mean()
            mu_ijk = (Lm[:, i] * Lm[:, j] * Lm[:, k]).mean()
            if abs(mu_ij) > 1e-6:
                estimates.append(mu_ijk / mu_ij)
        if estimates:
            accuracies[k] = np.clip((np.median(estimates) + 1) / 2, 0.05, 0.95)
        else:
            accuracies[k] = 0.5

    # Bayes rule: combine LF votes weighted by log-odds accuracy
    log_odds = np.zeros(n)
    for k in range(m):
        a = accuracies[k]
        log_ratio = np.log(a / (1 - a))
        log_odds += L[:, k] * log_ratio

    soft_labels = _sigmoid(log_odds)
    return soft_labels, accuracies


def apply_accuracies(
    L: np.ndarray,
    accuracies: np.ndarray,
    prevalence: float = 0.5,
    activity_log_odds: np.ndarray | None = None,
) -> np.ndarray:
    """Apply pre-computed LF accuracies to a full label matrix via Bayes rule."""
    n, m = L.shape
    prevalence = np.clip(prevalence, 1e-4, 1 - 1e-4)
    log_odds = np.full(n, np.log(prevalence / (1 - prevalence)))

    for k in range(m):
        a = np.clip(accuracies[k], 0.05, 0.95)
        if abs(a - 0.5) < 0.001:
            continue
        log_ratio = np.log(a / (1 - a))
        log_odds += L[:, k].astype(np.float64) * log_ratio

    if activity_log_odds is not None:
        log_odds += activity_log_odds

    return _sigmoid(log_odds).astype(np.float32)


def compute_activity_log_odds(
    agnostic_lfs: np.ndarray,
    activity_weight: float = 1.0,
) -> np.ndarray:
    """Convert species-agnostic LF votes to log-odds activity score.

    Each agnostic LF voting -1 contributes negative evidence (suppression).
    All abstaining (0) = neutral. More -1 votes = stronger suppression.
    """
    n_negative = (agnostic_lfs == -1).sum(axis=1).astype(np.float64)
    return -activity_weight * n_negative


# ---------------------------------------------------------------------------
# Per-species fusion worker
# ---------------------------------------------------------------------------


def _fuse_one_species(args: tuple) -> tuple[int, str, np.ndarray, np.ndarray]:
    """Fuse labels for a single species with negative sampling."""
    (species_idx, species_code, label_matrix, activity_log_odds,
     prevalence, neg_ratio) = args
    n_tokens, m_lfs = label_matrix.shape

    # Positive = any species-specific probe votes +1 (first 2 are always BirdNET, Perch)
    has_positive = (label_matrix[:, 0] == 1) | (label_matrix[:, 1] == 1)
    n_pos = has_positive.sum()

    if n_pos == 0:
        soft = apply_accuracies(
            label_matrix,
            np.full(m_lfs, 0.5),
            prevalence=prevalence,
            activity_log_odds=activity_log_odds,
        )
        return (
            species_idx,
            species_code,
            soft,
            np.full(m_lfs, 0.5, dtype=np.float64),
        )

    neg_mask = ~has_positive
    n_neg_available = neg_mask.sum()
    n_neg_sample = min(neg_ratio * n_pos, n_neg_available, 50_000)

    if n_neg_sample < n_neg_available:
        rng = np.random.RandomState(species_idx)
        neg_indices = rng.choice(
            np.where(neg_mask)[0], size=int(n_neg_sample), replace=False
        )
        sample_mask = np.zeros(n_tokens, dtype=bool)
        sample_mask[has_positive] = True
        sample_mask[neg_indices] = True
    else:
        sample_mask = np.ones(n_tokens, dtype=bool)

    L_sample = label_matrix[sample_mask]
    _, accuracies = flying_squid(L_sample)

    soft_labels = apply_accuracies(
        label_matrix, accuracies,
        prevalence=prevalence,
        activity_log_odds=activity_log_odds,
    )

    return species_idx, species_code, soft_labels, accuracies


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def fuse(
    split: Annotated[
        str, typer.Option(help="Data split: focal or soundscape")
    ] = "focal",
    probe_dir: Annotated[
        Path, typer.Option(help="Dir with birdnet/, perch/, indices/ probe .npz files")
    ] = DEFAULT_PROBE_DIR,
    data_dir: Annotated[
        Path, typer.Option(help="BirdCLEF data dir (train.csv, taxonomy.csv)")
    ] = DEFAULT_DATA_DIR,
    exp010_dir: Annotated[
        Path, typer.Option(help="Exp 010 labels root")
    ] = DEFAULT_EXP010_DIR,
    output_dir: Annotated[
        Path, typer.Option(help="Output dir for fused labels")
    ] = DEFAULT_OUTPUT_DIR,
    n_workers: Annotated[
        int, typer.Option(help="Parallel workers for per-species fusion")
    ] = 4,
    neg_ratio: Annotated[
        int, typer.Option(help="Negative:positive sampling ratio")
    ] = 10,
    pos_pct: Annotated[
        float, typer.Option(help="Probe percentile for +1 vote (top X%%)")
    ] = 95.0,
    neg_pct: Annotated[
        float, typer.Option(help="Probe percentile for -1 vote (below X%%)")
    ] = 50.0,
    ndsi_neg: Annotated[
        float, typer.Option(help="NDSI threshold for -1 (anthropophony)")
    ] = -0.2,
    bi_neg: Annotated[
        float, typer.Option(help="Bioacoustic index threshold for -1 (silent)")
    ] = 100.0,
) -> None:
    """Build label matrix and run FlyingSquid per species."""
    if split not in ("focal", "soundscape"):
        raise typer.BadParameter(f"split must be 'focal' or 'soundscape', got {split!r}")

    is_focal = split == "focal"
    lf_names = FOCAL_LF_NAMES if is_focal else SOUNDSCAPE_LF_NAMES
    n_species_lfs = len(lf_names)

    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    paths = _data_paths(exp010_dir, split)

    console.rule(f"[bold]Label Fusion — {split}")
    console.print(f"Species-specific LFs: {n_species_lfs} ({', '.join(lf_names)})")

    # ── 1. Load metadata ──
    console.rule("[bold]Loading metadata")
    file_to_species = load_file_to_species(data_dir)
    species_list = sorted(set(file_to_species.values()))
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    n_species = len(species_list)
    console.print(f"Species: {n_species}, Recordings: {len(file_to_species)}")

    # ── 2. Load probes ──
    console.rule("[bold]Loading probes")
    bn_coef, bn_intercept, bn_species = load_probe(
        probe_dir / "birdnet" / "species_probe.npz"
    )
    pc_coef, pc_intercept, pc_species = load_probe(
        probe_dir / "perch" / "species_probe.npz"
    )
    assert bn_species == pc_species == species_list, "Species list mismatch"
    console.print(
        f"BirdNET probe: ({bn_coef.shape[0]}, {bn_coef.shape[1]}), "
        f"Perch probe: ({pc_coef.shape[0]}, {pc_coef.shape[1]})"
    )

    # Load indices probe
    idx_probe_path = probe_dir / "indices" / "species_probe.npz"
    idx_coef, idx_intercept, idx_species, idx_feat_mean, idx_feat_std = (
        load_indices_probe(idx_probe_path)
    )
    assert idx_species == species_list, "Indices probe species list mismatch"
    console.print(f"Indices probe: ({idx_coef.shape[0]}, {idx_coef.shape[1]})")

    # ── 3. Score BirdNET embeddings ──
    console.rule("[bold]Scoring BirdNET embeddings")
    bn_logits, bn_files, bn_starts = score_embeddings(
        paths["birdnet_embed"], bn_coef, bn_intercept, split
    )
    n_tokens = len(bn_files)
    console.print(f"BirdNET: {n_tokens} tokens scored")
    console.print(
        f"  Logit range: [{bn_logits.min():.4f}, {bn_logits.max():.4f}], "
        f"std={bn_logits.std():.4f}"
    )

    # ── 4. Score Perch embeddings ──
    console.rule("[bold]Scoring Perch embeddings")
    pc_logits, pc_files, pc_starts = score_embeddings(
        paths["perch_embed"], pc_coef, pc_intercept, split
    )
    console.print(f"Perch: {len(pc_files)} tokens scored")

    # ── 5. Score indices probe ──
    console.rule("[bold]Scoring indices probe")
    idx_logits, idx_files, idx_starts = score_indices_probe(
        paths["indices"], idx_coef, idx_intercept,
        idx_feat_mean, idx_feat_std, split
    )
    console.print(f"Indices: {len(idx_files)} tokens scored")
    console.print(
        f"  Logit range: [{idx_logits.min():.4f}, {idx_logits.max():.4f}], "
        f"std={idx_logits.std():.4f}"
    )

    # ── 6. Build vote lookups ──
    console.rule("[bold]Building vote lookups")

    # Perch: forward-fill 5s -> 1s
    perch_lookup = build_perch_lookup(
        pc_logits, pc_files, pc_starts, pos_pct, neg_pct
    )
    console.print(f"Perch lookup: {len(perch_lookup)} 1s bins")
    del pc_logits

    # Indices probe: 3s/1s -> direct lookup
    indices_lookup = build_indices_lookup(
        idx_logits, idx_files, idx_starts, pos_pct, neg_pct
    )
    console.print(f"Indices lookup: {len(indices_lookup)} 1s bins")
    del idx_logits

    # ── 7. Threshold BirdNET logits ──
    console.rule("[bold]Thresholding BirdNET logits")
    console.print(f"  pos_pct={pos_pct} (top {100-pos_pct:.0f}%), neg_pct={neg_pct}")
    bn_votes = threshold_votes_percentile(bn_logits, pos_pct, neg_pct)
    del bn_logits
    for label, val in [("positive(+1)", 1), ("negative(-1)", -1), ("abstain(0)", 0)]:
        pct = 100 * (bn_votes == val).mean()
        console.print(f"  BirdNET {label}: {pct:.1f}%")

    # ── 8. Build species-agnostic activity modulation ──
    console.rule("[bold]Building activity modulation (negative-only)")
    indices_df = load_acoustic_indices(paths["indices"], split)
    agnostic_lfs = build_species_agnostic_lfs(
        bn_files, bn_starts, indices_df,
        ndsi_neg=ndsi_neg, bi_neg=bi_neg,
    )
    del indices_df

    activity_log_odds = compute_activity_log_odds(agnostic_lfs, activity_weight=1.0)
    console.print(
        f"Activity log-odds: mean={activity_log_odds.mean():.2f}, "
        f"suppressed tokens (any -1): "
        f"{100*(activity_log_odds < 0).mean():.1f}%"
    )
    del agnostic_lfs

    # ── 9. Build recording-label LF (focal only) ──
    if is_focal:
        console.rule("[bold]Building recording-label LF")
        token_primary_species = np.array(
            [species_to_idx.get(file_to_species.get(f, ""), -1) for f in bn_files],
            dtype=np.int32,
        )
        n_labeled = (token_primary_species >= 0).sum()
        console.print(
            f"Recording label LF: {n_labeled}/{n_tokens} tokens have primary species "
            f"({100*n_labeled/n_tokens:.1f}%)"
        )
    else:
        console.print("[dim]Soundscape split: no recording-label LF[/dim]")
        token_primary_species = None

    # ── 10. Compute per-species prevalence ──
    console.rule("[bold]Computing species prevalence")
    species_rec_counts = Counter(file_to_species.values())
    total_recordings = len(set(bn_files))
    species_prevalence = {}
    for s_code in species_list:
        frac = species_rec_counts.get(s_code, 0) / max(total_recordings, 1)
        species_prevalence[s_code] = np.clip(frac * 0.3, 0.001, 0.1)
    prev_vals = list(species_prevalence.values())
    console.print(
        f"Prevalence: mean={np.mean(prev_vals):.4f}, "
        f"range=[{min(prev_vals):.4f}, {max(prev_vals):.4f}]"
    )

    # ── 11. Verify file key consistency ──
    console.rule("[bold]File key consistency check")
    bn_file_set = set(bn_files)
    pc_file_set = set(pc_files)
    idx_file_set = set(idx_files)
    console.print(f"  BirdNET files: {len(bn_file_set)}")
    console.print(f"  Perch files: {len(pc_file_set)}")
    console.print(f"  Indices files: {len(idx_file_set)}")
    overlap_bn_idx = len(bn_file_set & idx_file_set)
    overlap_bn_pc = len(bn_file_set & pc_file_set)
    console.print(f"  BirdNET ^ Indices: {overlap_bn_idx} ({100*overlap_bn_idx/max(len(bn_file_set),1):.1f}%)")
    console.print(f"  BirdNET ^ Perch: {overlap_bn_pc} ({100*overlap_bn_pc/max(len(bn_file_set),1):.1f}%)")

    # ── 12. Assemble per-species label matrices & fuse ──
    console.rule("[bold]Per-species label fusion")

    # Pre-build Perch and indices vote matrices
    perch_votes_all = np.zeros((n_tokens, n_species), dtype=np.int8)
    console.print("Filling Perch forward-fill votes...")
    for i in range(n_tokens):
        key = (bn_files[i], int(bn_starts[i]))
        pv = perch_lookup.get(key)
        if pv is not None:
            perch_votes_all[i] = pv
    del perch_lookup
    n_perch_matched = (perch_votes_all != 0).any(axis=1).sum()
    console.print(
        f"  Perch coverage: {n_perch_matched}/{n_tokens} tokens "
        f"({100*n_perch_matched/n_tokens:.1f}%)"
    )

    indices_votes_all = np.zeros((n_tokens, n_species), dtype=np.int8)
    console.print("Filling indices probe votes...")
    for i in range(n_tokens):
        key = (bn_files[i], int(bn_starts[i]))
        iv = indices_lookup.get(key)
        if iv is not None:
            indices_votes_all[i] = iv
    del indices_lookup
    n_idx_matched = (indices_votes_all != 0).any(axis=1).sum()
    console.print(
        f"  Indices coverage: {n_idx_matched}/{n_tokens} tokens "
        f"({100*n_idx_matched/n_tokens:.1f}%)"
    )

    # Build task list
    tasks = []
    for s_idx, s_code in enumerate(species_list):
        L = np.zeros((n_tokens, n_species_lfs), dtype=np.int8)
        L[:, 0] = bn_votes[:, s_idx]          # LF0: BirdNET
        L[:, 1] = perch_votes_all[:, s_idx]    # LF1: Perch
        L[:, 2] = indices_votes_all[:, s_idx]  # LF2: indices probe
        if is_focal:
            L[token_primary_species == s_idx, 3] = 1  # LF3: recording label
        tasks.append((
            s_idx, s_code, L, activity_log_odds,
            species_prevalence[s_code], neg_ratio,
        ))

    del bn_votes, perch_votes_all, indices_votes_all

    console.print(f"Fusing {n_species} species with {n_workers} workers...")
    all_soft_labels = np.zeros((n_tokens, n_species), dtype=np.float32)
    all_accuracies = np.zeros((n_species, n_species_lfs), dtype=np.float64)

    if n_workers <= 1:
        for task in track(tasks, description="Fusing species"):
            s_idx, s_code, soft, acc = _fuse_one_species(task)
            all_soft_labels[:, s_idx] = soft
            all_accuracies[s_idx] = acc
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for s_idx, s_code, soft, acc in track(
                pool.map(_fuse_one_species, tasks),
                total=n_species,
                description="Fusing species",
            ):
                all_soft_labels[:, s_idx] = soft
                all_accuracies[s_idx] = acc

    # ── 13. Save outputs ──
    console.rule("[bold]Saving outputs")

    label_df = pl.DataFrame(
        {"file": bn_files, "start_time": bn_starts}
    ).with_columns(
        [
            pl.Series(name=species_list[i], values=all_soft_labels[:, i])
            for i in range(n_species)
        ]
    )
    soft_labels_path = split_output_dir / "soft_labels.parquet"
    label_df.write_parquet(soft_labels_path, compression="zstd")
    console.print(f"Soft labels: {soft_labels_path} ({label_df.shape})")

    acc_df = pl.DataFrame({"species": species_list}).with_columns(
        [
            pl.Series(name=lf_names[k], values=all_accuracies[:, k])
            for k in range(n_species_lfs)
        ]
    )
    acc_path = split_output_dir / "lf_accuracies.parquet"
    acc_df.write_parquet(acc_path)
    console.print(f"LF accuracies: {acc_path}")

    # Print summary
    console.rule("[bold]Summary")
    console.print(f"Split: {split}")
    console.print(f"Tokens: {n_tokens}")
    console.print(f"Species: {n_species}")
    console.print(f"Soft label stats:")
    console.print(f"  mean: {all_soft_labels.mean():.4f}")
    console.print(f"  near 0 (<0.1): {(all_soft_labels < 0.1).mean()*100:.1f}%")
    console.print(f"  near 1 (>0.9): {(all_soft_labels > 0.9).mean()*100:.1f}%")
    console.print(f"  ambiguous (0.3-0.7): {((all_soft_labels > 0.3) & (all_soft_labels < 0.7)).mean()*100:.1f}%")
    console.print()
    console.print("Mean LF accuracies across species:")
    for k in range(n_species_lfs):
        acc_vals = all_accuracies[:, k]
        console.print(
            f"  {lf_names[k]:30s}: "
            f"mean={acc_vals.mean():.3f}, "
            f"std={acc_vals.std():.3f}, "
            f"range=[{acc_vals.min():.3f}, {acc_vals.max():.3f}]"
        )


if __name__ == "__main__":
    app()
