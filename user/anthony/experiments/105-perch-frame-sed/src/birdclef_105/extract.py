"""Stage 1: extract Perch v2 pooled embeddings for labeled soundscape windows.

Reads the soundscape labels CSV + nocall parquet, applies the exp 102 seed=42
file-level 50/50 split, iterates windows, runs Perch ONNX in batches of 8,
writes a single shard to the configured output directory.

Only labeled-soundscape + nocall windows for phase 1. Focal chunks are
deferred to a follow-up extraction pass if needed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from birdclef_105 import audio as audio_mod
from birdclef_105 import cache as cache_mod
from birdclef_105 import split as split_mod
from birdclef_105.perch_onnx import PerchSession

logger = logging.getLogger("birdclef_105.extract")


def _time_to_sec(t: str) -> int:
    parts = str(t).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def build_species_list(train_csv: str) -> list[str]:
    """Sorted unique primary_label from the BirdCLEF train CSV.

    Matches exp 102's `build_species_list` — critical for label alignment.
    """
    df = pd.read_csv(train_csv)
    return sorted(df["primary_label"].astype(str).unique().tolist())


def build_window_index(
    soundscape_dir: str,
    soundscape_labels_csv: str,
    nocall_parquet: str | None,
    species_list: list[str],
    split_seed: int,
    limit_files: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of (filename, window_idx, start_sec, source, split, label_*) rows.

    Labels are encoded as semicolon-joined species IDs in a 'label_ids' column;
    the numpy `labels` array is constructed later during extraction so we can
    pass it directly to `write_shard` with the correct num_classes.

    Args:
        soundscape_dir: Directory containing soundscape .ogg files.
        soundscape_labels_csv: Path to train_soundscapes_labels.csv.
        nocall_parquet: Optional path to the nocall index parquet from exp 102.
        species_list: Sorted species list (matches train.csv).
        split_seed: Seed for the deterministic file-level split (must be 42 to
            match exp 102).
        limit_files: If set, keep only this many distinct filenames total (pilot
            mode). Selection is deterministic — takes the first N by sorted name.

    Returns:
        DataFrame with columns:
          filename, window_idx, start_sec, source, split, label_ids, n_positive_labels
    """
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    sdir = Path(soundscape_dir)

    train_files, val_files = split_mod.split_soundscape_files(
        soundscape_labels_csv, seed=split_seed
    )
    all_labeled_files = sorted(train_files | val_files)

    if limit_files is not None:
        kept = set(all_labeled_files[:limit_files])
        train_files = train_files & kept
        val_files = val_files & kept

    rows: list[dict] = []

    # Labeled soundscape windows.
    labels_df = pd.read_csv(soundscape_labels_csv).drop_duplicates(
        subset=["filename", "start"]
    )
    for _, row in labels_df.iterrows():
        filename = row["filename"]
        if filename not in train_files and filename not in val_files:
            continue
        if not (sdir / filename).exists():
            logger.warning("Skipping missing soundscape file: %s", filename)
            continue

        start_sec = _time_to_sec(row["start"])
        ids = [
            s.strip()
            for s in str(row["primary_label"]).split(";")
            if s.strip() in species_to_idx
        ]
        if not ids:
            # Window labels no species in our taxonomy — skip (mirrors exp 102).
            continue

        rows.append(
            dict(
                filename=filename,
                window_idx=len(rows),  # filled-in properly after we finalize order
                start_sec=float(start_sec),
                source="soundscape",
                split="train" if filename in train_files else "val",
                label_ids=";".join(ids),
                n_positive_labels=len(ids),
            )
        )

    # No-call windows.
    if nocall_parquet and Path(nocall_parquet).exists():
        nocall = pd.read_parquet(nocall_parquet)
        for _, row in nocall.iterrows():
            filename = row["filename"]
            if filename not in train_files and filename not in val_files:
                continue
            if not (sdir / filename).exists():
                continue
            rows.append(
                dict(
                    filename=filename,
                    window_idx=len(rows),
                    start_sec=float(row["chunk_start"]),
                    source="nocall",
                    split="train" if filename in train_files else "val",
                    label_ids="",
                    n_positive_labels=0,
                )
            )
    else:
        logger.warning("No nocall parquet at %s — skipping nocall windows", nocall_parquet)

    df = pd.DataFrame(rows)
    df["window_idx"] = np.arange(len(df))  # canonical order
    return df


def _encode_labels(label_ids: list[str], species_to_idx: dict[str, int], num_classes: int) -> np.ndarray:
    """Encode one window's label_ids string into a multi-hot int8 vector."""
    out = np.zeros(num_classes, dtype=np.int8)
    for sid in label_ids:
        if sid in species_to_idx:
            out[species_to_idx[sid]] = 1
    return out


def run(config_path: str, limit: int | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    perch_cfg = cfg["perch"]
    cache_cfg = cfg["cache"]

    # Build species list (source of truth for num_classes).
    species_list = build_species_list(data_cfg["train_csv"])
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)
    logger.info("Taxonomy: %d species from %s", num_classes, data_cfg["train_csv"])

    # Build window index (filtered by split and optional --limit).
    index_df = build_window_index(
        soundscape_dir=data_cfg["soundscape_dir"],
        soundscape_labels_csv=data_cfg["soundscape_labels_csv"],
        nocall_parquet=data_cfg.get("nocall_parquet"),
        species_list=species_list,
        split_seed=data_cfg["split_seed"],
        limit_files=limit,
    )
    if len(index_df) == 0:
        logger.error("No windows to extract — check soundscape_dir, labels CSV, and --limit")
        return 2
    logger.info(
        "Windows to extract: %d (%s)",
        len(index_df),
        index_df["source"].value_counts().to_dict(),
    )
    logger.info("Split: %s", index_df["split"].value_counts().to_dict())

    # Load Perch ONNX session.
    logger.info("Loading Perch ONNX: %s :: %s", perch_cfg["hf_repo"], perch_cfg["hf_filename"])
    perch = PerchSession(
        repo_id=perch_cfg["hf_repo"],
        filename=perch_cfg["hf_filename"],
    )

    # Allocate output arrays.
    embeddings = np.zeros((len(index_df), 1536), dtype=np.float32)
    labels = np.zeros((len(index_df), num_classes), dtype=np.int8)

    window_duration = data_cfg["chunk_duration"]
    window_samples = data_cfg["window_samples"]
    batch_size = int(perch_cfg["batch_size"])
    sdir = Path(data_cfg["soundscape_dir"])

    t0 = time.time()
    # Iterate in file-order to amortize soundfile.open overhead across windows from the same file.
    grouped = index_df.sort_values(["filename", "start_sec"]).reset_index(drop=False)
    buffer_waveforms: list[np.ndarray] = []
    buffer_row_positions: list[int] = []  # positions in the ORIGINAL index_df (grouped["index"])

    def flush():
        if not buffer_waveforms:
            return
        batch = np.stack(buffer_waveforms, axis=0).astype(np.float32, copy=False)
        out = perch.run(batch, outputs=["embedding"])
        emb = out["embedding"]  # (B, 1536)
        for pos, vec in zip(buffer_row_positions, emb):
            embeddings[pos] = vec
        buffer_waveforms.clear()
        buffer_row_positions.clear()

    for _, row in grouped.iterrows():
        orig_pos = int(row["index"])
        path = sdir / row["filename"]
        try:
            wav = audio_mod.load_audio_32k(
                str(path),
                start_sec=float(row["start_sec"]),
                duration_sec=float(window_duration),
            )
        except Exception as e:
            logger.warning("Audio load failed for %s @ %s: %s", row["filename"], row["start_sec"], e)
            continue

        # Pad / truncate to exactly window_samples.
        if len(wav) < window_samples:
            wav = np.pad(wav, (0, window_samples - len(wav)))
        else:
            wav = wav[:window_samples]

        # Encode labels.
        label_ids = [s for s in str(row["label_ids"]).split(";") if s]
        labels[orig_pos] = _encode_labels(label_ids, species_to_idx, num_classes)

        buffer_waveforms.append(wav)
        buffer_row_positions.append(orig_pos)
        if len(buffer_waveforms) >= batch_size:
            flush()
    flush()

    elapsed = time.time() - t0
    logger.info("Extracted %d windows in %.1f s (%.2f win/s)",
                len(index_df), elapsed, len(index_df) / max(elapsed, 1e-6))

    # Select output directory: pilot vs full.
    if limit is not None:
        out_dir = Path(cache_cfg["pilot_dir"]) / "soundscape_plus_nocall"
    else:
        out_dir = Path(cache_cfg["output_dir"]) / "soundscape_plus_nocall"

    # Write shard — drop the helper label_ids column before serialization.
    index_to_write = index_df.drop(columns=["label_ids"]).copy()
    cache_mod.write_shard(
        out_dir=out_dir,
        index_df=index_to_write,
        embeddings=embeddings,
        labels=labels,
        species_list=species_list,
    )
    logger.info("Shard written to %s", out_dir)
    return 0
