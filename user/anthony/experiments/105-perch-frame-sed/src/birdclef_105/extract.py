"""Stage 1: extract Perch v2 features.

Two sources:

1. **soundscape** — labeled soundscape windows + nocall windows, seed=42 file-level
   50/50 split (mirrors exp 102). Single shard, fast (~2 min).

2. **focal** — all 5s chunks of every focal recording in `train.csv`, stratified-
   by-primary_label split (mirrors exp 102 Run N). Sharded via `--num-shards` /
   `--shard-index` for SLURM array-job parallelism, since a single run would take
   6+ hours.

Both sources request `embedding` and (optionally) `spatial_embedding` in the
same Perch forward pass, so frame features are essentially free on disk.
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


def _build_focal_index(
    train_csv: str,
    species_list: list[str],
    shard_index: int,
    num_shards: int,
    split_seed: int,
    val_fraction: float,
) -> pd.DataFrame:
    """One row per focal recording ASSIGNED TO THIS SHARD.

    Windowing into 5s chunks happens later during extraction; the index only
    holds recording-level info plus pre-computed split assignment.

    Columns returned:
        filename, primary_label, split, shard_index
    """
    species_set = set(species_list)
    df = pd.read_csv(train_csv)
    df["primary_label"] = df["primary_label"].astype(str)

    # Drop any rows whose primary_label is outside our 206-taxon list.
    before = len(df)
    df = df[df["primary_label"].isin(species_set)].reset_index(drop=True)
    if len(df) != before:
        logger.warning(
            "Filtered %d focal rows whose primary_label was outside the 206-taxon list",
            before - len(df),
        )

    train_files, val_files = split_mod.stratified_focal_split(
        train_csv=train_csv, val_fraction=val_fraction, seed=split_seed
    )

    def _split_for(f: str) -> str:
        if f in train_files:
            return "train"
        if f in val_files:
            return "val"
        return "drop"

    df["split"] = df["filename"].map(_split_for)
    df = df[df["split"] != "drop"].reset_index(drop=True)

    # Assign shard by filename hash — deterministic, species-uncorrelated.
    df = df.sort_values("filename").reset_index(drop=True)
    df["shard_index"] = np.arange(len(df)) % num_shards
    df = df[df["shard_index"] == shard_index].reset_index(drop=True)
    return df[["filename", "primary_label", "split", "shard_index"]]


def run(
    config_path: str,
    limit: int | None = None,
    source: str = "soundscape",
    shard_index: int = 0,
    num_shards: int = 1,
) -> int:
    """Dispatcher — delegates to the right source-specific routine."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if source == "soundscape":
        return _run_soundscape(cfg, limit=limit)
    if source == "focal":
        return _run_focal(cfg, shard_index=shard_index, num_shards=num_shards)
    logger.error("Unknown source %r; expected 'soundscape' or 'focal'", source)
    return 2


def _make_perch(perch_cfg: dict) -> PerchSession:
    logger.info("Loading Perch ONNX: %s :: %s", perch_cfg["hf_repo"], perch_cfg["hf_filename"])
    return PerchSession(
        repo_id=perch_cfg["hf_repo"],
        filename=perch_cfg["hf_filename"],
    )


def _run_soundscape(cfg: dict, limit: int | None) -> int:
    data_cfg = cfg["data"]
    perch_cfg = cfg["perch"]
    cache_cfg = cfg["cache"]
    extract_cfg = cfg.get("extract", {}) or {}
    extract_frame_features = bool(extract_cfg.get("frame_features", False))
    capture_perch_logits = bool(extract_cfg.get("capture_perch_logits", False))

    species_list = build_species_list(data_cfg["train_csv"])
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)
    logger.info("Taxonomy: %d species from %s", num_classes, data_cfg["train_csv"])

    index_df = build_window_index(
        soundscape_dir=data_cfg["soundscape_dir"],
        soundscape_labels_csv=data_cfg["soundscape_labels_csv"],
        nocall_parquet=data_cfg.get("nocall_parquet"),
        species_list=species_list,
        split_seed=data_cfg["split_seed"],
        limit_files=limit,
    )
    if len(index_df) == 0:
        logger.error("No windows to extract")
        return 2
    logger.info("Windows: %d (%s)", len(index_df), index_df["source"].value_counts().to_dict())
    logger.info("Split: %s", index_df["split"].value_counts().to_dict())

    perch = _make_perch(perch_cfg)
    window_samples = data_cfg["window_samples"]
    window_duration = data_cfg["chunk_duration"]
    batch_size = int(perch_cfg["batch_size"])
    sdir = Path(data_cfg["soundscape_dir"])

    embeddings = np.zeros((len(index_df), 1536), dtype=np.float32)
    spatial = np.zeros((len(index_df), 16, 4, 1536), dtype=np.float32) if extract_frame_features else None
    perch_logits: np.ndarray | None = None  # allocated lazily after first batch (dim from model)
    labels = np.zeros((len(index_df), num_classes), dtype=np.int8)

    out_names = ["embedding"]
    if extract_frame_features:
        out_names.append("spatial_embedding")
    if capture_perch_logits:
        out_names.append("label")

    t0 = time.time()
    grouped = index_df.sort_values(["filename", "start_sec"]).reset_index(drop=False)
    buf_wavs: list[np.ndarray] = []
    buf_pos: list[int] = []

    def flush():
        nonlocal perch_logits
        if not buf_wavs:
            return
        batch = np.stack(buf_wavs, axis=0).astype(np.float32, copy=False)
        out = perch.run(batch, outputs=out_names)
        if capture_perch_logits and perch_logits is None:
            perch_logits = np.zeros((len(index_df), out["label"].shape[1]), dtype=np.float32)
        for k, pos in enumerate(buf_pos):
            embeddings[pos] = out["embedding"][k]
            if spatial is not None:
                spatial[pos] = out["spatial_embedding"][k]
            if capture_perch_logits:
                perch_logits[pos] = out["label"][k]
        buf_wavs.clear()
        buf_pos.clear()

    for _, row in grouped.iterrows():
        orig_pos = int(row["index"])
        try:
            wav = audio_mod.load_audio_32k(
                str(sdir / row["filename"]),
                start_sec=float(row["start_sec"]),
                duration_sec=float(window_duration),
            )
        except Exception as e:
            logger.warning("Load failed %s @ %.1f: %s", row["filename"], row["start_sec"], e)
            continue

        if len(wav) < window_samples:
            wav = np.pad(wav, (0, window_samples - len(wav)))
        else:
            wav = wav[:window_samples]

        label_ids = [s for s in str(row["label_ids"]).split(";") if s]
        labels[orig_pos] = _encode_labels(label_ids, species_to_idx, num_classes)

        buf_wavs.append(wav)
        buf_pos.append(orig_pos)
        if len(buf_wavs) >= batch_size:
            flush()
    flush()

    elapsed = time.time() - t0
    logger.info(
        "Extracted %d windows in %.1f s (%.2f win/s)",
        len(index_df), elapsed, len(index_df) / max(elapsed, 1e-6),
    )

    out_dir = (Path(cache_cfg["pilot_dir"]) if limit is not None else Path(cache_cfg["output_dir"])) / "soundscape_plus_nocall"
    index_to_write = index_df.drop(columns=["label_ids"]).copy()
    cache_mod.write_shard(
        out_dir=out_dir,
        index_df=index_to_write,
        embeddings=embeddings,
        labels=labels,
        species_list=species_list,
        spatial=spatial,
        perch_logits=perch_logits,
    )
    logger.info(
        "Shard written to %s (spatial=%s, perch_logits=%s)",
        out_dir, spatial is not None, perch_logits is not None,
    )
    return 0


def _run_focal(cfg: dict, shard_index: int, num_shards: int) -> int:
    data_cfg = cfg["data"]
    perch_cfg = cfg["perch"]
    cache_cfg = cfg["cache"]
    focal_cfg = cfg.get("focal", {})
    extract_cfg = cfg.get("extract", {}) or {}
    extract_frame_features = bool(extract_cfg.get("frame_features", False))
    capture_perch_logits = bool(extract_cfg.get("capture_perch_logits", False))

    audio_dir = Path(focal_cfg["audio_dir"])
    val_fraction = float(focal_cfg.get("val_fraction", 0.1))
    window_samples = data_cfg["window_samples"]
    batch_size = int(perch_cfg["batch_size"])
    split_seed = int(data_cfg["split_seed"])

    species_list = build_species_list(data_cfg["train_csv"])
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)
    logger.info("Taxonomy: %d species", num_classes)

    recordings = _build_focal_index(
        train_csv=data_cfg["train_csv"],
        species_list=species_list,
        shard_index=shard_index,
        num_shards=num_shards,
        split_seed=split_seed,
        val_fraction=val_fraction,
    )
    if len(recordings) == 0:
        logger.error("No recordings for shard %d/%d", shard_index, num_shards)
        return 2
    logger.info(
        "Shard %d/%d: %d recordings (train=%d val=%d)",
        shard_index,
        num_shards,
        len(recordings),
        int((recordings["split"] == "train").sum()),
        int((recordings["split"] == "val").sum()),
    )

    perch = _make_perch(perch_cfg)
    out_names = ["embedding"]
    if extract_frame_features:
        out_names.append("spatial_embedding")
    if capture_perch_logits:
        out_names.append("label")

    # Unknown a priori how many windows per shard — grow lists, stack once at end.
    emb_chunks: list[np.ndarray] = []
    spatial_chunks: list[np.ndarray] | None = [] if extract_frame_features else None
    logit_chunks: list[np.ndarray] | None = [] if capture_perch_logits else None
    label_chunks: list[np.ndarray] = []
    index_rows: list[dict] = []

    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []  # holds full row info for each buffered window

    def flush():
        if not buf_wavs:
            return
        batch = np.stack(buf_wavs, axis=0).astype(np.float32, copy=False)
        out = perch.run(batch, outputs=out_names)
        emb_chunks.append(out["embedding"].astype(np.float32))
        if spatial_chunks is not None:
            spatial_chunks.append(out["spatial_embedding"].astype(np.float32))
        if logit_chunks is not None:
            logit_chunks.append(out["label"].astype(np.float32))
        for k, meta in enumerate(buf_meta):
            lab = np.zeros(num_classes, dtype=np.int8)
            lab[species_to_idx[meta["primary_label"]]] = 1
            label_chunks.append(lab)
            index_rows.append(
                dict(
                    filename=meta["filename"],
                    window_idx=meta["window_idx"],
                    start_sec=meta["start_sec"],
                    source="focal",
                    split=meta["split"],
                    n_positive_labels=1,
                )
            )
        buf_wavs.clear()
        buf_meta.clear()

    t0 = time.time()
    n_failed = 0
    for _, rec in recordings.iterrows():
        path = audio_dir / rec["filename"]
        try:
            wav = audio_mod.load_audio_32k(str(path))  # full file
        except Exception as e:
            n_failed += 1
            logger.warning("Load failed %s: %s", rec["filename"], e)
            continue

        windows = audio_mod.window_audio(wav, window_samples=window_samples)
        for wi, win in enumerate(windows):
            buf_wavs.append(win)
            buf_meta.append(
                dict(
                    filename=rec["filename"],
                    window_idx=wi,
                    start_sec=float(wi * 5.0),
                    split=rec["split"],
                    primary_label=rec["primary_label"],
                )
            )
            if len(buf_wavs) >= batch_size:
                flush()
    flush()

    elapsed = time.time() - t0
    total_windows = sum(len(c) for c in emb_chunks)
    logger.info(
        "Shard %d: extracted %d windows from %d recordings in %.1f s (%.2f win/s, %d failed)",
        shard_index, total_windows, len(recordings), elapsed,
        total_windows / max(elapsed, 1e-6), n_failed,
    )
    if total_windows == 0:
        logger.error("Shard extracted 0 windows")
        return 3

    embeddings = np.concatenate(emb_chunks, axis=0)
    spatial = np.concatenate(spatial_chunks, axis=0) if spatial_chunks is not None else None
    perch_logits = np.concatenate(logit_chunks, axis=0) if logit_chunks is not None else None
    labels = np.stack(label_chunks, axis=0)
    index_df = pd.DataFrame(index_rows)

    out_dir = Path(cache_cfg["output_dir"]) / "focal" / f"shard-{shard_index:03d}"
    cache_mod.write_shard(
        out_dir=out_dir,
        index_df=index_df,
        embeddings=embeddings,
        labels=labels,
        species_list=species_list,
        spatial=spatial,
        perch_logits=perch_logits,
    )
    logger.info(
        "Shard written to %s (spatial=%s, perch_logits=%s)",
        out_dir, spatial is not None, perch_logits is not None,
    )
    return 0
