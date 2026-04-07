"""Extract Perch v2 embedding + label outputs to parquet.

Three sources:

1. **soundscape-labeled** — the 66 labeled soundscape files (739 windows),
   with ground-truth species from train_soundscapes_labels.csv.

2. **soundscape-unlabeled** — all remaining soundscape files (~10,592 files,
   ~127k windows). No ground-truth labels. Sharded for parallelism.

3. **focal** — all focal recordings from train.csv (~35,549 files). Sharded.

Output format (parquet per shard):
    filename        str     — audio filename
    window_idx      int     — 0-based window within the file
    start_sec       float   — window start in seconds
    source          str     — "soundscape-labeled" / "soundscape-unlabeled" / "focal"
    split           str     — "train" / "val" (for labeled data) or "" (unlabeled)
    primary_label   str     — ground-truth label (focal/labeled) or "" (unlabeled)
    ground_truth    str     — semicolon-joined species IDs or "" (unlabeled)
    embedding       list[float32] — Perch pooled embedding (1536-d)
    perch_logits    list[float32] — Perch label output (14795-d)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from birdclef_106 import audio as audio_mod
from birdclef_106.perch_onnx import PerchSession

logger = logging.getLogger("birdclef_106.extract")

WINDOW_SAMPLES = 160000  # 5s at 32kHz


def _time_to_sec(t: str) -> int:
    parts = str(t).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _make_perch(cfg: dict) -> PerchSession:
    logger.info("Loading Perch ONNX: %s :: %s", cfg["hf_repo"], cfg["hf_filename"])
    return PerchSession(repo_id=cfg["hf_repo"], filename=cfg["hf_filename"])


def _process_batch(
    perch: PerchSession,
    buf_wavs: list[np.ndarray],
    buf_meta: list[dict],
    meta_rows: list[dict],
    emb_list: list[np.ndarray],
    logit_list: list[np.ndarray],
) -> None:
    """Run a batch through Perch and append results to accumulator lists."""
    if not buf_wavs:
        return
    batch = np.stack(buf_wavs, axis=0).astype(np.float32, copy=False)
    out = perch.run(batch, outputs=["embedding", "label"])
    for k, meta in enumerate(buf_meta):
        meta_rows.append(meta)
        emb_list.append(out["embedding"][k].copy())
        logit_list.append(out["label"][k].copy())
    buf_wavs.clear()
    buf_meta.clear()


def _write_parquet(
    output_path: str,
    meta_rows: list[dict],
    emb_list: list[np.ndarray],
    logit_list: list[np.ndarray],
) -> None:
    """Write metadata + vector columns to parquet via polars."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n = len(meta_rows)
    if n == 0:
        logger.warning("No rows to write")
        return

    # Build metadata DataFrame
    meta_df = pl.DataFrame(meta_rows)

    # Stack vectors into 2-D arrays, then convert to polars list columns
    emb_matrix = np.stack(emb_list)       # (N, 1536)
    logit_matrix = np.stack(logit_list)   # (N, 14795)

    vectors_df = pl.DataFrame({
        "embedding": emb_matrix.tolist(),
        "perch_logits": logit_matrix.tolist(),
    }).cast({
        "embedding": pl.List(pl.Float32),
        "perch_logits": pl.List(pl.Float32),
    })

    df = pl.concat([meta_df, vectors_df], how="horizontal")
    df.write_parquet(output_path)
    logger.info("Written: %s (%d rows, %.1f MB)", output_path, n, Path(output_path).stat().st_size / 1e6)


def extract_soundscape_labeled(cfg: dict, output_path: str) -> int:
    """Extract features from labeled soundscape windows."""
    data = cfg["data"]
    perch = _make_perch(cfg["perch"])
    batch_size = cfg["perch"]["batch_size"]
    sdir = Path(data["soundscape_dir"])

    labels_df = pl.read_csv(data["soundscape_labels_csv"], infer_schema_length=10000).unique(
        subset=["filename", "start"]
    )
    logger.info("Labeled windows: %d (after dedup)", len(labels_df))

    # Build split (reuse exp 102/105 seed=42 split)
    all_files = sorted(labels_df["filename"].unique().to_list())
    rng = np.random.RandomState(data["split_seed"])
    rng.shuffle(all_files)
    mid = len(all_files) // 2
    train_files = set(all_files[:mid])

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    logit_list: list[np.ndarray] = []
    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []
    n_skipped = 0

    t0 = time.time()
    for row in labels_df.iter_rows(named=True):
        fname = row["filename"]
        path = sdir / fname
        if not path.exists():
            n_skipped += 1
            continue
        start_sec = _time_to_sec(row["start"])
        try:
            wav = audio_mod.load_audio_32k(str(path), start_sec=start_sec, duration_sec=5.0)
        except Exception as e:
            logger.warning("Load failed %s @ %d: %s", fname, start_sec, e)
            n_skipped += 1
            continue

        if len(wav) < WINDOW_SAMPLES:
            wav = np.pad(wav, (0, WINDOW_SAMPLES - len(wav)))
        else:
            wav = wav[:WINDOW_SAMPLES]

        split = "train" if fname in train_files else "val"
        gt = str(row["primary_label"]).strip()

        buf_wavs.append(wav)
        buf_meta.append(dict(
            filename=fname,
            window_idx=int(start_sec // 5),
            start_sec=float(start_sec),
            source="soundscape-labeled",
            split=split,
            primary_label=gt.split(";")[0] if gt else "",
            ground_truth=gt,
        ))
        if len(buf_wavs) >= batch_size:
            _process_batch(perch, buf_wavs, buf_meta, meta_rows, emb_list, logit_list)

    _process_batch(perch, buf_wavs, buf_meta, meta_rows, emb_list, logit_list)
    elapsed = time.time() - t0
    logger.info(
        "Labeled soundscapes: %d windows in %.1fs (skipped %d)",
        len(meta_rows), elapsed, n_skipped,
    )

    _write_parquet(output_path, meta_rows, emb_list, logit_list)
    return 0


def extract_soundscape_unlabeled(
    cfg: dict, output_path: str, shard_index: int = 0, num_shards: int = 1
) -> int:
    """Extract features from unlabeled soundscape files (all 12 windows per file)."""
    data = cfg["data"]
    perch = _make_perch(cfg["perch"])
    batch_size = cfg["perch"]["batch_size"]
    sdir = Path(data["soundscape_dir"])

    # Get labeled filenames to exclude
    labels_df = pl.read_csv(data["soundscape_labels_csv"], infer_schema_length=10000)
    labeled_files = set(labels_df["filename"].unique().to_list())

    # All soundscape files minus labeled ones
    all_files = sorted(p.name for p in sdir.glob("*.ogg"))
    unlabeled_files = [f for f in all_files if f not in labeled_files]
    logger.info("Total soundscape files: %d, unlabeled: %d", len(all_files), len(unlabeled_files))

    # Shard assignment
    unlabeled_files = [f for i, f in enumerate(unlabeled_files) if i % num_shards == shard_index]
    logger.info("Shard %d/%d: %d files", shard_index, num_shards, len(unlabeled_files))

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    logit_list: list[np.ndarray] = []
    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []
    n_failed = 0

    t0 = time.time()
    for fi, fname in enumerate(unlabeled_files):
        if fi % 500 == 0 and fi > 0:
            logger.info("Progress: %d/%d files, %d windows", fi, len(unlabeled_files), len(meta_rows))

        path = sdir / fname
        try:
            wav = audio_mod.load_audio_32k(str(path))
        except Exception as e:
            logger.warning("Load failed %s: %s", fname, e)
            n_failed += 1
            continue

        windows = audio_mod.window_audio(wav, window_samples=WINDOW_SAMPLES)
        for wi, win in enumerate(windows):
            buf_wavs.append(win)
            buf_meta.append(dict(
                filename=fname,
                window_idx=wi,
                start_sec=float(wi * 5.0),
                source="soundscape-unlabeled",
                split="",
                primary_label="",
                ground_truth="",
            ))
            if len(buf_wavs) >= batch_size:
                _process_batch(perch, buf_wavs, buf_meta, meta_rows, emb_list, logit_list)

    _process_batch(perch, buf_wavs, buf_meta, meta_rows, emb_list, logit_list)
    elapsed = time.time() - t0
    logger.info(
        "Unlabeled shard %d: %d windows from %d files in %.1fs (%d failed)",
        shard_index, len(meta_rows), len(unlabeled_files), elapsed, n_failed,
    )

    _write_parquet(output_path, meta_rows, emb_list, logit_list)
    return 0


def extract_focal(
    cfg: dict, output_path: str, shard_index: int = 0, num_shards: int = 1
) -> int:
    """Extract features from focal recordings."""
    data = cfg["data"]
    focal = cfg["focal"]
    perch = _make_perch(cfg["perch"])
    batch_size = cfg["perch"]["batch_size"]
    audio_dir = Path(focal["audio_dir"])

    train_df = pl.read_csv(data["train_csv"], infer_schema_length=10000).sort("filename")

    # Shard by index
    train_df = train_df.with_row_index("_idx").filter(
        pl.col("_idx") % num_shards == shard_index
    ).drop("_idx")
    logger.info("Focal shard %d/%d: %d recordings", shard_index, num_shards, len(train_df))

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    logit_list: list[np.ndarray] = []
    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []
    n_failed = 0

    t0 = time.time()
    for fi, rec in enumerate(train_df.iter_rows(named=True)):
        if fi % 1000 == 0 and fi > 0:
            logger.info("Progress: %d/%d files, %d windows", fi, len(train_df), len(meta_rows))

        path = audio_dir / rec["filename"]
        try:
            wav = audio_mod.load_audio_32k(str(path))
        except Exception as e:
            logger.warning("Load failed %s: %s", rec["filename"], e)
            n_failed += 1
            continue

        primary = str(rec["primary_label"])
        secondary = str(rec.get("secondary_labels", ""))
        gt = primary
        if secondary and secondary not in ("", "None", "nan", "null"):
            gt = f"{primary};{secondary}"

        windows = audio_mod.window_audio(wav, window_samples=WINDOW_SAMPLES)
        for wi, win in enumerate(windows):
            buf_wavs.append(win)
            buf_meta.append(dict(
                filename=rec["filename"],
                window_idx=wi,
                start_sec=float(wi * 5.0),
                source="focal",
                split="",
                primary_label=primary,
                ground_truth=gt,
            ))
            if len(buf_wavs) >= batch_size:
                _process_batch(perch, buf_wavs, buf_meta, meta_rows, emb_list, logit_list)

    _process_batch(perch, buf_wavs, buf_meta, meta_rows, emb_list, logit_list)
    elapsed = time.time() - t0
    logger.info(
        "Focal shard %d: %d windows from %d recordings in %.1fs (%d failed)",
        shard_index, len(meta_rows), len(train_df), elapsed, n_failed,
    )

    _write_parquet(output_path, meta_rows, emb_list, logit_list)
    return 0
