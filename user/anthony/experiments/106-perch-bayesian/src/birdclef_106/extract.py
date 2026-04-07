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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
    """Write metadata + vector columns to parquet via pyarrow."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n = len(meta_rows)
    if n == 0:
        logger.warning("No rows to write")
        return

    # Build metadata columns
    meta_df = pd.DataFrame(meta_rows)

    # Build arrow arrays for vector columns (avoids Python list overhead)
    emb_array = pa.FixedSizeListArray.from_arrays(
        pa.array(np.stack(emb_list).ravel(), type=pa.float32()),
        list_size=emb_list[0].shape[0],
    )
    logit_array = pa.FixedSizeListArray.from_arrays(
        pa.array(np.stack(logit_list).ravel(), type=pa.float32()),
        list_size=logit_list[0].shape[0],
    )

    table = pa.Table.from_pandas(meta_df)
    table = table.append_column("embedding", emb_array)
    table = table.append_column("perch_logits", logit_array)

    pq.write_table(table, output_path)
    logger.info("Written: %s (%d rows)", output_path, n)


def extract_soundscape_labeled(cfg: dict, output_path: str) -> int:
    """Extract features from labeled soundscape windows."""
    data = cfg["data"]
    perch = _make_perch(cfg["perch"])
    batch_size = cfg["perch"]["batch_size"]
    sdir = Path(data["soundscape_dir"])

    labels_df = pd.read_csv(data["soundscape_labels_csv"]).drop_duplicates(
        subset=["filename", "start"]
    )
    logger.info("Labeled windows: %d (after dedup)", len(labels_df))

    # Build split (reuse exp 102/105 seed=42 split)
    all_files = sorted(labels_df["filename"].unique())
    rng = np.random.RandomState(data["split_seed"])
    rng.shuffle(all_files)
    mid = len(all_files) // 2
    train_files = set(all_files[:mid])
    val_files = set(all_files[mid:])

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    logit_list: list[np.ndarray] = []
    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []
    n_skipped = 0

    t0 = time.time()
    for _, row in labels_df.iterrows():
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
    labels_df = pd.read_csv(data["soundscape_labels_csv"])
    labeled_files = set(labels_df["filename"].unique())

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

    train_df = pd.read_csv(data["train_csv"])
    train_df = train_df.sort_values("filename").reset_index(drop=True)

    # Shard by index
    train_df = train_df.iloc[shard_index::num_shards].reset_index(drop=True)
    logger.info("Focal shard %d/%d: %d recordings", shard_index, num_shards, len(train_df))

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    logit_list: list[np.ndarray] = []
    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []
    n_failed = 0

    t0 = time.time()
    for fi, (_, rec) in enumerate(train_df.iterrows()):
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
        if secondary and secondary != "nan":
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
