"""BirdNET embeddings for the labeled windows via bioacoustics-model-zoo (bmz).

BirdNET is not a `birdclef.encoders.AudioEncoder` (it is a TFLite model served
through bmz, native 3 s windows), so it runs in its own TF venv
(`sbatch/_setup-venv-birdnet.sh`) and is imported lazily — the torch venv never
touches TensorFlow.

Each labeled 5 s window [t, t+5) is represented by the mean of the BirdNET 3 s
windows whose start falls in that span (BirdNET embeds at a 1 s step by default),
giving one 1024-d vector per labeled window in the unified schema.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from birdclef_300.data_loading import labeled_parquet_path
from birdclef_300.extract import _write_unified_parquet, labeled_window_rows

logger = logging.getLogger("birdclef_300.extract_birdnet")

WINDOW_SEC = 5.0


def _file_embeddings(model, path: str):
    """Run BirdNET on one file -> (start_times, embeddings) sorted by start_time."""
    embed_df = model.embed([path], return_preds=False)
    embed_df = embed_df.reset_index()
    # bmz indexes embeddings by (file, start_time, end_time).
    start_col = "start_time" if "start_time" in embed_df.columns else embed_df.columns[1]
    feat_cols = [c for c in embed_df.columns if isinstance(c, int) or str(c).isdigit()]
    if not feat_cols:  # fall back: everything past the index columns is a feature
        feat_cols = [c for c in embed_df.columns if c not in ("file", start_col, "end_time")]
    starts = embed_df[start_col].to_numpy(dtype=np.float32)
    feats = embed_df[feat_cols].to_numpy(dtype=np.float32)
    order = np.argsort(starts)
    return starts[order], feats[order]


def extract_birdnet_labeled(
    cfg: dict,
    variant: str = "frozen",
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> int:
    import bioacoustics_model_zoo as bmz

    sdir = Path(cfg["data"]["soundscape_dir"])
    output_path = output_path or labeled_parquet_path(cfg["output"]["base_dir"], "birdnet", variant)

    rows = labeled_window_rows(cfg, limit=limit)
    logger.info("BirdNET: %d labeled windows over %d files",
                len(rows), len({r["filename"] for r in rows}))

    model = bmz.list_models()["BirdNET"]()

    # Embed each soundscape file once, then pool 3 s windows into each 5 s window.
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    n_skipped = 0

    for row in rows:
        fname = row["filename"]
        if fname not in cache:
            path = sdir / fname
            if not path.exists():
                cache[fname] = (np.empty(0), np.empty((0, 0)))
            else:
                try:
                    cache[fname] = _file_embeddings(model, str(path))
                except Exception as e:  # noqa: BLE001
                    logger.warning("BirdNET embed failed for %s: %s", fname, e)
                    cache[fname] = (np.empty(0), np.empty((0, 0)))
        starts, feats = cache[fname]
        if len(starts) == 0:
            n_skipped += 1
            continue
        t = row["start_sec"]
        sel = (starts >= t) & (starts < t + WINDOW_SEC)
        if not sel.any():  # nearest 3 s window if none start inside the 5 s span
            sel = np.zeros(len(starts), dtype=bool)
            sel[int(np.argmin(np.abs(starts - t)))] = True
        meta_rows.append({k: row[k] for k in
                          ("filename", "window_idx", "start_sec", "split", "primary_label", "ground_truth")})
        emb_list.append(feats[sel].mean(axis=0).astype(np.float32))

    logger.info("BirdNET: %d windows embedded (skipped %d)", len(meta_rows), n_skipped)
    _write_unified_parquet(output_path, meta_rows, emb_list, "birdnet", variant)
    return 0


def extract_birdnet_focal(
    cfg: dict,
    variant: str = "frozen",
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
) -> int:
    """BirdNET focal embeddings for a shard of the training recordings -> deploy cache.

    Tiles each recording into 5 s windows; each window is the mean of the BirdNET 3 s
    windows whose start falls inside it (the same pooling as the soundscape path).
    Writes one shard in the flat deploy layout and links the soundscape parquet.
    """
    import bioacoustics_model_zoo as bmz

    from birdclef_300.extract import (
        _focal_ground_truth,
        focal_shard_path,
        link_soundscape_parquet,
    )

    audio_dir = Path(cfg["focal"]["audio_dir"]).expanduser()
    train_df = pl.read_csv(cfg["data"]["train_csv"], infer_schema_length=10000).sort("filename")
    train_df = train_df.with_row_index("_idx").filter(
        pl.col("_idx") % num_shards == shard_index
    ).drop("_idx")
    if limit is not None:
        train_df = train_df.head(limit)
    logger.info("BirdNET focal shard %d/%d: %d recordings", shard_index, num_shards, len(train_df))

    model = bmz.list_models()["BirdNET"]()

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    n_failed = 0
    t0 = time.time()
    for fi, rec in enumerate(train_df.iter_rows(named=True)):
        if fi % 500 == 0 and fi > 0:
            logger.info("Progress: %d/%d files, %d windows", fi, len(train_df), len(meta_rows))
        path = audio_dir / rec["filename"]
        if not path.exists():
            n_failed += 1
            continue
        try:
            starts, feats = _file_embeddings(model, str(path))
        except Exception as e:  # noqa: BLE001
            logger.warning("BirdNET embed failed for %s: %s", rec["filename"], e)
            n_failed += 1
            continue
        if len(starts) == 0:
            n_failed += 1
            continue
        primary, gt = _focal_ground_truth(rec)
        n_windows = int(starts.max() // WINDOW_SEC) + 1
        for wi in range(n_windows):
            t = wi * WINDOW_SEC
            sel = (starts >= t) & (starts < t + WINDOW_SEC)
            if not sel.any():  # 3 s windows don't tile 5 s evenly; skip empty spans
                continue
            meta_rows.append(dict(
                filename=rec["filename"],
                window_idx=wi,
                start_sec=float(t),
                split="",
                primary_label=primary,
                ground_truth=gt,
            ))
            emb_list.append(feats[sel].mean(axis=0).astype(np.float32))

    logger.info("BirdNET focal shard %d: %d windows from %d recordings in %.1fs (%d failed)",
                shard_index, len(meta_rows), len(train_df), time.time() - t0, n_failed)
    out = focal_shard_path(cfg, "birdnet", shard_index)
    _write_unified_parquet(out, meta_rows, emb_list, "birdnet", variant, source="focal")
    link_soundscape_parquet(cfg, "birdnet", variant)
    return 0
