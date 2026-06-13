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
from pathlib import Path

import numpy as np

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
