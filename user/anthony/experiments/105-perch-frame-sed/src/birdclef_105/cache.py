"""Feature cache on disk.

A shard is a directory containing:
    meta.json       — species list, num_windows, dtypes, created_at
    index.parquet   — one row per window (filename, window_idx, start_sec,
                      source, split, n_positive_labels)
    embedding.npy   — float16 [N, 1536]
    labels.npy      — int8    [N, num_classes]

Rows in the index file are in the same order as rows in the numpy arrays.
Loading uses memmap so shard reads are zero-copy.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_shard(
    out_dir: str | Path,
    index_df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    species_list: list[str],
    spatial: np.ndarray | None = None,
) -> None:
    """Write a feature shard to disk.

    Args:
        out_dir: Directory to create (will be mkdir'd).
        index_df: One row per window with columns
            ['filename', 'window_idx', 'start_sec', 'source', 'split', 'n_positive_labels'].
        embeddings: `(N, 1536)` array — will be cast to float16.
        labels: `(N, num_classes)` array — will be cast to int8.
        species_list: Sorted list of species IDs matching the label column order.
        spatial: Optional `(N, 16, 4, 1536)` frame-feature array — cast to float16.
            When present, written alongside as `spatial.npy` and recorded in meta.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = len(index_df)
    if embeddings.shape[0] != n or labels.shape[0] != n:
        raise ValueError(
            f"Shape mismatch: index has {n} rows, "
            f"embeddings has {embeddings.shape[0]}, labels has {labels.shape[0]}"
        )
    if embeddings.shape[1] != 1536:
        raise ValueError(f"Expected embedding dim 1536, got {embeddings.shape[1]}")
    if labels.shape[1] != len(species_list):
        raise ValueError(
            f"Labels have {labels.shape[1]} classes but species_list has {len(species_list)}"
        )
    if spatial is not None:
        if spatial.shape[0] != n:
            raise ValueError(
                f"Spatial has {spatial.shape[0]} rows, expected {n}"
            )
        if spatial.shape[1:] != (16, 4, 1536):
            raise ValueError(
                f"Expected spatial shape (N, 16, 4, 1536), got (N, {spatial.shape[1:]})"
            )

    np.save(out / "embedding.npy", embeddings.astype(np.float16, copy=False))
    np.save(out / "labels.npy", labels.astype(np.int8, copy=False))
    index_df.to_parquet(out / "index.parquet", index=False)
    if spatial is not None:
        np.save(out / "spatial.npy", spatial.astype(np.float16, copy=False))

    meta = {
        "num_windows": int(n),
        "num_classes": int(labels.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "species_list": list(species_list),
        "embedding_dtype": "float16",
        "labels_dtype": "int8",
        "has_spatial": spatial is not None,
        "spatial_shape": list(spatial.shape[1:]) if spatial is not None else None,
        "spatial_dtype": "float16" if spatial is not None else None,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def read_shard(in_dir: str | Path) -> dict:
    """Read a feature shard from disk using memmap for arrays.

    Returns a dict with keys: `index` (DataFrame), `embeddings` (memmap ndarray),
    `labels` (memmap ndarray), `spatial` (memmap ndarray or None),
    `species_list` (list[str]), `meta` (dict).
    """
    path = Path(in_dir)
    meta = json.loads((path / "meta.json").read_text())
    index_df = pd.read_parquet(path / "index.parquet")
    embeddings = np.load(path / "embedding.npy", mmap_mode="r")
    labels = np.load(path / "labels.npy", mmap_mode="r")
    spatial = None
    if meta.get("has_spatial") and (path / "spatial.npy").exists():
        spatial = np.load(path / "spatial.npy", mmap_mode="r")
    return {
        "index": index_df,
        "embeddings": embeddings,
        "labels": labels,
        "spatial": spatial,
        "species_list": meta["species_list"],
        "meta": meta,
    }
