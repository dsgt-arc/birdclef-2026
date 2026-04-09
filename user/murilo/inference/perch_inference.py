#!/usr/bin/env python3
"""Perch v2 TF inference: extract embeddings and logits from soundscapes and focal audio.

Runs the frozen Perch v2 TensorFlow SavedModel on 5-second audio chunks,
saving 1536-dim embeddings and 14,795-class raw logits to parquet.

Usage:
    python user/murilo/scripts/perch_inference.py \
        --source train_soundscapes \
        --batch-size 32

    python user/murilo/scripts/perch_inference.py \
        --source all \
        --max-files 100 \
        --batch-size 16

Outputs:
    shared/inference/birdclef-2026/perch-v2-tf/{source}/
        embeddings.parquet  — (N, 5 meta cols + 1536 embedding cols)
        logits.parquet      — (N, 5 meta cols + 14795 logit cols)

Requires: tensorflow, soundfile, pandas, pyarrow, numpy, tqdm
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC  # 160,000

DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
MODEL_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2-cpu")
OUTPUT_BASE = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/inference/birdclef-2026/perch-v2-tf"
)

FLUSH_EVERY = 200  # Flush to parquet every N files


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_and_chunk(audio_path: Path) -> tuple[np.ndarray, list[dict]]:
    """Load audio file, split into non-overlapping 5s chunks.

    Returns:
        chunks: (N_chunks, WINDOW_SAMPLES) float32 array
        metadata: list of dicts with filename, source, chunk_idx, start_sec, end_sec
    """
    y, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)

    # Resample if needed (all competition data is 32kHz, but be safe)
    if file_sr != SR:
        import torchaudio

        y_tensor = __import__("torch").from_numpy(y).float()
        y_tensor = torchaudio.functional.resample(y_tensor, orig_freq=file_sr, new_freq=SR)
        y = y_tensor.numpy()

    total_samples = len(y)
    n_chunks = max(1, (total_samples + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES)

    # Pad to fill last chunk
    padded_len = n_chunks * WINDOW_SAMPLES
    if total_samples < padded_len:
        y = np.pad(y, (0, padded_len - total_samples))
    elif total_samples > padded_len:
        y = y[:padded_len]

    chunks = y.reshape(n_chunks, WINDOW_SAMPLES)
    metadata = []
    for i in range(n_chunks):
        metadata.append(
            {
                "chunk_idx": i,
                "start_sec": float(i * WINDOW_SEC),
                "end_sec": float((i + 1) * WINDOW_SEC),
            }
        )
    return chunks, metadata


# --------------------------------------------------------------------------- #
# File enumeration
# --------------------------------------------------------------------------- #
def enumerate_train_soundscapes() -> list[dict]:
    """List all soundscape files (labeled + unlabeled)."""
    ss_dir = DATA_DIR / "train_soundscapes"
    files = sorted(ss_dir.glob("*.ogg"))
    return [{"path": f, "filename": f.name, "source": "train_soundscapes"} for f in files]


def enumerate_train_audio() -> list[dict]:
    """List all focal recordings (organized in species subdirectories)."""
    ta_dir = DATA_DIR / "train_audio"
    files = sorted(ta_dir.glob("*/*.ogg"))
    return [
        {"path": f, "filename": f"{f.parent.name}/{f.name}", "source": "train_audio"} for f in files
    ]


# --------------------------------------------------------------------------- #
# Parquet I/O
# --------------------------------------------------------------------------- #
def flush_to_parquet(
    embeddings: list[np.ndarray],
    logits: list[np.ndarray],
    metadata: list[dict],
    emb_path: Path,
    logits_path: Path,
    emb_columns: list[str] | None = None,
    logit_columns: list[str] | None = None,
):
    """Append accumulated results to parquet files."""
    if not embeddings:
        return

    emb_arr = np.concatenate(embeddings, axis=0)
    logit_arr = np.concatenate(logits, axis=0)
    meta_df = pd.DataFrame(metadata)

    # Embedding parquet
    emb_cols = emb_columns or [f"emb_{i}" for i in range(emb_arr.shape[1])]
    emb_df = pd.concat([meta_df, pd.DataFrame(emb_arr, columns=emb_cols)], axis=1)

    # Logits parquet
    log_cols = logit_columns or [f"logit_{i}" for i in range(logit_arr.shape[1])]
    log_df = pd.concat([meta_df, pd.DataFrame(logit_arr, columns=log_cols)], axis=1)

    # Append or create
    for df, path in [(emb_df, emb_path), (log_df, logits_path)]:
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False, engine="pyarrow")


# --------------------------------------------------------------------------- #
# Main inference loop
# --------------------------------------------------------------------------- #
def run_inference(
    perch_fn,
    file_list: list[dict],
    output_dir: Path,
    batch_size: int = 32,
    max_files: int | None = None,
):
    """Run Perch TF inference on a list of audio files.

    Args:
        perch_fn: TF serving function (model.signatures["serving_default"])
        file_list: list of {path, filename, source} dicts
        output_dir: where to save parquet files
        batch_size: number of 5s chunks per TF call
        max_files: limit for debugging
    """
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / "embeddings.parquet"
    logits_path = output_dir / "logits.parquet"

    if max_files:
        file_list = file_list[:max_files]

    # Get label column names from a single forward pass
    dummy = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    dummy_out = perch_fn(inputs=tf.convert_to_tensor(dummy))
    n_emb = dummy_out["embedding"].shape[1]
    n_logits = dummy_out["label"].shape[1]
    emb_columns = [f"emb_{i}" for i in range(n_emb)]
    logit_columns = [f"logit_{i}" for i in range(n_logits)]
    print(f"  Perch outputs: embedding={n_emb}-dim, logits={n_logits}-dim")

    # Accumulators (flushed periodically)
    acc_emb = []
    acc_logits = []
    acc_meta = []
    total_chunks = 0
    t_start = time.time()

    for file_idx, file_info in enumerate(
        tqdm(file_list, desc=f"Perch inference ({output_dir.name})")
    ):
        chunks, chunk_meta = load_and_chunk(file_info["path"])

        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            outputs = perch_fn(inputs=tf.convert_to_tensor(batch))
            emb = outputs["embedding"].numpy().astype(np.float32)
            logit = outputs["label"].numpy().astype(np.float32)

            acc_emb.append(emb)
            acc_logits.append(logit)

            for i in range(batch_start, batch_end):
                meta = chunk_meta[i].copy()
                meta["filename"] = file_info["filename"]
                meta["source"] = file_info["source"]
                acc_meta.append(meta)

            total_chunks += batch_end - batch_start

        # Periodic flush
        if (file_idx + 1) % FLUSH_EVERY == 0:
            elapsed = time.time() - t_start
            rate = total_chunks / elapsed
            eta = (len(file_list) - file_idx - 1) * (elapsed / (file_idx + 1))
            print(
                f"  [{file_idx+1}/{len(file_list)}] {total_chunks} chunks, "
                f"{rate:.1f} chunks/s, ETA {eta/3600:.1f}h — flushing..."
            )
            flush_to_parquet(
                acc_emb,
                acc_logits,
                acc_meta,
                emb_path,
                logits_path,
                emb_columns,
                logit_columns,
            )
            acc_emb.clear()
            acc_logits.clear()
            acc_meta.clear()
            gc.collect()

    # Final flush
    if acc_emb:
        flush_to_parquet(
            acc_emb,
            acc_logits,
            acc_meta,
            emb_path,
            logits_path,
            emb_columns,
            logit_columns,
        )

    elapsed = time.time() - t_start
    print(
        f"  Done: {total_chunks} chunks from {len(file_list)} files in {elapsed/3600:.2f}h "
        f"({total_chunks/elapsed:.1f} chunks/s)"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Perch v2 TF inference")
    parser.add_argument(
        "--source",
        choices=["train_soundscapes", "train_audio", "all"],
        required=True,
        help="Which data to process",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Chunks per TF call")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files (debug)")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Perch TF model path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    # Lazy TF import (avoid loading TF if just checking --help)
    print("Loading TensorFlow...")
    t0 = time.time()
    # Suppress TF verbose logging
    import os

    import tensorflow as tf

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")

    # Configure TF threading for CPU parallelism
    n_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 8))
    tf.config.threading.set_inter_op_parallelism_threads(n_threads)
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)

    print(f"  TF {tf.__version__} loaded in {time.time()-t0:.1f}s (threads={n_threads})")

    # Load Perch model
    print(f"Loading Perch v2 TF model from {args.model_dir}...")
    t0 = time.time()
    model = tf.saved_model.load(args.model_dir)
    perch_fn = model.signatures["serving_default"]
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Build file lists
    sources = []
    if args.source in ("train_soundscapes", "all"):
        sources.append(("train_soundscapes", enumerate_train_soundscapes()))
    if args.source in ("train_audio", "all"):
        sources.append(("train_audio", enumerate_train_audio()))

    for source_name, file_list in sources:
        output_dir = OUTPUT_BASE / source_name
        emb_path = output_dir / "embeddings.parquet"

        if emb_path.exists() and not args.overwrite:
            print(f"  Skipping {source_name} — output exists. Use --overwrite to rerun.")
            continue

        if args.overwrite and emb_path.exists():
            # Remove existing files to start fresh
            for f in output_dir.glob("*.parquet"):
                f.unlink()

        n_files = min(len(file_list), args.max_files) if args.max_files else len(file_list)
        print(f"\nProcessing {source_name}: {n_files} files (of {len(file_list)} total)")
        run_inference(
            perch_fn,
            file_list,
            output_dir,
            batch_size=args.batch_size,
            max_files=args.max_files,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
