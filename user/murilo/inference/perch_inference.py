#!/usr/bin/env python3
"""Perch v2 TF inference: extract embeddings and logits from soundscapes and focal audio.

Runs the frozen Perch v2 TensorFlow SavedModel on 5-second audio chunks,
saving 1536-dim embeddings and 14,795-class raw logits to parquet.

Usage:
    python perch_inference.py --source train_soundscapes --batch-size 32
    python perch_inference.py --source train_audio --batch-size 16

    # Resume after interrupted job (default — skips already-processed files):
    python perch_inference.py --source train_audio

    # Start fresh, discarding any partial results:
    python perch_inference.py --source train_audio --overwrite

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
# Chunked parquet I/O (memory-safe)
# --------------------------------------------------------------------------- #
def flush_chunk(
    embeddings: list[np.ndarray],
    logits: list[np.ndarray],
    metadata: list[dict],
    output_dir: Path,
    chunk_idx: int,
    emb_columns: list[str],
    logit_columns: list[str],
):
    """Write one batch of results to numbered chunk parquet files.

    Each flush writes a small independent file (~100MB). No existing files are
    reloaded, so memory stays constant regardless of total dataset size.
    """
    if not embeddings:
        return

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    emb_arr = np.concatenate(embeddings, axis=0)
    logit_arr = np.concatenate(logits, axis=0)
    meta_df = pd.DataFrame(metadata)

    emb_df = pd.concat([meta_df, pd.DataFrame(emb_arr, columns=emb_columns)], axis=1)
    log_df = pd.concat([meta_df, pd.DataFrame(logit_arr, columns=logit_columns)], axis=1)

    emb_df.to_parquet(chunks_dir / f"emb_{chunk_idx:04d}.parquet", index=False, engine="pyarrow")
    log_df.to_parquet(chunks_dir / f"log_{chunk_idx:04d}.parquet", index=False, engine="pyarrow")


def get_processed_filenames(output_dir: Path) -> set[str]:
    """Get filenames already processed from main files and/or chunk files.

    Reads only the 'filename' column (a few MB) rather than full data.
    """
    filenames: set[str] = set()

    # Check main embeddings file (much smaller than logits)
    emb_path = output_dir / "embeddings.parquet"
    if emb_path.exists():
        try:
            df = pd.read_parquet(emb_path, columns=["filename"])
            filenames.update(df["filename"].unique())
            print(f"  Found {len(filenames):,} files in existing {emb_path.name}")
        except Exception as e:
            print(f"  Warning: could not read {emb_path}: {e}")

    # Check chunk files
    chunks_dir = output_dir / "chunks"
    if chunks_dir.exists():
        chunk_files = sorted(chunks_dir.glob("emb_*.parquet"))
        for f in chunk_files:
            try:
                df = pd.read_parquet(f, columns=["filename"])
                filenames.update(df["filename"].unique())
            except Exception as e:
                print(f"  Warning: could not read chunk {f.name}: {e}")
        if chunk_files:
            print(f"  Found {len(chunk_files)} chunk files in {chunks_dir}")

    return filenames


def merge_chunks(output_dir: Path):
    """Merge main parquet + chunk files into final output using streaming writes.

    Memory-efficient: reads one source file at a time in 50K-row batches,
    so peak memory is ~100-200 MB regardless of total dataset size.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    chunks_dir = output_dir / "chunks"
    if not chunks_dir.exists():
        return

    for name, prefix in [("embeddings", "emb"), ("logits", "log")]:
        main_path = output_dir / f"{name}.parquet"
        chunk_files = sorted(chunks_dir.glob(f"{prefix}_*.parquet"))

        if not chunk_files:
            continue

        # Collect all sources: existing main file first, then chunks in order
        sources: list[Path] = []
        if main_path.exists():
            sources.append(main_path)
        sources.extend(chunk_files)

        print(f"  Merging {name}: {len(sources)} source(s)...")
        tmp_path = output_dir / f"{name}_merged.parquet"

        writer = None
        total_rows = 0
        for src in sources:
            pf = pq.ParquetFile(src)
            for batch in pf.iter_batches(batch_size=50_000):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(str(tmp_path), table.schema)
                writer.write_table(table)
                total_rows += len(batch)

        if writer:
            writer.close()
            # Atomic replace
            if main_path.exists():
                main_path.unlink()
            tmp_path.rename(main_path)
            size_gb = main_path.stat().st_size / 1e9
            print(f"    {name}.parquet: {total_rows:,} rows, {size_gb:.2f} GB")

        # Clean up chunk files
        for f in chunk_files:
            f.unlink()

    # Remove chunks dir if empty
    if chunks_dir.exists() and not any(chunks_dir.iterdir()):
        chunks_dir.rmdir()
        print("  Cleaned up chunks directory")


# --------------------------------------------------------------------------- #
# Main inference loop
# --------------------------------------------------------------------------- #
def run_inference(
    perch_fn,
    file_list: list[dict],
    output_dir: Path,
    batch_size: int = 32,
    max_files: int | None = None,
    skip_filenames: set[str] | None = None,
):
    """Run Perch TF inference on a list of audio files.

    Args:
        perch_fn: TF serving function (model.signatures["serving_default"])
        file_list: list of {path, filename, source} dicts
        output_dir: where to save parquet files
        batch_size: number of 5s chunks per TF call
        max_files: limit for debugging
        skip_filenames: set of filenames to skip (for resume)
    """
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)

    if max_files:
        file_list = file_list[:max_files]

    # Filter out already-processed files
    if skip_filenames:
        original_count = len(file_list)
        file_list = [f for f in file_list if f["filename"] not in skip_filenames]
        skipped = original_count - len(file_list)
        print(f"  Resuming: {skipped:,} already done, {len(file_list):,} remaining")

    if not file_list:
        print("  All files already processed — nothing to do.")
        return

    # Get label column names from a single forward pass
    dummy = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    dummy_out = perch_fn(inputs=tf.convert_to_tensor(dummy))
    n_emb = dummy_out["embedding"].shape[1]
    n_logits = dummy_out["label"].shape[1]
    emb_columns = [f"emb_{i}" for i in range(n_emb)]
    logit_columns = [f"logit_{i}" for i in range(n_logits)]
    print(f"  Perch outputs: embedding={n_emb}-dim, logits={n_logits}-dim")

    # Determine starting chunk index (continue numbering from existing chunks)
    chunks_dir = output_dir / "chunks"
    if chunks_dir.exists():
        existing_chunks = list(chunks_dir.glob("emb_*.parquet"))
        chunk_idx = len(existing_chunks)
    else:
        chunk_idx = 0

    # Accumulators (flushed periodically as independent chunk files)
    acc_emb: list[np.ndarray] = []
    acc_logits: list[np.ndarray] = []
    acc_meta: list[dict] = []
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

        # Periodic flush to independent chunk file (no reload)
        if (file_idx + 1) % FLUSH_EVERY == 0:
            elapsed = time.time() - t_start
            rate = total_chunks / elapsed if elapsed > 0 else 0
            remaining_files = len(file_list) - file_idx - 1
            eta = remaining_files * (elapsed / (file_idx + 1)) if file_idx > 0 else 0
            print(
                f"  [{file_idx+1}/{len(file_list)}] {total_chunks} chunks, "
                f"{rate:.1f} chunks/s, ETA {eta/3600:.1f}h — flushing chunk {chunk_idx}..."
            )
            flush_chunk(
                acc_emb,
                acc_logits,
                acc_meta,
                output_dir,
                chunk_idx,
                emb_columns,
                logit_columns,
            )
            chunk_idx += 1
            acc_emb.clear()
            acc_logits.clear()
            acc_meta.clear()
            gc.collect()

    # Final flush
    if acc_emb:
        print(f"  Final flush: chunk {chunk_idx} ({total_chunks} total new chunks)")
        flush_chunk(
            acc_emb,
            acc_logits,
            acc_meta,
            output_dir,
            chunk_idx,
            emb_columns,
            logit_columns,
        )

    elapsed = time.time() - t_start
    rate = total_chunks / elapsed if elapsed > 0 else 0
    print(
        f"  Done: {total_chunks:,} chunks from {len(file_list):,} files "
        f"in {elapsed/3600:.2f}h ({rate:.1f} chunks/s)"
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard partial results and start fresh",
    )
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

        if args.overwrite:
            # Clean everything for fresh start
            if output_dir.exists():
                for f in output_dir.glob("*.parquet"):
                    f.unlink()
                chunks_dir = output_dir / "chunks"
                if chunks_dir.exists():
                    for f in chunks_dir.glob("*.parquet"):
                        f.unlink()
                    chunks_dir.rmdir()
            skip_filenames = None
        else:
            # Resume: find already-processed files
            skip_filenames = get_processed_filenames(output_dir)

        n_files = min(len(file_list), args.max_files) if args.max_files else len(file_list)
        print(f"\nProcessing {source_name}: {n_files} files (of {len(file_list)} total)")

        run_inference(
            perch_fn,
            file_list,
            output_dir,
            batch_size=args.batch_size,
            max_files=args.max_files,
            skip_filenames=skip_filenames,
        )

        # Merge chunk files into final parquet
        chunks_dir = output_dir / "chunks"
        if chunks_dir.exists() and any(chunks_dir.glob("*.parquet")):
            print(f"\nMerging chunks for {source_name}...")
            merge_chunks(output_dir)
            print()

    print("\nAll done.")


if __name__ == "__main__":
    main()
