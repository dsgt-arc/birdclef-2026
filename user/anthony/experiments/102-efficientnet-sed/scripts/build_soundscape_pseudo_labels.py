#!/usr/bin/env python3
"""Generate pseudo-labels for unlabeled soundscapes using ONNX model on CPU.

Designed to run as a SLURM array job — each task processes a shard of the
soundscape files. Results are saved as per-shard parquets then merged.

Each 60s soundscape file is split into 12 × 5s windows. For each window,
the ONNX model outputs clip-level probabilities for all 206 species.

Usage (single):
    python scripts/build_soundscape_pseudo_labels.py \
        --onnx-dir artifacts/weights/lora-attention/ \
        --output-dir artifacts/pseudo_labels/parts/

Usage (array job — SLURM_ARRAY_TASK_ID selects the shard):
    # See sbatch/build-pseudo-labels.sbatch
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

SOUNDSCAPE_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes")
LABELS_CSV = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes_labels.csv")
CHUNK_DURATION = 5.0


def compute_mel(audio, mel_cfg):
    """Compute normalized log-mel (same as Kaggle submission kernel)."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"],
        fmax=mel_cfg["fmax"],
        power=2.0,
    )
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)

    target_frames = mel_cfg.get("target_frames", 625)
    T = mel.shape[1]
    if T > target_frames:
        mel = mel[:, :target_frames]
    elif T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def load_audio_chunk(audio_path, sr, start_s, duration_s):
    """Load a chunk of audio and resample to target sr."""
    info = sf.info(audio_path)
    source_sr = info.samplerate
    total_duration = info.frames / source_sr

    actual_duration = min(duration_s, total_duration - start_s)
    if actual_duration <= 0:
        return np.zeros(int(sr * duration_s), dtype=np.float32)

    audio, _ = sf.read(
        audio_path, dtype="float32",
        start=int(start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    target_samples = int(sr * duration_s)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def main():
    parser = argparse.ArgumentParser(description="Build soundscape pseudo-labels")
    parser.add_argument("--onnx-dir", required=True, help="Dir with model.onnx + mel_config.json + species_list.json")
    parser.add_argument("--output-dir", default="artifacts/pseudo_labels/parts/")
    parser.add_argument("--n-shards", type=int, default=50, help="Total number of shards (array tasks)")
    parser.add_argument("--shard-id", type=int, default=None, help="This shard (0-indexed). Uses SLURM_ARRAY_TASK_ID if not set.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Determine shard
    shard_id = args.shard_id
    if shard_id is None:
        shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_shards = args.n_shards

    # Load ONNX model
    import onnxruntime as ort

    onnx_dir = Path(args.onnx_dir)
    with open(onnx_dir / "mel_config.json") as f:
        mel_cfg = json.load(f)
    with open(onnx_dir / "species_list.json") as f:
        species_list = json.load(f)

    session = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_is_probs = mel_cfg.get("output_is_probs", False)

    logger.info(f"ONNX loaded: {len(species_list)} species, output_is_probs={output_is_probs}")

    # Get list of unlabeled soundscape files (exclude labeled ones)
    labeled_files = set()
    if LABELS_CSV.exists():
        labels_df = pd.read_csv(LABELS_CSV)
        labeled_files = set(labels_df["filename"].unique())
    logger.info(f"Excluding {len(labeled_files)} labeled soundscape files")

    all_files = sorted(f.name for f in SOUNDSCAPE_DIR.glob("*.ogg"))
    unlabeled_files = [f for f in all_files if f not in labeled_files]
    logger.info(f"Total soundscape files: {len(all_files)}, unlabeled: {len(unlabeled_files)}")

    # Shard the file list
    shard_files = [f for i, f in enumerate(unlabeled_files) if i % n_shards == shard_id]
    logger.info(f"Shard {shard_id}/{n_shards}: processing {len(shard_files)} files")

    sr = mel_cfg["sr"]
    results = []
    t0 = time.time()

    for file_idx, filename in enumerate(shard_files):
        audio_path = str(SOUNDSCAPE_DIR / filename)

        try:
            info = sf.info(audio_path)
            duration = info.frames / info.samplerate
        except Exception as e:
            logger.warning(f"Skipping {filename}: {e}")
            continue

        # Process each 5s window
        n_windows = int(duration / CHUNK_DURATION)
        for win_idx in range(n_windows):
            start_s = win_idx * CHUNK_DURATION
            audio = load_audio_chunk(audio_path, sr, start_s, CHUNK_DURATION)
            mel = compute_mel(audio, mel_cfg)

            clip_probs = session.run(None, {input_name: mel})[0]  # (1, C)
            if not output_is_probs:
                clip_probs = sigmoid(clip_probs)
            probs = clip_probs[0]  # (C,)

            row = {
                "filename": filename,
                "chunk_start": start_s,
                "max_prob": float(probs.max()),
                "top_species": species_list[probs.argmax()],
                "top_prob": float(probs.max()),
                "n_species_above_01": int((probs > 0.1).sum()),
                "n_species_above_02": int((probs > 0.2).sum()),
            }
            # Store all species probabilities as individual columns
            for i, species in enumerate(species_list):
                row[species] = float(probs[i])
            results.append(row)

        if (file_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (file_idx + 1) / elapsed
            eta = (len(shard_files) - file_idx - 1) / max(rate, 0.01)
            logger.info(f"  [{file_idx+1}/{len(shard_files)}] {rate:.1f} files/sec, ETA {eta/60:.0f} min")

    # Save shard results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{shard_id:04d}.parquet"

    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    logger.info(f"Shard {shard_id} done: {len(df)} rows, {len(shard_files)} files in {elapsed/60:.1f} min → {output_path}")

    # Summary stats (only for this shard)
    if len(df) > 0:
        print(f"\n=== Shard {shard_id} Summary ===")
        print(f"Windows: {len(df)}")
        print(f"max_prob: mean={df['max_prob'].mean():.3f}, median={df['max_prob'].median():.3f}")
        print(f"Species above 0.1: mean={df['n_species_above_01'].mean():.1f}")
        print(f"Species above 0.2: mean={df['n_species_above_02'].mean():.1f}")


if __name__ == "__main__":
    main()
