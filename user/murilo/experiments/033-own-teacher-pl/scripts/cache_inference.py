"""Cache distilled-sed-kd-5fold ONNX predictions on all soundscapes.

Writes per-fold `logits.parquet` files to
`shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/fold{0..4}/`
in the SAME schema as the existing v5j-4fold cache (filename, source, chunk_idx,
start_sec, end_sec, *234_species_cols; raw pre-blended logits, no sigmoid).

The `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` script consumes
this cache directly when invoked with `configs/pl_gen_v2.yaml`.

Architecture choice:
- The 5 ONNX files at `experiments/010-submission-pipeline/artifacts/weights/distilled-sed-kd-5fold/`
  each output pre-blended LOGITS (0.5 * clip + 0.5 * frame_max in logit space) per the
  exp 032 export wrapper.
- Per-file batching: load audio once, compute 12 mels once, run through 5 sessions.
  Saves 5× on audio I/O.
- No TTA, no smoothing, no sigmoid — matches v5j-4fold cache generation.

Usage:
    # Full run on all 10,658 SS files (~30-60 min on rtx6000)
    python cache_inference.py

    # Smoke test on first 5 files (~1 min)
    python cache_inference.py --limit 5 --output-suffix _smoke

    # Custom output dir (default: shared/inference/.../distilled-sed-kd-5fold/...)
    python cache_inference.py --output-dir /tmp/test_cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# ----------------------------------------------------------------------
# Paths + constants
# ----------------------------------------------------------------------
REPO = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026")
WEIGHTS_DIR = (
    REPO
    / "user/murilo/experiments/010-submission-pipeline/artifacts/weights/distilled-sed-kd-5fold"
)
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
SS_DIR = DATA_DIR / "train_soundscapes"
DEFAULT_OUTPUT_DIR = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes"
)
SR = 32000
SEGMENT_DURATION_S = 5.0
WINDOW_SAMPLES = int(SR * SEGMENT_DURATION_S)
N_WINDOWS_PER_FILE = 12  # 60-sec soundscapes / 5-sec windows
N_FOLDS = 5

SOURCE_NAME = "distilled-sed-kd-5fold"


# ----------------------------------------------------------------------
# Kernel-faithful audio loader (mirrors submission.py:load_audio_chunk)
# ----------------------------------------------------------------------


def kernel_load_audio_chunk(audio_path: Path, start_s: float) -> np.ndarray:
    """Load a 5-sec window from a 32kHz soundscape file. Pads on short files."""
    target_samples = WINDOW_SAMPLES
    info = sf.info(str(audio_path))
    source_sr = info.samplerate
    total_duration = info.frames / source_sr
    actual_duration = max(0.0, min(SEGMENT_DURATION_S, total_duration - start_s))
    if actual_duration <= 0:
        return np.zeros(target_samples, dtype=np.float32)
    audio, _ = sf.read(
        str(audio_path),
        dtype="float32",
        start=int(start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=SR)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return audio.astype(np.float32)


# ----------------------------------------------------------------------
# Tucker mel (mirrors submission.py:compute_mel_tucker via torchaudio)
# ----------------------------------------------------------------------


def make_tucker_mel(mel_cfg: dict) -> torch.nn.Module:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_cfg["sr"],
        n_fft=mel_cfg.get("n_fft", 2048),
        hop_length=mel_cfg.get("hop_length", 512),
        n_mels=mel_cfg.get("n_mels", 256),
        f_min=mel_cfg.get("fmin", 20),
        f_max=mel_cfg.get("fmax", 16000),
        power=mel_cfg.get("power", 2.0),
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=mel_cfg.get("top_db", 80.0)
    )
    return torch.nn.Sequential(mel_transform, db_transform)


def kernel_compute_mel(
    audio_batch: np.ndarray, mel_module: torch.nn.Module, mel_cfg: dict
) -> np.ndarray:
    """Batched mel: (B, samples) → (B, 1, n_mels, T) after per-sample z-score."""
    wave = torch.from_numpy(audio_batch).float()  # (B, samples)
    mel = mel_module(wave)  # (B, n_mels, T)
    # Per-sample z-score (matches submission.py:compute_mel_tucker)
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std = mel.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    mel = (mel - mean) / std

    target_frames = mel_cfg.get("target_frames", 313)
    T = mel.shape[-1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel = mel[..., start : start + target_frames]
    elif T < target_frames:
        mel = torch.nn.functional.pad(mel, (0, target_frames - T))
    return mel.unsqueeze(1).numpy().astype(np.float32)  # (B, 1, n_mels, T)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N soundscape files (smoke testing). Default: all 10,658.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to fold dirs (e.g., '_smoke' → fold0_smoke/). Default: ''.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    import onnxruntime as ort

    # ------------------------------------------------------------------
    # Load mel config + species list (shared across folds)
    # ------------------------------------------------------------------
    mel_cfg_path = WEIGHTS_DIR / "mel_config.json"
    species_list_path = WEIGHTS_DIR / "species_list.json"
    if not mel_cfg_path.exists():
        raise FileNotFoundError(
            f"Expected {mel_cfg_path} (created by export_distilled_sed.py). "
            "Re-run the export wrapper if missing."
        )
    with open(mel_cfg_path) as f:
        mel_cfg = json.load(f)
    with open(species_list_path) as f:
        species_list_data = json.load(f)
    # species_list.json might be {"species": [...]} or a bare list
    if isinstance(species_list_data, dict):
        species = species_list_data.get("species") or species_list_data.get("birdclef2026_species")
    else:
        species = species_list_data
    assert (
        species is not None and len(species) == 234
    ), f"Expected 234 species, got {len(species) if species else 0}"
    print(
        f"Loaded mel_cfg: type={mel_cfg.get('type')} sr={mel_cfg.get('sr')} n_mels={mel_cfg.get('n_mels')}"
    )
    print(f"Species count: {len(species)}")

    # ------------------------------------------------------------------
    # Load 5 ONNX sessions
    # ------------------------------------------------------------------
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (
        ["CPUExecutionProvider"] if args.cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    sessions = []
    for fold in range(N_FOLDS):
        path = WEIGHTS_DIR / f"model_fold{fold}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Missing ONNX: {path}")
        sess = ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)
        sessions.append(sess)
        print(f"  Loaded fold {fold} ({path.stat().st_size / 1e6:.1f} MB)")
    input_name = sessions[0].get_inputs()[0].name
    print(f"ONNX input name: {input_name}")
    print(f"Providers in use: {sessions[0].get_providers()}")

    # ------------------------------------------------------------------
    # Build mel module (CPU; mel compute is cheap vs ONNX)
    # ------------------------------------------------------------------
    mel_module = make_tucker_mel(mel_cfg)
    mel_module.eval()

    # ------------------------------------------------------------------
    # Discover soundscape files
    # ------------------------------------------------------------------
    ss_files = sorted([p.name for p in SS_DIR.glob("*.ogg")])
    if args.limit is not None:
        ss_files = ss_files[: args.limit]
    print(f"Soundscape files to process: {len(ss_files):,}")

    # ------------------------------------------------------------------
    # Output dirs
    # ------------------------------------------------------------------
    fold_dirs = []
    for fold in range(N_FOLDS):
        fd = args.output_dir / f"fold{fold}{args.output_suffix}"
        fd.mkdir(parents=True, exist_ok=True)
        fold_dirs.append(fd)
        print(f"  Output fold{fold}: {fd}")

    # ------------------------------------------------------------------
    # Inference loop — per file, batch 12 windows, run through 5 folds
    # ------------------------------------------------------------------
    rows_per_fold: list[list[dict]] = [[] for _ in range(N_FOLDS)]
    t0 = time.time()
    with torch.no_grad():
        for fname in tqdm(ss_files, desc="Caching"):
            audio_path = SS_DIR / fname
            # Load 12 windows + compute 12 mels in a batch
            audio_batch = np.stack(
                [
                    kernel_load_audio_chunk(audio_path, chunk_idx * SEGMENT_DURATION_S)
                    for chunk_idx in range(N_WINDOWS_PER_FILE)
                ],
                axis=0,
            )  # (12, samples)
            mel_batch = kernel_compute_mel(audio_batch, mel_module, mel_cfg)  # (12, 1, n_mels, T)

            # Run through 5 folds
            for fold_idx, sess in enumerate(sessions):
                logits = sess.run(None, {input_name: mel_batch})[0]  # (12, 234)
                # Build rows for this file × this fold
                for chunk_idx in range(N_WINDOWS_PER_FILE):
                    row = {
                        "filename": fname,
                        "source": SOURCE_NAME,
                        "chunk_idx": chunk_idx,
                        "start_sec": float(chunk_idx * SEGMENT_DURATION_S),
                        "end_sec": float((chunk_idx + 1) * SEGMENT_DURATION_S),
                    }
                    for sp_idx, sp_name in enumerate(species):
                        row[sp_name] = float(logits[chunk_idx, sp_idx])
                    rows_per_fold[fold_idx].append(row)
    elapsed = time.time() - t0
    print(f"\nInference complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Write per-fold parquets
    # ------------------------------------------------------------------
    for fold_idx in range(N_FOLDS):
        df = pd.DataFrame(rows_per_fold[fold_idx])
        out_path = fold_dirs[fold_idx] / "logits.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  fold{fold_idx}: {len(df):,} rows → {out_path}")
        print(
            f"    cols: {len(df.columns)}, value range: [{df.select_dtypes(include='number').min().min():.2f}, {df.select_dtypes(include='number').max().max():.2f}]"
        )

    print(
        f"\nDone. Total cached: {sum(len(r) for r in rows_per_fold):,} predictions ({N_FOLDS} folds × {len(ss_files)} files × {N_WINDOWS_PER_FILE} windows)."
    )


if __name__ == "__main__":
    sys.exit(main())
