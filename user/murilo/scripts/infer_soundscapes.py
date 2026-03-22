#!/usr/bin/env python3
"""Run ONNX model inference on BirdCLEF 2026 soundscapes.

Processes soundscape audio files through an ONNX model and saves per-segment
predictions as npz. Reuses the mel preprocessing from the exp 010 submission
pipeline.

Usage:
    # Labeled soundscapes (66 files)
    python scripts/infer_soundscapes.py --model protoclr-2026-soundscape --split labeled

    # All soundscapes (labeled + unlabeled)
    python scripts/infer_soundscapes.py --model protoclr-2026-soundscape --split all

    # All 3 models on labeled data
    python scripts/infer_soundscapes.py --model all --split labeled

    # Dry run
    python scripts/infer_soundscapes.py --model protoclr-2026-soundscape --split labeled --dry-run

Output (per model):
    /storage/project/.../shared/inference/birdclef-2026/{model_slug}/{split}.npz
    Contents: predictions (n_segments, 234), seg_keys (n_segments, 2), species (234,)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_DIR = Path("/storage/project/ps-clef2026_birdclef-0")
DATA_DIR = PROJECT_DIR / "shared/data/birdclef-2026"
WEIGHTS_DIR = (
    PROJECT_DIR
    / "mgustineli3/birdclef-2026/user/murilo/experiments"
    / "010-submission-pipeline/artifacts/weights"
)
OUTPUT_BASE = PROJECT_DIR / "shared/inference/birdclef-2026"

SOUNDSCAPE_DIR = DATA_DIR / "train_soundscapes"
LABELS_CSV = DATA_DIR / "train_soundscapes_labels.csv"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"

MODEL_SLUGS = [
    "protoclr-2026-soundscape",
    "convnext-2026-soundscape",
    "eat-2026-soundscape",
]


# ---------------------------------------------------------------------------
# Mel preprocessing (from exp 010 submission.py)
# ---------------------------------------------------------------------------


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_mel(audio, cfg):
    mel_type = cfg.get("type", "mel")

    if mel_type == "protoclr_mel":
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=cfg["sr"],
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
            fmin=cfg.get("fmin", 50),
            fmax=cfg.get("fmax", 8000),
        )
        mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_std = (mel_norm - cfg.get("protoclr_mean", 0.5347)) / cfg.get("protoclr_std", 0.0772)
        target = cfg.get("target_frames", 250)
        T = mel_std.shape[1]
        if target < T:
            mel_std = mel_std[:, :target]
        elif target > T:
            mel_std = np.pad(mel_std, ((0, 0), (0, target - T)))
        return mel_std[np.newaxis, np.newaxis].astype(np.float32)

    elif mel_type == "kaldi_fbank":
        import torch
        import torchaudio

        waveform = torch.from_numpy(audio).float()
        waveform = waveform - waveform.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            htk_compat=True,
            sample_frequency=cfg["sr"],
            use_energy=False,
            window_type="hanning",
            num_mel_bins=cfg["num_mel_bins"],
            dither=0.0,
            frame_shift=cfg["frame_shift"],
        )
        target = cfg.get("target_length", 512)
        if mel.shape[0] < target:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target - mel.shape[0]))
        elif mel.shape[0] > target:
            mel = mel[:target]
        mel = (mel - cfg["norm_mean"]) / (cfg["norm_std"] * 2)
        return mel.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)

    else:  # mel (ConvNeXt)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=cfg["sr"],
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
            fmin=cfg.get("fmin", 0),
            fmax=cfg.get("fmax"),
            htk=cfg.get("htk", True),
        )
        mel = np.log(mel + 1e-6)
        mel = (mel - cfg["norm_mean"]) / (cfg["norm_std"] * 2)
        target = cfg.get("target_frames", 500)
        T = mel.shape[1]
        if target < T:
            s = (T - target) // 2
            mel = mel[:, s : s + target]
        elif target > T:
            mel = np.pad(mel, ((0, 0), (0, target - T)))
        return mel[np.newaxis, np.newaxis].astype(np.float32)


def load_audio_chunk(path, sr, start_s, dur=5.0):
    info = sf.info(path)
    audio, _ = sf.read(
        path,
        dtype="float32",
        start=int(start_s * info.samplerate),
        frames=int(dur * info.samplerate) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if info.samplerate != sr:
        audio = librosa.resample(audio, orig_sr=info.samplerate, target_sr=sr)
    target_samples = int(sr * dur)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return audio[:target_samples].astype(np.float32)


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------


def get_labeled_files():
    """Return set of filenames that have labels."""
    import csv

    files = set()
    with open(LABELS_CSV) as f:
        for row in csv.DictReader(f):
            files.add(row["filename"])
    return sorted(files)


def get_soundscape_files(split):
    """Return list of soundscape filenames for the given split."""
    all_files = sorted(p.name for p in SOUNDSCAPE_DIR.glob("*.ogg"))
    if split == "labeled":
        labeled = set(get_labeled_files())
        return [f for f in all_files if f in labeled]
    elif split == "unlabeled":
        labeled = set(get_labeled_files())
        return [f for f in all_files if f not in labeled]
    else:  # all
        return all_files


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(model_slug, split, dry_run=False):
    """Run ONNX inference for one model on the given split."""
    import onnxruntime as ort

    model_dir = WEIGHTS_DIR / model_slug
    if not model_dir.exists():
        logger.error(f"Model dir not found: {model_dir}")
        return

    with open(model_dir / "mel_config.json") as f:
        mel_cfg = json.load(f)
    with open(model_dir / "species_mapping.json") as f:
        sp_map = json.load(f)

    # Build species list (all 234 from taxonomy)
    taxonomy = np.loadtxt(TAXONOMY_CSV, delimiter=",", dtype=str, skiprows=1, usecols=0)
    all_species = sorted(taxonomy.tolist())
    sp2idx = {sp: i for i, sp in enumerate(all_species)}

    idx_map = sp_map.get("idx_to_name_2026") or sp_map["idx_2025_to_name_2026"]

    files = get_soundscape_files(split)
    logger.info(f"{model_slug} | split={split} | {len(files)} files")

    if dry_run:
        logger.info("[DRY-RUN] Would process %d files", len(files))
        return

    # ONNX session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts)
    input_name = session.get_inputs()[0].name
    sr = mel_cfg["sr"]

    # Collect predictions
    all_preds = []
    all_keys = []  # (filename, end_sec)

    t0 = time.time()
    for fi, fn in enumerate(files):
        audio_path = str(SOUNDSCAPE_DIR / fn)
        info = sf.info(audio_path)
        total_dur = info.frames / info.samplerate
        n_windows = int(total_dur / 5.0)

        for wi in range(n_windows):
            start_s = wi * 5.0
            end_sec = int(start_s + 5)
            audio = load_audio_chunk(audio_path, sr, start_s)
            mel = compute_mel(audio, mel_cfg)
            logits = session.run(None, {input_name: mel})[0].flatten()
            probs = sigmoid(logits)

            # Map to 234-species vector
            row = np.zeros(len(all_species), dtype=np.float32)
            for model_idx_str, sp_name in idx_map.items():
                if sp_name in sp2idx:
                    row[sp2idx[sp_name]] = probs[int(model_idx_str)]

            all_preds.append(row)
            all_keys.append((fn, end_sec))

        if (fi + 1) % 10 == 0 or fi == len(files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            logger.info(f"  {fi+1}/{len(files)} files ({rate:.1f} files/s)")

    predictions = np.array(all_preds, dtype=np.float32)
    seg_keys = np.array(all_keys, dtype=object)

    # Save
    out_dir = OUTPUT_BASE / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.npz"
    np.savez_compressed(
        out_path,
        predictions=predictions,
        seg_keys=seg_keys,
        species=np.array(all_species),
    )
    elapsed = time.time() - t0
    logger.info(f"Saved {out_path}: {predictions.shape} predictions, {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model slug (e.g., protoclr-2026-soundscape) or 'all'",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["labeled", "unlabeled", "all"],
        help="Which soundscape files to process",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")
    args = parser.parse_args()

    slugs = MODEL_SLUGS if args.model == "all" else [args.model]

    for slug in slugs:
        if slug not in MODEL_SLUGS:
            logger.error(f"Unknown model: {slug}. Choose from: {MODEL_SLUGS}")
            sys.exit(1)
        run_inference(slug, args.split, args.dry_run)


if __name__ == "__main__":
    main()
