#!/usr/bin/env python3
"""Teacher ensemble pseudo-labeling for BirdCLEF 2026 soundscapes.

Runs exp 012 ONNX teacher models on soundscape audio to produce soft pseudo-labels.
Supports two modes:
  - Mode A (direct): classify original audio, weighted ensemble across models
  - Mode B (mixit): classify original + 4 MixIT channels per model, max-pool per model,
    then weighted ensemble (Bird MixIT paper Section 7.1)

Usage:
    python pseudo_label.py --config config/mode_a.yaml --split labeled
    python pseudo_label.py --config config/mode_a.yaml --split unlabeled
    python pseudo_label.py --config config/mode_b.yaml --split labeled
    python pseudo_label.py --config config/mode_a.yaml --split labeled --max-files 5
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WINDOW_DURATION = 5.0


# ---------------------------------------------------------------------------
# Mel preprocessing (from exp 010 submission.py)
# ---------------------------------------------------------------------------


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
        if T > target:
            mel_std = mel_std[:, :target]
        elif T < target:
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
        if T > target:
            s = (T - target) // 2
            mel = mel[:, s : s + target]
        elif T < target:
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


def load_mixit_chunk(npz_path, channel_idx, sr_target, start_s, dur=5.0):
    """Load a 5s chunk from a MixIT separated channel."""
    data = np.load(npz_path)
    channels = data["channels"]  # (4, n_samples) at 22050 Hz
    mixit_sr = 22050
    start_sample = int(start_s * mixit_sr)
    end_sample = start_sample + int(dur * mixit_sr)
    audio = channels[channel_idx, start_sample:end_sample].astype(np.float32)

    # Resample to model's target sr if needed
    if mixit_sr != sr_target:
        audio = librosa.resample(audio, orig_sr=mixit_sr, target_sr=sr_target)

    target_samples = int(sr_target * dur)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return audio[:target_samples]


# ---------------------------------------------------------------------------
# Teacher model wrapper
# ---------------------------------------------------------------------------


class TeacherModel:
    """Wraps an ONNX teacher model with its mel config and species mapping."""

    def __init__(self, name, model_dir, use_gpu=True):
        self.name = name
        self.model_dir = Path(model_dir)

        with open(self.model_dir / "mel_config.json") as f:
            self.mel_cfg = json.load(f)
        with open(self.model_dir / "species_mapping.json") as f:
            sp_map = json.load(f)

        self.idx_map = sp_map.get("idx_to_name_2026") or sp_map["idx_2025_to_name_2026"]
        self.sr = self.mel_cfg["sr"]

        # ONNX session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            sess_options=opts,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer_logits(self, audio):
        """Run inference on a single audio chunk, return raw logits (234,)."""
        mel = compute_mel(audio, self.mel_cfg)
        logits = self.session.run(None, {self.input_name: mel})[0].flatten()
        return logits

    def infer_logits_mapped(self, audio, n_species, sp2idx):
        """Run inference and map model outputs to the full 234-species vector."""
        logits_model = self.infer_logits(audio)
        logits_full = np.full(n_species, -10.0, dtype=np.float32)  # low default = ~sigmoid(0)
        for model_idx_str, sp_name in self.idx_map.items():
            if sp_name in sp2idx:
                logits_full[sp2idx[sp_name]] = logits_model[int(model_idx_str)]
        return logits_full


# ---------------------------------------------------------------------------
# File listing
# ---------------------------------------------------------------------------


def get_labeled_files(data_dir):
    labels_csv = Path(data_dir) / "train_soundscapes_labels.csv"
    files = set()
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            files.add(row["filename"])
    return sorted(files)


def get_soundscape_files(data_dir, split):
    soundscape_dir = Path(data_dir) / "train_soundscapes"
    all_files = sorted(p.name for p in soundscape_dir.glob("*.ogg"))
    labeled = set(get_labeled_files(data_dir))
    if split == "labeled":
        return [f for f in all_files if f in labeled]
    elif split == "unlabeled":
        return [f for f in all_files if f not in labeled]
    else:
        return all_files


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def run_pseudo_labeling(config, split, max_files=None):
    data_dir = Path(config["data"]["data_dir"])
    weights_dir = Path(config["data"]["weights_dir"])
    mode = config["mode"]
    temperature = config.get("temperature", 1.0)
    out_base = Path(config["output"]["base_dir"])
    run_name = config["output"]["run_name"]
    out_dir = out_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load taxonomy for species list and taxon mapping
    taxonomy = pd.read_csv(data_dir / "taxonomy.csv")
    all_species = sorted(taxonomy["primary_label"].astype(str).tolist())
    sp2idx = {sp: i for i, sp in enumerate(all_species)}
    sp2taxon = dict(
        zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"], strict=False)
    )
    n_species = len(all_species)

    # Build per-species weight vectors for each teacher
    teacher_cfgs = config["teachers"]
    teacher_names = list(teacher_cfgs.keys())

    # Load teachers
    teachers = {}
    for tname in teacher_names:
        slug = teacher_cfgs[tname]["slug"]
        model_dir = weights_dir / slug
        teachers[tname] = TeacherModel(tname, model_dir, use_gpu=True)
        logger.info(f"Loaded teacher: {tname} ({slug})")

    # Build per-species weight matrix: (n_teachers, n_species)
    weight_matrix = np.zeros((len(teacher_names), n_species), dtype=np.float32)
    for ti, tname in enumerate(teacher_names):
        taxon_weights = teacher_cfgs[tname]["weights"]
        for sp_idx, sp in enumerate(all_species):
            taxon = sp2taxon.get(sp, "Aves")
            weight_matrix[ti, sp_idx] = taxon_weights.get(taxon, 0.0)

    # Normalize weights per species (so they sum to 1)
    weight_sums = weight_matrix.sum(axis=0, keepdims=True)
    weight_sums = np.maximum(weight_sums, 1e-8)
    weight_matrix = weight_matrix / weight_sums

    # MixIT config
    mixit_dir = None
    n_mixit_channels = 0
    if mode == "mixit":
        mixit_cfg = config["mixit"]
        mixit_dir = Path(mixit_cfg["separated_dir"])
        n_mixit_channels = mixit_cfg.get("n_channels", 4)

    # Get files
    soundscape_dir = data_dir / "train_soundscapes"
    files = get_soundscape_files(data_dir, split)
    if max_files:
        files = files[:max_files]
    logger.info(f"Mode: {mode}, split: {split}, files: {len(files)}, T: {temperature}")

    # Storage for per-model logits and final pseudo-labels
    all_seg_keys = []
    per_model_logits = {tname: [] for tname in teacher_names}
    all_probs = []

    t0 = time.time()
    for fi, fn in enumerate(tqdm(files, desc="Pseudo-labeling")):
        audio_path = str(soundscape_dir / fn)
        info = sf.info(audio_path)
        total_dur = info.frames / info.samplerate
        n_windows = int(total_dur / WINDOW_DURATION)

        # Load MixIT channels for this file (if Mode B)
        mixit_npz_path = None
        if mode == "mixit" and mixit_dir is not None:
            stem = Path(fn).stem
            mixit_npz_path = mixit_dir / f"{stem}.npz"
            if not mixit_npz_path.exists():
                mixit_npz_path = None  # fallback to Mode A for this file

        for wi in range(n_windows):
            start_s = wi * WINDOW_DURATION
            end_sec = int(start_s + WINDOW_DURATION)
            all_seg_keys.append((fn, end_sec))

            # Per-teacher logits for this segment
            segment_logits = np.zeros((len(teacher_names), n_species), dtype=np.float32)

            for ti, tname in enumerate(teacher_names):
                teacher = teachers[tname]
                sr = teacher.sr

                # Original audio
                audio_orig = load_audio_chunk(audio_path, sr, start_s)
                logits_orig = teacher.infer_logits_mapped(audio_orig, n_species, sp2idx)

                if mode == "mixit" and mixit_npz_path is not None:
                    # Mode B: run on 4 MixIT channels + original, max-pool
                    channel_logits = [logits_orig]
                    for ch in range(n_mixit_channels):
                        audio_ch = load_mixit_chunk(mixit_npz_path, ch, sr, start_s)
                        logits_ch = teacher.infer_logits_mapped(audio_ch, n_species, sp2idx)
                        channel_logits.append(logits_ch)

                    # Max-pool across channels (paper Section 7.1)
                    # Max on logits, then apply sigmoid
                    segment_logits[ti] = np.max(channel_logits, axis=0)
                else:
                    # Mode A: original only
                    segment_logits[ti] = logits_orig

                per_model_logits[tname].append(segment_logits[ti].copy())

            # Temperature-scaled sigmoid + weighted average
            probs_per_teacher = sigmoid(segment_logits / temperature)  # (n_teachers, n_species)
            ensemble_probs = (probs_per_teacher * weight_matrix).sum(axis=0)  # (n_species,)
            all_probs.append(ensemble_probs)

        if (fi + 1) % 100 == 0 or fi == len(files) - 1:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            logger.info(f"  {fi+1}/{len(files)} files ({rate:.1f} files/s)")

    # Save per-model logits
    logits_dir = out_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    seg_keys_arr = np.array(all_seg_keys, dtype=object)
    for tname in teacher_names:
        logits_arr = np.array(per_model_logits[tname], dtype=np.float32)
        np.savez_compressed(
            logits_dir / f"{tname}.npz",
            logits=logits_arr,
            seg_keys=seg_keys_arr,
            species=np.array(all_species),
        )
        logger.info(f"Saved {tname} logits: {logits_arr.shape}")

    # Save pseudo-labels as parquet
    probs_arr = np.array(all_probs, dtype=np.float32)
    df_data = {
        "filename": [k[0] for k in all_seg_keys],
        "end_sec": [int(k[1]) for k in all_seg_keys],
    }
    for sp_idx, sp in enumerate(all_species):
        df_data[sp] = probs_arr[:, sp_idx]

    df = pd.DataFrame(df_data)
    parquet_path = out_dir / "pseudo_labels.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved pseudo-labels: {parquet_path} ({len(df)} segments)")

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "config": config,
                "split": split,
                "n_files": len(files),
                "n_segments": len(all_seg_keys),
                "temperature": temperature,
            },
            f,
            indent=2,
        )

    elapsed = time.time() - t0
    logger.info(f"Done: {len(files)} files, {len(all_seg_keys)} segments in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--split", required=True, choices=["labeled", "unlabeled", "all"])
    parser.add_argument("--max-files", type=int, default=None, help="Limit files (for testing)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_pseudo_labeling(config, args.split, args.max_files)


if __name__ == "__main__":
    main()
