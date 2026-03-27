#!/usr/bin/env python3
"""Inspect frame-level predictions from attention model on focal recordings.

Runs the trained attention model on focal recordings and outputs per-frame
species probabilities. Shows whether the model cleanly identifies call vs
silence frames within each 5s clip.

Usage:
    python scripts/inspect_frame_predictions.py \
        --checkpoint logs/lora-attention/checkpoints/best.ckpt \
        --config configs/run-i-efficientnet-attention.yaml \
        --n-samples 20
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from nnAudio.features import MelSpectrogram

logger = logging.getLogger(__name__)


def load_audio_chunk(audio_path, target_sr, start_sample=0, num_samples=None):
    info = sf.info(audio_path)
    source_sr = info.samplerate
    src_start = int(start_sample * source_sr / target_sr)
    src_frames = int(num_samples * source_sr / target_sr) + 2 if num_samples else -1
    audio, _ = sf.read(audio_path, dtype="float32", start=src_start, frames=src_frames)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio).float()
    if source_sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=source_sr, new_freq=target_sr)
    if num_samples and audio_tensor.shape[0] > num_samples:
        audio_tensor = audio_tensor[:num_samples]
    return audio_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--output", default="artifacts/frame_predictions_sample.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})

    # Build model
    from birdclef_102.model import EfficientNetClassifier
    from birdclef_102.data import build_species_list

    species_list = build_species_list(data_cfg["metadata_csv"])
    num_classes = len(species_list)

    mel_spec = MelSpectrogram(
        sr=data_cfg["sample_rate"],
        n_fft=data_cfg.get("n_fft", 2048),
        win_length=data_cfg.get("n_fft", 2048),
        hop_length=data_cfg.get("hop_length", 256),
        n_mels=data_cfg["n_mels"],
        fmin=50, fmax=14000,
        trainable_mel=False, trainable_STFT=False,
    )

    model = EfficientNetClassifier(
        model_name=model_cfg["model_name"],
        num_classes=num_classes,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        backbone_mode=model_cfg.get("backbone_mode", "lora"),
        pooling_mode="attention",
        head_hidden_dim=model_cfg.get("head_hidden_dim", 512),
        head_dropout=model_cfg.get("head_dropout", 0.3),
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        lora_targets=lora_cfg.get("target_modules"),
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    model.cuda()
    logger.info("Model loaded")

    # Sample some focal recordings
    train_df = pd.read_csv(data_cfg["metadata_csv"])
    audio_dir = Path(data_cfg["audio_dir"])
    samples = train_df.sample(n=args.n_samples, random_state=42)

    sr = data_cfg["sample_rate"]
    chunk_samples = int(5.0 * sr)
    target_frames = data_cfg["target_frames"]

    results = []
    for _, row in samples.iterrows():
        fpath = audio_dir / row["filename"]
        if not fpath.exists():
            continue

        species = str(row["primary_label"])
        species_idx = species_list.index(species) if species in species_list else -1

        # Load first 5s
        waveform = load_audio_chunk(str(fpath), sr, num_samples=chunk_samples)
        if waveform.shape[0] < chunk_samples:
            waveform = torch.nn.functional.pad(waveform, (0, chunk_samples - waveform.shape[0]))

        with torch.no_grad():
            waveform_gpu = waveform.unsqueeze(0).cuda()
            mel = model.compute_mel(waveform_gpu)
            features = model._extract_features(mel)  # (1, D, H, W)

            # Get frame features and per-frame predictions
            frame_features = features.mean(dim=2).permute(0, 2, 1)  # (1, T, D)
            if model.head.projection is not None:
                frame_features = model.head.projection(frame_features)
            frame_logits = model.head.frame_head(frame_features)  # (1, T, C)
            frame_probs = torch.sigmoid(frame_logits)  # (1, T, C)

            # Clip-level via attention (training mode) and max (inference mode)
            att_logits = model.head.att_head(frame_features)
            att_weights = torch.softmax(torch.tanh(att_logits), dim=1)
            clip_att = (att_weights * frame_probs).sum(dim=1)  # (1, C)
            clip_max = frame_probs.max(dim=1).values  # (1, C)

        frame_probs_np = frame_probs[0].cpu().numpy()  # (T, C)
        n_frames = frame_probs_np.shape[0]
        frame_dur = 5.0 / n_frames

        # Get predictions for the labeled species
        if species_idx >= 0:
            species_frame_probs = frame_probs_np[:, species_idx].tolist()
            species_clip_att = float(clip_att[0, species_idx].cpu())
            species_clip_max = float(clip_max[0, species_idx].cpu())
        else:
            species_frame_probs = []
            species_clip_att = 0.0
            species_clip_max = 0.0

        # Top predicted species per frame
        top_per_frame = []
        for t in range(n_frames):
            top_idx = frame_probs_np[t].argmax()
            top_per_frame.append({
                "frame": t,
                "time_sec": round(t * frame_dur, 2),
                "species": species_list[top_idx],
                "prob": round(float(frame_probs_np[t, top_idx]), 4),
            })

        # Max prob across all species per frame (is anything calling?)
        max_any_species = frame_probs_np.max(axis=1).tolist()

        result = {
            "filename": row["filename"],
            "labeled_species": species,
            "n_frames": n_frames,
            "frame_duration_sec": round(frame_dur, 3),
            "labeled_species_frame_probs": [round(p, 4) for p in species_frame_probs],
            "labeled_species_clip_attention": round(species_clip_att, 4),
            "labeled_species_clip_max": round(species_clip_max, 4),
            "max_any_species_per_frame": [round(p, 4) for p in max_any_species],
            "top_species_per_frame": top_per_frame,
        }
        results.append(result)

        # Print summary
        if species_frame_probs:
            probs = np.array(species_frame_probs)
            active_frames = (probs > 0.3).sum()
            print(f"{row['filename']:50s} species={species:>10s} "
                  f"clip_max={species_clip_max:.3f} clip_att={species_clip_att:.3f} "
                  f"frames>{0.3}: {active_frames}/{n_frames} "
                  f"frame_probs: {' '.join(f'{p:.2f}' for p in species_frame_probs)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} samples to {output_path}")

    # Summary stats
    print(f"\n=== Summary ({len(results)} samples) ===")
    all_max = [r["labeled_species_clip_max"] for r in results if r["labeled_species_frame_probs"]]
    all_active_frac = []
    for r in results:
        if r["labeled_species_frame_probs"]:
            probs = np.array(r["labeled_species_frame_probs"])
            all_active_frac.append((probs > 0.3).mean())
    print(f"Clip max prob (labeled species): mean={np.mean(all_max):.3f}, median={np.median(all_max):.3f}")
    print(f"Active frame fraction (>0.3): mean={np.mean(all_active_frac):.2f}, median={np.median(all_active_frac):.2f}")
    print(f"Frame resolution: {results[0]['n_frames']} frames / 5s = {results[0]['frame_duration_sec']}s per frame")


if __name__ == "__main__":
    main()
