#!/usr/bin/env python3
"""Generate teacher predictions parquet for focal data cleaning.

Runs the trained attention model on all focal chunks and saves per-chunk
frame-level predictions. Used to hard-filter noisy chunks where the teacher
disagrees with the clip-level label.

Output columns:
  filename, chunk_start, teacher_prob, teacher_prob_att, active_frames,
  total_frames, max_other_prob, frame_probs_labeled

Usage:
    python scripts/build_teacher_predictions.py \
        --checkpoint logs/lora-attention/checkpoints/best.ckpt \
        --config configs/run-i-efficientnet-attention.yaml \
        --output artifacts/teacher_predictions.parquet
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from birdclef_102.data import build_species_list, load_audio_chunk

logger = logging.getLogger(__name__)


class FocalInferenceDataset(Dataset):
    """Lightweight dataset for teacher inference on all focal chunks."""

    def __init__(self, activity_parquet, metadata_csv, audio_dir, species_list,
                 sample_rate=32000, chunk_duration=5.0):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}

        # Build filename -> (relpath, species_idx) from train.csv
        train_df = pd.read_csv(metadata_csv)
        self.file_info = {}
        for _, row in train_df.iterrows():
            relpath = row["filename"]
            basename = Path(relpath).name
            species = str(row["primary_label"])
            species_idx = self.species_to_idx.get(species, -1)
            self.file_info[basename] = (relpath, species_idx)

        # Load chunk index from activity parquet
        activity = pd.read_parquet(activity_parquet)
        self.chunks = []
        for _, row in activity.iterrows():
            basename = row["filename"]
            if basename not in self.file_info:
                continue
            relpath, species_idx = self.file_info[basename]
            self.chunks.append((
                str(self.audio_dir / relpath),
                int(row["chunk_start"] * sample_rate),
                basename,
                float(row["chunk_start"]),
                species_idx,
            ))

        logger.info(f"Loaded {len(self.chunks)} chunks for inference")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        audio_path, start_sample, filename, chunk_start, species_idx = self.chunks[idx]

        waveform = load_audio_chunk(
            audio_path, self.sample_rate,
            start_sample=start_sample, num_samples=self.chunk_samples,
        )
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        return {
            "waveform": waveform,
            "filename": filename,
            "chunk_start": chunk_start,
            "species_idx": species_idx,
        }


def collate_fn(batch):
    return {
        "waveform": torch.stack([b["waveform"] for b in batch]),
        "filename": [b["filename"] for b in batch],
        "chunk_start": [b["chunk_start"] for b in batch],
        "species_idx": torch.tensor([b["species_idx"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser(description="Build teacher predictions parquet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--activity-parquet", default="artifacts/focal_activity.parquet")
    parser.add_argument("--output", default="artifacts/teacher_predictions.parquet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})

    # Build model
    from nnAudio.features import MelSpectrogram
    from birdclef_102.model import EfficientNetClassifier

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
        head_dropout=0.0,
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=0.0,
        lora_targets=lora_cfg.get("target_modules"),
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval().cuda()
    logger.info(f"Model loaded: {args.checkpoint}")

    # Build dataset
    dataset = FocalInferenceDataset(
        activity_parquet=args.activity_parquet,
        metadata_csv=data_cfg["metadata_csv"],
        audio_dir=data_cfg["audio_dir"],
        species_list=species_list,
        sample_rate=data_cfg["sample_rate"],
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=collate_fn, pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Inference
    results = []
    t0 = time.time()
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        waveforms = batch["waveform"].cuda()
        species_idxs = batch["species_idx"]
        B = waveforms.shape[0]

        with torch.no_grad(), torch.amp.autocast("cuda"):
            mel = model.compute_mel(waveforms)
            features = model._extract_features(mel)
            frame_features = features.mean(dim=2).permute(0, 2, 1)  # (B, T, D)

            if model.head.projection is not None:
                frame_features = model.head.projection(frame_features)

            frame_logits = model.head.frame_head(frame_features)  # (B, T, C)
            frame_probs = torch.sigmoid(frame_logits.float())  # (B, T, C)

            # Clip-level: max over frames (inference mode)
            clip_max = frame_probs.max(dim=1).values  # (B, C)

            # Clip-level: attention-weighted (training mode)
            att_logits = model.head.att_head(frame_features)
            att_weights = torch.softmax(torch.tanh(att_logits.float()), dim=1)
            clip_att = (att_weights * frame_probs).sum(dim=1)  # (B, C)

        frame_probs_cpu = frame_probs.cpu().numpy()
        clip_max_cpu = clip_max.cpu().numpy()
        clip_att_cpu = clip_att.cpu().numpy()
        T = frame_probs_cpu.shape[1]

        for i in range(B):
            sidx = species_idxs[i].item()
            if sidx < 0:
                continue

            fp = frame_probs_cpu[i, :, sidx]  # (T,)
            teacher_prob = float(clip_max_cpu[i, sidx])
            teacher_prob_att = float(clip_att_cpu[i, sidx])
            active = int((fp > 0.3).sum())

            # Max probability of any other species
            other_max = clip_max_cpu[i].copy()
            other_max[sidx] = 0.0
            max_other = float(other_max.max())

            results.append({
                "filename": batch["filename"][i],
                "chunk_start": batch["chunk_start"][i],
                "teacher_prob": teacher_prob,
                "teacher_prob_att": teacher_prob_att,
                "active_frames": active,
                "total_frames": T,
                "max_other_prob": max_other,
                "frame_probs_labeled": fp.astype(np.float16).tolist(),
            })

        if (batch_idx + 1) % 100 == 0 or batch_idx == total_batches - 1:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) * args.batch_size / elapsed
            eta = (total_batches - batch_idx - 1) * args.batch_size / max(rate, 1)
            logger.info(
                f"Batch {batch_idx + 1}/{total_batches} "
                f"({rate:.0f} samples/sec, ETA {eta/60:.0f} min)"
            )

    # Save
    df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")

    # Summary stats
    print(f"\n=== Teacher Predictions Summary ===")
    print(f"Total chunks: {len(df)}")
    print(f"Frames per chunk: {df['total_frames'].iloc[0]}")
    print(f"\nteacher_prob distribution:")
    print(f"  mean={df['teacher_prob'].mean():.3f}, median={df['teacher_prob'].median():.3f}")
    print(f"  p10={df['teacher_prob'].quantile(0.1):.3f}, p25={df['teacher_prob'].quantile(0.25):.3f}")
    print(f"  p75={df['teacher_prob'].quantile(0.75):.3f}, p90={df['teacher_prob'].quantile(0.9):.3f}")
    print(f"\nactive_frames distribution:")
    print(f"  mean={df['active_frames'].mean():.1f}/{df['total_frames'].iloc[0]}")
    print(f"  zero active: {(df['active_frames'] == 0).sum()} ({100*(df['active_frames'] == 0).mean():.1f}%)")
    print(f"\nFiltering preview (chunks that would be DROPPED):")
    for threshold in [0.1, 0.2, 0.3, 0.5]:
        dropped = (df["teacher_prob"] < threshold).sum()
        print(f"  teacher_prob < {threshold}: {dropped} ({100*dropped/len(df):.1f}%)")
    print(f"\nMislabel candidates (max_other_prob > teacher_prob):")
    mislabel = (df["max_other_prob"] > df["teacher_prob"]).sum()
    print(f"  {mislabel} ({100*mislabel/len(df):.1f}%)")


if __name__ == "__main__":
    main()
