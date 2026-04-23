#!/usr/bin/env python3
"""Batch pre-computation of per-sample losses and predictions.

Runs inference on the validation set, saves per-sample loss, predictions, and
logits to parquet + numpy files for fast browsing in the Marimo explorer app.

Usage:
    python precompute_losses.py \
        --config ../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml \
        --checkpoint /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt \
        --output losses_v5j.parquet \
        --device cuda

    # CPU (slower but works without GPU):
    python precompute_losses.py --config ... --checkpoint ... --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# Add src to path
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from model_loader import build_mel_transform, build_model, load_config  # noqa: E402

from birdclef.data.sed import (  # noqa: E402
    SEDFocalDataset,
    SEDSoundscapeDataset,
    _split_focal_dataframe,
    _split_soundscape_by_file,
)
from birdclef.data.soundscape import build_full_species_list  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_val_dataset(
    config: dict,
    species_list: list[str],
    data_dir: Path,
) -> tuple[ConcatDataset, list[dict]]:
    """Reconstruct the exact validation dataset from config.

    Replicates the split logic from SEDDataModule.setup() with seed=42.

    Returns:
        (val_dataset, sample_metadata) where sample_metadata is a list of dicts
        with keys: filename, source, primary_label, all_labels.
    """
    sample_rate = config["data"]["sample_rate"]
    window_duration = config["data"]["window_duration"]
    val_fraction = config["data"].get("val_fraction", 0.1)
    seed = config["data"].get("seed", 42)

    # Focal data
    train_csv = data_dir / "train.csv"
    focal_df = pd.read_csv(train_csv)
    focal_train, focal_val, _ = _split_focal_dataframe(
        focal_df, val_fraction=val_fraction, test_fraction=0.0, seed=seed
    )

    focal_val_dataset = SEDFocalDataset(
        audio_dir=str(data_dir / "train_audio"),
        metadata_df=focal_val,
        species_list=species_list,
        sample_rate=sample_rate,
        window_duration=window_duration,
        transform=None,  # Raw waveforms — mel computed separately
    )

    # Soundscape data
    soundscape_labels_csv = data_dir / "train_soundscapes_labels.csv"
    datasets = [focal_val_dataset]
    metadata = []

    # Collect focal metadata
    for i in range(len(focal_val_dataset)):
        rec = focal_val_dataset.recordings[i]
        metadata.append(
            {
                "filename": str(rec[0]),
                "source": "focal",
                "primary_label": rec[1],
                "all_labels": rec[1],
            }
        )

    if soundscape_labels_csv.exists():
        sc_df = pd.read_csv(soundscape_labels_csv)
        sc_train, sc_val, _ = _split_soundscape_by_file(
            sc_df,
            soundscape_dir=str(data_dir / "train_soundscapes"),
            n_val_files=config["data"].get("n_val_soundscape_files", 8),
            n_test_files=0,
            seed=seed + 1,
        )

        if len(sc_val) > 0:
            sc_val_dataset = SEDSoundscapeDataset(
                soundscape_dir=str(data_dir / "train_soundscapes"),
                labels_df=sc_val,
                species_list=species_list,
                sample_rate=sample_rate,
                target_duration=5.0,
                context_duration=window_duration,
                transform=None,
            )
            datasets.append(sc_val_dataset)

            for i in range(len(sc_val_dataset)):
                chunk = sc_val_dataset.chunks[i]
                metadata.append(
                    {
                        "filename": str(chunk[0]),
                        "source": "soundscape",
                        "primary_label": chunk[2] if len(chunk) > 2 else "unknown",
                        "all_labels": chunk[2] if len(chunk) > 2 else "unknown",
                    }
                )

    val_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return val_dataset, metadata


def precompute_losses(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 4,
) -> pd.DataFrame:
    """Run inference on all val samples and save results."""
    config = load_config(config_path)
    data_dir = Path(config["data"]["data_dir"]).expanduser()
    taxonomy_csv = data_dir / "taxonomy.csv"
    species_list = build_full_species_list(taxonomy_csv)
    num_classes = len(species_list)

    # Build model
    model = build_model(config, num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model.to(device)

    # Build mel transform
    mel_transform = build_mel_transform(config)
    mel_transform.to(device)

    # Build val dataset
    val_dataset, sample_metadata = build_val_dataset(config, species_list, data_dir)
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # Run inference
    all_logits = []
    all_losses = []

    training_cfg = config.get("training", {})
    blend_clip = training_cfg.get("blend_clip", 1.0)
    blend_frame = training_cfg.get("blend_frame", 0.0)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing losses"):
            waveform, labels, primary_idx = batch
            waveform = waveform.to(device)
            labels = labels.to(device)

            # Mel transform
            mel = mel_transform(waveform)

            # Forward
            clipwise_logits, framewise_logits = model(mel)

            # Blended logits (matching inference behavior)
            if blend_frame > 0:
                framewise_max = framewise_logits.max(dim=-1).values
                blended_logits = blend_clip * clipwise_logits + blend_frame * framewise_max
            else:
                blended_logits = clipwise_logits

            # Per-sample loss (mean across classes)
            per_sample_loss = F.binary_cross_entropy_with_logits(
                blended_logits, labels, reduction="none"
            ).mean(dim=-1)

            all_logits.append(blended_logits.cpu().numpy())
            all_losses.append(per_sample_loss.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)  # (N, 234)
    all_losses = np.concatenate(all_losses, axis=0)  # (N,)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid

    # Build results dataframe
    rows = []
    for i in range(len(all_logits)):
        probs_i = all_probs[i]
        meta = (
            sample_metadata[i]
            if i < len(sample_metadata)
            else {
                "filename": f"unknown_{i}",
                "source": "unknown",
                "primary_label": "unknown",
                "all_labels": "unknown",
            }
        )

        # Top-5 predictions
        top5_idx = np.argsort(probs_i)[::-1][:5]
        top5_species = [species_list[j] for j in top5_idx]
        top5_probs = [float(probs_i[j]) for j in top5_idx]

        # Ground truth species prob and rank
        primary = meta["primary_label"]
        if primary in species_list:
            gt_idx = species_list.index(primary)
            gt_prob = float(probs_i[gt_idx])
            gt_rank = int((probs_i > probs_i[gt_idx]).sum()) + 1
        else:
            gt_prob = 0.0
            gt_rank = num_classes

        rows.append(
            {
                "idx": i,
                "filename": meta["filename"],
                "source": meta["source"],
                "primary_label": primary,
                "all_labels": meta["all_labels"],
                "loss": float(all_losses[i]),
                "gt_prob": gt_prob,
                "gt_rank": gt_rank,
                "top1_species": top5_species[0],
                "top1_prob": top5_probs[0],
                "top5_species": ";".join(top5_species),
                "top5_probs": ";".join(f"{p:.4f}" for p in top5_probs),
            }
        )

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} samples to {output_path}")

    # Save raw logits as numpy array for deeper analysis
    logits_path = output_path.with_name(output_path.stem + "_logits.npy")
    np.save(logits_path, all_logits)
    logger.info(f"Saved logits ({all_logits.shape}) to {logits_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Pre-compute per-sample losses for val set")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="losses.parquet", help="Output parquet path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    precompute_losses(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
