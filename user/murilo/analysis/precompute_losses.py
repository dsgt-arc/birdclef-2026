#!/usr/bin/env python3
"""Batch pre-computation of per-sample losses and predictions.

Runs inference on the validation OR training set, saves per-sample loss,
predictions, and logits to parquet + numpy files for fast browsing in the
Marimo explorer app and per-frame analysis.

The training-split mode (--data-split=train, added 2026-04-26 for D1 diagnosis)
runs deterministic forward passes (no mixup, no augmentation) on the same focal
+ labeled-soundscape rows the model was trained on, in order to inspect
attention behavior on highest-loss-during-training and lowest-loss-during-training
examples (Salman-style).

For AttHead-family heads (head type starts with "att_head"), a forward hook on
the attention Conv1D captures the learned attention pre-activation — saved
to an extra `*_perframe_att.npy` so per_frame_analysis.py can compute the
real `softmax(tanh(att_raw))` attention weights instead of LSE-implicit.

Usage:
    # Val (existing behavior):
    python precompute_losses.py \
        --config ../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml \
        --checkpoint /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt \
        --output losses_v5j.parquet \
        --device cuda

    # Train (D1 diagnosis):
    python precompute_losses.py \
        --config ../experiments/021-sed-training/config/hgnet_b0_atthead_v6k.yaml \
        --checkpoint /storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/021-sed-training/checkpoints/2026-hgnet-b0-atthead-v6k/best-epochepoch=19.ckpt \
        --data-split train \
        --output data/losses_v6k_train.parquet \
        --device cuda

    # CPU (slower but works without GPU):
    python precompute_losses.py --config ... --checkpoint ... --device cpu
"""

import argparse
import logging
import random
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


def build_split_dataset(
    config: dict,
    species_list: list[str],
    data_dir: Path,
    split: str = "val",
) -> tuple[ConcatDataset, list[dict]]:
    """Reconstruct the focal + soundscape dataset for the requested split.

    Replicates the split logic from SEDDataModule.setup() with seed=42.

    Args:
        split: "val" (default, existing behavior) or "train" (D1 diagnosis).
            Both modes use the same `_split_focal_dataframe` /
            `_split_soundscape_by_file` calls — we just pick the
            corresponding output dataframe.

    Returns:
        (dataset, sample_metadata) where sample_metadata is a list of dicts
        with keys: filename, source, primary_label, all_labels.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split}")

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
    focal_split_df = focal_train if split == "train" else focal_val

    focal_dataset = SEDFocalDataset(
        audio_dir=str(data_dir / "train_audio"),
        metadata_df=focal_split_df,
        species_list=species_list,
        sample_rate=sample_rate,
        window_duration=window_duration,
        transform=None,  # Raw waveforms — mel computed separately
    )

    # Soundscape data
    soundscape_labels_csv = data_dir / "train_soundscapes_labels.csv"
    datasets = [focal_dataset]
    metadata = []

    # Collect focal metadata with deterministic crop offsets
    window_samples = int(window_duration * sample_rate)
    rng = random.Random(seed)
    for i in range(len(focal_dataset)):
        audio_path, total_samples, label_vec, primary_idx = focal_dataset.recordings[i]
        primary_label = species_list[primary_idx] if primary_idx < len(species_list) else "unknown"
        # Deterministic crop offset (matches what the dataset will produce with this seed)
        if total_samples > window_samples:
            offset = rng.randint(0, total_samples - window_samples)
        else:
            offset = 0
        offset_sec = offset / sample_rate
        metadata.append(
            {
                "filename": str(audio_path),
                "source": "focal",
                "primary_label": primary_label,
                "all_labels": primary_label,
                "offset_sec": offset_sec,
            }
        )

    if soundscape_labels_csv.exists():
        sc_df = pd.read_csv(soundscape_labels_csv)
        sc_train, sc_val, _ = _split_soundscape_by_file(
            sc_df,
            soundscape_val_files=config["data"].get(
                "soundscape_val_files",
                config["data"].get("n_val_soundscape_files", 8),
            ),
            soundscape_test_files=0,
            seed=seed + 1,
        )
        sc_split = sc_train if split == "train" else sc_val

        if len(sc_split) > 0:
            sc_dataset = SEDSoundscapeDataset(
                soundscape_dir=str(data_dir / "train_soundscapes"),
                labels_df=sc_split,
                species_list=species_list,
                sample_rate=sample_rate,
                target_duration=5.0,
                context_duration=window_duration,
                transform=None,
            )
            datasets.append(sc_dataset)

            for i in range(len(sc_dataset)):
                audio_path, start_sample, label_vec, primary_idx = sc_dataset.chunks[i]
                primary_label = (
                    species_list[primary_idx] if primary_idx < len(species_list) else "unknown"
                )
                # Collect all positive species from multi-hot label vector
                active = [species_list[j] for j in range(len(species_list)) if label_vec[j] > 0.5]
                # Soundscape offset: center of the context window
                extra_context = int((window_duration - 5.0) / 2 * sample_rate)
                load_start = max(0, start_sample - extra_context)
                metadata.append(
                    {
                        "filename": str(audio_path),
                        "source": "soundscape",
                        "primary_label": primary_label,
                        "all_labels": ";".join(active) if active else primary_label,
                        "offset_sec": load_start / sample_rate,
                    }
                )

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return full_dataset, metadata


def build_val_dataset(
    config: dict,
    species_list: list[str],
    data_dir: Path,
) -> tuple[ConcatDataset, list[dict]]:
    """Backward-compatible wrapper — calls build_split_dataset(split='val')."""
    return build_split_dataset(config, species_list, data_dir, split="val")


def _find_attention_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """Locate the AttHead.attention_conv module so we can register a forward hook.

    Returns None if the model's head isn't AttHead-family. Looks at the head
    submodule (named `sed_head` on `SEDFinetuneModule`, or `head` on simpler
    wrappers) for either `attention_conv` (AttHead, AttHeadMultiContext) or
    `att_conv` (AttBlockV2). LSEHead has no attention module — returns None.
    """
    head = getattr(model, "sed_head", None) or getattr(model, "head", None)
    if head is None:
        return None
    for attr in ("attention_conv", "att_conv"):
        if hasattr(head, attr):
            return getattr(head, attr)
    return None


def precompute_losses(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int = 64,
    device: str = "cuda",
    num_workers: int = 4,
    data_split: str = "val",
) -> pd.DataFrame:
    """Run inference on all samples in the requested split and save results.

    Args:
        data_split: "val" (default) or "train". Train mode runs deterministic
            inference over the same focal + labeled-SS rows that were used for
            training v5j-family models — used by the D1 diagnosis to inspect
            attention behavior on highest/lowest-loss training examples.
    """
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

    # If the head is AttHead-family, register a forward hook on the attention
    # Conv1D so we can capture per-frame attention pre-activation alongside
    # the framewise logits. For LSEHead this returns None (no hook).
    head_type_cfg = config.get("sed_head", {}).get("type", "unknown")
    attention_module = _find_attention_module(model)
    captured_att: list[np.ndarray] = []
    att_hook_handle = None
    if attention_module is not None:
        logger.info(
            f"Detected AttHead-family head (sed_head.type={head_type_cfg}); "
            f"capturing attention via forward hook on {type(attention_module).__name__}."
        )

        def _att_hook(_module, _inputs, output):
            # output shape: (B, num_classes, T) — pre-tanh, pre-softmax
            captured_att.append(output.detach().cpu().to(torch.float16).numpy())

        att_hook_handle = attention_module.register_forward_hook(_att_hook)
    else:
        # Loud warning if the config CLAIMS AttHead-family but we couldn't find
        # the attention module — easy to silently miss otherwise.
        if head_type_cfg in ("att_head", "att_block_v2", "att_head_multi_context"):
            logger.warning(
                f"sed_head.type={head_type_cfg} but no attention_conv/att_conv "
                f"found on model.sed_head or model.head. Attention will NOT be "
                f"captured. Inspect the model structure and update "
                f"_find_attention_module()."
            )
        else:
            logger.info(
                f"sed_head.type={head_type_cfg} — no AttHead-family head; "
                f"skipping attention capture (LSE-implicit attention will be "
                f"derived in per_frame_analysis.py)."
            )

    # Build mel transform
    mel_transform = build_mel_transform(config)
    mel_transform.to(device)

    # Build dataset for requested split
    full_dataset, sample_metadata = build_split_dataset(
        config, species_list, data_dir, split=data_split
    )
    logger.info(f"{data_split.capitalize()} dataset: {len(full_dataset)} samples")

    # Seed the global random module so SEDFocalDataset's random crops are
    # deterministic and match the offsets stored in sample_metadata.
    seed = config["data"].get("seed", 42)
    import random as py_random

    py_random.seed(seed)

    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Must be 0 for deterministic random crops
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # Run inference
    all_logits = []
    all_framewise_logits = []
    all_losses = []

    training_cfg = config.get("training", {})
    blend_clip = training_cfg.get("blend_clip", 1.0)
    blend_frame = training_cfg.get("blend_frame", 0.0)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Computing losses ({data_split})"):
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
            # framewise_logits shape: (B, n_classes, T) — last dim is time frames
            all_framewise_logits.append(framewise_logits.cpu().numpy().astype(np.float16))
            all_losses.append(per_sample_loss.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)  # (N, 234)
    all_framewise_logits = np.concatenate(all_framewise_logits, axis=0)  # (N, 234, T)
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
                "offset_sec": meta.get("offset_sec", 0.0),
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

    # Save raw clipwise logits for deeper analysis
    logits_path = output_path.with_name(output_path.stem + "_logits.npy")
    np.save(logits_path, all_logits)
    logger.info(f"Saved clipwise logits ({all_logits.shape}) to {logits_path}")

    # Save per-frame logits for LSE-attention / per-frame analysis (float16 to save space)
    perframe_path = output_path.with_name(output_path.stem + "_perframe_logits.npy")
    np.save(perframe_path, all_framewise_logits)
    logger.info(f"Saved per-frame logits ({all_framewise_logits.shape}) to {perframe_path}")

    # If we captured AttHead attention, save it alongside (pre-tanh, pre-softmax).
    # per_frame_analysis.py applies tanh+softmax to recover real attention weights.
    if att_hook_handle is not None:
        att_hook_handle.remove()
        if captured_att:
            all_att_raw = np.concatenate(captured_att, axis=0)
            att_path = output_path.with_name(output_path.stem + "_perframe_att.npy")
            np.save(att_path, all_att_raw)
            logger.info(
                f"Saved per-frame attention pre-activation ({all_att_raw.shape}) to {att_path}"
            )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute per-sample losses for val or training set"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="losses.parquet", help="Output parquet path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--data-split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Which split to run inference on. 'val' = existing behavior; "
        "'train' = D1 diagnosis (deterministic forward pass on training data).",
    )
    args = parser.parse_args()

    precompute_losses(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        data_split=args.data_split,
    )


if __name__ == "__main__":
    main()
