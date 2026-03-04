#!/usr/bin/env python3
"""Inference script: extract embeddings and predictions from fine-tuned models.

Processes train_audio and/or train_soundscapes through a fine-tuned model,
saving per-chunk embeddings and species probability predictions as parquet files.

Usage:
    python infer.py --config config/protoclr_lora_attn.yaml --source train_audio
    python infer.py --config config/protoclr_lora_attn.yaml --source train_soundscapes
    python infer.py --config config/protoclr_lora_attn.yaml --source all
    python infer.py --config config/protoclr_lora_attn.yaml --source train_audio --max-files 10

    # GPU profiling (quick check with a few files)
    python infer.py --config config/protoclr_lora_attn.yaml --source train_audio --max-files 10 --gpu-check

    # Overwrite existing output
    python infer.py --config config/protoclr_lora_attn.yaml --source train_audio --overwrite
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from birdclef.data.birdclef2025 import build_species_list  # noqa: E402
from birdclef.data.inference import InferenceChunkDataset, inference_collate_fn  # noqa: E402
from birdclef.encoders.protoclr import ProtoCLREncoder  # noqa: E402
from birdclef.lightning.lora_finetune import LoRAFinetuneModule  # noqa: E402
from birdclef.preprocessing.protoclr_mel import ProtoCLRMelSpectrogram  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


# ============================================================================
# File enumeration
# ============================================================================


def enumerate_train_audio(data_dir: Path) -> list[dict]:
    """Build file list from train.csv for train_audio/ files."""
    metadata_csv = data_dir / "train.csv"
    df = pd.read_csv(metadata_csv)
    file_list = []
    for _, row in df.iterrows():
        audio_path = data_dir / "train_audio" / row["filename"]
        if audio_path.exists():
            file_list.append(
                {
                    "path": audio_path,
                    "filename": row["filename"],
                    "source": "train_audio",
                }
            )
    logger.info(f"train_audio: {len(file_list)} files from {len(df)} metadata rows")
    return file_list


def enumerate_train_soundscapes(data_dir: Path) -> list[dict]:
    """Build file list by globbing train_soundscapes/*.ogg."""
    soundscape_dir = data_dir / "train_soundscapes"
    files = sorted(soundscape_dir.glob("*.ogg"))
    file_list = [
        {
            "path": f,
            "filename": f.name,
            "source": "train_soundscapes",
        }
        for f in files
    ]
    logger.info(f"train_soundscapes: {len(file_list)} files")
    return file_list


# ============================================================================
# Model loading
# ============================================================================


def load_model(config: dict) -> LoRAFinetuneModule:
    """Load fine-tuned model from checkpoint.

    Follows the same pattern as train.py:
    1. Build ProtoCLREncoder (downloads from HuggingFace)
    2. Apply LoRA structure if lora config present
    3. Load checkpoint weights via LoRAFinetuneModule.load_from_checkpoint
    """
    from peft import LoraConfig, get_peft_model

    # 1. Build encoder
    encoder = ProtoCLREncoder(
        model_name=config["encoder"]["model_name"],
        checkpoint=config["encoder"].get("checkpoint", "protoclr.pth"),
    )
    logger.info(f"Encoder embed_dim: {encoder.embed_dim}")

    # 2. Apply LoRA structure
    if "lora" in config:
        lora_cfg = config["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
        encoder.model = get_peft_model(encoder.model, lora_config)
        logger.info("Applied LoRA adapters")
        encoder.model.print_trainable_parameters()

    # 3. Load checkpoint
    ckpt_path = config["model"]["checkpoint_path"]
    logger.info(f"Loading checkpoint: {ckpt_path}")

    module = LoRAFinetuneModule.load_from_checkpoint(
        ckpt_path,
        encoder=encoder,
        num_classes=config["model"]["num_classes"],
        embed_dim=config["encoder"].get("embed_dim", 384),
        hidden_dim=config["model"].get("hidden_dim", 0),
        threshold=config["model"].get("threshold", 0.5),
        map_location="cpu",
    )
    module.eval()
    logger.info("Model loaded and set to eval mode")

    return module


# ============================================================================
# Inference loop
# ============================================================================


@torch.no_grad()
def run_inference(
    module: LoRAFinetuneModule,
    dataloader: DataLoader,
    device: torch.device,
    gpu_monitor=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run inference, extracting embeddings and predictions.

    Args:
        module: Fine-tuned model module.
        dataloader: DataLoader yielding (mel, metadata) tuples.
        device: Device to run inference on.
        gpu_monitor: Optional InferenceGPUMonitor for GPU profiling.

    Returns:
        embeddings: (N, embed_dim) float32 numpy array
        probabilities: (N, num_classes) float32 numpy array
        metadata: dict of lists (filename, source, chunk_idx, start_sec, end_sec)
    """
    module = module.to(device)

    all_embeddings = []
    all_probs = []
    all_metadata = {
        "filename": [],
        "source": [],
        "chunk_idx": [],
        "start_sec": [],
        "end_sec": [],
    }

    for mel, metadata in tqdm(dataloader, desc="Inference"):
        if gpu_monitor:
            gpu_monitor.batch_start()

        mel = mel.to(device)

        with torch.amp.autocast("cuda"):
            embeddings = module.encoder(mel)  # (batch, embed_dim)
            logits = module.head(embeddings)  # (batch, num_classes)

        probs = torch.sigmoid(logits)

        # Store as float32 on CPU
        all_embeddings.append(embeddings.float().cpu().numpy())
        all_probs.append(probs.float().cpu().numpy())

        if gpu_monitor:
            gpu_monitor.batch_end(batch_size=mel.shape[0])

        for key in all_metadata:
            all_metadata[key].extend(metadata[key])

    embeddings = np.concatenate(all_embeddings, axis=0)
    probabilities = np.concatenate(all_probs, axis=0)

    logger.info(
        f"Inference complete: {embeddings.shape[0]} chunks, "
        f"embedding dim={embeddings.shape[1]}, "
        f"num_classes={probabilities.shape[1]}"
    )

    return embeddings, probabilities, all_metadata


# ============================================================================
# Save results
# ============================================================================


def save_results(
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    metadata: dict,
    species_list: list[str],
    output_dir: Path,
):
    """Save embeddings and predictions as parquet files.

    Args:
        embeddings: (N, embed_dim) array
        probabilities: (N, num_classes) array
        metadata: dict with filename, source, chunk_idx, start_sec, end_sec
        species_list: ordered list of species names (column headers for predictions)
        output_dir: directory to write parquet files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n = embeddings.shape[0]

    # Base metadata columns
    base_df = pd.DataFrame(
        {
            "filename": metadata["filename"],
            "source": metadata["source"],
            "chunk_idx": metadata["chunk_idx"],
            "start_sec": metadata["start_sec"],
            "end_sec": metadata["end_sec"],
        }
    )

    # Embeddings parquet: embedding stored as list[float32]
    emb_df = base_df.copy()
    emb_df["embedding"] = [embeddings[i].tolist() for i in range(n)]

    emb_path = output_dir / "embeddings.parquet"
    emb_df.to_parquet(emb_path, index=False, engine="pyarrow")
    logger.info(f"Saved embeddings: {emb_path} ({n} rows, dim={embeddings.shape[1]})")

    # Predictions parquet: one column per species
    species_df = pd.DataFrame(probabilities.astype(np.float32), columns=species_list)
    pred_df = pd.concat([base_df, species_df], axis=1)

    pred_path = output_dir / "predictions.parquet"
    pred_df.to_parquet(pred_path, index=False, engine="pyarrow")
    logger.info(f"Saved predictions: {pred_path} ({n} rows, {len(species_list)} species)")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings and predictions from fine-tuned models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["train_audio", "train_soundscapes", "all"],
        help="Data source to process",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files (for debugging)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers from config",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip sources with existing results)",
    )
    parser.add_argument(
        "--gpu-check",
        action="store_true",
        help="Enable GPU utilization profiling (use with --max-files for quick check)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_name = config["run_name"]
    data_dir = Path(config["data"]["data_dir"]).expanduser()
    num_workers = (
        args.num_workers if args.num_workers is not None else config["data"]["num_workers"]
    )

    logger.info(f"Run name: {run_name}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Source: {args.source}")

    # Determine sources to process
    sources = []
    if args.source in ("train_audio", "all"):
        sources.append("train_audio")
    if args.source in ("train_soundscapes", "all"):
        sources.append("train_soundscapes")

    # Build species list
    metadata_csv = data_dir / "train.csv"
    species_list = build_species_list(metadata_csv)
    logger.info(f"Species: {len(species_list)}")

    # Build mel transform
    mel_transform = ProtoCLRMelSpectrogram(
        sample_rate=config["data"]["sample_rate"],
    )

    # GPU monitor (optional)
    gpu_monitor = None
    if args.gpu_check:
        from birdclef.callbacks.gpu_monitor import InferenceGPUMonitor

        gpu_monitor = InferenceGPUMonitor(
            batch_size=config["data"]["batch_size"],
            num_workers=num_workers,
        )
        logger.info("GPU profiling enabled (--gpu-check)")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    module = load_model(config)

    # Output base directory: /storage/project/<project>/shared/inference/<dataset>/<run_name>/
    dataset_name = config["dataset_name"]
    output_base = (
        Path("/storage/project/ps-clef2026_birdclef-0/shared/inference") / dataset_name / run_name
    )
    logger.info(f"Output: {output_base}")

    # Process each source
    for source in sources:
        output_dir = output_base / source

        # Skip if output already exists (unless --overwrite or --gpu-check)
        if (
            not args.overwrite
            and not args.gpu_check
            and (output_dir / "embeddings.parquet").exists()
        ):
            logger.info(f"Skipping {source}: output already exists at {output_dir}")
            logger.info("  Use --overwrite to re-run and replace existing results")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {source}")
        logger.info(f"{'='*60}")

        # Enumerate files
        if source == "train_audio":
            file_list = enumerate_train_audio(data_dir)
        else:
            file_list = enumerate_train_soundscapes(data_dir)

        # Apply max-files limit
        if args.max_files is not None:
            file_list = file_list[: args.max_files]
            logger.info(f"Limited to {len(file_list)} files (--max-files)")

        # Build dataset and dataloader
        dataset = InferenceChunkDataset(
            file_list=file_list,
            sample_rate=config["data"]["sample_rate"],
            chunk_duration=config["data"]["chunk_duration"],
            transform=mel_transform,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=inference_collate_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        # Run inference
        t0 = time.time()
        embeddings, probabilities, metadata = run_inference(
            module,
            dataloader,
            device,
            gpu_monitor=gpu_monitor,
        )
        elapsed = time.time() - t0
        logger.info(f"Inference time: {elapsed:.1f}s ({len(dataset)/elapsed:.1f} chunks/s)")

        # Save results (skip when only profiling GPU)
        if not args.gpu_check or args.overwrite:
            save_results(
                embeddings=embeddings,
                probabilities=probabilities,
                metadata=metadata,
                species_list=species_list,
                output_dir=output_dir,
            )

    # Print GPU profiling summary
    if gpu_monitor:
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_file = summary_dir / f"{run_name}_{job_id}.txt"
        gpu_monitor.print_summary(save_path=summary_file)
        gpu_monitor.shutdown()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
