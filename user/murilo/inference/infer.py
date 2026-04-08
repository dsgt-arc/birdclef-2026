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

from birdclef.data.dataset import build_species_list  # noqa: E402
from birdclef.data.inference import InferenceChunkDataset, inference_collate_fn  # noqa: E402

# ProtoCLR imports (always available)
from birdclef.encoders.protoclr import ProtoCLREncoder  # noqa: E402
from birdclef.lightning.lora_finetune import LoRAFinetuneModule  # noqa: E402
from birdclef.preprocessing.protoclr_mel import ProtoCLRMelSpectrogram  # noqa: E402

# SED imports are lazy (inside load_sed_model) to avoid requiring timm/nnAudio
# when running ProtoCLR-only inference.

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


def build_species_list_taxonomy(data_dir: Path) -> list[str]:
    """Build 234-species list from taxonomy.csv (includes soundscape-only species).

    train.csv only has 206 species (focal recordings). SED models are trained
    on 234 classes including 28 soundscape-only species from taxonomy.csv.
    """
    df = pd.read_csv(data_dir / "taxonomy.csv")
    species = sorted(df["primary_label"].astype(str).unique().tolist())
    logger.info(f"Found {len(species)} species in taxonomy.csv")
    return species


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


def load_sed_model(config: dict):
    """Load SED model from checkpoint.

    Reconstructs encoder, sed_head, and mel_transform from config, then loads
    checkpoint weights. Returns (module, mel_transform) since mel runs on GPU
    separately from the module.
    """
    from birdclef.encoders.efficientnet import EfficientNetEncoder
    from birdclef.heads.sed_head import LSEHead
    from birdclef.lightning.sed_finetune import SEDFinetuneModule
    from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

    # 1. Build encoder
    enc_cfg = config["encoder"]
    encoder = EfficientNetEncoder(
        model_name=enc_cfg["model_name"],
        pretrained=enc_cfg.get("pretrained", False),
    )
    logger.info(f"Encoder: {enc_cfg['model_name']}, embed_dim={encoder.embed_dim}")

    # 2. Build SED head
    head_cfg = config["sed_head"]
    sed_head = LSEHead(
        in_channels=encoder.embed_dim,
        num_classes=config["model"]["num_classes"],
        hidden_dim=head_cfg["hidden_dim"],
        dropout=head_cfg.get("dropout", 0.5),
        gem_p=head_cfg.get("gem_p", 3.0),
        lse_temperature=head_cfg.get("lse_temperature", 1.0),
        freq_pool=head_cfg.get("freq_pool", "gem"),
    )

    # 3. Build mel transform (GPU-side, applied in inference loop)
    mel_cfg = config["mel"]
    sample_rate = config["data"]["sample_rate"]
    hop_length = mel_cfg.get("hop_length", 313)
    chunk_duration = config["data"]["chunk_duration"]
    target_frames = int(chunk_duration * sample_rate / hop_length)

    mel_transform = NNAudioMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=mel_cfg.get("n_mels", 256),
        target_frames=target_frames,
        n_fft=mel_cfg.get("n_fft"),
        hop_length=hop_length,
        fmax=mel_cfg.get("fmax"),
        norm_type=mel_cfg.get("norm_type", "amplitude_to_db"),
        top_db=mel_cfg.get("top_db", 80.0),
    )

    # 4. Load checkpoint
    ckpt_path = config["model"]["checkpoint_path"]
    logger.info(f"Loading checkpoint: {ckpt_path}")

    # Pass mel_transform to the module so its state_dict keys are loaded
    # from the checkpoint. We still use it separately in the inference loop.
    module = SEDFinetuneModule.load_from_checkpoint(
        ckpt_path,
        encoder=encoder,
        sed_head=sed_head,
        mel_transform=mel_transform,
        num_classes=config["model"]["num_classes"],
        map_location="cpu",
    )
    module.eval()
    logger.info("SED model loaded and set to eval mode")

    # Return the mel_transform from the module (now has loaded state)
    return module, module.mel_transform


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


@torch.no_grad()
def run_sed_inference(
    module,
    mel_transform,
    dataloader: DataLoader,
    device: torch.device,
    framewise_logits_path: Path | None = None,
    species_list: list[str] | None = None,
    gpu_monitor=None,
) -> tuple[np.ndarray, np.ndarray, dict, int]:
    """Run SED inference: extract embeddings, clip logits, and frame logits.

    Unlike run_inference(), this receives raw waveforms from the DataLoader
    and applies the mel transform on GPU. The LSEHead forward pass is split
    to extract the pre-head backbone embedding.

    Framewise logits are written incrementally to parquet (too large for RAM).

    Args:
        module: SEDFinetuneModule with encoder and sed_head.
        mel_transform: NNAudioMelSpectrogram (moved to GPU).
        dataloader: DataLoader yielding (waveforms, metadata) tuples.
        device: Device to run inference on.
        framewise_logits_path: Path to write framewise logits parquet (streamed).
        species_list: Species names for framewise logits columns.
        gpu_monitor: Optional InferenceGPUMonitor.

    Returns:
        embeddings: (N, embed_dim) float32 — clip-level backbone embedding
        clip_logits: (N, num_classes) float32 — raw clipwise logits
        metadata: dict of lists
        num_frames: number of temporal frames per chunk
    """
    module = module.to(device)
    mel_transform = mel_transform.to(device)

    all_embeddings = []
    all_clip_logits = []
    all_metadata = {
        "filename": [],
        "source": [],
        "chunk_idx": [],
        "start_sec": [],
        "end_sec": [],
    }

    head = module.sed_head
    num_frames = 0

    # Stream framewise logits to parquet in chunks to avoid OOM
    frame_logits_parts = []
    frame_metadata_parts = []
    FLUSH_EVERY = 500  # flush to disk every N batches

    for batch_idx, (waveforms, metadata) in enumerate(tqdm(dataloader, desc="SED Inference")):
        if gpu_monitor:
            gpu_monitor.batch_start()

        waveforms = waveforms.to(device)

        # GPU mel: compute in float32 (matches training behavior)
        with torch.amp.autocast("cuda", enabled=False):
            mel = mel_transform(waveforms.float())

        with torch.amp.autocast("cuda"):
            # Backbone: (B, 1, n_mels, T_mel) -> (B, C, F', T')
            features = module.encoder.forward_features(mel)

            # Split LSEHead forward to extract pre-head embedding
            # Freq pool: (B, C, F', T') -> (B, C, T')
            if head.freq_pool_type == "mean":
                h = features.mean(dim=2)
            else:
                h = head.gem(features)

            # (B, C, T') -> (B, T', C) — frame-level embeddings
            h = h.permute(0, 2, 1)

            # Classification head: (B, T', C) -> (B, T', num_classes)
            frame_logits = head.cls_fc(h)

            # LSE temporal pool: (B, T', num_classes) -> (B, num_classes)
            clip_logits = head.lse_pool(frame_logits)

            # Clip-level embedding: mean pool over time
            clip_emb = h.mean(dim=1)  # (B, C)

        # Store clip-level outputs (small, safe in RAM)
        all_embeddings.append(clip_emb.float().cpu().numpy())
        all_clip_logits.append(clip_logits.float().cpu().numpy())

        # Buffer framewise logits for incremental writing
        if framewise_logits_path is not None:
            fl_np = frame_logits.float().cpu().numpy()  # (B, T, C)
            num_frames = fl_np.shape[1]
            frame_logits_parts.append(fl_np)
            frame_metadata_parts.append({k: list(v) for k, v in metadata.items()})

            if (batch_idx + 1) % FLUSH_EVERY == 0:
                _flush_framewise_logits(
                    frame_logits_parts,
                    frame_metadata_parts,
                    framewise_logits_path,
                    species_list,
                )
                frame_logits_parts.clear()
                frame_metadata_parts.clear()

        if gpu_monitor:
            gpu_monitor.batch_end(batch_size=waveforms.shape[0])

        for key in all_metadata:
            all_metadata[key].extend(metadata[key])

    # Flush remaining framewise logits
    if framewise_logits_path is not None and frame_logits_parts:
        _flush_framewise_logits(
            frame_logits_parts,
            frame_metadata_parts,
            framewise_logits_path,
            species_list,
        )
        frame_logits_parts.clear()
        frame_metadata_parts.clear()

    embeddings = np.concatenate(all_embeddings, axis=0)
    clip_logits_arr = np.concatenate(all_clip_logits, axis=0)

    logger.info(
        f"SED inference complete: {embeddings.shape[0]} chunks, "
        f"embedding dim={embeddings.shape[1]}, "
        f"num_classes={clip_logits_arr.shape[1]}, "
        f"num_frames={num_frames}"
    )

    return embeddings, clip_logits_arr, all_metadata, num_frames


def _flush_framewise_logits(
    frame_logits_parts: list[np.ndarray],
    frame_metadata_parts: list[dict],
    output_path: Path,
    species_list: list[str],
):
    """Append framewise logits to a parquet file incrementally.

    Explodes each chunk into T rows (one per frame) with a frame_idx column
    and one column per species. This avoids storing large nested lists.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for fl_batch, meta_batch in zip(frame_logits_parts, frame_metadata_parts, strict=False):
        # fl_batch: (B, T, num_classes)
        B, T, C = fl_batch.shape
        for i in range(B):
            for t in range(T):
                row = {
                    "filename": meta_batch["filename"][i],
                    "source": meta_batch["source"][i],
                    "chunk_idx": meta_batch["chunk_idx"][i],
                    "start_sec": meta_batch["start_sec"][i],
                    "end_sec": meta_batch["end_sec"][i],
                    "frame_idx": t,
                }
                for s, species in enumerate(species_list):
                    row[species] = float(fl_batch[i, t, s])
                rows.append(row)

    if not rows:
        return

    table = pa.Table.from_pylist(rows)

    if output_path.exists():
        existing = pq.read_table(output_path)
        table = pa.concat_tables([existing, table])

    pq.write_table(table, output_path)
    logger.info(f"  Flushed framewise logits: {len(rows)} rows → {output_path}")


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


def save_sed_results(
    embeddings: np.ndarray,
    clip_logits: np.ndarray,
    metadata: dict,
    species_list: list[str],
    output_dir: Path,
):
    """Save SED inference outputs (embeddings + clip logits) as parquet files.

    Framewise logits are written incrementally during inference (see
    run_sed_inference + _flush_framewise_logits) and are NOT handled here.

    Args:
        embeddings: (N, embed_dim) clip-level embeddings
        clip_logits: (N, num_classes) raw clipwise logits
        metadata: dict with filename, source, chunk_idx, start_sec, end_sec
        species_list: ordered list of species names (column headers for logits)
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

    # 1. embeddings.parquet: clip-level embedding as list[float32]
    emb_df = base_df.copy()
    emb_df["embedding"] = [embeddings[i].tolist() for i in range(n)]

    emb_path = output_dir / "embeddings.parquet"
    emb_df.to_parquet(emb_path, index=False, engine="pyarrow")
    logger.info(f"Saved embeddings: {emb_path} ({n} rows, dim={embeddings.shape[1]})")

    # 2. logits.parquet: raw clipwise logits (one column per species)
    species_df = pd.DataFrame(clip_logits.astype(np.float32), columns=species_list)
    logits_df = pd.concat([base_df, species_df], axis=1)

    logits_path = output_dir / "logits.parquet"
    logits_df.to_parquet(logits_path, index=False, engine="pyarrow")
    logger.info(f"Saved logits: {logits_path} ({n} rows, {len(species_list)} species)")


# ============================================================================
# CLI
# ============================================================================


def _process_source(
    source: str,
    config: dict,
    model_type: str,
    species_list: list[str],
    data_dir: Path,
    output_dir: Path,
    module,
    gpu_mel_transform,
    device: torch.device,
    num_workers: int,
    gpu_monitor,
    args,
):
    """Process a single data source (train_audio or train_soundscapes)."""
    # Skip if output already exists (unless --overwrite or --gpu-check)
    check_file = "logits.parquet" if model_type == "sed" else "embeddings.parquet"
    if not args.overwrite and not args.gpu_check and (output_dir / check_file).exists():
        logger.info(f"Skipping {source}: output already exists at {output_dir}")
        logger.info("  Use --overwrite to re-run and replace existing results")
        return

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

    # For SED: no transform in DataLoader (raw waveforms, mel on GPU)
    # For ProtoCLR: mel transform in DataLoader workers (CPU)
    if model_type == "sed":
        dataset_transform = None
    else:
        dataset_transform = ProtoCLRMelSpectrogram(
            sample_rate=config["data"]["sample_rate"],
        )

    # Build dataset and dataloader
    dataset = InferenceChunkDataset(
        file_list=file_list,
        sample_rate=config["data"]["sample_rate"],
        chunk_duration=config["data"]["chunk_duration"],
        transform=dataset_transform,
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

    if model_type == "sed":
        # Prepare framewise logits path (written incrementally during inference)
        output_dir.mkdir(parents=True, exist_ok=True)
        framewise_path = output_dir / "framewise_logits.parquet"
        if framewise_path.exists():
            framewise_path.unlink()  # remove stale file before streaming

        embeddings, clip_logits, metadata, num_frames = run_sed_inference(
            module,
            gpu_mel_transform,
            dataloader,
            device,
            framewise_logits_path=framewise_path,
            species_list=species_list,
            gpu_monitor=gpu_monitor,
        )
    else:
        embeddings, probabilities, metadata = run_inference(
            module, dataloader, device, gpu_monitor=gpu_monitor
        )

    elapsed = time.time() - t0
    logger.info(f"Inference time: {elapsed:.1f}s ({len(dataset)/elapsed:.1f} chunks/s)")

    # Save results (skip when only profiling GPU)
    if not args.gpu_check or args.overwrite:
        if model_type == "sed":
            save_sed_results(
                embeddings=embeddings,
                clip_logits=clip_logits,
                metadata=metadata,
                species_list=species_list,
                output_dir=output_dir,
            )
        else:
            save_results(
                embeddings=embeddings,
                probabilities=probabilities,
                metadata=metadata,
                species_list=species_list,
                output_dir=output_dir,
            )


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
    model_type = config.get("model_type", "protoclr")
    num_workers = (
        args.num_workers if args.num_workers is not None else config["data"]["num_workers"]
    )

    logger.info(f"Run name: {run_name}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Source: {args.source}")

    # Determine sources to process
    sources = []
    if args.source in ("train_audio", "all"):
        sources.append("train_audio")
    if args.source in ("train_soundscapes", "all"):
        sources.append("train_soundscapes")

    # Build species list
    if model_type == "sed":
        species_list = build_species_list_taxonomy(data_dir)  # 234 species
    else:
        metadata_csv = data_dir / "train.csv"
        species_list = build_species_list(metadata_csv)  # 206 species
    logger.info(f"Species: {len(species_list)}")

    # GPU monitor (optional)
    gpu_monitor = None
    if args.gpu_check:
        from birdclef.callbacks.gpu_monitor import InferenceGPUMonitor

        gpu_monitor = InferenceGPUMonitor(
            batch_size=config["data"]["batch_size"],
            num_workers=num_workers,
        )
        logger.info("GPU profiling enabled (--gpu-check)")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Output base directory
    dataset_name = config["dataset_name"]
    output_base = (
        Path("/storage/project/ps-clef2026_birdclef-0/shared/inference") / dataset_name / run_name
    )
    logger.info(f"Output: {output_base}")

    # Determine fold configs (4-fold or single)
    folds = config["model"].get("folds")

    if folds:
        # ---- 4-fold: process each fold sequentially ----
        for fold_idx, fold_cfg in enumerate(folds):
            logger.info(f"\n{'#'*60}")
            logger.info(f"Fold {fold_idx}")
            logger.info(f"{'#'*60}")

            # Override checkpoint path for this fold
            config["model"]["checkpoint_path"] = fold_cfg["checkpoint_path"]
            module, gpu_mel_transform = load_sed_model(config)

            for source in sources:
                output_dir = output_base / source / f"fold{fold_idx}"
                _process_source(
                    source=source,
                    config=config,
                    model_type=model_type,
                    species_list=species_list,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    module=module,
                    gpu_mel_transform=gpu_mel_transform,
                    device=device,
                    num_workers=num_workers,
                    gpu_monitor=gpu_monitor,
                    args=args,
                )

            # Free GPU memory between folds
            del module, gpu_mel_transform
            torch.cuda.empty_cache()

    else:
        # ---- Single model ----
        if model_type == "sed":
            module, gpu_mel_transform = load_sed_model(config)
        else:
            module = load_model(config)
            gpu_mel_transform = None

        for source in sources:
            output_dir = output_base / source
            _process_source(
                source=source,
                config=config,
                model_type=model_type,
                species_list=species_list,
                data_dir=data_dir,
                output_dir=output_dir,
                module=module,
                gpu_mel_transform=gpu_mel_transform,
                device=device,
                num_workers=num_workers,
                gpu_monitor=gpu_monitor,
                args=args,
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
