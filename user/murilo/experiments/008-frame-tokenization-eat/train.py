#!/usr/bin/env python3
"""Training script for Experiment 008: Frame Tokenization for EAT on BirdCLEF 2025.

Replaces EAT's 16x16 square patch embedding with full-frequency vertical frame
tokenization (W x 128, W in {2, 4}) while LoRA fine-tuning the pretrained
transformer blocks.

Usage:
    python train.py --config config/config_a_baseline.yaml [--fast-dev-run]
    python train.py --config config/config_c1a_frame_2x128.yaml [--fast-dev-run]
    python train.py --config config/config_c1a_frame_2x128.yaml --gpu-check 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from user-level .env (must be before W&B imports)
_user_dir = Path(__file__).parent.parent.parent
load_dotenv(_user_dir / ".env")

import numpy as np  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger  # noqa: E402

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from birdclef.data import BirdCLEF2025DataModule, build_species_list  # noqa: E402
from birdclef.encoders.eat import EATEncoder  # noqa: E402
from birdclef.lightning.lora_finetune import LoRAFinetuneModule  # noqa: E402
from birdclef.lora.peft import apply_lora  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "008-frame-tokenization-eat"


# ============================================================================
# Sincos positional encoding (copied from EAT's model_core.py to avoid
# fragile HF cache dependency — 3 pure-numpy functions + 1 nn.Module wrapper)
# ============================================================================


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Sincos positional encoding for a 1D grid.

    Args:
        embed_dim: output dimension for each position
        pos: positions to encode, shape (M,)

    Returns:
        Positional embeddings of shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Sincos positional encoding for a 2D grid.

    Args:
        embed_dim: output dimension (split evenly between H and W)
        grid: meshgrid array of shape [2, 1, H, W]

    Returns:
        Positional embeddings of shape (H*W, embed_dim)
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """Generate 2D sincos positional embeddings for a given grid size.

    Args:
        embed_dim: embedding dimension
        grid_size: (H, W) tuple
        cls_token: if True, prepend a zero vector for the CLS token

    Returns:
        Positional embeddings of shape [H*W, embed_dim] or [1+H*W, embed_dim]
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first in meshgrid
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class FixedPositionalEncoder(nn.Module):
    """Wrapper that returns a fixed positional encoding tensor.

    Uses register_buffer so the tensor moves to GPU with the model.
    """

    def __init__(self, pos_embed):
        super().__init__()
        self.register_buffer("positions", pos_embed)

    def forward(self, x, padding_mask):
        return self.positions


# ============================================================================
# Encoder setup
# ============================================================================


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def build_encoder(config: dict, precomputed_features: bool = False):
    """Build EAT encoder from config."""
    return EATEncoder(
        model_name=config["encoder"].get("model_name", "worstchan/EAT-base_epoch30_pretrain"),
        target_length=config["encoder"].get("target_length", 512),
        precomputed_features=precomputed_features,
    )


# ============================================================================
# Model modification: patch embedding + positional encoding
# ============================================================================


def modify_eat_model(encoder, config: dict):
    """Modify EAT's patch embedding and positional encoding based on config.

    Handles 3 patch_embed types:
    - "pretrained": freeze existing Conv2d, keep original sincos pos encoding
    - "random_square": replace Conv2d(1,768,16,16) with random init (trainable),
                       regenerate sincos for (32, 8) grid
    - "frame": replace with Conv2d(1,768,(W,128),(W,128)) random init (trainable),
               regenerate sincos for (512/W, 1) grid

    Args:
        encoder: EATEncoder with loaded pretrained model
        config: full config dict with patch_embed section

    Returns:
        Modified encoder
    """
    patch_cfg = config["patch_embed"]
    patch_type = patch_cfg["type"]
    patch_size = tuple(patch_cfg["patch_size"])  # (time, freq)
    freeze = patch_cfg["freeze"]

    # Access the EAT nn.Module backbone
    eat_model = encoder.model.model  # EATEncoder.model (EATModel) .model (EAT)
    embed_dim = 768
    target_length = config["encoder"].get("target_length", 512)

    if patch_type == "pretrained":
        # Config A: freeze existing pretrained patch embedding, keep original pos encoding
        logger.info("Patch embed: using pretrained 16x16 (frozen)")
        for param in eat_model.local_encoder.parameters():
            param.requires_grad = False

    elif patch_type in ("random_square", "frame"):
        # Config B/C: replace patch embedding with new random-init Conv2d
        kernel_size = patch_size
        stride = patch_size

        new_proj = nn.Conv2d(1, embed_dim, kernel_size=kernel_size, stride=stride)
        nn.init.xavier_uniform_(new_proj.weight)
        nn.init.zeros_(new_proj.bias)

        eat_model.local_encoder.proj = new_proj
        logger.info(
            f"Patch embed: replaced with random Conv2d(1, {embed_dim}, "
            f"kernel_size={kernel_size}, stride={stride})"
        )

        # Compute new grid size for positional encoding
        grid_h = target_length // patch_size[0]  # time dimension
        grid_w = 128 // patch_size[1]  # freq dimension
        grid_size = (grid_h, grid_w)

        # Regenerate sincos 2D positional encoding for the new grid
        pos_embed_np = get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=True)
        pos_embed_tensor = torch.from_numpy(pos_embed_np).float().unsqueeze(0)
        # Shape: (1, 1+grid_h*grid_w, embed_dim)

        eat_model.fixed_positional_encoder = FixedPositionalEncoder(pos_embed_tensor)
        logger.info(
            f"Positional encoding: regenerated sincos for grid {grid_size} "
            f"({grid_h * grid_w} tokens + CLS)"
        )

        if freeze:
            for param in eat_model.local_encoder.parameters():
                param.requires_grad = False
            logger.info("Patch embed: frozen")
        else:
            logger.info("Patch embed: trainable")
    else:
        raise ValueError(f"Unknown patch_embed type: {patch_type}")

    return encoder


def apply_partial_unfreeze(encoder, config: dict):
    """Unfreeze early transformer blocks for full-rank training.

    Called AFTER apply_lora(). Sets requires_grad=True for all parameters in
    the specified blocks, so they train at full rank while remaining blocks
    use LoRA. The LoRA adapters on the unfrozen blocks are redundant but harmless.

    Access path after PEFT wrapping:
        encoder.model.base_model.model.model.blocks[i]

    Args:
        encoder: EATEncoder with PEFT-wrapped model
        config: full config dict with model.partial_unfreeze_blocks

    Returns:
        encoder (modified in place)
    """
    unfreeze_blocks = config.get("model", {}).get("partial_unfreeze_blocks", [])

    if not unfreeze_blocks:
        return encoder

    # After PEFT wrapping: encoder.model is PeftModel
    # base_model.model is EATModel, .model is EAT backbone
    blocks = encoder.model.base_model.model.model.blocks

    for block_idx in unfreeze_blocks:
        block = blocks[block_idx]
        unfrozen_count = 0
        for param in block.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        logger.info(
            f"Partial unfreeze: block {block_idx} — "
            f"set {unfrozen_count} params to requires_grad=True"
        )

    return encoder


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Frame tokenization EAT fine-tuning on BirdCLEF 2025"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_a_baseline.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help=f"PACE project allocation name (default: {DEFAULT_PROJECT_NAME})",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run quick validation with 5 batches",
    )
    parser.add_argument(
        "--gpu-check",
        type=int,
        default=0,
        help="Run N epochs for GPU utilization check, print report, then exit",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers from config",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.fast_dev_run and args.gpu_check > 0:
        parser.error("--fast-dev-run and --gpu-check are mutually exclusive")

    gpu_check_mode = args.gpu_check > 0

    # Derive project directory from project name
    project_dir = Path(f"/storage/project/{args.project_name}/mgustineli3")

    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    run_name = config["run_name"]
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Run name: {run_name}")

    # Override max_epochs for GPU check mode
    if gpu_check_mode:
        config["training"]["max_epochs"] = args.gpu_check
        logger.info(f"GPU check mode: running {args.gpu_check} epoch(s)")

    # Override num_workers if specified
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
        logger.info(f"Overriding num_workers to {args.num_workers}")

    # Resolve data paths
    data_dir = Path(config["data"]["data_dir"]).expanduser()
    audio_dir = data_dir / "train_audio"
    metadata_csv = data_dir / "train.csv"
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Audio dir: {audio_dir}")

    # Build species list
    species_list = build_species_list(metadata_csv)
    num_classes = len(species_list)
    logger.info(f"Classes: {num_classes}")

    # Initialize encoder (with fbank transform offloaded to dataloader for EAT)
    audio_transform = None
    if config["encoder"]["type"] == "eat":
        from functools import partial

        from birdclef.encoders.eat import waveform_to_fbank

        audio_transform = partial(
            waveform_to_fbank,
            target_length=config["encoder"].get("target_length", 512),
        )

    logger.info(f"Initializing {config['encoder']['type'].upper()} encoder...")
    encoder = build_encoder(config, precomputed_features=audio_transform is not None)

    # Modify patch embedding and positional encoding (before LoRA)
    logger.info("Modifying patch embedding...")
    patch_cfg = config["patch_embed"]
    logger.info(
        f"  type={patch_cfg['type']}, patch_size={patch_cfg['patch_size']}, "
        f"freeze={patch_cfg['freeze']}"
    )
    encoder = modify_eat_model(encoder, config)

    # Apply LoRA adapters
    logger.info("Applying LoRA adapters...")
    lora_cfg = config["lora"]
    logger.info(
        f"  rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}, " f"dropout={lora_cfg['dropout']}"
    )
    logger.info(f"  target_modules={lora_cfg['target_modules']}")
    encoder = apply_lora(encoder, config)

    # Apply partial unfreeze (after LoRA, for config C3 only)
    unfreeze_blocks = config.get("model", {}).get("partial_unfreeze_blocks", [])
    if unfreeze_blocks:
        logger.info(f"Applying partial unfreeze on blocks: {unfreeze_blocks}")
    encoder = apply_partial_unfreeze(encoder, config)

    # Initialize Lightning module
    logger.info("Initializing LoRAFinetuneModule...")
    pl_module = LoRAFinetuneModule(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=config["encoder"].get("embed_dim", 768),
        hidden_dim=config["training"].get("hidden_dim", 0),
        dropout=config["training"].get("dropout", 0.1),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        max_epochs=config["training"]["max_epochs"],
        warmup_epochs=config["training"]["warmup_epochs"],
        threshold=config["training"]["threshold"],
    )

    # Initialize data module
    logger.info("Initializing BirdCLEF2025DataModule...")
    data_module = BirdCLEF2025DataModule(
        audio_dir=str(audio_dir),
        metadata_csv=str(metadata_csv),
        species_list=species_list,
        sample_rate=config["data"]["sample_rate"],
        chunk_duration=config["data"]["chunk_duration"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        val_fraction=config["data"].get("val_fraction", 0.1),
        test_fraction=config["data"].get("test_fraction", 0.1),
        seed=config["data"].get("seed", 42),
        transform=audio_transform,
    )

    # Setup callbacks
    callbacks = []
    checkpoint_callback = None
    gpu_monitor = None

    if gpu_check_mode:
        # GPU check mode: only GPU monitor + LR monitor
        from birdclef.callbacks.gpu_monitor import GPUUtilizationMonitor

        gpu_monitor = GPUUtilizationMonitor(log_every_n_steps=10)
        callbacks.append(gpu_monitor)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        logger.info("GPU check mode: skipping checkpoints, early stopping, and device stats")
    else:
        # Normal training: full callback set

        # Model checkpoint — save to project shared storage
        checkpoint_dir = project_dir / "experiments" / EXPERIMENT_NAME / "checkpoints" / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            monitor=config["callbacks"]["model_checkpoint"]["monitor"],
            mode=config["callbacks"]["model_checkpoint"]["mode"],
            save_top_k=config["callbacks"]["model_checkpoint"]["save_top_k"],
            save_last=config["callbacks"]["model_checkpoint"]["save_last"],
            filename=config["callbacks"]["model_checkpoint"]["filename"],
            dirpath=checkpoint_dir,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=config["callbacks"]["early_stopping"]["monitor"],
            mode=config["callbacks"]["early_stopping"]["mode"],
            patience=config["callbacks"]["early_stopping"]["patience"],
            min_delta=config["callbacks"]["early_stopping"]["min_delta"],
            verbose=True,
        )
        callbacks.append(early_stopping)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # GPU stats monitor
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)

    # Setup loggers
    loggers = []
    experiment_name = f"exp008-{run_name}"

    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=Path(__file__).parent / "outputs",
        name="tensorboard",
        version=experiment_name,
    )
    loggers.append(tb_logger)

    if args.fast_dev_run or gpu_check_mode:
        logger.info(f"{'GPU check' if gpu_check_mode else 'Fast dev run'} — skipping W&B logging")
    else:
        # W&B logger (only for full training runs)
        logger.info("Initializing W&B logger...")
        wandb_logger = WandbLogger(
            entity=config["logging"]["entity"],
            project=config["logging"]["project"],
            name=experiment_name,
            save_dir=Path(__file__).parent / config["logging"]["save_dir"],
            log_model=config["logging"]["log_model"],
        )
        # Log full config + LoRA details + patch embed details
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(
            {
                "lora/rank": lora_cfg["rank"],
                "lora/alpha": lora_cfg["alpha"],
                "lora/dropout": lora_cfg["dropout"],
                "lora/target_modules": lora_cfg["target_modules"],
                "patch_embed/type": patch_cfg["type"],
                "patch_embed/patch_size": patch_cfg["patch_size"],
                "patch_embed/freeze": patch_cfg["freeze"],
            }
        )
        loggers.append(wandb_logger)

    # Initialize trainer
    logger.info(f"Using {args.devices} GPU(s)")
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=config["training"]["precision"],
        gradient_clip_val=config["training"]["grad_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        fast_dev_run=5 if args.fast_dev_run else False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(pl_module, data_module)

    # GPU check mode: print summary, save to file, and exit early
    if gpu_check_mode:
        import os

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        summary_file = summary_dir / f"{run_name}_{job_id}.txt"
        gpu_monitor.print_summary(save_path=summary_file)
        logger.info("GPU check complete!")
        return

    # Evaluate on held-out test set using best checkpoint
    if not args.fast_dev_run and checkpoint_callback.best_model_path:
        logger.info(
            f"Evaluating best checkpoint on test set: {checkpoint_callback.best_model_path}"
        )
        test_results = trainer.test(
            pl_module, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
        )
    elif args.fast_dev_run:
        logger.info("Fast dev run — running test step for pipeline validation")
        test_results = trainer.test(pl_module, datamodule=data_module)
    else:
        test_results = None

    # Save results
    if not args.fast_dev_run:
        results_dir = Path(__file__).parent / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{run_name}.json"

        raw_results = {
            "config_path": str(config_path),
            "run_name": run_name,
            "best_val_cmap": float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None,
            "best_checkpoint_path": str(checkpoint_callback.best_model_path),
            "stopped_epoch": trainer.current_epoch,
            "num_classes": num_classes,
            "encoder_type": config["encoder"]["type"],
            "patch_embed": {
                "type": patch_cfg["type"],
                "patch_size": patch_cfg["patch_size"],
                "freeze": patch_cfg["freeze"],
            },
            "lora": {
                "rank": lora_cfg["rank"],
                "alpha": lora_cfg["alpha"],
                "dropout": lora_cfg["dropout"],
                "target_modules": lora_cfg["target_modules"],
            },
            "partial_unfreeze_blocks": unfreeze_blocks,
            "test_metrics": test_results[0] if test_results else None,
        }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
