#!/usr/bin/env python3
"""Training script for Experiment 004: Frame-Based Tokenization.

Trains FrameTokenizedViT models with configurable patch shapes on BirdSet
datasets. Tests whether vertical frame patches (128xW) improve cmAP over
standard 16x16 patches when training from scratch.

Usage:
    python train.py --config config/vertical_128x2.yaml [--fast-dev-run]
    python train.py --config config/baseline_16x16.yaml [--fast-dev-run]
    python train.py --config config/vertical_128x2.yaml --project-name my-project
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file (must be before W&B imports)
load_dotenv()

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger  # noqa: E402

from birdset_benchmark.datasets import get_num_classes  # noqa: E402
from birdset_benchmark.lightning.birdset_datamodule import BirdSetDataModule  # noqa: E402
from birdset_benchmark.lightning.supervised_vit import SupervisedViTModule  # noqa: E402
from birdset_benchmark.models.frame_vit import FrameTokenizedViT  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# =============================================================================
# Constants
# =============================================================================
DEFAULT_PROJECT_NAME = "ps-dsgt_clef2026-0"  # PACE project allocation name
EXPERIMENT_NAME = "004-frame-tokenization"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train FrameTokenizedViT on BirdSet")
    parser.add_argument(
        "--config",
        type=str,
        default="config/vertical_128x2.yaml",
        help="Path to config file (default: config/vertical_128x2.yaml)",
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
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers from config (useful for debugging)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Training strategy: 'auto', 'ddp', 'ddp_spawn', etc. (default: auto)",
    )
    args = parser.parse_args()

    # Derive project directory from project name
    project_dir = Path(f"/storage/project/{args.project_name}/shared/birdclef-2026")

    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    run_name = config["run_name"]
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Run name: {run_name}")

    # Override num_workers if specified
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
        logger.info(f"Overriding num_workers to {args.num_workers}")

    # Get number of classes for dataset
    dataset_name = config["data"]["dataset_name"]
    num_classes = get_num_classes(dataset_name)
    logger.info(f"Dataset: {dataset_name}, Classes: {num_classes}")

    # Initialize model — FrameTokenizedViT with configurable patch_size
    model_name = config["model"]["model_name"]
    pretrained = config["model"]["pretrained"]
    patch_size = tuple(config["model"]["patch_size"])
    img_size = tuple(config["model"]["img_size"])
    logger.info(
        f"Initializing model: {model_name} (pretrained={pretrained}, patch_size={patch_size})..."
    )
    model = FrameTokenizedViT(
        model_name=model_name,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        pretrained=pretrained,
        drop_rate=config["model"]["drop_rate"],
        sample_rate=config["data"]["sample_rate"],
        n_mels=config["data"]["n_mels"],
        frame_shift_ms=config["data"]["frame_shift_ms"],
        target_frames=config["data"]["target_frames"],
        norm_mean=config["data"]["norm_mean"],
        norm_std=config["data"]["norm_std"],
    )

    # Log patch info
    freq_patches = img_size[0] // patch_size[0]
    time_patches = img_size[1] // patch_size[1]
    num_tokens = freq_patches * time_patches
    logger.info(f"Patch grid: {freq_patches}x{time_patches} = {num_tokens} tokens")

    # Initialize Lightning module
    logger.info("Initializing PyTorch Lightning module...")
    pl_module = SupervisedViTModule(
        model=model,
        num_classes=num_classes,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        max_epochs=config["training"]["max_epochs"],
        warmup_epochs=config["training"]["warmup_epochs"],
        threshold=config["training"]["threshold"],
    )

    # Initialize data module
    logger.info("Initializing data module...")
    data_module = BirdSetDataModule(
        dataset_name=config["data"]["dataset_name"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        target_sr=config["data"]["sample_rate"],
        target_duration=config["data"]["target_duration"],
        n_mels=config["data"]["n_mels"],
        frame_shift_ms=config["data"]["frame_shift_ms"],
        target_frames=config["data"]["target_frames"],
        norm_mean=config["data"]["norm_mean"],
        norm_std=config["data"]["norm_std"],
        pin_memory=config["data"]["pin_memory"],
        persistent_workers=config["data"]["persistent_workers"],
        prefetch_factor=config["data"]["prefetch_factor"],
        waveform_only=True,  # nnAudio computes mel on GPU inside model
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint - save to project shared storage (more quota than home)
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

    # GPU stats monitor (logs GPU utilization to W&B)
    device_stats = DeviceStatsMonitor()
    callbacks.append(device_stats)

    # Setup loggers
    loggers = []
    experiment_name = f"exp004-{run_name}-{dataset_name}"

    # TensorBoard logger (always enabled)
    logger.info("Initializing TensorBoard logger...")
    tb_logger = TensorBoardLogger(
        save_dir=Path(__file__).parent / "outputs",
        name="tensorboard",
        version=experiment_name,
    )
    loggers.append(tb_logger)

    if not args.fast_dev_run:
        # W&B logger (only for full training runs)
        logger.info("Initializing W&B logger...")
        wandb_logger = WandbLogger(
            entity=config["logging"]["entity"],
            project=config["logging"]["project"],
            name=experiment_name,
            save_dir=Path(__file__).parent / config["logging"]["save_dir"],
            log_model=config["logging"]["log_model"],
        )
        # Log config to W&B (include patch_size for easy filtering)
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(
            {
                "patch_size_str": f"{patch_size[0]}x{patch_size[1]}",
                "num_tokens": num_tokens,
            }
        )
        loggers.append(wandb_logger)
    else:
        logger.info("Fast dev run - skipping W&B logging")

    # Initialize trainer
    logger.info("Initializing PyTorch Lightning trainer...")
    logger.info(f"Using {args.devices} GPU(s) with strategy: {args.strategy}")
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=args.strategy if args.devices > 1 else "auto",
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

    # Test on best checkpoint (or current model for fast_dev_run)
    if not args.fast_dev_run:
        logger.info("Testing on best checkpoint...")
        test_results = trainer.test(pl_module, data_module, ckpt_path="best")
    else:
        # In fast_dev_run, test with current model to validate test_step works
        logger.info("Testing current model (fast dev run)...")
        test_results = trainer.test(pl_module, data_module)

    # Save raw test results for save_metrics.py to consume
    results_dir = Path(__file__).parent / "outputs" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{run_name}.json"

    raw_results = {
        "test_metrics": test_results[0] if test_results else {},
        "config_path": str(config_path),
        "run_name": run_name,
        "best_epoch": checkpoint_callback.best_model_score is not None
        and int(Path(checkpoint_callback.best_model_path).stem.split("epoch=")[-1])
        or None,
        "best_val_cmap": float(checkpoint_callback.best_model_score)
        if checkpoint_callback.best_model_score is not None
        else None,
        "best_checkpoint_path": str(checkpoint_callback.best_model_path),
        "stopped_epoch": trainer.current_epoch,
        "num_params": sum(p.numel() for p in model.parameters()),
        "patch_size": list(patch_size),
        "num_tokens": num_tokens,
    }
    with open(results_file, "w") as f:
        json.dump(raw_results, f, indent=2)
    logger.info(f"Raw test results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
