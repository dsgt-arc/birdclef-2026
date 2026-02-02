#!/usr/bin/env python3
"""Training script for Experiment 002: Supervised ViT Baseline.

Usage:
    python train.py [--config config.yaml] [--fast-dev-run]
"""

import argparse
import logging
import sys
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
from pytorch_lightning.loggers import WandbLogger  # noqa: E402

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from birdset_benchmark.datasets import get_num_classes  # noqa: E402
from birdset_benchmark.lightning.birdset_datamodule import BirdSetDataModule  # noqa: E402
from birdset_benchmark.lightning.supervised_vit import SupervisedViTModule  # noqa: E402
from birdset_benchmark.models.vit_audio import ViTAudio  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train Supervised ViT Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
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

    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Override num_workers if specified
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
        logger.info(f"Overriding num_workers to {args.num_workers}")

    # Get number of classes for dataset
    dataset_name = config["data"]["dataset_name"]
    num_classes = get_num_classes(dataset_name)
    logger.info(f"Dataset: {dataset_name}, Classes: {num_classes}")

    # Initialize model
    logger.info("Initializing ViT-Audio model...")
    model = ViTAudio(
        num_classes=num_classes,
        img_size=tuple(config["model"]["img_size"]),
        patch_size=config["model"]["patch_size"],
        pretrained=config["model"]["pretrained"],
        drop_rate=config["model"]["drop_rate"],
    )

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
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=config["callbacks"]["model_checkpoint"]["monitor"],
        mode=config["callbacks"]["model_checkpoint"]["mode"],
        save_top_k=config["callbacks"]["model_checkpoint"]["save_top_k"],
        save_last=config["callbacks"]["model_checkpoint"]["save_last"],
        filename=config["callbacks"]["model_checkpoint"]["filename"],
        dirpath=Path(__file__).parent / "outputs" / "checkpoints",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

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

    # Setup logger
    if not args.fast_dev_run:
        logger.info("Initializing W&B logger...")
        experiment_name = f"exp002-{dataset_name}"
        wandb_logger = WandbLogger(
            entity=config["logging"]["entity"],
            project=config["logging"]["project"],
            name=experiment_name,
            save_dir=Path(__file__).parent / config["logging"]["save_dir"],
            log_model=config["logging"]["log_model"],
        )
        # Log config to W&B
        wandb_logger.experiment.config.update(config)
    else:
        logger.info("Fast dev run - skipping W&B logging")
        wandb_logger = None

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
        logger=wandb_logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        fast_dev_run=5 if args.fast_dev_run else False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(pl_module, data_module)

    # Test on best checkpoint
    if not args.fast_dev_run:
        logger.info("Testing on best checkpoint...")
        trainer.test(pl_module, data_module, ckpt_path="best")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
