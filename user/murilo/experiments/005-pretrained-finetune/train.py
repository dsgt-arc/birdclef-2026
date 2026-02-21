#!/usr/bin/env python3
"""Training script for Experiment 005: Pretrained Audio Model Fine-tuning.

Fine-tunes pretrained audio encoders (EAT, BEATs) on BirdCLEF 2025 data
with configurable frozen/unfrozen training modes.

Usage:
    python train.py --config config/eat_frozen.yaml [--fast-dev-run]
    python train.py --config config/eat_finetune.yaml [--fast-dev-run]
"""

import argparse
import json
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
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger  # noqa: E402

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from birdclef.data import BirdCLEF2025DataModule, build_species_list  # noqa: E402
from birdclef.lightning.pretrained_finetune import PretrainedFinetuneModule  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-dsgt_clef2026-0"
EXPERIMENT_NAME = "005-pretrained-finetune"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def build_encoder(config: dict):
    """Build encoder based on config."""
    encoder_type = config["encoder"]["type"]

    if encoder_type == "eat":
        from birdclef.encoders.eat import EATEncoder

        return EATEncoder(
            model_name=config["encoder"].get("model_name", "worstchan/EAT-base_epoch30_pretrain"),
            target_length=config["encoder"].get("target_length", 512),
        )
    elif encoder_type == "beats":
        from birdclef.encoders.beats import BEATsEncoder

        return BEATsEncoder(
            checkpoint_path=config["encoder"]["checkpoint_path"],
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained audio encoders on BirdCLEF 2025"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/eat_frozen.yaml",
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

    # Initialize encoder
    logger.info(f"Initializing {config['encoder']['type'].upper()} encoder...")
    encoder = build_encoder(config)

    # Initialize Lightning module
    logger.info("Initializing PretrainedFinetuneModule...")
    pl_module = PretrainedFinetuneModule(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=config["encoder"].get("embed_dim", 768),
        hidden_dim=config["training"].get("hidden_dim", 0),
        dropout=config["training"].get("dropout", 0.1),
        lr=config["training"]["lr"],
        encoder_lr=config["training"].get("encoder_lr", 1e-5),
        weight_decay=config["training"]["weight_decay"],
        max_epochs=config["training"]["max_epochs"],
        warmup_epochs=config["training"]["warmup_epochs"],
        freeze_encoder=config["training"]["freeze_encoder"],
        unfreeze_epoch=config["training"].get("unfreeze_epoch", 0),
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
        seed=config["data"].get("seed", 42),
    )

    # Setup callbacks
    callbacks = []

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
    experiment_name = f"exp005-{run_name}"

    # TensorBoard logger (always enabled)
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
        wandb_logger.experiment.config.update(config)
        loggers.append(wandb_logger)
    else:
        logger.info("Fast dev run — skipping W&B logging")

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
            "freeze_encoder": config["training"]["freeze_encoder"],
        }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
