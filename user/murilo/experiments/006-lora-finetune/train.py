#!/usr/bin/env python3
"""Training script for Experiment 006: LoRA Fine-Tuning EAT on BirdCLEF 2025.

Applies PEFT LoRA adapters to the EAT encoder's attention (and optionally MLP)
layers, training ~0.7% of parameters instead of all 90M.

Usage:
    python train.py --config config/eat_lora_attn.yaml [--fast-dev-run]
    python train.py --config config/eat_lora_all.yaml [--fast-dev-run]
    python train.py --config config/eat_lora_attn.yaml --gpu-check 3
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
from birdclef.encoders.eat import EATEncoder  # noqa: E402
from birdclef.lightning.lora_finetune import LoRAFinetuneModule  # noqa: E402
from birdclef.lora.peft import apply_lora  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-dsgt_clef2026-0"
EXPERIMENT_NAME = "006-lora-finetune"


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
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune EAT encoder on BirdCLEF 2025")
    parser.add_argument(
        "--config",
        type=str,
        default="config/eat_lora_attn.yaml",
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
    project_dir = Path(f"/storage/project/{args.project_name}/shared/birdclef-2026")

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

    # Apply LoRA adapters
    logger.info("Applying LoRA adapters...")
    lora_cfg = config["lora"]
    logger.info(
        f"  rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}, " f"dropout={lora_cfg['dropout']}"
    )
    logger.info(f"  target_modules={lora_cfg['target_modules']}")
    encoder = apply_lora(encoder, config)

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
    experiment_name = f"exp006-{run_name}"

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
        # Log full config + LoRA details
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(
            {
                "lora/rank": lora_cfg["rank"],
                "lora/alpha": lora_cfg["alpha"],
                "lora/dropout": lora_cfg["dropout"],
                "lora/target_modules": lora_cfg["target_modules"],
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
            "lora": {
                "rank": lora_cfg["rank"],
                "alpha": lora_cfg["alpha"],
                "dropout": lora_cfg["dropout"],
                "target_modules": lora_cfg["target_modules"],
            },
            "test_metrics": test_results[0] if test_results else None,
        }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
