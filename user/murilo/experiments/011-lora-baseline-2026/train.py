#!/usr/bin/env python3
"""Unified training script for Experiment 011: LoRA Baseline on BirdCLEF 2026.

Supports all three encoder architectures via a single script, dispatching
based on `config["encoder"]["type"]`:
  - eat:      EAT-base (AudioSet pretrained), LoRA on attention layers
  - convnext: ConvNeXt-Base (ImageNet pretrained), LoRA on MLP layers
  - protoclr: ProtoCLR/CvT-13 (custom audio pretrained), LoRA on attention layers

Usage:
    python train.py --config config/eat_lora_attn_mlphead.yaml [--fast-dev-run]
    python train.py --config config/convnext_lora_mlp_mlphead.yaml [--fast-dev-run]
    python train.py --config config/protoclr_lora_attn.yaml [--fast-dev-run]
    python train.py --config config/eat_lora_attn_mlphead.yaml --gpu-check 3
"""

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from user-level .env (must be before W&B imports)
_user_dir = Path(__file__).parent.parent.parent
load_dotenv(_user_dir / ".env")

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
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

from birdclef.data import BirdCLEFDataModule, build_species_list  # noqa: E402
from birdclef.lightning.lora_finetune import LoRAFinetuneModule  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "011-lora-baseline-2026"


# ============================================================================
# Encoder + transform + LoRA setup (dispatched by encoder type)
# ============================================================================


def build_encoder(config: dict):
    """Build encoder from config, dispatching on encoder type."""
    encoder_type = config["encoder"]["type"]

    if encoder_type == "eat":
        from birdclef.encoders.eat import EATEncoder

        # precomputed_features=True because fbank transform runs in dataloader
        return EATEncoder(
            model_name=config["encoder"].get("model_name", "worstchan/EAT-base_epoch30_pretrain"),
            target_length=config["encoder"].get("target_length", 512),
            precomputed_features=True,
        )
    elif encoder_type == "convnext":
        from birdclef.encoders.convnext import ConvNeXtEncoder

        return ConvNeXtEncoder(
            model_name=config["encoder"]["model_name"],
            pretrained=config["encoder"].get("pretrained", True),
        )
    elif encoder_type == "protoclr":
        from birdclef.encoders.protoclr import ProtoCLREncoder

        return ProtoCLREncoder(
            model_name=config["encoder"]["model_name"],
            checkpoint=config["encoder"].get("checkpoint", "protoclr.pth"),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def build_audio_transform(config: dict):
    """Build the audio transform for the dataloader, dispatching on encoder type."""
    encoder_type = config["encoder"]["type"]

    if encoder_type == "eat":
        from birdclef.encoders.eat import waveform_to_fbank

        return partial(
            waveform_to_fbank,
            target_length=config["encoder"].get("target_length", 512),
        )
    elif encoder_type == "convnext":
        from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

        return NNAudioMelSpectrogram(
            sample_rate=config["data"]["sample_rate"],
        )
    elif encoder_type == "protoclr":
        from birdclef.preprocessing.protoclr_mel import ProtoCLRMelSpectrogram

        return ProtoCLRMelSpectrogram(
            sample_rate=config["data"]["sample_rate"],
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def apply_lora_to_encoder(encoder, config: dict):
    """Apply PEFT LoRA adapters to the encoder, dispatching on encoder type.

    Each encoder stores its pretrained weights on a different attribute:
      - EAT:      encoder.model     (HuggingFace model)
      - ConvNeXt: encoder.backbone  (timm model)
      - ProtoCLR: encoder.model     (CvT model)
    """
    lora_cfg = config["lora"]
    encoder_type = config["encoder"]["type"]

    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    if encoder_type == "eat":
        encoder.model = get_peft_model(encoder.model, lora_config)
        encoder.model.print_trainable_parameters()
    elif encoder_type == "convnext":
        encoder.backbone = get_peft_model(encoder.backbone, lora_config)
        encoder.backbone.print_trainable_parameters()
    elif encoder_type == "protoclr":
        encoder.model = get_peft_model(encoder.model, lora_config)
        encoder.model.print_trainable_parameters()
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder


# ============================================================================
# Main
# ============================================================================


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune encoders on BirdCLEF 2026 (EAT, ConvNeXt, ProtoCLR)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., config/eat_lora_attn_mlphead.yaml)",
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
    encoder_type = config["encoder"]["type"]
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Encoder type: {encoder_type}")

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

    # Build audio transform (applied in dataloader workers on CPU)
    audio_transform = build_audio_transform(config)
    logger.info(f"Audio transform: {type(audio_transform).__name__}")

    # Initialize encoder
    logger.info(f"Initializing {encoder_type.upper()} encoder...")
    encoder = build_encoder(config)
    logger.info(f"Encoder embed_dim: {config['encoder'].get('embed_dim', 'auto')}")

    # Determine fine-tuning mode from config
    use_lora = "lora" in config
    lora_cfg = config.get("lora")

    if use_lora:
        # LoRA fine-tuning: apply PEFT adapters
        logger.info("Fine-tuning mode: LoRA")
        logger.info(
            f"  rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}, "
            f"dropout={lora_cfg['dropout']}"
        )
        logger.info(f"  target_modules={lora_cfg['target_modules']}")
        encoder = apply_lora_to_encoder(encoder, config)
    else:
        # Full fine-tuning: all parameters trainable
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        logger.info("Fine-tuning mode: full (all parameters trainable)")
        logger.info(f"  trainable params: {trainable_params:,} / {total_params:,}")

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
    logger.info("Initializing BirdCLEFDataModule...")
    data_module = BirdCLEFDataModule(
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
    experiment_name = f"exp011-{run_name}"

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
        # Log full config + fine-tuning details
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update({"finetune_mode": "lora" if use_lora else "full"})
        if use_lora:
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
            "finetune_mode": "lora" if use_lora else "full",
            "best_val_cmap": float(checkpoint_callback.best_model_score)
            if checkpoint_callback.best_model_score is not None
            else None,
            "best_checkpoint_path": str(checkpoint_callback.best_model_path),
            "stopped_epoch": trainer.current_epoch,
            "num_classes": num_classes,
            "encoder_type": encoder_type,
            "test_metrics": test_results[0] if test_results else None,
        }
        if use_lora:
            raw_results["lora"] = {
                "rank": lora_cfg["rank"],
                "alpha": lora_cfg["alpha"],
                "dropout": lora_cfg["dropout"],
                "target_modules": lora_cfg["target_modules"],
            }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
