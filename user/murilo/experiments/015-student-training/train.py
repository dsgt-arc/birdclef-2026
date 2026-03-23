#!/usr/bin/env python3
"""Training script for Experiment 015: Student Training on Pseudo-Labeled Soundscapes.

Loads exp 012 checkpoints (already 234-class heads) and fine-tunes on mixed
focal + labeled soundscape + pseudo-labeled soundscape data.

Supports all three encoder architectures via config dispatch:
  - eat:      EAT-base (AudioSet pretrained), LoRA on attention layers
  - convnext: ConvNeXt-Base (ImageNet pretrained), LoRA on MLP layers
  - protoclr: ProtoCLR/CvT-13 (Xeno-Canto pretrained), LoRA on attention layers

Usage:
    python train.py --config config/protoclr_student.yaml
    python train.py --config config/convnext_student.yaml
    python train.py --config config/eat_student.yaml
    python train.py --config config/protoclr_student.yaml --fast-dev-run
    python train.py --config config/protoclr_student.yaml --gpu-check 2
    python train.py --config config/protoclr_student.yaml --pseudo-weight 0.25
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

from birdclef.data.soundscape import build_full_species_list  # noqa: E402
from birdclef.data.student import StudentDataModule  # noqa: E402
from birdclef.lightning.student_finetune import StudentFinetuneModule  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "015-student-training"


# ============================================================================
# Encoder + transform + LoRA setup (same as exp 012)
# ============================================================================


def build_encoder(config: dict):
    """Build encoder from config, dispatching on encoder type."""
    encoder_type = config["encoder"]["type"]

    if encoder_type == "eat":
        from birdclef.encoders.eat import EATEncoder

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
    """Apply PEFT LoRA adapters to the encoder, dispatching on encoder type."""
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
        description="Student training on focal + labeled + pseudo-labeled soundscapes"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., config/protoclr_student.yaml)",
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
        "--pseudo-weight",
        type=float,
        default=None,
        help="Override pseudo-label weight from config",
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

    # Override from CLI
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
        logger.info(f"Overriding num_workers to {args.num_workers}")

    pseudo_weight = config["pseudo_labels"]["weight"]
    if args.pseudo_weight is not None:
        pseudo_weight = args.pseudo_weight
        logger.info(f"Overriding pseudo-label weight to {pseudo_weight}")

    # ========================================================================
    # Resolve data paths and species lists
    # ========================================================================

    data_dir = Path(config["data"]["data_dir"]).expanduser()
    taxonomy_csv = data_dir / "taxonomy.csv"
    pseudo_label_dir = Path(config["pseudo_labels"]["parquet_dir"])

    # Build species list
    species_234 = build_full_species_list(taxonomy_csv)
    num_classes = len(species_234)
    logger.info(f"Full species list: {num_classes} classes")
    logger.info(f"Pseudo-label dir: {pseudo_label_dir}")
    logger.info(f"Pseudo-label weight: {pseudo_weight}")

    # ========================================================================
    # Build encoder + LoRA
    # ========================================================================

    audio_transform = build_audio_transform(config)
    logger.info(f"Audio transform: {type(audio_transform).__name__}")

    logger.info(f"Initializing {encoder_type.upper()} encoder...")
    encoder = build_encoder(config)

    lora_cfg = config["lora"]
    logger.info(
        f"LoRA: rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}, "
        f"dropout={lora_cfg['dropout']}, targets={lora_cfg['target_modules']}"
    )
    encoder = apply_lora_to_encoder(encoder, config)

    # ========================================================================
    # Build Lightning module (234 classes)
    # ========================================================================

    hidden_dim = config["training"].get("hidden_dim", 0)
    embed_dim = config["encoder"].get("embed_dim", 768)

    logger.info("Initializing StudentFinetuneModule with 234 classes...")
    pl_module = StudentFinetuneModule(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=config["training"].get("dropout", 0.1),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        max_epochs=config["training"]["max_epochs"],
        warmup_epochs=config["training"]["warmup_epochs"],
        threshold=config["training"]["threshold"],
    )

    # ========================================================================
    # Load exp 012 checkpoint (already 234 classes — no head remapping)
    # ========================================================================

    checkpoint_path = config["checkpoint"]["resume_from"]
    logger.info(f"Loading exp 012 checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Load directly — exp 012 checkpoints already have 234-class heads
    missing, unexpected = pl_module.load_state_dict(state_dict, strict=False)

    # Filter out expected mismatches (metrics state)
    real_missing = [
        k
        for k in missing
        if not any(m in k for m in ["cmap", "auroc", "f1_macro", "f1_micro", "top1", "top5"])
    ]
    if real_missing:
        logger.warning(f"Missing keys (non-metric): {real_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:10]}")

    logger.info("Exp 012 checkpoint loaded successfully")

    # ========================================================================
    # Initialize data module
    # ========================================================================

    logger.info("Initializing StudentDataModule...")
    data_module = StudentDataModule(
        data_dir=str(data_dir),
        taxonomy_csv=str(taxonomy_csv),
        pseudo_label_dir=str(pseudo_label_dir),
        pseudo_label_weight=pseudo_weight,
        sample_rate=config["data"]["sample_rate"],
        chunk_duration=config["data"]["chunk_duration"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        val_fraction=config["data"].get("val_fraction", 0.1),
        test_fraction=config["data"].get("test_fraction", 0.1),
        soundscape_val_files=config["data"].get("soundscape_val_files", 8),
        soundscape_test_files=config["data"].get("soundscape_test_files", 8),
        seed=config["data"].get("seed", 42),
        transform=audio_transform,
    )

    # ========================================================================
    # Callbacks
    # ========================================================================

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
        # Full training mode: all callbacks
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

    # ========================================================================
    # Loggers
    # ========================================================================

    loggers = []
    experiment_name = f"exp015-{run_name}"

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=Path(__file__).parent / "outputs",
        name="tensorboard",
        version=experiment_name,
    )
    loggers.append(tb_logger)

    if args.fast_dev_run or gpu_check_mode:
        logger.info(f"{'GPU check' if gpu_check_mode else 'Fast dev run'} — skipping W&B logging")
    else:
        logger.info("Initializing W&B logger...")
        wandb_logger = WandbLogger(
            entity=config["logging"]["entity"],
            project=config["logging"]["project"],
            name=experiment_name,
            save_dir=Path(__file__).parent / config["logging"]["save_dir"],
            log_model=config["logging"]["log_model"],
        )
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(
            {
                "finetune_mode": "student_training",
                "num_classes": num_classes,
                "pseudo_label_dir": str(pseudo_label_dir),
                "pseudo_label_weight": pseudo_weight,
                "checkpoint_from": checkpoint_path,
            }
        )
        loggers.append(wandb_logger)

    # ========================================================================
    # Trainer
    # ========================================================================

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

    # ========================================================================
    # Train
    # ========================================================================

    logger.info("Starting training...")
    trainer.fit(pl_module, data_module)

    # ========================================================================
    # GPU check mode: print summary, save report + plot, and exit
    # ========================================================================

    if gpu_check_mode:
        import os

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        summary_file = summary_dir / f"{run_name}_{job_id}.txt"
        gpu_monitor.print_summary(save_path=summary_file)
        logger.info("GPU check complete!")
        return

    # ========================================================================
    # Evaluate on test set
    # ========================================================================

    test_results = None
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

    # ========================================================================
    # Save results
    # ========================================================================

    if not args.fast_dev_run:
        results_dir = Path(__file__).parent / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{run_name}.json"

        raw_results = {
            "config_path": str(config_path),
            "run_name": run_name,
            "finetune_mode": "student_training",
            "num_classes": num_classes,
            "checkpoint_from": checkpoint_path,
            "pseudo_label_dir": str(pseudo_label_dir),
            "pseudo_label_weight": pseudo_weight,
            "best_val_score": (
                float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None
                else None
            ),
            "best_checkpoint_path": str(checkpoint_callback.best_model_path),
            "stopped_epoch": trainer.current_epoch,
            "encoder_type": encoder_type,
            "test_metrics": test_results[0] if test_results else None,
            "lora": {
                "rank": lora_cfg["rank"],
                "alpha": lora_cfg["alpha"],
                "dropout": lora_cfg["dropout"],
                "target_modules": lora_cfg["target_modules"],
            },
        }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
