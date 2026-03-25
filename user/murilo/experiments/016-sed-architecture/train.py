#!/usr/bin/env python3
"""Training script for Experiment 016: SED Architecture with EfficientNet-B0.

Trains an EfficientNet-B0 backbone with SED attention head from scratch
(ImageNet pretrained, no prior BirdCLEF checkpoint). Uses 10-second input
windows, dual soft CE loss, and spectrogram mixup.

Usage:
    python train.py --config config/efficientnet_b0_sed.yaml
    python train.py --config config/efficientnet_b0_sed.yaml --fast-dev-run
    python train.py --config config/efficientnet_b0_sed.yaml --gpu-check
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

from birdclef.data.sed import SEDDataModule  # noqa: E402
from birdclef.data.soundscape import build_full_species_list  # noqa: E402
from birdclef.encoders.efficientnet import EfficientNetEncoder  # noqa: E402
from birdclef.heads.sed_head import AttHead  # noqa: E402
from birdclef.lightning.sed_finetune import SEDFinetuneModule  # noqa: E402
from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "016-sed-architecture"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SED architecture training with EfficientNet-B0")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--project-name", type=str, default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--gpu-check", type=int, nargs="?", const=1, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    if args.fast_dev_run and args.gpu_check > 0:
        parser.error("--fast-dev-run and --gpu-check are mutually exclusive")

    gpu_check_mode = args.gpu_check > 0
    project_dir = Path(f"/storage/project/{args.project_name}/mgustineli3")

    # Load config
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    run_name = config["run_name"]

    logger.info(f"Config: {config_path}")
    logger.info(f"Run name: {run_name}")

    if gpu_check_mode:
        config["training"]["max_epochs"] = args.gpu_check
        logger.info(f"GPU check mode: {args.gpu_check} epoch(s)")

    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    # ========================================================================
    # Species list
    # ========================================================================

    data_dir = Path(config["data"]["data_dir"]).expanduser()
    taxonomy_csv = data_dir / "taxonomy.csv"
    species_list = build_full_species_list(taxonomy_csv)
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes} classes")

    # ========================================================================
    # Build encoder + SED head
    # ========================================================================

    logger.info("Building EfficientNet encoder...")
    encoder = EfficientNetEncoder(
        model_name=config["encoder"]["model_name"],
        pretrained=config["encoder"].get("pretrained", True),
    )

    # Use encoder's actual embed_dim (not config) — features_only=True may differ
    embed_dim = encoder.embed_dim
    logger.info(f"Encoder embed_dim: {embed_dim}")

    logger.info("Building SED attention head...")
    sed_head = AttHead(
        in_channels=embed_dim,
        num_classes=num_classes,
        hidden_dim=config["sed_head"]["hidden_dim"],
        dropout=config["sed_head"]["dropout"],
        gem_p=config["sed_head"]["gem_p"],
    )

    # ========================================================================
    # Build mel spectrogram transform (10s → 1000 frames at 32kHz)
    # ========================================================================

    sample_rate = config["data"]["sample_rate"]
    window_duration = config["data"]["window_duration"]
    target_frames = int(window_duration * sample_rate / (sample_rate * 0.01))  # 10ms hop

    audio_transform = NNAudioMelSpectrogram(
        sample_rate=sample_rate,
        target_frames=target_frames,
    )
    logger.info(f"Mel transform: sr={sample_rate}, target_frames={target_frames}")

    # ========================================================================
    # Build Lightning module
    # ========================================================================

    training_cfg = config["training"]
    raw_waveform_mixup = training_cfg.get("raw_waveform_mixup", False)

    # When raw_waveform_mixup is enabled, the mel transform moves to the model
    # (GPU-side) so mixup can apply on raw waveforms before mel conversion.
    # Use a separate instance for the model to avoid sharing nn.Module state
    # between GPU (model) and CPU (DataLoader eval workers).
    gpu_mel_transform = None
    if raw_waveform_mixup:
        gpu_mel_transform = NNAudioMelSpectrogram(
            sample_rate=sample_rate,
            target_frames=target_frames,
        )
        logger.info("Raw waveform mixup: mel transform on GPU, mixup before mel")

    pl_module = SEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        lr=training_cfg["lr"],
        head_lr=training_cfg["head_lr"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=training_cfg["max_epochs"],
        warmup_epochs=training_cfg["warmup_epochs"],
        threshold=training_cfg["threshold"],
        clipwise_weight=training_cfg["clipwise_weight"],
        framewise_weight=training_cfg["framewise_weight"],
        blend_clip=training_cfg["blend_clip"],
        blend_frame=training_cfg["blend_frame"],
        mixup_prob=training_cfg.get("mixup_prob", 0.5),
        mixup_alpha=training_cfg.get("mixup_alpha", 0.5),
        loss_type=training_cfg.get("loss_type", "soft_ce"),
        label_smoothing=training_cfg.get("label_smoothing", 0.0),
        mel_transform=gpu_mel_transform,
    )

    # ========================================================================
    # Build data module
    # ========================================================================

    # When raw_waveform_mixup is enabled:
    #   - train datasets get transform=None (return raw waveforms)
    #   - val/test datasets get eval_transform=audio_transform (return mel)
    # When disabled (default): all datasets use audio_transform (backward compatible)
    train_transform = None if raw_waveform_mixup else audio_transform

    data_module = SEDDataModule(
        data_dir=str(data_dir),
        taxonomy_csv=str(taxonomy_csv),
        sample_rate=sample_rate,
        window_duration=window_duration,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        val_fraction=config["data"].get("val_fraction", 0.1),
        test_fraction=config["data"].get("test_fraction", 0.1),
        soundscape_val_files=config["data"].get("soundscape_val_files", 8),
        soundscape_test_files=config["data"].get("soundscape_test_files", 8),
        seed=config["data"].get("seed", 42),
        transform=train_transform,
        eval_transform=audio_transform,
    )

    # ========================================================================
    # Callbacks
    # ========================================================================

    callbacks = []
    checkpoint_callback = None
    gpu_monitor = None

    if gpu_check_mode:
        from birdclef.callbacks.gpu_monitor import GPUUtilizationMonitor

        gpu_monitor = GPUUtilizationMonitor(log_every_n_steps=10)
        callbacks.append(gpu_monitor)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        logger.info("GPU check mode: skipping checkpoints, early stopping, and device stats")
    else:
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
        logger.info(f"Checkpoints: {checkpoint_dir}")

        early_stopping = EarlyStopping(
            monitor=config["callbacks"]["early_stopping"]["monitor"],
            mode=config["callbacks"]["early_stopping"]["mode"],
            patience=config["callbacks"]["early_stopping"]["patience"],
            min_delta=config["callbacks"]["early_stopping"]["min_delta"],
            verbose=True,
        )
        callbacks.append(early_stopping)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)

    # ========================================================================
    # Loggers
    # ========================================================================

    loggers = []
    experiment_name = f"exp016-{run_name}"

    tb_logger = TensorBoardLogger(
        save_dir=Path(__file__).parent / "outputs",
        name="tensorboard",
        version=experiment_name,
    )
    loggers.append(tb_logger)

    if args.fast_dev_run or gpu_check_mode:
        logger.info(f"{'GPU check' if gpu_check_mode else 'Fast dev run'} — skipping W&B logging")
    else:
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
                "architecture": "sed",
                "num_classes": num_classes,
                "encoder_name": config["encoder"]["model_name"],
            }
        )
        loggers.append(wandb_logger)

    # ========================================================================
    # Trainer
    # ========================================================================

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

    # GPU check: print summary and exit
    if gpu_check_mode:
        import os

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        summary_file = summary_dir / f"{run_name}_{job_id}.txt"
        gpu_monitor.print_summary(save_path=summary_file)
        logger.info("GPU check complete!")
        return

    # ========================================================================
    # Test
    # ========================================================================

    test_results = None
    if not args.fast_dev_run and checkpoint_callback.best_model_path:
        logger.info(f"Testing: {checkpoint_callback.best_model_path}")
        test_results = trainer.test(
            pl_module, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
        )
    elif args.fast_dev_run:
        logger.info("Fast dev run — running test step")
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
            "architecture": "sed",
            "num_classes": num_classes,
            "encoder": config["encoder"]["model_name"],
            "best_val_score": (
                float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None
                else None
            ),
            "best_checkpoint_path": str(checkpoint_callback.best_model_path),
            "stopped_epoch": trainer.current_epoch,
            "test_metrics": test_results[0] if test_results else None,
        }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results: {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
