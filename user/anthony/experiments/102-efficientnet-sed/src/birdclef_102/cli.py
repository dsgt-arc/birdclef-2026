"""CLI entry point for EfficientNet-B1 fine-tuning with optional LoRA.

Forked from exp 012. Reads a YAML config that controls backbone mode
(frozen/lora/full), LoRA parameters, and training settings.
"""

import argparse
import logging

import pytorch_lightning as pl
import yaml
from nnAudio.features import MelSpectrogram

from birdclef_009.data.finetune import BirdCLEFDataModule, build_species_list
from birdclef_009.training.finetune import FinetuneLightningModule
from birdclef_009.utils.gpu_monitor import GPUUtilizationMonitor

from .model import EfficientNetClassifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="EfficientNet-B1 Fine-Tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--overfit-batches", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    cb_cfg = cfg.get("callbacks", {})
    lora_cfg = cfg.get("lora", {})

    # Build species list
    species_list = build_species_list(data_cfg["metadata_csv"])
    num_classes = len(species_list)
    logger.info(f"Number of classes: {num_classes}")

    # Build nnAudio mel layer
    n_fft = data_cfg.get("n_fft", 2048)
    hop_length = data_cfg.get("hop_length", 256)
    mel_spec = MelSpectrogram(
        sr=data_cfg["sample_rate"],
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=data_cfg["n_mels"],
        fmin=50,
        fmax=14000,
        trainable_mel=False,
        trainable_STFT=False,
    )

    # Build model
    backbone_mode = model_cfg.get("backbone_mode", "frozen")
    model = EfficientNetClassifier(
        model_name=model_cfg["model_name"],
        num_classes=num_classes,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        head_dropout=model_cfg.get("head_dropout", 0.3),
        backbone_mode=backbone_mode,
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        lora_targets=lora_cfg.get("target_modules"),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"Backbone mode: {backbone_mode}")

    # Build Lightning module
    lightning_module = FinetuneLightningModule(
        model=model,
        num_classes=num_classes,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg.get("warmup_epochs", 0),
        freeze_epochs=0,
        gamma=train_cfg.get("gamma", 2.0),
        augmentation=train_cfg.get("augmentation"),
    )

    # Build data module
    data_module = BirdCLEFDataModule(
        audio_dir=data_cfg["audio_dir"],
        metadata_csv=data_cfg["metadata_csv"],
        species_list=species_list,
        sample_rate=data_cfg["sample_rate"],
        chunk_duration=data_cfg["chunk_duration"],
        num_tokens=data_cfg.get("target_frames", 1000),
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_fraction=data_cfg.get("val_fraction", 0.1),
    )

    # Run name for logging
    run_name = args.run_name or backbone_mode
    log_dir = f"logs/{run_name}"

    # Callbacks
    callbacks = [
        GPUUtilizationMonitor(log_every_n_steps=10),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if not args.overfit_batches and not args.fast_dev_run:
        es_cfg = cb_cfg.get("early_stopping")
        if es_cfg:
            callbacks.append(pl.callbacks.EarlyStopping(**es_cfg))

        mc_cfg = cb_cfg.get("model_checkpoint")
        if mc_cfg:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"{log_dir}/checkpoints",
                    filename=f"{run_name}-{{epoch:02d}}-{{val/cmap:.4f}}",
                    save_last=True,
                    **mc_cfg,
                )
            )

    max_epochs = 200 if args.overfit_batches else train_cfg["max_epochs"]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=train_cfg.get("precision", "bf16-mixed"),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir=log_dir,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
