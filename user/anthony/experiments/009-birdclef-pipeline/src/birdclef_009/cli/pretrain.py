"""CLI entry point for SSAMBA MSPM pretraining."""

import argparse
import logging

import pytorch_lightning as pl
import torch
import yaml
from nnAudio.features import MelSpectrogram
from ssamba_hf.config import SSAMBAConfig

from ..data.pretrain import SoundscapeDataModule
from ..models.pretrain import SSAMBAPretrainModel
from ..training.pretrain import PretrainLightningModule
from ..utils.gpu_monitor import GPUUtilizationMonitor

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SSAMBA MSPM Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 1 batch for sanity check")
    parser.add_argument("--overfit-batches", type=int, default=0, help="Overfit on N batches (0=disabled)")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Build nnAudio mel layer
    mel_spec = MelSpectrogram(
        sr=data_cfg["sample_rate"],
        n_fft=1024,
        win_length=1024,
        hop_length=int(data_cfg["sample_rate"] * data_cfg["chunk_duration"] / data_cfg["target_frames"]),
        n_mels=data_cfg["n_mels"],
        fmin=50,
        fmax=14000,
        trainable_mel=False,
        trainable_STFT=False,
    )

    # Build SSAMBA config
    config = SSAMBAConfig(
        img_size=tuple(model_cfg["img_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
    )

    # Build model
    model = SSAMBAPretrainModel(
        config=config,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        norm_mean=data_cfg.get("norm_mean", -4.268),
        norm_std=data_cfg.get("norm_std", 4.569),
        mask_ratio=model_cfg.get("mask_ratio", 0.78),
        mask_mode=model_cfg.get("mask_mode", "span"),
        mask_span_min=model_cfg.get("mask_span_min", 5),
        mask_span_max=model_cfg.get("mask_span_max", 15),
        grid_jitter=model_cfg.get("grid_jitter", 5),
        normalize_targets=model_cfg.get("normalize_targets", True),
        loss_fn=model_cfg.get("loss_fn", "mse"),
        decoder_depth=model_cfg.get("decoder_depth", 2),
        jitter_weight=train_cfg.get("jitter_weight", 0.0),
        block_size=model_cfg.get("block_size", 50),
        shuffle_ratio=model_cfg.get("shuffle_ratio", 0.75),
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Build Lightning module
    lightning_module = PretrainLightningModule(
        model=model,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        jitter_weight=train_cfg.get("jitter_weight", 1.0),
    )

    # Build data module
    data_module = SoundscapeDataModule(
        soundscape_dir=data_cfg["soundscape_dir"],
        sample_rate=data_cfg["sample_rate"],
        chunk_duration=data_cfg["chunk_duration"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    # Callbacks
    callbacks = [
        GPUUtilizationMonitor(log_every_n_steps=10),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            dirpath="logs/pretrain/checkpoints",
            filename="pretrain-{epoch:02d}-{train/loss:.4f}",
            save_top_k=3,
            monitor="train/loss_epoch",
            mode="min",
            every_n_epochs=5,
        ),
    ]

    # Trainer
    max_epochs = train_cfg["max_epochs"]
    if args.overfit_batches > 0:
        max_epochs = 200

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=train_cfg.get("precision", "bf16-mixed"),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir="logs/pretrain",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
