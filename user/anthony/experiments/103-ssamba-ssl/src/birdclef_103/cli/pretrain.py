"""CLI entry point for SSAMBA SSL pretraining."""

import argparse
import logging
import os

import pytorch_lightning as pl
import torch
import yaml
from nnAudio.features import MelSpectrogram
from ssamba_hf.config import SSAMBAConfig

from ..data.pretrain import SSLDataModule
from ..models.pretrain import SSAMBAPretrainModel
from ..training.pretrain import PretrainLightningModule
from ..utils.gpu_monitor import GPUUtilizationMonitor

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SSAMBA SSL Pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 1 batch for sanity check")
    parser.add_argument("--overfit-batches", type=int, default=0, help="Overfit on N batches (0=disabled)")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint ('last' to auto-find)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint directory")
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
        n_fft=data_cfg.get("n_fft", 4096),
        win_length=data_cfg.get("win_length", 4096),
        hop_length=data_cfg.get("hop_length", 320),
        n_mels=data_cfg["n_mels"],
        fmin=data_cfg.get("fmin", 50),
        fmax=data_cfg.get("fmax", 16000),
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

    # Disable grid jitter for overfit tests (otherwise targets shift each step,
    # preventing memorization and making the test meaningless)
    grid_jitter = model_cfg.get("grid_jitter", 5)
    if args.overfit_batches > 0 and grid_jitter > 0:
        logger.info(f"Overfit mode: disabling grid_jitter (was {grid_jitter})")
        grid_jitter = 0

    # Build model
    model = SSAMBAPretrainModel(
        config=config,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        mel_norm=model_cfg.get("mel_norm", "pcen"),
        mask_ratio=model_cfg.get("mask_ratio", 0.40),
        mask_mode=model_cfg.get("mask_mode", "random"),
        mask_span_min=model_cfg.get("mask_span_min", 5),
        mask_span_max=model_cfg.get("mask_span_max", 15),
        grid_jitter=grid_jitter,
        normalize_targets=model_cfg.get("normalize_targets", True),
        loss_fn=model_cfg.get("loss_fn", "mse"),
        decoder_depth=model_cfg.get("decoder_depth", 2),
        nce_weight=model_cfg.get("nce_weight", 1.0),
        mse_weight=model_cfg.get("mse_weight", 10.0),
        nce_pool=model_cfg.get("nce_pool", "token"),
        nce_num_negatives=model_cfg.get("nce_num_negatives", 100),
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Checkpoint and log directories
    ckpt_dir = args.ckpt_dir or "logs/pretrain/checkpoints"
    log_dir = os.path.dirname(ckpt_dir)

    # Resolve --ckpt-path=last
    ckpt_path = args.ckpt_path
    if ckpt_path == "last":
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            ckpt_path = last_ckpt
            logger.info(f"Resuming from {ckpt_path}")
        else:
            logger.info("No last.ckpt found, starting from scratch")
            ckpt_path = None

    # Resolve actual max_epochs (overfit mode overrides to 200)
    max_epochs = train_cfg["max_epochs"]
    is_overfit = args.overfit_batches > 0
    if is_overfit:
        max_epochs = 200

    # Build Lightning module
    weight_decay = train_cfg["weight_decay"]
    if is_overfit:
        logger.info(f"Overfit mode: disabling weight_decay (was {weight_decay})")
        weight_decay = 0.0

    lightning_module = PretrainLightningModule(
        model=model,
        lr=train_cfg["lr"],
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        warmup_epochs=train_cfg.get("warmup_epochs", 5) if not is_overfit else 0,
        nce_weight=model_cfg.get("nce_weight", 1.0),
        mse_weight=model_cfg.get("mse_weight", 10.0),
    )

    # Build data module
    data_module = SSLDataModule(
        soundscape_dir=data_cfg.get("soundscape_dir"),
        focal_dir=data_cfg.get("focal_dir"),
        sample_rate=data_cfg["sample_rate"],
        chunk_duration=data_cfg["chunk_duration"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_fraction=data_cfg.get("val_fraction", 0.05),
    )

    # Callbacks
    callbacks = [
        GPUUtilizationMonitor(log_every_n_steps=10),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    if not is_overfit and not args.fast_dev_run:
        callbacks.extend([
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="ssl-{epoch:02d}-{val/loss:.4f}",
                save_top_k=3,
                monitor="val/loss",
                mode="min",
                every_n_epochs=1,
                save_last=True,
            ),
            pl.callbacks.EarlyStopping(
                monitor="val/loss",
                patience=5,
                mode="min",
                verbose=True,
            ),
        ])

    # Fixed-version logger so resumes append to the same TensorBoard run
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name="lightning_logs",
        version="run",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=train_cfg.get("precision", "16-mixed"),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=log_dir,
        log_every_n_steps=1 if is_overfit else 10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1 if is_overfit else train_cfg.get("accumulate_grad_batches", 1),
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
