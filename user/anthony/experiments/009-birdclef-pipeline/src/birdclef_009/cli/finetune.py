"""CLI entry point for SSAMBA SED fine-tuning."""

import argparse
import logging

import pytorch_lightning as pl
import torch
import yaml
from nnAudio.features import MelSpectrogram
from ssamba_hf.config import SSAMBAConfig

from ..data.finetune import BirdCLEFDataModule, build_species_list
from ..models.finetune import SSAMBAClassifier
from ..training.finetune import FinetuneLightningModule
from ..utils.gpu_monitor import GPUUtilizationMonitor
from ..utils.pretrained import load_mspm_backbone, load_pretrained_backbone

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SSAMBA SED Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from Lightning checkpoint")
    parser.add_argument("--pretrained-ckpt", type=str, default=None, help="Pretrained MSPM or SSAMBA checkpoint")
    parser.add_argument("--overfit-batches", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    cb_cfg = cfg.get("callbacks", {})

    # Build species list
    species_list = build_species_list(data_cfg["metadata_csv"])
    num_classes = len(species_list)
    logger.info(f"Number of classes: {num_classes}")

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
    drop_path_rate = model_cfg.get("drop_path_rate", 0.0)

    config = SSAMBAConfig(
        img_size=tuple(model_cfg["img_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
    )

    # Build model
    model = SSAMBAClassifier(
        config=config,
        num_classes=num_classes,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        head_dropout=model_cfg.get("head_dropout", 0.3),
    )

    # Apply stochastic depth (linearly increasing per layer)
    if drop_path_rate > 0:
        model.backbone.set_drop_path_rate(drop_path_rate)

    # Load pretrained weights
    pretrained_ckpt = args.pretrained_ckpt or train_cfg.get("pretrained_ckpt")
    if pretrained_ckpt:
        # Detect checkpoint type: Lightning (.ckpt) vs original SSAMBA (.pth)
        if pretrained_ckpt.endswith(".pth"):
            load_pretrained_backbone(model, pretrained_ckpt)
        else:
            load_mspm_backbone(model, pretrained_ckpt)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Build Lightning module
    freeze_epochs = 0 if args.overfit_batches else train_cfg.get("freeze_epochs", 0)
    lightning_module = FinetuneLightningModule(
        model=model,
        num_classes=num_classes,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        freeze_epochs=freeze_epochs,
        gamma=train_cfg.get("gamma", 2.0),
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

    # Callbacks
    callbacks = [
        GPUUtilizationMonitor(log_every_n_steps=10),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Skip early stopping and checkpointing for overfit/fast-dev-run
    if not args.overfit_batches and not args.fast_dev_run:
        es_cfg = cb_cfg.get("early_stopping")
        if es_cfg:
            callbacks.append(pl.callbacks.EarlyStopping(**es_cfg))

        mc_cfg = cb_cfg.get("model_checkpoint")
        if mc_cfg:
            callbacks.append(pl.callbacks.ModelCheckpoint(
                dirpath="logs/finetune/checkpoints",
                filename="finetune-{epoch:02d}-{val/cmap:.4f}",
                **mc_cfg,
            ))

    max_epochs = 200 if args.overfit_batches else train_cfg["max_epochs"]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=train_cfg.get("precision", "bf16-mixed"),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        default_root_dir="logs/finetune",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
