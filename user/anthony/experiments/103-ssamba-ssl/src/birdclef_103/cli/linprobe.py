"""CLI entry point for linear probe evaluation of SSL pretrained SSAMBA."""

import argparse
import logging
import os

import pytorch_lightning as pl
import yaml
from nnAudio.features import MelSpectrogram
from ssamba_hf.config import SSAMBAConfig

from ..data.focal import FocalDataModule
from ..models.classifier import SSAMBALinearProbeClassifier
from ..training.linprobe import LinProbeLightningModule

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SSAMBA SSL Linear Probe")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 1 batch for sanity check")
    parser.add_argument("--overfit-batches", type=int, default=0, help="Overfit on N batches")
    parser.add_argument("--ssl-checkpoint", type=str, default=None, help="Override SSL checkpoint path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Build nnAudio mel layer (runs on GPU inside the model)
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

    # Build data module
    data_module = FocalDataModule(
        train_csv=data_cfg["train_csv"],
        taxonomy_csv=data_cfg["taxonomy_csv"],
        audio_dir=data_cfg["audio_dir"],
        sample_rate=data_cfg["sample_rate"],
        chunk_duration=data_cfg["chunk_duration"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_fraction=data_cfg.get("val_fraction", 0.20),
    )
    num_classes = data_module.num_classes

    # Build classifier from pretrain checkpoint
    ssl_ckpt = args.ssl_checkpoint or model_cfg["ssl_checkpoint"]
    logger.info(f"Loading SSL checkpoint: {ssl_ckpt}")
    classifier = SSAMBALinearProbeClassifier.from_pretrain_checkpoint(
        ckpt_path=ssl_ckpt,
        config=config,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        num_classes=num_classes,
        head_dropout=model_cfg.get("head_dropout", 0.3),
    )

    # Build Lightning module
    lightning_module = LinProbeLightningModule(
        model=classifier,
        num_classes=num_classes,
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg.get("warmup_epochs", 2),
    )

    # Callbacks
    ckpt_dir = "logs/linprobe-103c/checkpoints"
    log_dir = os.path.dirname(ckpt_dir)

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="linprobe-{epoch:02d}-{val/cmap:.4f}",
            save_top_k=1,
            monitor="val/cmap",
            mode="max",
            every_n_epochs=1,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/cmap",
            patience=5,
            mode="max",
            verbose=True,
        ),
    ]

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name="lightning_logs",
        version="linprobe",
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        precision=train_cfg.get("precision", "32-true"),
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=log_dir,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module, data_module)


if __name__ == "__main__":
    main()
