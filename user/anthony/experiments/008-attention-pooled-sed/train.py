"""SSAMBA attention-pooled SED training on BirdSet HSN.

Usage:
    uv run python train.py --config config-sed-128x1.yaml
    uv run python train.py --config config-sed-128x1.yaml --fast-dev-run
    uv run python train.py --config config-sed-128x1.yaml --overfit-batches 4
"""

import argparse
import logging

import pytorch_lightning as pl
import yaml
from nnAudio.features import MelSpectrogram
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ssamba_hf.config import SSAMBAConfig

from src.data import BirdSetStreamingDataModule
from src.gpu_monitor import GPUUtilizationMonitor
from src.lightning_module import SEDLightningModule
from src.model import SSAMBASEDClassifier
from src.pretrained import load_pretrained_backbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config-sed-128x1.yaml")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from Lightning checkpoint")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--pretrained-ckpt", type=str, default=None, help="Path to SSAMBA pre-trained checkpoint")
    parser.add_argument("--freeze-epochs", type=int, default=None, help="Epochs to freeze backbone")
    parser.add_argument("--overfit-batches", type=int, default=0, help="Overfit on N batches (0=disabled)")
    parser.add_argument("--run-name", type=str, default="sed-128x1", help="Run name for logs/checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation")
    cb_cfg = cfg.get("callbacks", {})

    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size

    # Build SSAMBA config
    ssamba_config = SSAMBAConfig(
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        img_size=tuple(model_cfg["img_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        use_cls_token=True,
        use_middle_cls_token=True,
        rms_norm=False,
    )

    # Build data module
    dm = BirdSetStreamingDataModule(
        dataset_name=data_cfg["dataset_name"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        sample_rate=data_cfg["sample_rate"],
        chunk_duration=data_cfg.get("chunk_duration", 10.0),
        num_tokens=data_cfg["target_frames"],
    )
    num_classes = dm.num_classes

    # Build nnAudio mel spectrogram layer (GPU-accelerated, non-trainable)
    mel_spec = MelSpectrogram(
        sr=data_cfg["sample_rate"],
        n_fft=800,
        win_length=800,
        hop_length=320,
        n_mels=data_cfg["n_mels"],
        htk=True,
        window="hann",
        center=False,
        fmin=0,
        fmax=None,
        trainable_mel=False,
        trainable_STFT=False,
    )

    # Build model
    classifier = SSAMBASEDClassifier(
        config=ssamba_config,
        num_classes=num_classes,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        norm_mean=data_cfg["norm_mean"],
        norm_std=data_cfg["norm_std"],
        head_dropout=model_cfg.get("head_dropout", 0.3),
    )

    # Apply stochastic depth schedule
    if model_cfg.get("drop_path_rate", 0.0) > 0:
        classifier.backbone.set_drop_path_rate(model_cfg["drop_path_rate"])

    # Load pre-trained weights
    pretrained_ckpt = args.pretrained_ckpt or train_cfg.get("pretrained_ckpt")
    if pretrained_ckpt:
        logger.info(f"Loading pre-trained weights from {pretrained_ckpt}")
        load_pretrained_backbone(classifier, pretrained_ckpt)

    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Determine freeze_epochs
    freeze_epochs = args.freeze_epochs
    if freeze_epochs is None:
        freeze_epochs = train_cfg.get("freeze_epochs", 0)

    # Disable augmentation for overfit test
    if args.overfit_batches > 0:
        aug_cfg = None
        logger.info("Overfit mode: augmentation disabled")

    # Build lightning module
    module = SEDLightningModule(
        model=classifier,
        num_classes=num_classes,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg["warmup_epochs"],
        threshold=train_cfg.get("threshold", 0.5),
        augmentation=aug_cfg,
        freeze_epochs=freeze_epochs,
        alpha=train_cfg.get("alpha", 1.0),
        gamma_clip=train_cfg.get("gamma_clip", 2.0),
        gamma_sed=train_cfg.get("gamma_sed", 2.0),
    )

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.overfit_batches == 0:
        callbacks.append(GPUUtilizationMonitor(log_every_n_steps=10))

        es_cfg = cb_cfg.get("early_stopping", {})
        if es_cfg:
            callbacks.append(
                EarlyStopping(
                    monitor=es_cfg.get("monitor", "val/cmap"),
                    mode=es_cfg.get("mode", "max"),
                    patience=es_cfg.get("patience", 15),
                    min_delta=es_cfg.get("min_delta", 0.001),
                )
            )

        ckpt_cfg = cb_cfg.get("model_checkpoint", {})
        checkpoint_callback = ModelCheckpoint(
            monitor=ckpt_cfg.get("monitor", "val/cmap"),
            mode=ckpt_cfg.get("mode", "max"),
            save_top_k=ckpt_cfg.get("save_top_k", 1),
            save_last=ckpt_cfg.get("save_last", True),
            filename="sed-hsn-{epoch:03d}-{val/cmap:.4f}",
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = None

    # Logger
    tb_logger = TensorBoardLogger(save_dir="logs", name=args.run_name)

    # Trainer
    max_epochs = train_cfg["max_epochs"]
    if args.overfit_batches > 0:
        max_epochs = 200

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        precision=train_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=train_cfg.get("grad_clip_val", 1.0),
        callbacks=callbacks,
        logger=tb_logger,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        enable_progress_bar=True,
    )

    logger.info("Starting training...")
    trainer.fit(module, dm, ckpt_path=args.ckpt_path)

    if not args.fast_dev_run and checkpoint_callback is not None:
        logger.info(f"Best model: {checkpoint_callback.best_model_path}")
        logger.info(f"Best val/cmap: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
