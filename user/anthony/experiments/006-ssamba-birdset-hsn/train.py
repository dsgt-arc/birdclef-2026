"""SSAMBA training on BirdSet HSN.

Usage:
    uv run python train.py --config config.yaml
    uv run python train.py --config config.yaml --fast-dev-run
"""

import argparse
import logging

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ssamba_hf.config import SSAMBAConfig

from src.data import BirdSetDataModule
from src.lightning_module import SSAMBALightningModule
from src.ssamba_classifier import SSAMBAClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--num-workers", type=int, default=None)
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
    dm = BirdSetDataModule(
        dataset_name=data_cfg["dataset_name"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        sample_rate=data_cfg["sample_rate"],
        target_duration=data_cfg["target_duration"],
        n_mels=data_cfg["n_mels"],
        frame_shift_ms=data_cfg["frame_shift_ms"],
        target_frames=data_cfg["target_frames"],
        norm_mean=data_cfg["norm_mean"],
        norm_std=data_cfg["norm_std"],
    )
    num_classes = dm.num_classes

    # Build model
    classifier = SSAMBAClassifier(
        config=ssamba_config,
        num_classes=num_classes,
        head_dropout=model_cfg.get("head_dropout", 0.3),
    )

    # Apply stochastic depth schedule across layers
    if model_cfg.get("drop_path_rate", 0.0) > 0:
        classifier.backbone.set_drop_path_rate(model_cfg["drop_path_rate"])

    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable")

    # Build lightning module
    module = SSAMBALightningModule(
        model=classifier,
        num_classes=num_classes,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg["warmup_epochs"],
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
        threshold=train_cfg.get("threshold", 0.5),
        augmentation=aug_cfg,
    )

    # Callbacks
    callbacks = [LearningRateMonitor(logging_interval="epoch")]

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
        filename="ssamba-hsn-{epoch:03d}-{val/cmap:.4f}",
    )
    callbacks.append(checkpoint_callback)

    # Logger
    tb_logger = TensorBoardLogger(save_dir="logs", name="ssamba-hsn")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        devices="auto",
        precision=train_cfg.get("precision", "16-mixed"),
        gradient_clip_val=train_cfg.get("grad_clip_val", 1.0),
        callbacks=callbacks,
        logger=tb_logger,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
    )

    logger.info("Starting training...")
    trainer.fit(module, dm, ckpt_path=args.ckpt_path)

    if not args.fast_dev_run:
        logger.info(f"Best model: {checkpoint_callback.best_model_path}")
        logger.info(f"Best val/cmap: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
