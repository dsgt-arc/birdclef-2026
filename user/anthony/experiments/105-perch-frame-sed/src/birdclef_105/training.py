"""Lightning training module and `run()` entry point for stage 2.

Trains a small head on cached Perch pooled embeddings using the same
FocalBCELoss + AdamW + cosine-with-warmup schedule as exp 102 Run N.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchmetrics
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from birdclef_105 import cache as cache_mod
from birdclef_105 import model as model_mod
from birdclef_105.data import CachedEmbeddingDataModule, FocalDataModule
from birdclef_105.losses import FocalBCELoss

logger = logging.getLogger("birdclef_105.training")


class HeadTrainer(pl.LightningModule):
    def __init__(
        self,
        head: torch.nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 50,
        warmup_epochs: int = 3,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["head"])
        self.head = head
        self.num_classes = num_classes
        self.loss_fn = FocalBCELoss(gamma=gamma, from_logits=True)

        self.train_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        self.val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )
        # Auxiliary soundscape val (dataloader_idx=1 when present)
        self.sc_val_cmap = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average="macro"
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.head(embedding)

    def training_step(self, batch, batch_idx):
        logits = self.head(batch["embedding"])
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)
        self.train_cmap.update(torch.sigmoid(logits), labels.round().int())
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/cmap", self.train_cmap.compute(), prog_bar=True)
        self.train_cmap.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.head(batch["embedding"])
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        if dataloader_idx == 0:
            # Primary focal val
            self.val_cmap.update(probs, labels.round().int())
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Auxiliary soundscape val (monitoring only)
            self.sc_val_cmap.update(probs, labels.round().int())
        return loss

    def on_validation_epoch_end(self):
        self.log("val/cmap", self.val_cmap.compute(), prog_bar=True)
        self.val_cmap.reset()
        # Soundscape val — may be empty if no soundscape dataloader provided
        try:
            sc = self.sc_val_cmap.compute()
            self.log("val/sc_cmap", sc, prog_bar=True)
        except (RuntimeError, ValueError):
            pass  # no soundscape samples this epoch
        self.sc_val_cmap.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        cosine_epochs = max(self.hparams.max_epochs - self.hparams.warmup_epochs, 1)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )
        if self.hparams.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.hparams.warmup_epochs],
            )
        else:
            scheduler = cosine
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _resolve_data(cfg: dict) -> tuple[pl.LightningDataModule, int]:
    """Pick the best data source available and return (DataModule, num_classes).

    Priority:
      1. Focal shards + soundscape auxiliary val  (matches exp 102 Run N)
      2. Single soundscape shard (fallback)
      3. Pilot shard (fallback)
    """
    cache_dir = Path(cfg["cache"]["output_dir"])
    focal_dir = cache_dir / "focal"
    soundscape_dir = cache_dir / "soundscape_plus_nocall"
    pilot_dir = Path(cfg["cache"]["pilot_dir"]) / "soundscape_plus_nocall"
    train_cfg = cfg["training"]

    if focal_dir.exists() and any(focal_dir.glob("shard-*/meta.json")):
        logger.info("Using focal shards at %s (+ soundscape auxiliary val)", focal_dir)
        dm = FocalDataModule(
            focal_dir=focal_dir,
            soundscape_dir=soundscape_dir if soundscape_dir.exists() else None,
            batch_size=train_cfg["batch_size"],
            num_workers=2,
        )
        dm.setup()
        return dm, dm.num_classes

    if (soundscape_dir / "meta.json").exists():
        logger.info("No focal shards — using soundscape shard at %s", soundscape_dir)
        dm = CachedEmbeddingDataModule(
            shard_dir=soundscape_dir,
            batch_size=train_cfg["batch_size"],
            num_workers=0,
        )
        dm.setup()
        return dm, dm.num_classes

    if (pilot_dir / "meta.json").exists():
        logger.info("Falling back to pilot shard at %s", pilot_dir)
        dm = CachedEmbeddingDataModule(
            shard_dir=pilot_dir,
            batch_size=train_cfg["batch_size"],
            num_workers=0,
        )
        dm.setup()
        return dm, dm.num_classes

    raise FileNotFoundError(
        f"No feature shards found. Checked:\n  {focal_dir}\n  {soundscape_dir}\n  {pilot_dir}\n"
        "Run `sbatch sbatch/extract-focal.sbatch` or `sbatch sbatch/extract-features.sbatch` first."
    )


def run(config_path: str, head: str = "linear", run_name: str | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    train_cfg = cfg["training"]

    dm, num_classes = _resolve_data(cfg)
    logger.info("Training data ready: %d classes", num_classes)

    # Build head.
    head_kwargs: dict = {}
    if head == "mlp" and "mlp" in cfg.get("model", {}):
        head_kwargs = {
            k: v for k, v in cfg["model"]["mlp"].items() if k in {"hidden_dim", "dropout"}
        }
    head_module = model_mod.build_head(
        name=head, embed_dim=1536, num_classes=num_classes, **head_kwargs
    )
    n_params = sum(p.numel() for p in head_module.parameters())
    logger.info("Head %s: %d parameters", head, n_params)

    trainer_module = HeadTrainer(
        head=head_module,
        num_classes=num_classes,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg["warmup_epochs"],
        gamma=train_cfg.get("gamma", 2.0),
    )

    tag = run_name or f"{head}"
    log_dir = Path("logs") / tag
    ckpt_dir = Path("artifacts") / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(save_dir=str(log_dir), name="", version="")
    early_stop = EarlyStopping(
        monitor=train_cfg["early_stopping"]["monitor"],
        mode=train_cfg["early_stopping"]["mode"],
        patience=train_cfg["early_stopping"]["patience"],
    )
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{tag}-{{epoch}}-{{val/cmap:.3f}}",
        monitor=train_cfg["early_stopping"]["monitor"],
        mode=train_cfg["early_stopping"]["mode"],
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        logger=tb_logger,
        callbacks=[early_stop, ckpt],
        log_every_n_steps=5,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(trainer_module, datamodule=dm)

    best = ckpt.best_model_score
    if best is not None:
        logger.info("Best val/cmap: %.4f (checkpoint: %s)", best, ckpt.best_model_path)
    return 0
