"""Linear probe training with focal loss via PyTorch Lightning.

Multi-label species probe (206 binary outputs) and multiclass taxa probe
(4-class), both using focal loss to handle extreme class imbalance.

CLI:
    weaksup-probe species birdnet --gamma 2.0 --epochs 50
    weaksup-probe taxa perch --gamma 2.0 --epochs 50
    weaksup-probe all birdnet
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import lightning as L
import numpy as np
import torch
import typer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor, nn
from torchmetrics.classification import (
    MultilabelAUROC,
    MulticlassAccuracy,
    MulticlassF1Score,
)

from weaksup.data import EmbeddingDataModule, IndicesDataModule
from weaksup.focal import FocalBCELoss, FocalCELoss

app = typer.Typer(pretty_exceptions_enable=False)

# --- Default paths ---
DEFAULT_DATA_DIR = Path("~/scratch/birdclef-2026/data/birdclef-2025").expanduser()
DEFAULT_OUTPUT_ROOT = Path("~/scratch/birdclef-2026/outputs/exp010/labels").expanduser()
DEFAULT_PROBE_DIR = Path("~/scratch/birdclef-2026/outputs/exp013/probes").expanduser()


def _embed_dir(source: str) -> Path:
    if source == "indices":
        return DEFAULT_OUTPUT_ROOT / "indices" / "train_audio" / "parts"
    return DEFAULT_OUTPUT_ROOT / source / "train_audio" / "parts" / "embed"


class SpeciesProbeModule(L.LightningModule):
    """Multi-label linear probe for 206 BirdCLEF species with focal BCE loss.

    Single linear layer: (N, D) -> (N, 206).
    Each output is an independent binary classifier.
    """

    def __init__(
        self,
        input_dim: int,
        n_species: int,
        alpha: Tensor | None = None,
        gamma: float = 2.0,
        C: float = 1.0,
        lr: float = 1e-3,
        max_steps: int = 500,
        species_counts: Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["alpha", "species_counts"])
        self.linear = nn.Linear(input_dim, n_species)
        self.criterion = FocalBCELoss(alpha=alpha, gamma=gamma)
        self.C = C
        self.lr = lr
        self.max_steps = max_steps

        self.val_auroc = MultilabelAUROC(num_labels=n_species, average="macro")
        self.val_auroc_per_species = MultilabelAUROC(num_labels=n_species, average=None)
        if species_counts is not None:
            self.register_buffer(
                "_freq_order", species_counts.argsort(descending=True)
            )
        else:
            self._freq_order = None

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        x, y_species, _y_taxa = batch
        logits = self(x)
        loss = self.criterion(logits, y_species)
        loss = loss + (0.5 / self.C) * self.linear.weight.pow(2).sum()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> None:
        x, y_species, _y_taxa = batch
        logits = self(x)
        loss = self.criterion(logits, y_species)
        self.log("val/loss", loss, prog_bar=True)
        self.val_auroc.update(logits, y_species.int())
        self.val_auroc_per_species.update(logits, y_species.int())

    def on_validation_epoch_end(self) -> None:
        auroc = self.val_auroc.compute()
        self.log("val/auroc_macro", auroc, prog_bar=True)
        self.val_auroc.reset()

        per_species = self.val_auroc_per_species.compute()
        self.val_auroc_per_species.reset()
        valid = ~per_species.isnan()
        if valid.any() and self._freq_order is not None:
            top20 = self._freq_order[:20]
            bot20 = self._freq_order[-20:]
            t_valid = valid[top20]
            b_valid = valid[bot20]
            if t_valid.any():
                self.log("val/auroc_top20", per_species[top20][t_valid].mean())
            if b_valid.any():
                self.log("val/auroc_bot20", per_species[bot20][b_valid].mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_steps, eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class TaxaProbeModule(L.LightningModule):
    """Multiclass linear probe for 4 taxa classes with focal CE loss.

    Single linear layer: (N, D) -> (N, 4).
    """

    def __init__(
        self,
        input_dim: int,
        n_taxa: int,
        alpha: Tensor | None = None,
        gamma: float = 2.0,
        C: float = 1.0,
        lr: float = 1e-3,
        max_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["alpha"])
        self.linear = nn.Linear(input_dim, n_taxa)
        self.criterion = FocalCELoss(alpha=alpha, gamma=gamma)
        self.C = C
        self.lr = lr
        self.max_steps = max_steps

        self.val_acc = MulticlassAccuracy(num_classes=n_taxa, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=n_taxa, average="macro")

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        x, _y_species, y_taxa = batch
        logits = self(x)
        loss = self.criterion(logits, y_taxa)
        loss = loss + (0.5 / self.C) * self.linear.weight.pow(2).sum()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> None:
        x, _y_species, y_taxa = batch
        logits = self(x)
        loss = self.criterion(logits, y_taxa)
        self.log("val/loss", loss, prog_bar=True)
        self.val_acc.update(logits, y_taxa)
        self.val_f1.update(logits, y_taxa)

    def on_validation_epoch_end(self) -> None:
        self.log("val/acc_macro", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1_macro", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_steps, eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def _build_trainer(
    output_dir: Path,
    name: str,
    max_epochs: int,
    monitor: str = "val/loss",
    mode: str = "min",
) -> L.Trainer:
    """Build a Lightning Trainer for CPU probe training."""
    return L.Trainer(
        accelerator="cpu",
        max_epochs=max_epochs,
        default_root_dir=output_dir,
        log_every_n_steps=10,
        logger=CSVLogger(save_dir=output_dir, name=name),
        callbacks=[
            RichProgressBar(),
            EarlyStopping(
                monitor=monitor,
                patience=7,
                mode=mode,
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath=output_dir / name / "checkpoints",
                filename=f"best-{{epoch:02d}}-{{{monitor}:.4f}}",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
            ),
        ],
        enable_model_summary=True,
    )


def _save_probe_results(
    module: L.LightningModule,
    best_ckpt_path: str | Path | None,
    dm: EmbeddingDataModule | IndicesDataModule,
    output_path: Path,
    probe_type: str,
) -> None:
    """Load best checkpoint weights into module and save as .npz."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_ckpt_path:
        print(f"Loading best checkpoint: {best_ckpt_path}")
        ckpt = torch.load(str(best_ckpt_path), map_location="cpu")
        module.load_state_dict(ckpt["state_dict"])
    else:
        print("No checkpoint found, using last model state")

    weight = module.linear.weight.detach().numpy()
    bias = module.linear.bias.detach().numpy()

    save_dict = dict(
        coef=weight,
        intercept=bias,
        gamma=module.criterion.gamma,
    )

    if probe_type == "species":
        save_dict["species_list"] = np.array(dm.species_list)
        if hasattr(module.criterion, "alpha") and module.criterion.alpha is not None:
            alpha = module.criterion.alpha
            if isinstance(alpha, Tensor):
                alpha = alpha.numpy()
            save_dict["alpha"] = alpha
        # Save feature normalization stats for indices probes
        if isinstance(dm, IndicesDataModule):
            from weaksup.data import INDICES_FEATURES
            save_dict["feature_mean"] = dm.feature_mean
            save_dict["feature_std"] = dm.feature_std
            save_dict["feature_names"] = np.array(INDICES_FEATURES)
    elif probe_type == "taxa":
        save_dict["taxa_list"] = np.array(dm.taxa_list)
        if module.criterion.alpha is not None:
            save_dict["alpha"] = module.criterion.alpha.numpy()

    np.savez(output_path, **save_dict)
    print(f"Saved probe weights to {output_path}")


# --- CLI ---


@app.command()
def species(
    source: Annotated[str, typer.Argument(help="Embedding source: birdnet, perch, or indices")],
    gamma: Annotated[float, typer.Option(help="Focal loss gamma")] = 2.0,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    epochs: Annotated[int, typer.Option(help="Max training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 4096,
    C: Annotated[float, typer.Option(help="Inverse L2 regularization")] = 1.0,
    balance: Annotated[str, typer.Option(help="Sampling balance: none/sqrt/inverse")] = "none",
    no_alpha: Annotated[bool, typer.Option("--no-alpha", help="Disable focal alpha (use with balanced sampling)")] = False,
    data_dir: Annotated[Path, typer.Option(help="BirdCLEF data dir")] = DEFAULT_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Output dir for probes")] = DEFAULT_PROBE_DIR,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Train multi-label species probe with focal BCE loss."""
    L.seed_everything(seed)

    embed_dir = _embed_dir(source)
    if source == "indices":
        dm = IndicesDataModule(
            indices_dir=embed_dir,
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        dm = EmbeddingDataModule(
            embed_dir=embed_dir,
            data_dir=data_dir,
            batch_size=batch_size,
            seed=seed,
        )
    dm.setup()

    alpha = None if no_alpha else torch.from_numpy(dm.species_pos_weight)
    n_train = len(dm.train_dataset)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    max_steps = steps_per_epoch * epochs
    module = SpeciesProbeModule(
        input_dim=dm.n_features,
        n_species=dm.n_species,
        alpha=alpha,
        gamma=gamma,
        C=C,
        lr=lr,
        max_steps=max_steps,
        species_counts=torch.from_numpy(dm.species_counts),
    )

    probe_dir = output_dir / source
    trainer = _build_trainer(
        probe_dir, "species", max_epochs=epochs,
        monitor="val/auroc_macro", mode="max",
    )
    if balance == "none":
        trainer.fit(module, datamodule=dm)
    else:
        trainer.fit(
            module,
            train_dataloaders=dm.species_balanced_train_dataloader(mode=balance),
            val_dataloaders=dm.val_dataloader(),
        )

    _save_probe_results(
        module,
        trainer.checkpoint_callback.best_model_path,
        dm,
        probe_dir / "species_probe.npz",
        "species",
    )


@app.command()
def taxa(
    source: Annotated[str, typer.Argument(help="Embedding source: birdnet or perch")],
    gamma: Annotated[float, typer.Option(help="Focal loss gamma")] = 2.0,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    epochs: Annotated[int, typer.Option(help="Max training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 4096,
    C: Annotated[float, typer.Option(help="Inverse L2 regularization")] = 1.0,
    data_dir: Annotated[Path, typer.Option(help="BirdCLEF data dir")] = DEFAULT_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Output dir for probes")] = DEFAULT_PROBE_DIR,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Train multiclass taxa probe with focal CE loss."""
    L.seed_everything(seed)

    embed_dir = _embed_dir(source)
    dm = EmbeddingDataModule(
        embed_dir=embed_dir,
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
    )
    dm.setup()

    alpha = torch.from_numpy(dm.taxa_class_weight)
    n_train = len(dm.train_dataset)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    max_steps = steps_per_epoch * epochs
    module = TaxaProbeModule(
        input_dim=dm.n_features,
        n_taxa=dm.n_taxa,
        alpha=alpha,
        gamma=gamma,
        C=C,
        lr=lr,
        max_steps=max_steps,
    )

    probe_dir = output_dir / source
    trainer = _build_trainer(
        probe_dir, "taxa", max_epochs=epochs,
        monitor="val/f1_macro", mode="max",
    )
    trainer.fit(
        module,
        train_dataloaders=dm.taxa_balanced_train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    _save_probe_results(
        module,
        trainer.checkpoint_callback.best_model_path,
        dm,
        probe_dir / "taxa_probe.npz",
        "taxa",
    )


@app.command()
def all(
    source: Annotated[str, typer.Argument(help="Embedding source: birdnet or perch")],
    gamma: Annotated[float, typer.Option(help="Focal loss gamma")] = 2.0,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    epochs: Annotated[int, typer.Option(help="Max training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 4096,
    C: Annotated[float, typer.Option(help="Inverse L2 regularization")] = 1.0,
    balance: Annotated[str, typer.Option(help="Species sampling balance: none/sqrt/inverse")] = "none",
    no_alpha: Annotated[bool, typer.Option("--no-alpha", help="Disable species focal alpha")] = False,
    data_dir: Annotated[Path, typer.Option(help="BirdCLEF data dir")] = DEFAULT_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Output dir for probes")] = DEFAULT_PROBE_DIR,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Train both species and taxa probes."""
    L.seed_everything(seed)

    embed_dir = _embed_dir(source)
    dm = EmbeddingDataModule(
        embed_dir=embed_dir,
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
    )
    dm.setup()

    probe_dir = output_dir / source
    n_train = len(dm.train_dataset)
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    max_steps = steps_per_epoch * epochs

    # Species probe
    print("\n=== Training species probe ===")
    alpha_species = None if no_alpha else torch.from_numpy(dm.species_pos_weight)
    species_module = SpeciesProbeModule(
        input_dim=dm.n_features,
        n_species=dm.n_species,
        alpha=alpha_species,
        gamma=gamma,
        C=C,
        lr=lr,
        max_steps=max_steps,
        species_counts=torch.from_numpy(dm.species_counts),
    )
    species_trainer = _build_trainer(
        probe_dir, "species", max_epochs=epochs,
        monitor="val/auroc_macro", mode="max",
    )
    if balance == "none":
        species_trainer.fit(species_module, datamodule=dm)
    else:
        species_trainer.fit(
            species_module,
            train_dataloaders=dm.species_balanced_train_dataloader(mode=balance),
            val_dataloaders=dm.val_dataloader(),
        )
    _save_probe_results(
        species_module,
        species_trainer.checkpoint_callback.best_model_path,
        dm,
        probe_dir / "species_probe.npz",
        "species",
    )

    # Taxa probe
    print("\n=== Training taxa probe ===")
    alpha_taxa = torch.from_numpy(dm.taxa_class_weight)
    taxa_module = TaxaProbeModule(
        input_dim=dm.n_features,
        n_taxa=dm.n_taxa,
        alpha=alpha_taxa,
        gamma=gamma,
        C=C,
        lr=lr,
        max_steps=max_steps,
    )
    taxa_trainer = _build_trainer(
        probe_dir, "taxa", max_epochs=epochs,
        monitor="val/f1_macro", mode="max",
    )
    taxa_trainer.fit(
        taxa_module,
        train_dataloaders=dm.taxa_balanced_train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    _save_probe_results(
        taxa_module,
        taxa_trainer.checkpoint_callback.best_model_path,
        dm,
        probe_dir / "taxa_probe.npz",
        "taxa",
    )

    print(f"\nAll probes saved to {probe_dir}")


if __name__ == "__main__":
    app()
