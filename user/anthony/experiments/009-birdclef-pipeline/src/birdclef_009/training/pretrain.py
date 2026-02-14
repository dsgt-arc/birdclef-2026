"""PyTorch Lightning module for SSAMBA MPG + JiTTER self-supervised pretraining."""

import pytorch_lightning as pl
import torch


class PretrainLightningModule(pl.LightningModule):
    """Lightning module for MPG + JiTTER pretraining.

    No validation — self-supervised objective only.
    Logs both MSE and L1 as monitoring metrics regardless of loss_fn.
    Periodically logs spectrogram visualizations to TensorBoard.

    Args:
        model: SSAMBAPretrainModel instance
        lr: Learning rate
        weight_decay: AdamW weight decay
        max_epochs: Max epochs for cosine annealing
        warmup_epochs: Linear warmup epochs
        jitter_weight: Weight for JiTTER loss (default 1.0)
        viz_every_n_epochs: Log spectrogram trios every N epochs (default 5)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
        jitter_weight: float = 1.0,
        viz_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.jitter_weight = jitter_weight
        self.viz_every_n_epochs = viz_every_n_epochs

    def on_before_optimizer_step(self, optimizer):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"))
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch, batch_idx):
        waveforms = batch["waveform"]
        out = self.model(waveforms)
        loss_mpg = out["loss_mpg"]
        loss_jitter = out["loss_jitter"]
        loss = loss_mpg + self.jitter_weight * loss_jitter

        # Primary losses
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_mpg", loss_mpg, on_step=True, on_epoch=True)
        self.log("train/loss_jitter", loss_jitter, on_step=True, on_epoch=True, prog_bar=True)

        # Cross-metrics: common ruler for comparing MSE vs L1 runs
        self.log("train/monitor_mse", out["monitor_mse"], on_step=False, on_epoch=True)
        self.log("train/monitor_l1", out["monitor_l1"], on_step=False, on_epoch=True)

        # Spectrogram visualization (first batch of selected epochs)
        if (batch_idx == 0
                and self.viz_every_n_epochs > 0
                and self.current_epoch % self.viz_every_n_epochs == 0
                and self.logger is not None
                and hasattr(self.logger.experiment, "add_image")):
            self._log_spectrograms(out)

        return loss

    def _log_spectrograms(self, out):
        """Log original/masked/reconstructed spectrogram trio to TensorBoard."""
        mel = out["viz_mel"][0, 0]         # (F, T)
        mask = out["viz_mask"]             # (N,)
        pred = out["viz_pred"][0]          # (M, patch_dim)
        target = out["viz_target"][0]      # (M, patch_dim)

        F_mel, T_mel = mel.shape
        patch_dim = self.model.patch_dim
        num_patches = mask.shape[0]

        # Build masked spectrogram (zero out masked positions)
        masked_mel = mel.clone()
        fpatch = self.model.backbone.config.fpatch
        tpatch = self.model.backbone.config.tpatch
        mask_indices = mask.nonzero(as_tuple=True)[0]
        for idx in mask_indices:
            t_start = idx * tpatch
            t_end = min(t_start + tpatch, T_mel)
            masked_mel[:fpatch, t_start:t_end] = 0

        # Build reconstructed spectrogram (fill masked positions with predictions)
        recon_mel = mel.clone()
        for i, idx in enumerate(mask_indices):
            if i >= pred.shape[0]:
                break
            t_start = idx * tpatch
            t_end = min(t_start + tpatch, T_mel)
            patch = pred[i].reshape(fpatch, tpatch)
            recon_mel[:fpatch, t_start:t_end] = patch[:, :t_end - t_start]

        # Normalize for visualization (shared scale across all three)
        vmin = mel.min()
        vmax = mel.max()
        def normalize(x):
            return ((x - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)

        # Stack as 3-row image: [original, masked, reconstructed]
        trio = torch.stack([normalize(mel), normalize(masked_mel), normalize(recon_mel)])
        # trio: (3, F, T) — treat as 3-channel image
        self.logger.experiment.add_image(
            "spectrogram/original_masked_recon",
            trio,
            global_step=self.global_step,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )

        if self.hparams.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=self.hparams.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[self.hparams.warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
