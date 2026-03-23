"""Student fine-tuning module with per-sample weighted loss.

Extends LoRAFinetuneModule to support mixed training on ground-truth labeled
data (weight=1.0) and pseudo-labeled data (weight<1.0). The per-sample weight
controls how much influence pseudo-labeled samples have on the loss.

Data flow: waveforms -> LoRA-adapted encoder -> classification head -> logits.
Loss: per-sample BCE with per-sample weight scaling.
"""

import logging

import torch
import torch.nn as nn

from birdclef.lightning.lora_finetune import LoRAFinetuneModule

logger = logging.getLogger(__name__)


class StudentFinetuneModule(LoRAFinetuneModule):
    """Fine-tune a LoRA-adapted encoder with per-sample weighted BCE loss.

    Extends LoRAFinetuneModule for student training on mixed labeled +
    pseudo-labeled data. Each batch sample carries a weight:
      - weight=1.0 for ground-truth labeled data (focal + labeled soundscapes)
      - weight<1.0 for pseudo-labeled data (unlabeled soundscapes)

    The training_step computes per-sample loss and applies the weight before
    averaging. Validation and test steps use unweighted loss (all ground truth).

    All constructor args are passed through to LoRAFinetuneModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override criterion to use reduction='none' for per-sample weighting
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def training_step(self, batch, batch_idx):
        waveform, labels, weights = batch
        logits = self(waveform)

        # Per-sample weighted loss
        per_sample_loss = self.criterion(logits, labels.float()).mean(dim=1)  # (batch,)
        loss = (per_sample_loss * weights).mean()

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        # Metrics: labels.int() rounds soft labels to 0/1 for metric computation.
        # This is approximate for pseudo-labeled samples but acceptable since
        # val metrics (ground truth only) are used for checkpoint selection.
        self.train_cmap.update(probs, labels.int())
        self.train_auroc.update(probs, labels.int())
        self.train_f1_macro.update(preds, labels.int())
        self.train_f1_micro.update(preds, labels.int())
        self.train_top1.update(probs, labels.int())
        self.train_top5.update(probs, labels.int())

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        waveform, labels, weights = batch
        logits = self(waveform)

        # Validation uses unweighted loss (all ground truth, weights=1.0)
        loss = self.criterion(logits, labels.float()).mean()

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.val_cmap.update(probs, labels.int())
        self.val_auroc.update(probs, labels.int())
        self.val_f1_macro.update(preds, labels.int())
        self.val_f1_micro.update(preds, labels.int())
        self.val_top1.update(probs, labels.int())
        self.val_top5.update(probs, labels.int())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        waveform, labels, weights = batch
        logits = self(waveform)

        loss = self.criterion(logits, labels.float()).mean()

        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).int()

        self.test_cmap.update(probs, labels.int())
        self.test_auroc.update(probs, labels.int())
        self.test_f1_macro.update(preds, labels.int())
        self.test_f1_micro.update(preds, labels.int())
        self.test_top1.update(probs, labels.int())
        self.test_top5.update(probs, labels.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return loss
