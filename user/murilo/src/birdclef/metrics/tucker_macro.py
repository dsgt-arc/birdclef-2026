"""Tucker-style macro AUROC and AP that skip 0-positive classes.

Background — `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`,
notebook lines 369-384:

    def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
        aucs = []
        for c in range(y_true.shape[1]):
            col = y_true[:, c]
            if col.sum() == 0 or col.sum() == len(col):
                continue
            try:
                aucs.append(roc_auc_score(col, y_pred[:, c]))
            except ValueError:
                continue
        return (np.mean(aucs) if aucs else float("nan")), len(aucs)

torchmetrics 1.9 `MultilabelAUROC(average="macro")` returns 0.0 (NOT NaN)
for classes with no positives and INCLUDES them in the macro denominator.
On exp 032's val fold (168 SS windows, 205 of 234 classes have 0 positives)
this drags the macro from 0.89 down to 0.11 — see
`scripts/diag_val_auroc.py` for the diagnostic that proved this.

This module provides drop-in `torchmetrics.Metric` subclasses matching
Tucker's behavior: per-class AUC/AP computed only on classes that have
both positives AND negatives in the accumulated state, macro-averaged
over those evaluable classes.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics import Metric


class _TuckerMacroBase(Metric):
    """Shared accumulator + skip-empty-class logic for AUC/AP variants.

    State is a list of (preds, target) tensor pairs across batches. We
    concatenate at `compute()` time, then iterate per-class, skipping
    classes with `col.sum() == 0` (no positives — AUC undefined) or
    `col.sum() == len(col)` (no negatives — AUC undefined).

    Subclasses override `_per_class_score(col_true, col_pred) -> float`.
    """

    # Concatenated state, no full state update needed each step.
    full_state_update: bool = False
    # Bounded in [0, 1] but we report NaN if no evaluable classes.
    is_differentiable: bool = False

    def __init__(self, num_labels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_labels = int(num_labels)
        # `default=[]` + `dist_reduce_fx="cat"` is the standard torchmetrics
        # pattern for accumulators that need the whole epoch's data at
        # compute time. Each entry is a per-batch tensor; dist sync gathers
        # across ranks then concatenates.
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim != 2 or preds.shape[1] != self.num_labels:
            raise ValueError(
                f"Expected preds shape (B, {self.num_labels}); got {tuple(preds.shape)}"
            )
        if target.shape != preds.shape:
            raise ValueError(
                f"preds and target shape mismatch: {tuple(preds.shape)} vs {tuple(target.shape)}"
            )
        # Detach + move-to-cpu lazily at compute time so multi-GPU sync still works.
        self.preds.append(preds.detach())
        self.target.append(target.detach())

    def _per_class_score(self, col_true: np.ndarray, col_pred: np.ndarray) -> float:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        if not self.target:
            return torch.tensor(float("nan"))
        preds = torch.cat(self.preds, dim=0).float().cpu().numpy()
        target = torch.cat(self.target, dim=0).float().cpu().numpy()
        scores: list[float] = []
        for c in range(self.num_labels):
            col = target[:, c]
            n_pos = float(col.sum())
            if n_pos == 0.0 or n_pos == float(len(col)):
                continue
            try:
                scores.append(self._per_class_score(col, preds[:, c]))
            except ValueError:
                continue
        if not scores:
            return torch.tensor(float("nan"))
        return torch.tensor(float(np.mean(scores)))

    def n_evaluable(self) -> int:
        """How many classes contributed to the most recent compute()."""
        if not self.target:
            return 0
        target = torch.cat(self.target, dim=0).cpu().numpy()
        col_sums = target.sum(axis=0)
        return int(((col_sums > 0) & (col_sums < len(target))).sum())


class TuckerMacroAUC(_TuckerMacroBase):
    """Macro-averaged ROC-AUC over evaluable classes (Tucker's compute_macro_auc).

    Drop-in replacement for `torchmetrics.classification.MultilabelAUROC`
    with `average="macro"`, except 0-positive and 1-only classes are
    skipped instead of being silently set to 0.0.

    Reference: `vendor/.../bc2026-distilled-sed/bc2026-distilled-sed.py` lines 369-384.
    """

    def _per_class_score(self, col_true: np.ndarray, col_pred: np.ndarray) -> float:
        return float(roc_auc_score(col_true, col_pred))


class TuckerMacroAP(_TuckerMacroBase):
    """Macro-averaged Average Precision over evaluable classes.

    Drop-in replacement for `torchmetrics.classification.MultilabelAveragePrecision`
    with `average="macro"`. Same skip-empty-class semantics as TuckerMacroAUC.

    Used as a Tucker-faithful `val/cmap` replacement.
    """

    def _per_class_score(self, col_true: np.ndarray, col_pred: np.ndarray) -> float:
        return float(average_precision_score(col_true, col_pred))
