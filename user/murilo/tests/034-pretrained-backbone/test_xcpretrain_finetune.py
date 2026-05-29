"""Phase B unit smoke tests for XCPretrainFinetuneModule (B.9.1).

Exercises the module's core logic on a synthetic exp032-shaped batch
(wavs (B,1,160000), labels (B,234), wts, mks) WITHOUT a Trainer (calls `_step`
directly to avoid self.log). The full training_step + logging path is exercised
by the fast-dev-run gate (B.9.2).

Covers: finite loss + finite grads; val AUROC update/compute; mass-preserving
label smoothing; frozen-backbone freezes backbone but not head; optimizer config.
"""

import torch

from birdclef.lightning.xcpretrain_finetune import XCPretrainFinetuneModule
from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier

NUM_CLASS = 234


def _model():
    return XCPretrainWaveClassifier(
        backbone="tf_efficientnetv2_s_in21k",
        pretrained=False,
        head_config={"p": 0.5, "num_class": NUM_CLASS, "train_period": 5.0, "infer_period": 5.0},
    )


def _module(freeze=False, rf=True):
    return XCPretrainFinetuneModule(
        model=_model(),
        num_classes=NUM_CLASS,
        freeze_backbone=freeze,
        use_random_filtering=rf,
        max_epochs=5,
    )


def _batch(B=2):
    wav = torch.randn(B, 1, 160_000)
    labels = (torch.rand(B, NUM_CLASS) > 0.9).float()
    wts = torch.ones(B, NUM_CLASS)
    mks = torch.ones(B, NUM_CLASS)
    return (wav, labels, wts, mks)


def test_training_step_finite_and_grad():
    torch.manual_seed(0)
    m = _module()
    m.train()
    loss, out, labels = m._step(_batch(), training=True)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(out) >= {"clipwise_logits_long", "clipwise_pred_long"}
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_validation_step_updates_auroc():
    torch.manual_seed(0)
    m = _module(rf=False)
    m.eval()
    with torch.no_grad():
        _, out, labels = m._step(_batch(B=4), training=False)
        m.val_auroc.update(out["clipwise_pred_long"], labels.int())
    auc = m.val_auroc.compute()
    assert auc.numel() == 1  # scalar (may be NaN if a fold lacks positives)


def test_label_smoothing_mass_preserving():
    m = _module()
    t = torch.zeros(2, NUM_CLASS)
    t[0, 5] = 1.0
    t[1, 3] = 1.0
    t[1, 7] = 1.0
    s = m._smooth(t)
    torch.testing.assert_close(s.sum(dim=-1), t.sum(dim=-1))  # row mass preserved
    assert (s > 0).all()  # negatives lifted off zero


def test_label_smoothing_zero_disables():
    m = XCPretrainFinetuneModule(model=_model(), num_classes=NUM_CLASS, label_smoothing=0.0)
    t = (torch.rand(2, NUM_CLASS) > 0.9).float()
    torch.testing.assert_close(m._smooth(t), t)


def test_freeze_backbone():
    m = _module(freeze=True)
    assert all(not p.requires_grad for p in m.model.backbone.parameters())
    assert any(p.requires_grad for p in m.model.head.parameters())


def test_configure_optimizers_standalone():
    m = _module()
    cfg = m.configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.AdamW)
    assert "lr_scheduler" in cfg and cfg["lr_scheduler"]["interval"] == "step"
