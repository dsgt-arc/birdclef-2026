"""Exp 036 Part 1 — full metric suite + logit_std on XCPretrainFinetuneModule.

Verifies the exp-032 metric suite (cmap/auroc/f1/top1/top5 + n_evaluable) was wired
onto both train+val splits (additively — exp 034's `self.val_auroc` name preserved),
computing clip-only from `clipwise_pred_long`/`clipwise_logits_long`. No Trainer:
calls `_step` + `_update_metrics` directly.
"""

import torch

from birdclef.lightning.xcpretrain_finetune import XCPretrainFinetuneModule
from birdclef.metrics import TuckerMacroAP, TuckerMacroAUC
from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier

NUM_CLASS = 234
_METRICS = ("cmap", "auroc", "f1_macro", "f1_micro", "top1", "top5")


def _module():
    model = XCPretrainWaveClassifier(
        backbone="tf_efficientnetv2_s_in21k",
        pretrained=False,
        head_config={"p": 0.5, "num_class": NUM_CLASS, "train_period": 5.0, "infer_period": 5.0},
    )
    return XCPretrainFinetuneModule(model=model, num_classes=NUM_CLASS, max_epochs=5)


def _batch(B=4):
    wav = torch.randn(B, 1, 160_000)
    labels = (torch.rand(B, NUM_CLASS) > 0.9).float()
    return (wav, labels, torch.ones(B, NUM_CLASS), torch.ones(B, NUM_CLASS))


def test_metric_attrs_exist_both_splits():
    m = _module()
    for split in ("train", "val"):
        for name in _METRICS:
            assert hasattr(m, f"{split}_{name}"), f"missing {split}_{name}"
    # exp 034 back-compat: the lone `val_auroc` name is preserved + is a TuckerMacroAUC.
    assert isinstance(m.val_auroc, TuckerMacroAUC)
    assert isinstance(m.val_cmap, TuckerMacroAP)


def test_update_and_compute_scalar():
    torch.manual_seed(0)
    m = _module()
    m.eval()
    with torch.no_grad():
        _, out, labels = m._step(_batch(), training=False)
        m._update_metrics("val", out, labels)
    for name in _METRICS:
        val = getattr(m, f"val_{name}").compute()
        assert val.numel() == 1, f"val_{name} not scalar"  # may be NaN if no evaluable classes
    assert isinstance(m.val_auroc.n_evaluable(), int)


def test_train_metrics_update_and_logit_std():
    torch.manual_seed(0)
    m = _module()
    m.train()
    loss, out, labels = m._step(_batch(), training=True)
    m._update_metrics("train", out, labels)
    for name in _METRICS:
        assert getattr(m, f"train_{name}").compute().numel() == 1
    # logit_std (hengck23 overfit proxy) is finite
    assert torch.isfinite(out["clipwise_logits_long"].std())


def test_clip_only_aggregation_preserved():
    """This module must read clip-only outputs (no clip+frame blend)."""
    m = _module()
    m.eval()
    with torch.no_grad():
        _, out, _ = m._step(_batch(B=2), training=False)
    # clipwise_pred_long is a probability in [0,1]; metrics consume it directly.
    p = out["clipwise_pred_long"]
    assert p.shape == (2, NUM_CLASS)
    assert (p >= 0).all() and (p <= 1).all()
