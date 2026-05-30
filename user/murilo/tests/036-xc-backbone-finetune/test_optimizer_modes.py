"""Exp 036 Part 2 — flexible configure_optimizers (A1 uniform / A2 discriminative / A3 unfreeze).

All standalone (no Trainer): configure_optimizers uses the max_epochs*100 fallback for
total_steps. Defaults must reproduce exp 034's single-group cosine (back-compat).
"""

from unittest.mock import PropertyMock, patch

import pytest
import torch

from birdclef.lightning.xcpretrain_finetune import XCPretrainFinetuneModule
from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier

NUM_CLASS = 234


def _module(**kw):
    model = XCPretrainWaveClassifier(
        backbone="tf_efficientnetv2_s_in21k",
        pretrained=False,
        head_config={"p": 0.5, "num_class": NUM_CLASS, "train_period": 5.0, "infer_period": 5.0},
    )
    return XCPretrainFinetuneModule(model=model, num_classes=NUM_CLASS, max_epochs=5, **kw)


def test_uniform_default_single_group_cosine():
    m = _module()
    cfg = m.configure_optimizers()
    opt = cfg["optimizer"]
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == m.lr
    assert isinstance(cfg["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
    assert cfg["lr_scheduler"]["interval"] == "step"


def test_param_groups_disjoint_and_complete():
    m = _module()
    bb, other = m._param_groups()
    bb_ids, other_ids = {id(p) for p in bb}, {id(p) for p in other}
    assert bb_ids.isdisjoint(other_ids)
    assert len(bb) + len(other) == len(list(m.parameters()))
    assert bb_ids == {id(p) for p in m.model.backbone.parameters()}


def test_discriminative_two_groups():
    m = _module(optimizer_mode="discriminative", backbone_lr_mult=0.1)
    opt = m.configure_optimizers()["optimizer"]
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs[0] == pytest.approx(m.lr * 0.1)  # backbone
    assert lrs[1] == pytest.approx(m.lr)  # head


def test_warmup_builds_sequentiallr():
    m = _module(warmup_epochs=2)  # total_steps fallback = 5*100=500; warmup=2*100=200 < 500
    sched = m.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)


def test_a3_freezes_backbone_at_construction_but_keeps_in_optimizer():
    m = _module(unfreeze_backbone_epoch=2)
    assert all(not p.requires_grad for p in m.model.backbone.parameters())  # frozen now
    opt = m.configure_optimizers()["optimizer"]
    n_in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_total = sum(p.numel() for p in m.parameters())
    assert n_in_opt == n_total  # frozen backbone INCLUDED → updates on later unfreeze


def test_a3_unfreeze_hook_flips_requires_grad():
    m = _module(unfreeze_backbone_epoch=2)
    with patch.object(
        XCPretrainFinetuneModule, "current_epoch", new_callable=PropertyMock, return_value=2
    ):
        m.on_train_epoch_start()
    assert all(p.requires_grad for p in m.model.backbone.parameters())


def test_a3_unfreeze_hook_noop_before_epoch():
    m = _module(unfreeze_backbone_epoch=5)
    with patch.object(
        XCPretrainFinetuneModule, "current_epoch", new_callable=PropertyMock, return_value=2
    ):
        m.on_train_epoch_start()
    assert all(not p.requires_grad for p in m.model.backbone.parameters())  # still frozen


def test_freeze_and_unfreeze_mutually_exclusive():
    with pytest.raises(ValueError):
        _module(freeze_backbone=True, unfreeze_backbone_epoch=2)
