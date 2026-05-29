"""Phase H unit tests for the Perch-KD branch of XCPretrainFinetuneModule.

Exercises `_step_kd` + the `kd_detach_cls_features` gradient routing WITHOUT
onnxruntime — PerchTeacher needs the 394 MB Perch ONNX + CUDA, so we inject a
fake teacher returning a random (B, 1536) clip embedding. train.py wires the
real teacher from the config; the math validated here is what matters:

- total = cls + alpha_distill · MSE(distill_head(feat), perch_emb)
- detach=True  → cls_loss alone produces NO backbone gradient (Tucker-faithful)
- detach=False → cls_loss alone DOES reach the backbone (joint fine-tune)
"""

import torch

from birdclef.lightning.sed_finetune import DistillHead
from birdclef.lightning.xcpretrain_finetune import XCPretrainFinetuneModule
from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier

NUM_CLASS = 234
EMBED_DIM = 1536


class _FakeTeacher:
    """Stand-in for PerchTeacher: returns a random (B, EMBED_DIM) clip embedding."""

    def embed(self, wave: torch.Tensor) -> torch.Tensor:
        return torch.randn(wave.shape[0], EMBED_DIM, device=wave.device)


def _model():
    return XCPretrainWaveClassifier(
        backbone="tf_efficientnetv2_s_in21k",
        pretrained=False,
        head_config={"p": 0.5, "num_class": NUM_CLASS, "train_period": 5.0, "infer_period": 5.0},
    )


def _kd_module(detach: bool):
    """Build a module with KD enabled via injection (no real ONNX teacher)."""
    m = XCPretrainFinetuneModule(model=_model(), num_classes=NUM_CLASS, max_epochs=5)
    backbone_dim = m.model.backbone.feature_info.channels()[-1]
    m.use_perch_distill = True
    m.alpha_distill = 1.0
    m.kd_detach_cls_features = detach
    m.distill_head = DistillHead(backbone_dim, EMBED_DIM)
    m.perch_teacher = _FakeTeacher()
    m.train()
    return m


def _batch(B=2):
    wav = torch.randn(B, 1, 160_000)
    labels = (torch.rand(B, NUM_CLASS) > 0.9).float()
    return (wav, labels, torch.ones(B, NUM_CLASS), torch.ones(B, NUM_CLASS))


def _backbone_has_grad(m) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.model.backbone.parameters())


def test_kd_step_finite_and_total():
    torch.manual_seed(0)
    m = _kd_module(detach=True)
    loss, out, _labels, cls_loss, dist_loss = m._step_kd(_batch())
    for t in (loss, cls_loss, dist_loss):
        assert t.ndim == 0 and torch.isfinite(t)
    assert torch.allclose(loss, cls_loss + m.alpha_distill * dist_loss)
    assert set(out) >= {"clipwise_logits_long", "clipwise_pred_long"}


def test_kd_grads_flow_to_backbone_and_distill_head():
    torch.manual_seed(0)
    m = _kd_module(detach=True)
    loss, *_ = m._step_kd(_batch())
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in m.distill_head.parameters()
    ), "distill_head must receive gradient"
    assert _backbone_has_grad(m), "backbone must receive gradient (via the Perch MSE branch)"


def test_detach_true_cls_loss_does_not_touch_backbone():
    torch.manual_seed(0)
    m = _kd_module(detach=True)
    _, _, _, cls_loss, _ = m._step_kd(_batch())
    m.zero_grad()
    cls_loss.backward()
    assert not _backbone_has_grad(m), "detach=True → SED head must not move the backbone"


def test_detach_false_cls_loss_reaches_backbone():
    torch.manual_seed(0)
    m = _kd_module(detach=False)
    _, _, _, cls_loss, _ = m._step_kd(_batch())
    m.zero_grad()
    cls_loss.backward()
    assert _backbone_has_grad(m), "detach=False → task gradient must fine-tune the backbone"


def test_kd_disabled_by_default():
    """Default module has no KD state and the plain _step path still works."""
    m = XCPretrainFinetuneModule(model=_model(), num_classes=NUM_CLASS, max_epochs=5)
    assert m.use_perch_distill is False
    assert not hasattr(m, "distill_head")
    m.train()
    loss, out, _ = m._step(_batch(), training=True)
    assert loss.ndim == 0 and torch.isfinite(loss)
