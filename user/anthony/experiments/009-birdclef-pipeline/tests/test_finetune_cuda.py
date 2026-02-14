"""CUDA tests for clip-level fine-tuning model."""

import pytest
import torch

from birdclef_009.models.finetune import SSAMBAClassifier
from birdclef_009.training.finetune import FinetuneLightningModule
from birdclef_009.utils.losses import FocalBCELoss

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

NUM_CLASSES = 10


@pytest.fixture
def finetune_model_gpu(tiny_config, mel_spec_layer):
    """Build finetune model on GPU."""
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=NUM_CLASSES,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        head_dropout=0.0,
    )
    return model.cuda()


def test_forward_on_gpu(finetune_model_gpu):
    """Classifier forward pass runs on GPU."""
    B = 2
    waveforms = torch.randn(B, 320000, device="cuda")
    clip_logits = finetune_model_gpu(waveforms)

    assert clip_logits.shape == (B, NUM_CLASSES)
    assert clip_logits.device.type == "cuda"
    assert torch.isfinite(clip_logits).all()


def test_forward_with_padding_mask_gpu(finetune_model_gpu):
    """Padding mask works correctly on GPU."""
    B = 2
    waveforms = torch.randn(B, 320000, device="cuda")
    padding_mask = torch.ones(B, 1000, dtype=torch.bool, device="cuda")
    padding_mask[0, 500:] = False

    clip_logits = finetune_model_gpu(waveforms, padding_mask=padding_mask)

    assert clip_logits.shape == (B, NUM_CLASSES)
    assert torch.isfinite(clip_logits).all()


def test_backward_on_gpu(finetune_model_gpu):
    """Gradients flow through classifier on GPU."""
    waveforms = torch.randn(2, 320000, device="cuda")
    clip_logits = finetune_model_gpu(waveforms)

    target = torch.zeros(2, NUM_CLASSES, device="cuda")
    target[0, 0] = 1.0
    target[1, 3] = 1.0

    loss_fn = FocalBCELoss(gamma=2.0, from_logits=True)
    loss = loss_fn(clip_logits, target)
    loss.backward()

    grads = [p.grad for p in finetune_model_gpu.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()


def test_bf16_forward(finetune_model_gpu):
    """Classifier forward pass works with bf16 mixed precision."""
    waveforms = torch.randn(2, 320000, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clip_logits = finetune_model_gpu(waveforms)

    assert torch.isfinite(clip_logits).all()


def test_bf16_backward(finetune_model_gpu):
    """Gradients flow correctly under bf16."""
    waveforms = torch.randn(2, 320000, device="cuda")
    target = torch.zeros(2, NUM_CLASSES, device="cuda")
    target[0, 0] = 1.0

    with torch.autocast("cuda", dtype=torch.bfloat16):
        clip_logits = finetune_model_gpu(waveforms)
        loss_fn = FocalBCELoss(gamma=2.0, from_logits=True)
        loss = loss_fn(clip_logits, target)

    loss.backward()
    grads = [p.grad for p in finetune_model_gpu.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_head_gets_gradient(finetune_model_gpu):
    """Classification head should receive gradients."""
    waveforms = torch.randn(2, 320000, device="cuda")
    clip_logits = finetune_model_gpu(waveforms)

    clip_target = torch.zeros(2, NUM_CLASSES, device="cuda")
    clip_target[0, 0] = 1.0

    loss_fn = FocalBCELoss(gamma=2.0, from_logits=True)
    loss = loss_fn(clip_logits, clip_target)
    loss.backward()

    for name, param in finetune_model_gpu.head.named_parameters():
        assert param.grad is not None, f"head.{name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"head.{name} has NaN gradient"


def test_lightning_training_step(tiny_config, mel_spec_layer):
    """FinetuneLightningModule training_step runs on GPU."""
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=NUM_CLASSES,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        head_dropout=0.0,
    )
    lightning_module = FinetuneLightningModule(
        model=model, num_classes=NUM_CLASSES,
        lr=1e-4, max_epochs=10, gamma=2.0,
    )
    lightning_module = lightning_module.cuda()

    B = 2
    batch = {
        "waveform": torch.randn(B, 320000, device="cuda"),
        "clip_labels": torch.zeros(B, NUM_CLASSES, device="cuda"),
        "padding_mask": torch.ones(B, 1000, dtype=torch.bool, device="cuda"),
    }
    batch["clip_labels"][0, 0] = 1.0

    loss = lightning_module.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_lightning_validation_step(tiny_config, mel_spec_layer):
    """FinetuneLightningModule validation_step runs on GPU."""
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=NUM_CLASSES,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
    )
    lightning_module = FinetuneLightningModule(
        model=model, num_classes=NUM_CLASSES,
        lr=1e-4, max_epochs=10,
    )
    lightning_module = lightning_module.cuda()

    B = 2
    batch = {
        "waveform": torch.randn(B, 320000, device="cuda"),
        "clip_labels": torch.zeros(B, NUM_CLASSES, device="cuda"),
        "padding_mask": torch.ones(B, 1000, dtype=torch.bool, device="cuda"),
    }
    batch["clip_labels"][0, 0] = 1.0

    loss = lightning_module.validation_step(batch, 0)
    assert torch.isfinite(loss)


def test_backbone_freeze_unfreeze(tiny_config, mel_spec_layer):
    """Backbone freeze/unfreeze works correctly on GPU."""
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=NUM_CLASSES,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
    )
    lightning_module = FinetuneLightningModule(
        model=model, num_classes=NUM_CLASSES,
        lr=1e-4, max_epochs=10, freeze_epochs=2,
    )
    lightning_module = lightning_module.cuda()

    # Simulate epoch 0 — should freeze
    lightning_module._current_fx_name = "on_train_epoch_start"
    lightning_module.on_train_epoch_start()

    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert frozen > 0, "No parameters frozen"
    assert trainable > 0, "All parameters frozen (head should be trainable)"

    # Verify training still works with frozen backbone
    B = 2
    batch = {
        "waveform": torch.randn(B, 320000, device="cuda"),
        "clip_labels": torch.zeros(B, NUM_CLASSES, device="cuda"),
        "padding_mask": torch.ones(B, 1000, dtype=torch.bool, device="cuda"),
    }
    batch["clip_labels"][0, 0] = 1.0

    loss = lightning_module.training_step(batch, 0)
    loss.backward()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None or (p.grad == 0).all(), f"Frozen param {name} has gradient"


def test_optimizer_step_updates_params(tiny_config, mel_spec_layer):
    """Full optimizer step works on GPU."""
    model = SSAMBAClassifier(
        config=tiny_config,
        num_classes=NUM_CLASSES,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        head_dropout=0.0,
    )
    lightning_module = FinetuneLightningModule(
        model=model, num_classes=NUM_CLASSES,
        lr=1e-3, max_epochs=10,
    )
    lightning_module = lightning_module.cuda()

    opt_config = lightning_module.configure_optimizers()
    optimizer = opt_config["optimizer"]

    initial_params = {
        name: p.clone() for name, p in model.named_parameters() if p.requires_grad
    }

    B = 4
    batch = {
        "waveform": torch.randn(B, 320000, device="cuda"),
        "clip_labels": torch.zeros(B, NUM_CLASSES, device="cuda"),
        "padding_mask": torch.ones(B, 1000, dtype=torch.bool, device="cuda"),
    }
    for i in range(B):
        batch["clip_labels"][i, i % NUM_CLASSES] = 1.0

    loss = lightning_module.training_step(batch, 0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    changed = 0
    for name, p in model.named_parameters():
        if name in initial_params and not torch.equal(p, initial_params[name]):
            changed += 1
    assert changed > 0, "No parameters updated after optimizer step"


def test_forward_windows_gpu(finetune_model_gpu):
    """forward_windows works on GPU."""
    waveforms = torch.randn(1, 320000, device="cuda")
    window_logits = finetune_model_gpu.forward_windows(waveforms, window_size=500)

    assert window_logits.shape == (1, 2, NUM_CLASSES)
    assert window_logits.device.type == "cuda"
    assert torch.isfinite(window_logits).all()
