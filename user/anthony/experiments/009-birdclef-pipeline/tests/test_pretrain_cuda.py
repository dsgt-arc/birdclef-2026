"""CUDA tests for MPG pretraining model with span masking."""

import pytest
import torch

from birdclef_009.models.pretrain import SSAMBAPretrainModel
from birdclef_009.training.pretrain import PretrainLightningModule

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture
def pretrain_model_gpu(tiny_config, mel_spec_layer):
    """Build pretrain model on GPU with span masking."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        mask_mode="span",
        mask_span_min=5,
        mask_span_max=15,
        grid_jitter=0,  # Disable for deterministic testing
    )
    return model.cuda()


def test_forward_on_gpu(pretrain_model_gpu):
    """Forward pass runs on GPU and returns correct types."""
    waveforms = torch.randn(2, 320000, device="cuda")
    out = pretrain_model_gpu(waveforms)

    assert out["loss_mpg"].device.type == "cuda"
    assert torch.isfinite(out["loss_mpg"])
    # JiTTER disabled by default
    assert out["loss_jitter"].item() == 0.0
    # Cross-metrics present
    assert torch.isfinite(out["monitor_mse"])
    assert torch.isfinite(out["monitor_l1"])


def test_backward_on_gpu(pretrain_model_gpu):
    """Gradients flow through MPG model on GPU."""
    waveforms = torch.randn(2, 320000, device="cuda")
    out = pretrain_model_gpu(waveforms)
    loss = out["loss_mpg"] + out["loss_jitter"]
    loss.backward()

    # Check at least some parameters have gradients
    grads = [p.grad for p in pretrain_model_gpu.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"

    # Check no NaN gradients
    for g in grads:
        assert torch.isfinite(g).all(), "NaN or Inf gradient detected"


def test_bf16_forward(pretrain_model_gpu):
    """Forward pass works with bf16 mixed precision."""
    waveforms = torch.randn(2, 320000, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pretrain_model_gpu(waveforms)

    assert torch.isfinite(out["loss_mpg"])


def test_bf16_backward(pretrain_model_gpu):
    """Gradients flow correctly under bf16 mixed precision."""
    waveforms = torch.randn(2, 320000, device="cuda")
    scaler = torch.amp.GradScaler("cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pretrain_model_gpu(waveforms)
        loss = out["loss_mpg"] + out["loss_jitter"]

    scaler.scale(loss).backward()

    grads = [p.grad for p in pretrain_model_gpu.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_mask_embed_gets_gradient(pretrain_model_gpu):
    """The learnable mask embedding should receive gradients."""
    waveforms = torch.randn(2, 320000, device="cuda")
    out = pretrain_model_gpu(waveforms)
    loss = out["loss_mpg"] + out["loss_jitter"]
    loss.backward()

    assert pretrain_model_gpu.mask_embed.grad is not None
    assert torch.isfinite(pretrain_model_gpu.mask_embed.grad).all()


def test_mpg_head_gets_gradient(pretrain_model_gpu):
    """MPG prediction head should receive gradients."""
    waveforms = torch.randn(2, 320000, device="cuda")
    out = pretrain_model_gpu(waveforms)
    loss = out["loss_mpg"] + out["loss_jitter"]
    loss.backward()

    for name, param in pretrain_model_gpu.gpredlayer.named_parameters():
        assert param.grad is not None, f"gpredlayer.{name} has no gradient"


def test_span_mask_on_gpu(pretrain_model_gpu):
    """Span mask generation works on GPU."""
    mask = pretrain_model_gpu._generate_span_mask(1000, torch.device("cuda"))
    assert mask.device.type == "cuda"
    assert mask.dtype == torch.bool
    # Should mask approximately 50% (mask_ratio=0.5)
    ratio = mask.sum().item() / 1000
    assert ratio >= 0.5, f"Ratio {ratio:.3f} below target"


def test_lightning_training_step(tiny_config, mel_spec_layer):
    """PretrainLightningModule training_step runs on GPU."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        mask_mode="span",
        grid_jitter=0,
    )
    lightning_module = PretrainLightningModule(
        model=model, lr=1e-4, max_epochs=10, jitter_weight=0.0,
    )
    lightning_module = lightning_module.cuda()

    batch = {"waveform": torch.randn(2, 320000, device="cuda")}
    loss = lightning_module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss.device.type == "cuda"


def test_optimizer_step(tiny_config, mel_spec_layer):
    """Full optimizer step works on GPU (forward + backward + update)."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        mask_mode="span",
        grid_jitter=0,
    )
    lightning_module = PretrainLightningModule(
        model=model, lr=1e-4, max_epochs=10, jitter_weight=0.0,
    )
    lightning_module = lightning_module.cuda()

    opt_config = lightning_module.configure_optimizers()
    optimizer = opt_config["optimizer"]

    # Capture initial param snapshot
    initial_params = {
        name: p.clone() for name, p in model.named_parameters() if p.requires_grad
    }

    batch = {"waveform": torch.randn(4, 320000, device="cuda")}
    loss = lightning_module.training_step(batch, 0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Verify at least some parameters changed
    changed = 0
    for name, p in model.named_parameters():
        if name in initial_params and not torch.equal(p, initial_params[name]):
            changed += 1
    assert changed > 0, "No parameters updated after optimizer step"
