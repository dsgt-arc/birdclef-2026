"""GPU forward/backward pass tests for both conditions.

Validates shapes, loss computation, gradient flow, and memory on actual GPU.
These tests catch dtype mismatches, OOM, broken CUDA kernels, and
frozen-param gradient issues.

Run with:
    uv run pytest tests/test_forward_gpu.py -v
"""

import pytest
import torch

from ssamba_hf import MAMBA_SSM_AVAILABLE
from src.data import _compute_padding_mask
from src.pretrained import load_pretrained_backbone
from src.ssamba_classifier import SSAMBAClassifier
from tests.conftest import requires_cuda, requires_mamba_ssm, requires_pretrained


@requires_cuda
@requires_mamba_ssm
class TestForwardConditionA:
    """Condition A: 16x16, 496 tokens on GPU with CUDA Mamba kernels."""

    def test_forward_shape(self, config_16x16, num_classes):
        model = SSAMBAClassifier(config_16x16, num_classes=num_classes).cuda()
        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 496, dtype=torch.bool, device="cuda")

        with torch.no_grad():
            logits = model(x, padding_mask=mask)

        assert logits.shape == (2, num_classes)
        assert logits.isfinite().all()

    def test_forward_no_mask(self, config_16x16, num_classes):
        """Forward without padding mask (all tokens pooled)."""
        model = SSAMBAClassifier(config_16x16, num_classes=num_classes).cuda()
        x = torch.randn(2, 1, 128, 1000, device="cuda")

        with torch.no_grad():
            logits = model(x, padding_mask=None)

        assert logits.shape == (2, num_classes)
        assert logits.isfinite().all()

    def test_backward(self, config_16x16, num_classes):
        """Full forward + backward pass, checking gradients flow."""
        model = SSAMBAClassifier(config_16x16, num_classes=num_classes).cuda()
        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 496, dtype=torch.bool, device="cuda")
        labels = torch.zeros(2, num_classes, device="cuda")
        labels[:, 0] = 1.0

        logits = model(x, padding_mask=mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        assert loss.isfinite()
        # Check gradients exist on backbone and head
        assert model.backbone.patch_embed.proj.weight.grad is not None
        assert model.head[1].weight.grad is not None  # Linear layer in Sequential

    def test_bf16_mixed(self, config_16x16, num_classes):
        """Forward pass with bf16 autocast (matches training precision)."""
        model = SSAMBAClassifier(config_16x16, num_classes=num_classes).cuda()
        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 496, dtype=torch.bool, device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, padding_mask=mask)

        assert logits.isfinite().all()

    def test_half_padded_mask(self, config_16x16, num_classes):
        """Forward with 5s-padded-to-10s mask (val scenario)."""
        model = SSAMBAClassifier(config_16x16, num_classes=num_classes).cuda()
        x = torch.randn(2, 1, 128, 1000, device="cuda")

        mask = _compute_padding_mask(500, 1000, (16, 16)).unsqueeze(0).expand(2, -1).cuda()
        assert mask.shape == (2, 496)
        assert mask[0].sum() == 8 * 31  # 8 freq * 31 time patches

        with torch.no_grad():
            logits = model(x, padding_mask=mask)

        assert logits.shape == (2, num_classes)
        assert logits.isfinite().all()


@requires_cuda
@requires_mamba_ssm
class TestForwardConditionB:
    """Condition B: 128x1, 1000 tokens on GPU with CUDA Mamba kernels."""

    def test_forward_shape(self, config_128x1, num_classes):
        model = SSAMBAClassifier(config_128x1, num_classes=num_classes).cuda()
        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")

        with torch.no_grad():
            logits = model(x, padding_mask=mask)

        assert logits.shape == (2, num_classes)
        assert logits.isfinite().all()

    def test_backward(self, config_128x1, num_classes):
        model = SSAMBAClassifier(config_128x1, num_classes=num_classes).cuda()
        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")
        labels = torch.zeros(2, num_classes, device="cuda")
        labels[:, 0] = 1.0

        logits = model(x, padding_mask=mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        assert loss.isfinite()
        assert model.backbone.patch_embed.proj.weight.grad is not None

    def test_token_count(self, config_128x1):
        """128x1 on (128, 1000) → exactly 1000 tokens."""
        from ssamba_hf import SSAMBAModel

        backbone = SSAMBAModel(config_128x1)
        assert backbone.patch_embed.num_patches == 1000
        assert backbone.patch_embed.f_dim == 1
        assert backbone.patch_embed.t_dim == 1000


@requires_cuda
@requires_mamba_ssm
@requires_pretrained
class TestPretrainedForwardGPU:
    """End-to-end: load pre-trained weights, forward on GPU."""

    def test_pretrained_16x16_forward(self, config_16x16, pretrained_ckpt_path, num_classes):
        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)
        load_pretrained_backbone(classifier, pretrained_ckpt_path)
        classifier = classifier.cuda()

        x = torch.randn(4, 1, 128, 1000, device="cuda")
        mask = torch.ones(4, 496, dtype=torch.bool, device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = classifier(x, padding_mask=mask)

        assert logits.shape == (4, num_classes)
        assert logits.isfinite().all()

    def test_pretrained_128x1_forward(self, config_128x1, pretrained_ckpt_path, num_classes):
        classifier = SSAMBAClassifier(config_128x1, num_classes=num_classes)
        load_pretrained_backbone(classifier, pretrained_ckpt_path)
        classifier = classifier.cuda()

        x = torch.randn(4, 1, 128, 1000, device="cuda")
        mask = torch.ones(4, 1000, dtype=torch.bool, device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = classifier(x, padding_mask=mask)

        assert logits.shape == (4, num_classes)
        assert logits.isfinite().all()

    def test_pretrained_backward_grads(self, config_16x16, pretrained_ckpt_path, num_classes):
        """Verify gradients flow through pre-trained layers."""
        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)
        load_pretrained_backbone(classifier, pretrained_ckpt_path)
        classifier = classifier.cuda()

        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 496, dtype=torch.bool, device="cuda")
        labels = torch.zeros(2, num_classes, device="cuda")
        labels[:, 0] = 1.0

        logits = classifier(x, padding_mask=mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        # Check gradients on pre-trained layers
        layer0_mixer = list(classifier.backbone.layers.layers[0].mixer.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer0_mixer), (
            "No gradients in first Mamba layer"
        )


@requires_cuda
@requires_mamba_ssm
class TestFreezeUnfreeze:
    """Test backbone freeze/unfreeze logic."""

    def test_freeze_leaves_head_trainable(self, config_16x16, num_classes):
        from src.lightning_module import SSAMBALightningModule

        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)
        module = SSAMBALightningModule(
            model=classifier, num_classes=num_classes, freeze_epochs=5
        )
        module._freeze_backbone()

        # Head should be trainable
        for name, p in classifier.named_parameters():
            if "head." in name:
                assert p.requires_grad, f"{name} should be trainable"
            elif "patch_embed." in name or "pos_embed" in name or "cls_token" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_unfreeze_all_trainable(self, config_16x16, num_classes):
        from src.lightning_module import SSAMBALightningModule

        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes)
        module = SSAMBALightningModule(
            model=classifier, num_classes=num_classes, freeze_epochs=5
        )
        module._freeze_backbone()
        module._unfreeze_backbone()

        for name, p in classifier.named_parameters():
            assert p.requires_grad, f"{name} should be trainable after unfreeze"

    def test_frozen_backward_grads_head_only(self, config_16x16, num_classes):
        """With frozen backbone, gradients should only reach head + patch_embed."""
        classifier = SSAMBAClassifier(config_16x16, num_classes=num_classes).cuda()
        from src.lightning_module import SSAMBALightningModule

        module = SSAMBALightningModule(
            model=classifier, num_classes=num_classes, freeze_epochs=5
        )
        module._freeze_backbone()

        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 496, dtype=torch.bool, device="cuda")
        labels = torch.zeros(2, num_classes, device="cuda")

        logits = classifier(x, padding_mask=mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        # Head should have gradients
        assert classifier.head[1].weight.grad is not None
        # patch_embed should have gradients (not frozen)
        assert classifier.backbone.patch_embed.proj.weight.grad is not None

    @requires_pretrained
    def test_frozen_pretrained_backward(self, config_128x1, pretrained_ckpt_path, num_classes):
        """Condition B frozen: patch_embed (random init) should get gradients even
        though backbone Mamba layers are frozen."""
        classifier = SSAMBAClassifier(config_128x1, num_classes=num_classes)
        load_pretrained_backbone(classifier, pretrained_ckpt_path)
        classifier = classifier.cuda()

        from src.lightning_module import SSAMBALightningModule

        module = SSAMBALightningModule(
            model=classifier, num_classes=num_classes, freeze_epochs=5
        )
        module._freeze_backbone()

        x = torch.randn(2, 1, 128, 1000, device="cuda")
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")
        labels = torch.zeros(2, num_classes, device="cuda")

        logits = classifier(x, padding_mask=mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()

        # patch_embed should have gradients despite backbone frozen
        assert classifier.backbone.patch_embed.proj.weight.grad is not None
        grad_norm = classifier.backbone.patch_embed.proj.weight.grad.abs().sum()
        assert grad_norm > 0, "patch_embed should receive nonzero gradients"


@requires_cuda
@requires_mamba_ssm
class TestEnvironment:
    """Validate the GPU environment is correctly configured."""

    def test_cuda_available(self):
        assert torch.cuda.is_available()

    def test_cuda_device_name(self):
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}")
        assert len(name) > 0

    def test_mamba_ssm_import(self):
        import mamba_ssm
        print(f"mamba-ssm version: {mamba_ssm.__version__}")

    def test_causal_conv1d_import(self):
        import causal_conv1d
        print(f"causal-conv1d version: {causal_conv1d.__version__}")

    def test_torch_cuda_version(self):
        print(f"torch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")

    def test_bf16_support(self):
        """bf16 required for training precision."""
        assert torch.cuda.is_bf16_supported(), "bf16 not supported on this GPU"

    def test_cuda_kernel_functional(self):
        """Smoke test: create a small model with CUDA kernels and forward pass."""
        from ssamba_hf import SSAMBAConfig, SSAMBAModel

        config = SSAMBAConfig(
            embed_dim=192, depth=2,
            img_size=(128, 100), patch_size=(16, 16),
            use_cls_token=True, use_middle_cls_token=True,
            rms_norm=False,
        )
        model = SSAMBAModel(config).cuda()
        x = torch.randn(1, 1, 128, 100, device="cuda")

        with torch.no_grad():
            out = model.forward_features(x)

        assert out.shape == (1, 192)
        assert out.isfinite().all()
