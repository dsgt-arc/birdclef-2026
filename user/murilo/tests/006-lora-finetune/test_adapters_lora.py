"""Test adapters-library LoRA integration on EAT encoder.

These tests require the `adapters` package. Mark as slow since they load
the full EAT-base checkpoint from HuggingFace.

Run from user/murilo/ directory:
    pytest tests/006-lora-finetune/test_adapters_lora.py -v -m slow
"""

import pytest
import torch

adapters = pytest.importorskip("adapters", reason="adapters library not installed")

LORA_CONFIG_ATTN = {
    "lora": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["qkv", "proj"],
    }
}

LORA_CONFIG_ALL = {
    "lora": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["qkv", "proj", "fc1", "fc2"],
    }
}


def _make_lora_encoder(target_modules):
    """Helper: create a fresh EAT encoder with LoRA applied."""
    from birdclef.encoders.eat import EATEncoder
    from birdclef.lora.adapters import apply_adapters_lora

    config = {
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": target_modules,
        }
    }
    encoder = EATEncoder(precomputed_features=True)
    return apply_adapters_lora(encoder, config)


@pytest.fixture(scope="module")
def lora_encoder_attn():
    """EAT encoder with LoRA on attention layers (qkv, proj). Loaded once."""
    return _make_lora_encoder(["qkv", "proj"])


class TestAdaptersInit:
    """Test that adapters.init() + LoRA injection works correctly."""

    @pytest.mark.slow
    def test_init_does_not_crash(self, lora_encoder_attn):
        """adapters.init() + add_adapter should complete without errors."""
        assert lora_encoder_attn is not None

    @pytest.mark.slow
    def test_lora_parameters_injected(self, lora_encoder_attn):
        """LoRA parameters should appear in the model after init."""
        lora_params = [
            n for n, _ in lora_encoder_attn.model.named_parameters() if "lora" in n.lower()
        ]
        assert len(lora_params) > 0, "No LoRA parameters found after adapter injection"

    @pytest.mark.slow
    def test_trainable_params_fraction(self, lora_encoder_attn):
        """Only LoRA params should be trainable (~0.3-3% of total)."""
        trainable = sum(p.numel() for p in lora_encoder_attn.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_encoder_attn.model.parameters())
        pct = trainable / total * 100
        assert 0.1 < pct < 5.0, f"Unexpected trainable fraction: {pct:.2f}%"

    @pytest.mark.slow
    def test_backbone_params_frozen(self, lora_encoder_attn):
        """Original backbone parameters should be frozen after train_adapter()."""
        for name, param in lora_encoder_attn.model.named_parameters():
            if "lora" not in name.lower() and "adapter" not in name.lower():
                assert not param.requires_grad, f"Non-adapter param {name} is trainable"


class TestAdaptersForward:
    """Test forward pass with LoRA-adapted EAT encoder."""

    @pytest.mark.slow
    def test_forward_pass_shape(self, lora_encoder_attn):
        """Forward pass should produce (batch, 768) embeddings."""
        lora_encoder_attn.eval()
        dummy = torch.randn(2, 1, 512, 128)
        with torch.no_grad():
            out = lora_encoder_attn(dummy)
        assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"

    @pytest.mark.slow
    def test_forward_no_nan(self, lora_encoder_attn):
        """Output should not contain NaN or Inf."""
        lora_encoder_attn.eval()
        dummy = torch.randn(2, 1, 512, 128)
        with torch.no_grad():
            out = lora_encoder_attn(dummy)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    @pytest.mark.slow
    def test_gradients_flow_through_lora(self, lora_encoder_attn):
        """Gradients should flow through LoRA parameters during backward."""
        lora_encoder_attn.train()
        dummy = torch.randn(2, 1, 512, 128)
        out = lora_encoder_attn(dummy)
        loss = out.sum()
        loss.backward()

        lora_with_grad = 0
        for name, param in lora_encoder_attn.model.named_parameters():
            if "lora" in name.lower() and param.grad is not None:
                lora_with_grad += 1

        assert lora_with_grad > 0, "No LoRA parameters received gradients"

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_gpu(self, lora_encoder_attn):
        """Forward pass should work on GPU."""
        device = torch.device("cuda:0")
        lora_encoder_attn.to(device)
        lora_encoder_attn.eval()

        dummy = torch.randn(2, 1, 512, 128, device=device)
        with torch.no_grad():
            out = lora_encoder_attn(dummy)

        assert out.device.type == "cuda"
        assert out.shape == (2, 768)

        # Move back to CPU for other tests
        lora_encoder_attn.to("cpu")


class TestAdaptersAllTargets:
    """Test LoRA on all linear layers (qkv + proj + fc1 + fc2)."""

    @pytest.mark.slow
    def test_all_targets_init(self):
        """LoRA on all targets should init without errors."""
        encoder = _make_lora_encoder(["qkv", "proj", "fc1", "fc2"])
        lora_params = [n for n, _ in encoder.model.named_parameters() if "lora" in n.lower()]
        assert len(lora_params) > 0

    @pytest.mark.slow
    def test_all_targets_more_params_than_attn_only(self):
        """LoRA on all targets should have more trainable params than attn-only."""
        enc_attn = _make_lora_encoder(["qkv", "proj"])
        enc_all = _make_lora_encoder(["qkv", "proj", "fc1", "fc2"])

        trainable_attn = sum(p.numel() for p in enc_attn.model.parameters() if p.requires_grad)
        trainable_all = sum(p.numel() for p in enc_all.model.parameters() if p.requires_grad)

        assert trainable_all > trainable_attn, (
            f"all-targets ({trainable_all:,}) should have more trainable params "
            f"than attn-only ({trainable_attn:,})"
        )
