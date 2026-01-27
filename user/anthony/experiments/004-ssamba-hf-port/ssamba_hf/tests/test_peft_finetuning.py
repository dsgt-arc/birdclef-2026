"""Tests for PEFT/LoRA fine-tuning integration with SSAMBA.

These tests verify:
1. LoRA layers are correctly injected into in_proj/out_proj modules
2. Parameter counting (trainable vs frozen) is correct
3. Gradients flow through LoRA layers during training
4. Forward pass works with LoRA adapters
5. Adapter weights can be saved and loaded
6. Memory efficiency features (mixed precision, gradient checkpointing) work
7. End-to-end training simulation passes
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ssamba_hf import SSAMBAConfig, SSAMBAModel

# Check PEFT availability
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

# Skip marker for tests requiring peft
requires_peft = pytest.mark.skipif(
    not PEFT_AVAILABLE,
    reason="peft not installed",
)


@pytest.fixture
def sample_spectrogram():
    """Create a reproducible test spectrogram."""
    rng = np.random.default_rng(42)
    spec = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    return torch.from_numpy(spec)


@pytest.fixture
def batch_spectrogram():
    """Create a batch of test spectrograms."""
    rng = np.random.default_rng(42)
    spec = rng.standard_normal((4, 1024, 128)).astype(np.float32)
    return torch.from_numpy(spec)


@pytest.fixture
def small_config():
    """Create a small model config for testing."""
    return SSAMBAConfig.small(use_cuda_kernels=False)


@pytest.fixture
def lora_config():
    """Standard LoRA config targeting in_proj and out_proj."""
    if not PEFT_AVAILABLE:
        return None
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["in_proj", "out_proj"],
        bias="none",
    )


@pytest.fixture
def model_with_lora(small_config, lora_config):
    """Create a model with LoRA adapters applied."""
    if not PEFT_AVAILABLE:
        return None
    model = SSAMBAModel(small_config)
    return get_peft_model(model, lora_config)


class TestLoRAInjection:
    """Test that LoRA layers are correctly injected."""

    @requires_peft
    def test_lora_modules_injected(self, model_with_lora, small_config):
        """LoRA layers should be injected into target modules."""
        # Check that we have lora_A and lora_B in the model
        lora_params = [n for n, _ in model_with_lora.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0, "No LoRA parameters found"

    @requires_peft
    def test_lora_injected_in_all_layers(self, model_with_lora, small_config):
        """LoRA should be injected in all Mamba layers."""
        # Count unique layer indices with LoRA params
        layer_indices = set()
        for name, _ in model_with_lora.named_parameters():
            if "lora_" in name and "layers.layers." in name:
                # Extract layer index from name like "base_model.model.layers.layers.0.mixer.in_proj.lora_A"
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts) and parts[i + 1] == "layers":
                        layer_idx = int(parts[i + 2])
                        layer_indices.add(layer_idx)
                        break

        assert len(layer_indices) == small_config.depth, (
            f"Expected LoRA in {small_config.depth} layers, found {len(layer_indices)}"
        )

    @requires_peft
    def test_in_proj_has_lora(self, model_with_lora):
        """in_proj modules should have LoRA adapters."""
        in_proj_lora = [n for n, _ in model_with_lora.named_parameters()
                       if "in_proj" in n and "lora_" in n]
        assert len(in_proj_lora) > 0, "No LoRA on in_proj"

    @requires_peft
    def test_out_proj_has_lora(self, model_with_lora):
        """out_proj modules should have LoRA adapters."""
        out_proj_lora = [n for n, _ in model_with_lora.named_parameters()
                        if "out_proj" in n and "lora_" in n]
        assert len(out_proj_lora) > 0, "No LoRA on out_proj"


class TestParameterCounting:
    """Test trainable vs frozen parameter counting."""

    @requires_peft
    def test_base_model_params_frozen(self, model_with_lora):
        """Base model parameters should be frozen."""
        frozen_params = sum(
            p.numel() for p in model_with_lora.parameters() if not p.requires_grad
        )
        assert frozen_params > 0, "Expected some frozen parameters"

    @requires_peft
    def test_lora_params_trainable(self, model_with_lora):
        """LoRA parameters should be trainable."""
        trainable_params = sum(
            p.numel() for p in model_with_lora.parameters() if p.requires_grad
        )
        assert trainable_params > 0, "Expected some trainable parameters"

    @requires_peft
    def test_trainable_ratio(self, model_with_lora):
        """Trainable parameters should be a small fraction of total."""
        trainable = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model_with_lora.parameters())
        ratio = trainable / total

        # LoRA with r=8 on in_proj/out_proj should be ~2-5% of total
        assert 0.01 < ratio < 0.10, f"Trainable ratio {ratio:.2%} outside expected range"

    @requires_peft
    def test_expected_trainable_count(self, model_with_lora, small_config):
        """Verify trainable parameter count is in expected range."""
        trainable = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)

        # For small config with r=8:
        # Each layer: in_proj (2*d_inner x r + r x d_model) + out_proj (d_model x r + r x d_inner)
        # d_inner = 384 * 2 = 768
        # in_proj: 384 -> 768*2 = 1536, LoRA A: 384*8=3072, LoRA B: 8*1536=12288
        # out_proj: 768 -> 384, LoRA A: 768*8=6144, LoRA B: 8*384=3072
        # Per layer: ~24576 params
        # 24 layers: ~589,824 params

        # Allow some variance for PEFT implementation details
        assert 400_000 < trainable < 800_000, (
            f"Trainable params {trainable} outside expected range (400K-800K)"
        )


class TestGradientFlow:
    """Test gradient flow through LoRA layers."""

    @requires_peft
    def test_gradients_computed_for_lora(self, model_with_lora, sample_spectrogram):
        """Gradients should be computed for LoRA parameters."""
        model_with_lora.train()

        # Forward pass
        output = model_with_lora(sample_spectrogram)

        # Backward pass
        loss = output.mean()
        loss.backward()

        # Check gradients exist for LoRA params
        lora_params_with_grad = [
            n for n, p in model_with_lora.named_parameters()
            if "lora_" in n and p.grad is not None
        ]
        assert len(lora_params_with_grad) > 0, "No gradients for LoRA parameters"

    @requires_peft
    def test_no_gradients_for_frozen(self, model_with_lora, sample_spectrogram):
        """Frozen parameters should not have gradients."""
        model_with_lora.train()

        output = model_with_lora(sample_spectrogram)
        loss = output.mean()
        loss.backward()

        # Check no gradients for frozen params
        for name, param in model_with_lora.named_parameters():
            if "lora_" not in name and not param.requires_grad:
                assert param.grad is None, f"Frozen param {name} has gradient"

    @requires_peft
    def test_gradients_nonzero(self, model_with_lora, sample_spectrogram):
        """LoRA gradients should be non-zero."""
        model_with_lora.train()

        output = model_with_lora(sample_spectrogram)
        loss = output.mean()
        loss.backward()

        nonzero_grads = 0
        for name, param in model_with_lora.named_parameters():
            if "lora_" in name and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    nonzero_grads += 1

        assert nonzero_grads > 0, "All LoRA gradients are zero"

    @requires_peft
    def test_optimizer_step_updates_lora(self, model_with_lora, sample_spectrogram):
        """Optimizer step should update LoRA parameters."""
        model_with_lora.train()

        # Get initial values
        initial_values = {}
        for name, param in model_with_lora.named_parameters():
            if "lora_A" in name and param.requires_grad:
                initial_values[name] = param.clone().detach()

        # Training step
        optimizer = torch.optim.Adam(model_with_lora.parameters(), lr=0.01)
        output = model_with_lora(sample_spectrogram)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # Check values changed
        changed = 0
        for name, param in model_with_lora.named_parameters():
            if name in initial_values:
                if not torch.allclose(initial_values[name], param):
                    changed += 1

        assert changed > 0, "No LoRA parameters changed after optimizer step"


class TestForwardPass:
    """Test forward pass with LoRA adapters."""

    @requires_peft
    def test_forward_produces_output(self, model_with_lora, sample_spectrogram):
        """Forward pass should produce valid output."""
        model_with_lora.eval()

        with torch.no_grad():
            output = model_with_lora(sample_spectrogram)

        assert output.shape == (1, 384), f"Unexpected output shape: {output.shape}"

    @requires_peft
    def test_forward_no_nan(self, model_with_lora, sample_spectrogram):
        """Forward pass should not produce NaN values."""
        model_with_lora.eval()

        with torch.no_grad():
            output = model_with_lora(sample_spectrogram)

        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"

    @requires_peft
    def test_forward_deterministic(self, model_with_lora, sample_spectrogram):
        """Forward pass should be deterministic in eval mode."""
        model_with_lora.eval()

        with torch.no_grad():
            out1 = model_with_lora(sample_spectrogram)
            out2 = model_with_lora(sample_spectrogram)

        assert torch.allclose(out1, out2), "Non-deterministic output"

    @requires_peft
    def test_batch_forward(self, model_with_lora, batch_spectrogram):
        """Forward pass should work with batched input."""
        model_with_lora.eval()

        with torch.no_grad():
            output = model_with_lora(batch_spectrogram)

        assert output.shape == (4, 384), f"Unexpected batch output shape: {output.shape}"

    @requires_peft
    def test_train_mode_forward(self, model_with_lora, sample_spectrogram):
        """Forward pass should work in train mode (with dropout)."""
        model_with_lora.train()

        output = model_with_lora(sample_spectrogram)

        assert output.shape == (1, 384)
        assert not torch.isnan(output).any()


class TestAdapterPersistence:
    """Test saving and loading LoRA adapter weights."""

    @requires_peft
    def test_save_adapter(self, model_with_lora, tmp_path, sample_spectrogram):
        """Adapter should be saveable."""
        save_path = tmp_path / "adapter"
        model_with_lora.save_pretrained(save_path)

        # Check files exist
        assert (save_path / "adapter_config.json").exists()
        assert (save_path / "adapter_model.safetensors").exists() or \
               (save_path / "adapter_model.bin").exists()

    @requires_peft
    def test_load_adapter(self, small_config, lora_config, tmp_path, sample_spectrogram):
        """Adapter should be loadable."""
        # Create and save model
        model = SSAMBAModel(small_config)
        peft_model = get_peft_model(model, lora_config)

        # Get output before save
        peft_model.eval()
        with torch.no_grad():
            original_output = peft_model(sample_spectrogram).clone()

        save_path = tmp_path / "adapter"
        peft_model.save_pretrained(save_path)

        # Load into new model
        base_model = SSAMBAModel(small_config)
        loaded_model = PeftModel.from_pretrained(base_model, save_path)

        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(sample_spectrogram)

        assert torch.allclose(original_output, loaded_output, atol=1e-6), (
            "Loaded model produces different output"
        )

    @requires_peft
    def test_adapter_size_reasonable(self, model_with_lora, tmp_path):
        """Saved adapter should be much smaller than full model."""
        import os

        save_path = tmp_path / "adapter"
        model_with_lora.save_pretrained(save_path)

        # Get adapter size
        adapter_size = 0
        for f in save_path.iterdir():
            if f.is_file():
                adapter_size += f.stat().st_size

        # Adapter should be < 10MB (full model is ~100MB)
        assert adapter_size < 10 * 1024 * 1024, (
            f"Adapter size {adapter_size / 1024 / 1024:.1f}MB is too large"
        )

    @requires_peft
    def test_merge_adapter(self, model_with_lora, sample_spectrogram):
        """Adapter should be mergeable into base model."""
        model_with_lora.eval()

        with torch.no_grad():
            output_before = model_with_lora(sample_spectrogram).clone()

        # Merge adapter
        merged_model = model_with_lora.merge_and_unload()

        with torch.no_grad():
            output_after = merged_model(sample_spectrogram)

        assert torch.allclose(output_before, output_after, atol=1e-5), (
            "Merged model produces different output"
        )


class TestMemoryEfficiency:
    """Test memory efficiency features."""

    @requires_peft
    def test_mixed_precision_forward(self, model_with_lora, sample_spectrogram):
        """Forward pass should work with mixed precision."""
        model_with_lora.eval()

        # Skip if CUDA not available for autocast
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for autocast test")

        model_with_lora = model_with_lora.cuda()
        spec = sample_spectrogram.cuda()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                output = model_with_lora(spec)

        assert not torch.isnan(output).any()
        assert output.shape == (1, 384)

    @requires_peft
    def test_gradient_checkpointing_compatible(self, small_config, lora_config, sample_spectrogram):
        """Model should be compatible with gradient checkpointing."""
        # Note: This tests that the model doesn't break with checkpointing
        # Actual memory savings would need profiling
        model = SSAMBAModel(small_config)
        peft_model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing if available
        if hasattr(peft_model, "gradient_checkpointing_enable"):
            peft_model.gradient_checkpointing_enable()

        peft_model.train()
        output = peft_model(sample_spectrogram)
        loss = output.mean()
        loss.backward()

        # Should complete without error
        assert True

    @requires_peft
    def test_cpu_memory_reasonable(self, model_with_lora, sample_spectrogram):
        """Training on CPU should not use excessive memory."""
        import gc

        gc.collect()

        model_with_lora.train()

        # Do a few training steps
        optimizer = torch.optim.Adam(model_with_lora.parameters(), lr=0.001)

        for _ in range(3):
            optimizer.zero_grad()
            output = model_with_lora(sample_spectrogram)
            loss = output.mean()
            loss.backward()
            optimizer.step()

        # Should complete without OOM
        assert True

    @requires_peft
    def test_lora_rank_affects_param_count(self, small_config):
        """Different LoRA ranks should produce different param counts."""
        model_r4 = SSAMBAModel(small_config)
        config_r4 = LoraConfig(r=4, target_modules=["in_proj", "out_proj"])
        peft_r4 = get_peft_model(model_r4, config_r4)
        params_r4 = sum(p.numel() for p in peft_r4.parameters() if p.requires_grad)

        model_r16 = SSAMBAModel(small_config)
        config_r16 = LoraConfig(r=16, target_modules=["in_proj", "out_proj"])
        peft_r16 = get_peft_model(model_r16, config_r16)
        params_r16 = sum(p.numel() for p in peft_r16.parameters() if p.requires_grad)

        assert params_r16 > params_r4, "Higher rank should have more params"
        # r=16 should have roughly 4x the params of r=4
        assert 3 < params_r16 / params_r4 < 5


class TestIntegration:
    """End-to-end integration tests."""

    @requires_peft
    def test_full_training_loop(self, model_with_lora, batch_spectrogram):
        """Complete training loop should work."""
        model_with_lora.train()

        # Create mock labels
        labels = torch.randint(0, 10, (4,))

        # Add a classification head
        classifier = nn.Linear(384, 10)
        optimizer = torch.optim.Adam(
            list(model_with_lora.parameters()) + list(classifier.parameters()),
            lr=0.001
        )

        losses = []
        for epoch in range(3):
            optimizer.zero_grad()

            embeddings = model_with_lora(batch_spectrogram)
            logits = classifier(embeddings)
            loss = nn.functional.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should generally decrease (not strictly required)
        assert losses[-1] < losses[0] * 2, "Loss exploded during training"

    @requires_peft
    def test_inference_after_training(self, model_with_lora, sample_spectrogram):
        """Model should work in eval mode after training."""
        model_with_lora.train()

        # Do some training
        optimizer = torch.optim.Adam(model_with_lora.parameters(), lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            output = model_with_lora(sample_spectrogram)
            loss = output.mean()
            loss.backward()
            optimizer.step()

        # Switch to eval
        model_with_lora.eval()

        with torch.no_grad():
            output = model_with_lora(sample_spectrogram)

        assert output.shape == (1, 384)
        assert not torch.isnan(output).any()
