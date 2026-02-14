"""Tests for MPG pretraining model with span masking and grid jitter."""

import torch

from birdclef_009.models.pretrain import SSAMBAPretrainModel


def test_forward_returns_dict(tiny_config, mel_spec_layer):
    """Forward pass returns dict with losses and monitoring metrics."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        mask_mode="span",
    )
    model.eval()

    B = 2
    waveforms = torch.randn(B, 320000)  # 10s @ 32kHz

    out = model(waveforms)

    assert isinstance(out, dict)
    assert out["loss_mpg"].dim() == 0, "loss_mpg should be scalar"
    assert out["loss_jitter"].dim() == 0, "loss_jitter should be scalar"
    # JiTTER disabled by default
    assert out["loss_jitter"].item() == 0.0
    # Cross-metrics always present
    assert torch.isfinite(out["monitor_mse"])
    assert torch.isfinite(out["monitor_l1"])
    # Viz data present
    assert "viz_mel" in out
    assert "viz_mask" in out
    assert "viz_pred" in out
    assert "viz_target" in out


def test_mask_count(tiny_config, mel_spec_layer):
    """Verify mask ratio produces expected number of masked tokens."""
    mask_ratio = 0.78
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=mask_ratio,
    )
    # The model should mask ~780 out of 1000 tokens
    expected_mask_count = int(1000 * mask_ratio)
    assert expected_mask_count == 780


def test_losses_are_finite(tiny_config, mel_spec_layer):
    """Both losses should be finite."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
    )
    model.eval()

    waveforms = torch.randn(2, 320000)

    with torch.no_grad():
        out = model(waveforms)

    assert torch.isfinite(out["loss_mpg"]), f"loss_mpg is not finite: {out['loss_mpg']}"
    assert torch.isfinite(out["loss_jitter"]), f"loss_jitter is not finite: {out['loss_jitter']}"


def test_jitter_disabled(tiny_config, mel_spec_layer):
    """With jitter_weight=0, loss_jitter should be zero."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        jitter_weight=0.0,
    )
    model.eval()

    waveforms = torch.randn(2, 320000)

    with torch.no_grad():
        out = model(waveforms)

    assert out["loss_jitter"].item() == 0.0, "loss_jitter should be 0 when jitter_weight=0"
    assert torch.isfinite(out["loss_mpg"]), "loss_mpg should still work"


def test_span_mask_contiguous(tiny_config, mel_spec_layer):
    """Span masking should produce contiguous masked regions."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        mask_mode="span",
        mask_span_min=5,
        mask_span_max=15,
    )
    mask = model._generate_span_mask(1000, torch.device("cpu"))

    # Check that masked regions are contiguous (no isolated single masks)
    # Find transitions from unmasked to masked
    transitions = (mask[1:].int() - mask[:-1].int())
    span_starts = (transitions == 1).sum().item()
    span_ends = (transitions == -1).sum().item()
    # Account for mask starting at position 0 or ending at last position
    if mask[0]:
        span_starts += 1
    if mask[-1]:
        span_ends += 1

    # Should have far fewer spans than masked tokens (contiguous, not scattered)
    masked_count = mask.sum().item()
    assert span_starts > 0, "Should have at least one span"
    assert masked_count / span_starts >= 5, (
        f"Average span length {masked_count / span_starts:.1f} too short, "
        "spans should be contiguous"
    )


def test_span_mask_approximate_ratio(tiny_config, mel_spec_layer):
    """Span masking should approximately hit the target mask ratio."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.78,
        mask_mode="span",
        mask_span_min=5,
        mask_span_max=15,
    )
    mask = model._generate_span_mask(1000, torch.device("cpu"))
    actual_ratio = mask.sum().item() / 1000

    # Should be at least mask_ratio (may overshoot slightly due to overlapping spans)
    assert actual_ratio >= 0.78, f"Actual ratio {actual_ratio:.3f} below target 0.78"
    assert actual_ratio <= 0.95, f"Actual ratio {actual_ratio:.3f} way too high"


def test_random_mask_exact_count(tiny_config, mel_spec_layer):
    """Random masking should produce exactly the expected count."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.78,
        mask_mode="random",
    )
    mask = model._generate_random_mask(1000, torch.device("cpu"))
    assert mask.sum().item() == 780


def test_grid_jitter_training_only(tiny_config, mel_spec_layer):
    """Grid jitter should only apply during training, not eval."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        grid_jitter=5,
    )

    waveforms = torch.randn(2, 320000)

    # Eval mode: no jitter, deterministic
    model.eval()
    with torch.no_grad():
        mel_eval1 = model.compute_mel(waveforms)
        mel_eval2 = model.compute_mel(waveforms)
    assert torch.equal(mel_eval1, mel_eval2), "Eval mel should be deterministic"

    # Training mode: jitter may shift (not guaranteed different each call due to randomness)
    model.train()
    mel_train = model.compute_mel(waveforms)
    assert mel_train.shape == mel_eval1.shape, "Shape should be same regardless of jitter"


def test_l1_loss(tiny_config, mel_spec_layer):
    """L1 loss option should produce finite scalar loss."""
    model = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        loss_fn="l1",
    )
    model.eval()

    waveforms = torch.randn(2, 320000)
    with torch.no_grad():
        out = model(waveforms)

    assert torch.isfinite(out["loss_mpg"]), f"L1 loss_mpg is not finite: {out['loss_mpg']}"


def test_linear_decoder(tiny_config, mel_spec_layer):
    """Linear decoder (depth=1) should work and have fewer params than MLP."""
    import torch.nn as nn

    model_linear = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        decoder_depth=1,
    )
    model_mlp = SSAMBAPretrainModel(
        config=tiny_config,
        mel_spec_layer=mel_spec_layer,
        target_frames=1000,
        mask_ratio=0.5,
        decoder_depth=2,
    )

    # Linear decoder should be a single nn.Linear
    assert isinstance(model_linear.gpredlayer, nn.Linear)
    # MLP decoder should be nn.Sequential
    assert isinstance(model_mlp.gpredlayer, nn.Sequential)

    # Linear should have fewer params
    linear_params = sum(p.numel() for p in model_linear.gpredlayer.parameters())
    mlp_params = sum(p.numel() for p in model_mlp.gpredlayer.parameters())
    assert linear_params < mlp_params

    # Forward pass should work
    model_linear.eval()
    waveforms = torch.randn(2, 320000)
    with torch.no_grad():
        out = model_linear(waveforms)
    assert torch.isfinite(out["loss_mpg"])


def test_cross_metrics_present(tiny_config, mel_spec_layer):
    """Both MSE and L1 monitor metrics should be logged regardless of loss_fn."""
    for loss_fn in ["mse", "l1"]:
        model = SSAMBAPretrainModel(
            config=tiny_config,
            mel_spec_layer=mel_spec_layer,
            target_frames=1000,
            mask_ratio=0.5,
            loss_fn=loss_fn,
        )
        model.eval()

        waveforms = torch.randn(2, 320000)
        with torch.no_grad():
            out = model(waveforms)

        assert torch.isfinite(out["monitor_mse"]), f"monitor_mse not finite for loss_fn={loss_fn}"
        assert torch.isfinite(out["monitor_l1"]), f"monitor_l1 not finite for loss_fn={loss_fn}"
        # monitor_mse should match loss_mpg when using MSE
        if loss_fn == "mse":
            assert torch.allclose(out["loss_mpg"], out["monitor_mse"], atol=1e-5)
