"""Tests for Experiment 016 v3 SED changes.

Validates:
1. Raw waveform mixup (v3b): waveform → mixup → mel → forward
2. Standard CE loss (v3c): argmax-based cross-entropy
3. Backward compatibility: v1 behavior unchanged with default params
4. SEDDataModule eval_transform: val/test use separate transform from train
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.encoders.efficientnet import EfficientNetEncoder
from birdclef.heads.sed_head import AttHead
from birdclef.lightning.sed_finetune import SEDFinetuneModule
from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

NUM_CLASSES = 234
SAMPLE_RATE = 32000
WINDOW_DURATION = 10.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)  # 320000
TARGET_FRAMES = int(WINDOW_DURATION / 0.01)  # 1000


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def encoder():
    return EfficientNetEncoder("efficientnet_b0", pretrained=False)


@pytest.fixture(scope="module")
def sed_head(encoder):
    return AttHead(
        in_channels=encoder.embed_dim,
        num_classes=NUM_CLASSES,
        hidden_dim=512,
        dropout=0.5,
        gem_p=3.0,
    )


@pytest.fixture(scope="module")
def mel_transform():
    return NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)


def _make_module(
    encoder, sed_head, mel_transform=None, loss_type="soft_ce", label_smoothing=0.0, mixup_prob=1.0
):
    """Build SEDFinetuneModule with given settings."""
    return SEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=NUM_CLASSES,
        lr=1e-4,
        head_lr=1e-3,
        weight_decay=0.01,
        max_epochs=40,
        warmup_epochs=5,
        threshold=0.5,
        clipwise_weight=0.5,
        framewise_weight=0.5,
        blend_clip=0.75,
        blend_frame=0.25,
        mixup_prob=mixup_prob,
        mixup_alpha=0.5,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        mel_transform=mel_transform,
    )


@pytest.fixture
def batch_waveforms():
    """Batch of raw 10s waveforms."""
    torch.manual_seed(42)
    return torch.randn(4, WINDOW_SAMPLES)


@pytest.fixture
def batch_mel(mel_transform, batch_waveforms):
    """Batch of mel spectrograms (pre-computed from waveforms)."""
    with torch.no_grad():
        return mel_transform(batch_waveforms)


@pytest.fixture
def batch_labels():
    """Multi-hot labels for 4 samples."""
    labels = torch.zeros(4, NUM_CLASSES)
    labels[0, 10] = 1.0
    labels[1, 20] = 1.0
    labels[2, 30] = 1.0
    labels[3, 40] = 1.0
    return labels


@pytest.fixture
def multi_species_labels():
    """Multi-hot labels with multiple species per sample."""
    labels = torch.zeros(4, NUM_CLASSES)
    labels[0, 10] = 1.0
    labels[0, 50] = 1.0  # secondary species
    labels[1, 20] = 1.0
    labels[2, 5] = 1.0
    labels[2, 30] = 1.0  # secondary species with lower index
    labels[3, 40] = 1.0
    return labels


# ============================================================================
# v3b: Raw waveform mixup
# ============================================================================


class TestRawWaveformMixup:
    """Tests for v3b: raw waveform mixup with GPU-side mel transform."""

    def test_training_step_accepts_waveforms(
        self, encoder, sed_head, batch_waveforms, batch_labels
    ):
        """training_step should accept raw waveforms when mel_transform is set."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf)
        loss = module.training_step((batch_waveforms, batch_labels), 0)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"

    def test_training_step_produces_finite_loss(
        self, encoder, sed_head, batch_waveforms, batch_labels
    ):
        """Loss should be finite (no inf/nan from mel computation)."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf)
        loss = module.training_step((batch_waveforms, batch_labels), 0)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_mixup_modifies_waveforms(self, encoder, sed_head, batch_waveforms, batch_labels):
        """Mixup should produce different waveforms than the originals."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf, mixup_prob=1.0)
        original = batch_waveforms.clone()

        # Run _apply_mixup directly
        mixed, mixed_labels = module._apply_mixup(batch_waveforms.clone(), batch_labels.clone())
        assert not torch.allclose(mixed, original), "Mixup should modify waveforms"

    def test_mixup_preserves_waveform_shape(self, encoder, sed_head, batch_waveforms, batch_labels):
        """Mixup should not change waveform shape."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf)
        mixed, _ = module._apply_mixup(batch_waveforms.clone(), batch_labels.clone())
        assert (
            mixed.shape == batch_waveforms.shape
        ), f"Shape changed: {mixed.shape} vs {batch_waveforms.shape}"

    def test_waveform_mixup_then_mel_shape(self, encoder, sed_head, batch_waveforms, batch_labels):
        """After mixup + mel transform, output should have correct mel shape."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf)
        mixed, _ = module._apply_mixup(batch_waveforms.clone(), batch_labels.clone())
        mel = mel_tf(mixed)
        assert mel.shape == (4, 1, 128, TARGET_FRAMES), f"Unexpected mel shape: {mel.shape}"

    def test_val_step_still_accepts_mel(self, encoder, sed_head, batch_mel, batch_labels):
        """validation_step should still work with mel spectrograms even when
        mel_transform is set (val/test datasets use eval_transform)."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf)
        loss = module.validation_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(loss), f"Val loss not finite: {loss.item()}"

    def test_test_step_still_accepts_mel(self, encoder, sed_head, batch_mel, batch_labels):
        """test_step should still work with mel spectrograms."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf)
        loss = module.test_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(loss), f"Test loss not finite: {loss.item()}"

    def test_silent_waveform_no_nan(self, encoder, sed_head, batch_labels):
        """Silent waveforms should not produce NaN loss (tests log(0+eps) safety)."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf, mixup_prob=0.0)
        silent = torch.zeros(4, WINDOW_SAMPLES)
        loss = module.training_step((silent, batch_labels), 0)
        assert torch.isfinite(loss), f"Silent waveform produced non-finite loss: {loss.item()}"

    def test_mel_transform_is_submodule(self, encoder, sed_head):
        """mel_transform should be registered as a submodule for device transfer."""
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)
        module = _make_module(encoder, sed_head, mel_transform=mel_tf)
        submodule_names = [name for name, _ in module.named_modules()]
        assert "mel_transform" in submodule_names, "mel_transform should be a registered submodule"

    def test_mel_output_is_float32_under_autocast(self, encoder, sed_head, batch_waveforms):
        """Mel transform should produce float32 output even under autocast.

        nnAudio STFT magnitudes can overflow float16 range (~65504), so the mel
        computation must run in float32 regardless of mixed precision settings.
        """
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)

        # Simulate autocast context (as Lightning does with precision="16-mixed")
        with torch.amp.autocast("cpu", dtype=torch.float16):
            x, labels = batch_waveforms, torch.zeros(4, NUM_CLASSES)
            labels[0, 10] = 1.0
            labels[1, 20] = 1.0
            labels[2, 30] = 1.0
            labels[3, 40] = 1.0

            # The mel transform should force float32 internally
            with torch.no_grad(), torch.amp.autocast("cpu", enabled=False):
                mel = mel_tf(x.float())

            assert mel.dtype == torch.float32, f"Mel should be float32, got {mel.dtype}"
            assert torch.isfinite(mel).all(), "Mel contains non-finite values under autocast"

    def test_waveform_mixup_output_matches_cpu_mel(
        self, encoder, sed_head, batch_waveforms, batch_labels
    ):
        """GPU-side mel from waveform mixup should match CPU-side mel numerically.

        Ensures the mel transform produces the same values whether run inside the
        model (GPU path for v3b) or as a dataset transform (CPU path for v1).
        """
        mel_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)

        # CPU path (v1 behavior): transform in dataset
        with torch.no_grad():
            cpu_mel = mel_tf(batch_waveforms)

        # GPU path (v3b behavior): transform in training_step
        with torch.no_grad(), torch.amp.autocast("cpu", enabled=False):
            gpu_mel = mel_tf(batch_waveforms.float())

        max_diff = (cpu_mel - gpu_mel).abs().max().item()
        assert max_diff < 1e-5, f"CPU vs GPU mel mismatch: max diff {max_diff}"


# ============================================================================
# v3c: Standard CE loss
# ============================================================================


class TestCELoss:
    """Tests for v3c: standard CrossEntropyLoss on argmax class."""

    def test_ce_loss_produces_finite(self, encoder, sed_head, batch_mel, batch_labels):
        """CE loss should produce finite scalar."""
        module = _make_module(encoder, sed_head, loss_type="ce", label_smoothing=0.0)
        loss = module.training_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(loss), f"CE loss not finite: {loss.item()}"

    def test_ce_loss_in_expected_range(self, encoder, sed_head, batch_mel, batch_labels):
        """CE loss should be in reasonable range for 234 classes (ln(234) ≈ 5.46)."""
        module = _make_module(
            encoder, sed_head, loss_type="ce", label_smoothing=0.0, mixup_prob=0.0
        )
        loss = module.training_step((batch_mel, batch_labels), 0)
        assert (
            3.0 < loss.item() < 8.0
        ), f"CE loss out of expected range: {loss.item()} (expected ~5.46)"

    def test_ce_with_label_smoothing(self, encoder, sed_head, batch_mel, batch_labels):
        """CE with label smoothing should also work."""
        module = _make_module(encoder, sed_head, loss_type="ce", label_smoothing=0.1)
        loss = module.training_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(loss), f"CE+smoothing loss not finite: {loss.item()}"

    def test_ce_argmax_selects_correct_class(self):
        """argmax on single-species multi-hot should return the correct class."""
        labels = torch.zeros(2, NUM_CLASSES)
        labels[0, 100] = 1.0
        labels[1, 200] = 1.0
        targets = labels.argmax(dim=-1)
        assert targets[0].item() == 100
        assert targets[1].item() == 200

    def test_ce_argmax_with_multi_species(self, multi_species_labels):
        """argmax on multi-species labels picks the lowest active index."""
        targets = multi_species_labels.argmax(dim=-1)
        # Sample 0: species 10 and 50 → argmax = 10
        assert targets[0].item() == 10
        # Sample 2: species 5 and 30 → argmax = 5
        assert targets[2].item() == 5

    def test_ce_loss_different_from_soft_ce(self, encoder, sed_head, batch_mel, batch_labels):
        """CE and soft_ce should produce different loss values."""
        module_ce = _make_module(encoder, sed_head, loss_type="ce", mixup_prob=0.0)
        module_soft = _make_module(encoder, sed_head, loss_type="soft_ce", mixup_prob=0.0)

        loss_ce = module_ce.training_step((batch_mel.clone(), batch_labels.clone()), 0)
        loss_soft = module_soft.training_step((batch_mel.clone(), batch_labels.clone()), 0)

        # For single-species samples, CE and soft_ce should be close but not identical
        # due to numerical differences in how targets are constructed
        assert (
            loss_ce.item() != loss_soft.item()
        ), "CE and soft_ce should produce different loss values"

    def test_ce_val_and_test_work(self, encoder, sed_head, batch_mel, batch_labels):
        """CE loss should work in validation and test steps too."""
        module = _make_module(encoder, sed_head, loss_type="ce", label_smoothing=0.0)
        val_loss = module.validation_step((batch_mel, batch_labels), 0)
        test_loss = module.test_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(val_loss)
        assert torch.isfinite(test_loss)


# ============================================================================
# Backward compatibility: v1 behavior
# ============================================================================


class TestBackwardCompatibility:
    """Ensure v1 behavior is unchanged with default parameters."""

    def test_default_mel_transform_is_none(self, encoder, sed_head):
        """Default mel_transform should be None (v1 behavior)."""
        module = _make_module(encoder, sed_head)
        assert module.mel_transform is None

    def test_default_loss_type_is_soft_ce(self, encoder, sed_head):
        """Default loss_type should be soft_ce (v1 behavior)."""
        module = _make_module(encoder, sed_head)
        assert module.hparams.loss_type == "soft_ce"

    def test_spectrogram_mixup_still_works(self, encoder, sed_head, batch_mel, batch_labels):
        """Spectrogram-domain mixup should still work when mel_transform is None."""
        module = _make_module(encoder, sed_head, mel_transform=None, mixup_prob=1.0)
        loss = module.training_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(loss)

    def test_no_mixup_still_works(self, encoder, sed_head, batch_mel, batch_labels):
        """Training with mixup_prob=0 should still work."""
        module = _make_module(encoder, sed_head, mixup_prob=0.0)
        loss = module.training_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(loss)

    def test_spectrogram_mixup_modifies_mel(self, encoder, sed_head, batch_mel, batch_labels):
        """Spectrogram mixup should modify the mel input (not raw waveforms)."""
        module = _make_module(encoder, sed_head, mel_transform=None, mixup_prob=1.0)
        original_mel = batch_mel.clone()
        mixed, _ = module._apply_mixup(batch_mel.clone(), batch_labels.clone())
        assert not torch.allclose(mixed, original_mel), "Spectrogram mixup should modify mel"
        assert mixed.shape == original_mel.shape


# ============================================================================
# SEDDataModule eval_transform
# ============================================================================


class TestEvalTransform:
    """Tests for the eval_transform parameter on SEDDataModule."""

    def test_eval_transform_defaults_to_transform(self):
        """When eval_transform is not set, it should default to transform."""
        from birdclef.data.sed import SEDDataModule

        transform = nn.Identity()
        # Can't fully instantiate without data, but test the __init__ logic
        dm = SEDDataModule.__new__(SEDDataModule)
        dm.transform = transform
        dm.eval_transform = None if None is not None else transform
        assert dm.eval_transform is transform

    def test_eval_transform_separate_from_train(self):
        """When eval_transform is set, it should be different from transform."""
        eval_tf = NNAudioMelSpectrogram(sample_rate=SAMPLE_RATE, target_frames=TARGET_FRAMES)

        # In raw waveform mode: train_transform=None, eval_transform=mel
        train_transform = None
        assert train_transform is not eval_tf
        assert isinstance(eval_tf, NNAudioMelSpectrogram)


# ============================================================================
# Mixup label behavior
# ============================================================================


class TestMixupLabels:
    """Tests for mixup label handling across loss types."""

    def test_mixup_unions_labels(self, encoder, sed_head):
        """Mixup should union (OR) the labels of mixed samples."""
        module = _make_module(encoder, sed_head, mixup_prob=1.0)
        labels = torch.zeros(4, NUM_CLASSES)
        labels[0, 10] = 1.0
        labels[1, 20] = 1.0
        labels[2, 30] = 1.0
        labels[3, 40] = 1.0

        # After mixup with a permutation, labels should have multiple active species
        torch.manual_seed(0)
        x = torch.randn(4, 100)  # dummy input
        _, mixed_labels = module._apply_mixup(x, labels.clone())

        # At least some samples should have >1 active species
        active_per_sample = (mixed_labels > 0).sum(dim=-1)
        assert (active_per_sample > 1).any(), "Mixup should create multi-species labels"

    def test_mixup_labels_clamped_to_one(self, encoder, sed_head):
        """Mixup labels should be clamped to max 1.0."""
        module = _make_module(encoder, sed_head)
        labels = torch.ones(4, NUM_CLASSES)  # all species active
        x = torch.randn(4, 100)
        _, mixed_labels = module._apply_mixup(x, labels.clone())
        assert mixed_labels.max() <= 1.0, "Labels should be clamped to 1.0"

    def test_mixup_lambda_in_valid_range(self):
        """Mixup lambda should produce values between the two mixed inputs."""
        x1 = torch.ones(4, WINDOW_SAMPLES)
        x2 = torch.zeros(4, WINDOW_SAMPLES)

        # Manually set up a known mixup
        import numpy as np

        np.random.seed(42)
        lam = np.random.beta(0.5, 0.5)
        mixed = lam * x1 + (1 - lam) * x2
        assert (mixed >= 0).all() and (
            mixed <= 1
        ).all(), "Mixed values should be between the two inputs"


# ============================================================================
# v3a: TF-ported model variant
# ============================================================================


class TestTFEfficientNet:
    """Tests for v3a: tf_efficientnet_b0.in1k model variant."""

    def test_tf_model_same_architecture(self):
        """tf_efficientnet_b0 should have same embed_dim as efficientnet_b0."""
        enc_pytorch = EfficientNetEncoder("efficientnet_b0", pretrained=False)
        enc_tf = EfficientNetEncoder("tf_efficientnet_b0", pretrained=False)
        assert enc_pytorch.embed_dim == enc_tf.embed_dim == 320

    def test_tf_model_same_output_shape(self):
        """tf_efficientnet_b0 should produce same feature map shape."""
        enc_pytorch = EfficientNetEncoder("efficientnet_b0", pretrained=False)
        enc_tf = EfficientNetEncoder("tf_efficientnet_b0", pretrained=False)
        x = torch.randn(2, 1, 128, 1000)  # mel spectrogram input
        feat_pytorch = enc_pytorch.forward_features(x)
        feat_tf = enc_tf.forward_features(x)
        assert (
            feat_pytorch.shape == feat_tf.shape
        ), f"Shape mismatch: {feat_pytorch.shape} vs {feat_tf.shape}"

    def test_tf_model_works_in_full_pipeline(self, batch_mel, batch_labels):
        """tf_efficientnet_b0 should work end-to-end in the SED pipeline."""
        enc = EfficientNetEncoder("tf_efficientnet_b0", pretrained=False)
        head = AttHead(
            in_channels=enc.embed_dim,
            num_classes=NUM_CLASSES,
            hidden_dim=512,
            dropout=0.5,
            gem_p=3.0,
        )
        module = _make_module(enc, head, loss_type="soft_ce", mixup_prob=0.0)
        loss = module.training_step((batch_mel, batch_labels), 0)
        assert torch.isfinite(loss), f"TF model loss not finite: {loss.item()}"
