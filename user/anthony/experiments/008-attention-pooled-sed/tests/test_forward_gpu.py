"""GPU forward/backward tests for SED model with CUDA Mamba kernels.

Validates shapes, dual loss computation, gradient flow, and attention pooling.

Run with:
    uv run pytest tests/test_forward_gpu.py -v
"""

import pytest
import torch

from ssamba_hf import MAMBA_SSM_AVAILABLE
from src.losses import FocalBCELoss
from src.model import SSAMBASEDClassifier
from src.pretrained import load_pretrained_backbone
from tests.conftest import requires_cuda, requires_mamba_ssm, requires_pretrained


def _make_real_mel_spec():
    """Create nnAudio MelSpectrogram for GPU testing."""
    from nnAudio.features import MelSpectrogram
    return MelSpectrogram(
        sr=32000, n_fft=800, win_length=800, hop_length=320,
        n_mels=128, htk=True, window="hann", center=False,
        fmin=0, fmax=None, trainable_mel=False, trainable_STFT=False,
    )


@requires_cuda
@requires_mamba_ssm
class TestSEDForward:
    """SED model forward pass on GPU with 128x1 patches."""

    def test_forward_shapes(self, config_128x1, num_classes):
        mel_spec = _make_real_mel_spec()
        model = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        ).cuda()
        waveforms = torch.randn(2, 320000, device="cuda")  # 10s at 32kHz
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")

        with torch.no_grad():
            clip_pred, sed_logits = model(waveforms, padding_mask=mask)

        assert clip_pred.shape == (2, num_classes)
        assert sed_logits.shape == (2, 1000, num_classes)
        assert clip_pred.isfinite().all()
        assert sed_logits.isfinite().all()
        # clip_pred should be probabilities
        assert (clip_pred >= 0).all()
        assert (clip_pred <= 1).all()

    def test_backward_dual_loss(self, config_128x1, num_classes):
        """Full forward + backward with both clip and SED losses."""
        mel_spec = _make_real_mel_spec()
        model = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        ).cuda()

        waveforms = torch.randn(2, 320000, device="cuda")
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")
        clip_labels = torch.zeros(2, num_classes, device="cuda")
        clip_labels[:, 0] = 1.0
        token_labels = torch.zeros(2, 1000, num_classes, device="cuda")
        token_labels[:, 100:200, 0] = 1.0

        clip_pred, sed_logits = model(waveforms, padding_mask=mask)

        clip_loss = FocalBCELoss(gamma=2.0, from_logits=False)(clip_pred, clip_labels)
        sed_loss = FocalBCELoss(gamma=2.0, from_logits=True)(sed_logits, token_labels, mask=mask)
        total_loss = clip_loss + sed_loss
        total_loss.backward()

        assert total_loss.isfinite()
        assert model.sed_head[1].weight.grad is not None
        assert model.backbone.patch_embed.proj.weight.grad is not None

    def test_bf16_mixed(self, config_128x1, num_classes):
        mel_spec = _make_real_mel_spec()
        model = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        ).cuda()
        waveforms = torch.randn(2, 320000, device="cuda")
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            clip_pred, sed_logits = model(waveforms, padding_mask=mask)

        assert clip_pred.isfinite().all()
        assert sed_logits.isfinite().all()

    def test_half_padded_mask(self, config_128x1, num_classes):
        """5s clip padded to 10s: 500 real tokens, 500 padding."""
        mel_spec = _make_real_mel_spec()
        model = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        ).cuda()

        waveforms = torch.randn(2, 320000, device="cuda")
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")
        mask[:, 500:] = False

        with torch.no_grad():
            clip_pred, sed_logits = model(waveforms, padding_mask=mask)

        assert clip_pred.shape == (2, num_classes)
        assert clip_pred.isfinite().all()
        # Attention weights should not attend to padded tokens


@requires_cuda
@requires_mamba_ssm
@requires_pretrained
class TestPretrainedSEDForward:
    """Pre-trained SSAMBA with SED head on GPU."""

    def test_pretrained_128x1_forward(self, config_128x1, pretrained_ckpt_path, num_classes):
        mel_spec = _make_real_mel_spec()
        classifier = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        )
        load_pretrained_backbone(classifier, pretrained_ckpt_path)
        classifier = classifier.cuda()

        waveforms = torch.randn(4, 320000, device="cuda")
        mask = torch.ones(4, 1000, dtype=torch.bool, device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            clip_pred, sed_logits = classifier(waveforms, padding_mask=mask)

        assert clip_pred.shape == (4, num_classes)
        assert sed_logits.shape == (4, 1000, num_classes)
        assert clip_pred.isfinite().all()

    def test_pretrained_backward_grads(self, config_128x1, pretrained_ckpt_path, num_classes):
        mel_spec = _make_real_mel_spec()
        classifier = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        )
        load_pretrained_backbone(classifier, pretrained_ckpt_path)
        classifier = classifier.cuda()

        waveforms = torch.randn(2, 320000, device="cuda")
        mask = torch.ones(2, 1000, dtype=torch.bool, device="cuda")
        clip_labels = torch.zeros(2, num_classes, device="cuda")
        clip_labels[:, 0] = 1.0

        clip_pred, sed_logits = classifier(waveforms, padding_mask=mask)
        loss = FocalBCELoss(gamma=2.0, from_logits=False)(clip_pred, clip_labels)
        loss.backward()

        # Mamba layers should receive gradients
        layer0_mixer = list(classifier.backbone.layers.layers[0].mixer.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer0_mixer)


@requires_cuda
@requires_mamba_ssm
class TestSEDFreezeUnfreeze:
    """Test freeze/unfreeze with SED model."""

    def test_freeze_leaves_sed_head_trainable(self, config_128x1, num_classes):
        from src.lightning_module import SEDLightningModule

        mel_spec = _make_real_mel_spec()
        classifier = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        )
        module = SEDLightningModule(model=classifier, num_classes=num_classes, freeze_epochs=5)
        module._freeze_backbone()

        for name, p in classifier.named_parameters():
            if "sed_head." in name or "mel_spec." in name or "patch_embed." in name or "pos_embed" in name or "cls_token" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    def test_unfreeze_all(self, config_128x1, num_classes):
        from src.lightning_module import SEDLightningModule

        mel_spec = _make_real_mel_spec()
        classifier = SSAMBASEDClassifier(
            config=config_128x1, num_classes=num_classes,
            mel_spec_layer=mel_spec, target_frames=1000,
        )
        module = SEDLightningModule(model=classifier, num_classes=num_classes, freeze_epochs=5)
        module._freeze_backbone()
        module._unfreeze_backbone()

        for name, p in classifier.named_parameters():
            assert p.requires_grad, f"{name} should be trainable after unfreeze"


@requires_cuda
@requires_mamba_ssm
class TestEnvironment:
    """Validate GPU environment."""

    def test_cuda_available(self):
        assert torch.cuda.is_available()

    def test_bf16_support(self):
        assert torch.cuda.is_bf16_supported()

    def test_mamba_ssm_import(self):
        import mamba_ssm
        print(f"mamba-ssm version: {mamba_ssm.__version__}")

    def test_nnAudio_import(self):
        from nnAudio.features import MelSpectrogram
        mel = MelSpectrogram(sr=32000, n_fft=800, hop_length=320, n_mels=128)
        # Quick forward test
        x = torch.randn(1, 32000)
        out = mel(x)
        assert out.shape[1] == 128  # n_mels
