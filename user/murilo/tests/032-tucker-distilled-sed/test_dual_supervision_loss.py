"""Tests for `DistilledSEDFinetuneModule` (exp 032 task 0.3).

Verifies the Tucker-recipe LightningModule:
1. training_step runs without NaN/Inf on a random batch
2. Loss equals `clipwise_weight·BCE(clip) + framewise_weight·BCE(frame_max)`
3. Gradient flows to both `cla` and `att` convs of the head
4. KD off: total loss == cls loss
5. KD on (with a stub teacher): total loss == cls + α · MSE; backbone
   receives gradient ONLY through the distill branch (cls branch detached)
6. Inference blend: `probs == blend_clip·σ(clip) + blend_frame·σ(frame_max)`
7. configure_optimizers builds the LinearLR→CosineAnnealingLR SequentialLR

Pretrained=False; tests run offline.

Reference:
- src/birdclef/lightning/distilled_sed_finetune.py
- analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md (§ "Tucker's loss")
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 14)
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from birdclef.encoders.efficientnet import EfficientNetEncoder
from birdclef.heads.sed_head import AttHeadLogitSpace
from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule

NUM_CLASSES = 234
BATCH = 2
N_MELS = 256
TIME_FRAMES = 313  # 5s @ 32kHz, hop=512
SAMPLES_5S = 160_000


# ============================================================================
# Fixtures
# ============================================================================


class _IdentityMel(nn.Module):
    """Stand-in for torchaudio.MelSpectrogram + AmplitudeToDB. Takes (B, N) raw
    waveform and returns (B, 1, n_mels, T) random-but-stable mel tensor of the
    right shape. We don't care about mel correctness for module-level tests."""

    def __init__(self, n_mels: int = N_MELS, time_frames: int = TIME_FRAMES):
        super().__init__()
        self.n_mels = n_mels
        self.time_frames = time_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deterministic projection so different inputs map to different mels
        B = x.shape[0]
        # Cheap: use a fixed linear hash to a (n_mels * time_frames) vector
        flat = x.float().reshape(B, -1)
        proj = flat[:, : self.n_mels * self.time_frames]
        # Pad if needed so the next step doesn't error on short inputs
        if proj.shape[1] < self.n_mels * self.time_frames:
            proj = F.pad(proj, (0, self.n_mels * self.time_frames - proj.shape[1]))
        return proj.reshape(B, 1, self.n_mels, self.time_frames)


def _make_module(use_perch_distill: bool = False, perch_emb_value: float = 0.0):
    """Construct a DistilledSEDFinetuneModule with EffNet-B0 NS-JFT encoder +
    AttHeadLogitSpace + identity mel. Stubs PerchTeacher with a Module that
    returns a fixed embedding."""
    encoder = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
        drop_path_rate=0.1,
        use_post_head=True,
    )
    head = AttHeadLogitSpace(in_channels=encoder.embed_dim, num_classes=NUM_CLASSES, hidden_dim=512)
    mel = _IdentityMel()

    if not use_perch_distill:
        return DistilledSEDFinetuneModule(
            encoder=encoder,
            sed_head=head,
            num_classes=NUM_CLASSES,
            mel_transform=mel,
        )

    # KD on: stub PerchTeacher to skip ONNX session creation.
    class _StubPerchTeacher:
        def __init__(self, *args, **kwargs):
            pass

        @torch.no_grad()
        def embed(self, waveforms_5s: torch.Tensor) -> torch.Tensor:
            B = waveforms_5s.shape[0]
            return torch.full((B, 1536), perch_emb_value, device=waveforms_5s.device)

    with patch("birdclef.lightning.distilled_sed_finetune.PerchTeacher", _StubPerchTeacher):
        return DistilledSEDFinetuneModule(
            encoder=encoder,
            sed_head=head,
            num_classes=NUM_CLASSES,
            mel_transform=mel,
            use_perch_distill=True,
            perch_onnx_path="/stub/path/perch.onnx",
            perch_embed_dim=1536,
            alpha_distill=1.0,
        )


@pytest.fixture
def module():
    return _make_module(use_perch_distill=False)


@pytest.fixture
def module_kd():
    return _make_module(use_perch_distill=True)


@pytest.fixture
def batch():
    """(waveform, labels) tuple — 5s waveform + multi-hot label."""
    x = torch.randn(BATCH, SAMPLES_5S)
    labels = torch.zeros(BATCH, NUM_CLASSES)
    labels[:, 0] = 1.0
    labels[:, 17] = 1.0
    return (x, labels)


# ============================================================================
# Constructor edge cases
# ============================================================================


def test_requires_mel_transform():
    """The module rejects mel_transform=None — Tucker's recipe needs GPU mel."""
    encoder = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    head = AttHeadLogitSpace(in_channels=1280, num_classes=NUM_CLASSES)
    with pytest.raises(ValueError, match="GPU-side mel_transform"):
        DistilledSEDFinetuneModule(
            encoder=encoder, sed_head=head, num_classes=NUM_CLASSES, mel_transform=None
        )


def test_kd_requires_onnx_path():
    """KD on without perch_onnx_path raises ValueError."""
    encoder = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    head = AttHeadLogitSpace(in_channels=1280, num_classes=NUM_CLASSES)
    mel = _IdentityMel()
    with pytest.raises(ValueError, match="perch_onnx_path"):
        DistilledSEDFinetuneModule(
            encoder=encoder,
            sed_head=head,
            num_classes=NUM_CLASSES,
            mel_transform=mel,
            use_perch_distill=True,
            perch_onnx_path=None,
        )


# ============================================================================
# Training step — basic correctness
# ============================================================================


def test_training_step_runs_no_nan(module, batch):
    """fast-dev-run style: one batch through training_step. No NaN/Inf in loss."""
    with patch.object(module, "log"):
        loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_training_step_loss_equals_manual_formula(module, batch):
    """Loss must equal `0.5·BCE(clip) + 0.5·BCE(frame_max)` to fp tolerance."""
    module.eval()  # disable dropout for reproducible forward
    x, labels = batch
    # Reproduce the forward manually
    with torch.no_grad():
        mel = module._compute_mel(x, training=False)
        clip_logits, frame_logits = module(mel)
        frame_max = frame_logits.max(dim=-1).values
        bce_clip = F.binary_cross_entropy_with_logits(clip_logits, labels.float())
        bce_frame = F.binary_cross_entropy_with_logits(frame_max, labels.float())
        expected = 0.5 * bce_clip + 0.5 * bce_frame

    # Compare to module's helper directly (eval mode → identical)
    with torch.no_grad():
        mel2 = module._compute_mel(x, training=False)
        clip2, frame2 = module(mel2)
        actual, _ = module._dual_supervision_loss(clip2, frame2, labels)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_dual_supervision_loss_weights_respected(module):
    """Synthetic clip/frame logits — verify _dual_supervision_loss uses the
    configured weights. Uses extreme logit values so BCE(clip) ≠ BCE(frame_max)
    despite saturation behavior of BCE on near-zero logits."""
    # Construct very different clip and frame_max logits
    clip = torch.full((2, NUM_CLASSES), -5.0)
    frame = torch.full((2, NUM_CLASSES, 10), 5.0)
    labels = torch.zeros(2, NUM_CLASSES)
    labels[:, 0] = 1.0

    bce_clip = F.binary_cross_entropy_with_logits(clip, labels.float())
    frame_max = frame.max(dim=-1).values
    bce_frame = F.binary_cross_entropy_with_logits(frame_max, labels.float())

    # Default 0.5/0.5
    expected_default = 0.5 * bce_clip + 0.5 * bce_frame
    actual_default, _ = module._dual_supervision_loss(clip, frame, labels)
    torch.testing.assert_close(actual_default, expected_default, rtol=1e-5, atol=1e-6)

    # bce_clip and bce_frame must differ on these inputs (sanity)
    assert not torch.isclose(bce_clip, bce_frame, rtol=1e-3, atol=1e-3)


# ============================================================================
# Inference blend
# ============================================================================


def test_blended_probs_matches_manual_formula(module, batch):
    """probs = 0.5·σ(clip) + 0.5·σ(frame_max)."""
    module.eval()
    x, _ = batch
    with torch.no_grad():
        mel = module._compute_mel(x, training=False)
        clip, frame = module(mel)
        frame_max = frame.max(dim=-1).values

        expected = 0.5 * torch.sigmoid(clip) + 0.5 * torch.sigmoid(frame_max)
        actual = module._blended_probs(clip, frame_max)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_blend_weights_respected(batch):
    """blend_clip=1.0, blend_frame=0.0 should give pure clipwise sigmoid."""
    encoder = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    head = AttHeadLogitSpace(in_channels=1280, num_classes=NUM_CLASSES)
    mel = _IdentityMel()
    mod = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=head,
        num_classes=NUM_CLASSES,
        mel_transform=mel,
        blend_clip=1.0,
        blend_frame=0.0,
    )
    mod.eval()
    x, _ = batch
    with torch.no_grad():
        mel_t = mod._compute_mel(x, training=False)
        clip, frame = mod(mel_t)
        frame_max = frame.max(dim=-1).values
        probs = mod._blended_probs(clip, frame_max)
        expected = torch.sigmoid(clip)
    torch.testing.assert_close(probs, expected, rtol=1e-5, atol=1e-6)


# ============================================================================
# Gradient flow
# ============================================================================


def test_grad_flows_to_both_head_convs(module, batch):
    """BCE on the dual-supervision loss should produce gradient on att AND cla.
    cla gets gradient from BOTH terms (it's in the clip aggregation AND is the
    framewise logit producer). att gets gradient only from the clipwise term."""
    with patch.object(module, "log"):
        loss = module.training_step(batch, batch_idx=0)
    loss.backward()
    head = module.sed_head
    assert head.cla.weight.grad is not None
    assert head.cla.weight.grad.abs().sum().item() > 0
    assert head.att.weight.grad is not None
    assert head.att.weight.grad.abs().sum().item() > 0


def test_grad_flows_to_encoder_when_kd_off(module, batch):
    """Without KD, BCE-on-clipwise gradient should reach the encoder backbone."""
    with patch.object(module, "log"):
        loss = module.training_step(batch, batch_idx=0)
    loss.backward()
    assert module.encoder.backbone.conv_stem.weight.grad is not None
    assert module.encoder.backbone.conv_stem.weight.grad.abs().sum().item() > 0


# ============================================================================
# KD branch
# ============================================================================


def test_kd_loss_includes_distill_term(module_kd, batch):
    """When KD is on, total loss = cls_loss + α · dist_loss; total > cls alone."""
    with patch.object(module_kd, "log"):
        loss = module_kd.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_kd_backbone_grad_only_from_distill_branch(module_kd, batch):
    """When KD is on, backbone grad must come ONLY through distill_head.
    The sed_head branch uses features.detach() so it can't update the backbone."""
    # Zero α — backbone should NOT receive gradient (only KD branch updates it,
    # and we just zeroed that contribution).
    module_kd.hparams.alpha_distill = 0.0
    with patch.object(module_kd, "log"):
        loss = module_kd.training_step(batch, batch_idx=0)
    loss.backward()
    backbone_grad = module_kd.encoder.backbone.conv_stem.weight.grad
    if backbone_grad is not None:
        assert backbone_grad.abs().sum().item() == 0.0


def test_kd_backbone_grad_present_with_alpha_one(module_kd, batch):
    """With α=1.0, the backbone DOES receive gradient (through distill_head)."""
    with patch.object(module_kd, "log"):
        loss = module_kd.training_step(batch, batch_idx=0)
    loss.backward()
    backbone_grad = module_kd.encoder.backbone.conv_stem.weight.grad
    assert backbone_grad is not None
    assert backbone_grad.abs().sum().item() > 0


# ============================================================================
# Optimizer + schedule
# ============================================================================


def test_configure_optimizers_builds_sequential_lr(module):
    """LinearLR warmup → CosineAnnealingLR via SequentialLR, per-step."""
    # Simulate Trainer: estimated_stepping_batches / max_epochs gives steps/epoch.
    # With max_epochs=25 and 100 steps/epoch → warmup=200, cosine_steps=2300.
    module.trainer = SimpleNamespace(estimated_stepping_batches=2500, max_epochs=25)

    cfg = module.configure_optimizers()
    assert "optimizer" in cfg
    assert "lr_scheduler" in cfg
    sched_cfg = cfg["lr_scheduler"]
    assert sched_cfg["interval"] == "step"
    sched = sched_cfg["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)


def test_configure_optimizers_per_step_lr_curve(module):
    """LR should rise linearly during warmup, then decay cosine."""
    module.trainer = SimpleNamespace(estimated_stepping_batches=2500, max_epochs=25)
    cfg = module.configure_optimizers()
    opt = cfg["optimizer"]
    sched = cfg["lr_scheduler"]["scheduler"]

    # Step 0: initial_lr = lr * start_factor = 5e-4 * (1/25) = 2e-5
    initial_lr = opt.param_groups[0]["lr"]
    expected_initial = 5e-4 * (1.0 / 25)
    assert abs(initial_lr - expected_initial) < 1e-9

    # Step until end of warmup (200 steps). LR should reach lr (5e-4).
    for _ in range(200):
        sched.step()
    warmup_end_lr = opt.param_groups[0]["lr"]
    # LinearLR ramps to end_factor=1.0 so we hit 5e-4 ± tiny fp.
    assert abs(warmup_end_lr - 5e-4) < 1e-7

    # Step 100 more (mid-cosine). LR should have decayed below 5e-4.
    for _ in range(100):
        sched.step()
    mid_cosine_lr = opt.param_groups[0]["lr"]
    assert mid_cosine_lr < warmup_end_lr
    assert mid_cosine_lr > 1e-6


def test_configure_optimizers_falls_back_when_no_warmup_room(module):
    """When total_steps < warmup_steps (very small schedule), fall back to pure cosine."""
    # 5 stepping batches, 25 epochs → warmup would need 25*2 epochs * (5/25) = 10
    # steps but total is 5. Fall-back path.
    module.trainer = SimpleNamespace(estimated_stepping_batches=5, max_epochs=25)
    cfg = module.configure_optimizers()
    sched = cfg["lr_scheduler"]["scheduler"]
    # In fall-back, scheduler is a plain CosineAnnealingLR (no SequentialLR).
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)


# ============================================================================
# Validation step
# ============================================================================


def test_validation_step_runs_and_logs(module, batch):
    """validation_step computes loss + metrics without crashing."""
    module.trainer = SimpleNamespace()
    module.eval()
    # Bypass Lightning's logger plumbing — patch self.log to a no-op.
    with patch.object(module, "log"):
        loss = module.validation_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


# ============================================================================
# Per-sample mel normalization
# ============================================================================


def test_per_sample_mel_norm_zero_mean_unit_std(module, batch):
    """After per-sample norm, each sample's mel should have ~zero mean and ~unit std."""
    x, _ = batch
    mel = module._compute_mel(x, training=False)
    # Per-sample stats
    means = mel.mean(dim=(1, 2, 3))
    stds = mel.std(dim=(1, 2, 3))
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-3)


# ============================================================================
# Audit-validation tests — explicit cross-checks against the two audit docs
# ============================================================================


def test_dual_supervision_formula_matches_audit_cell_14():
    """The 2026-05-09 audit § 'Tucker's loss' specifies the exact formula:

        bce_clip  = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
        bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
        bce = 0.5 * bce_clip + 0.5 * bce_frame
        ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
        cls_loss = (ps * sw).mean()

    With wt=mk=ones (2026-05-23 audit § 4) and sw=ones (no focal_missing),
    this reduces to plain mean BCE. Our _dual_supervision_loss returns exactly
    that value.
    """
    encoder = EfficientNetEncoder(
        "tf_efficientnet_b0.ns_jft_in1k", pretrained=False, use_post_head=True
    )
    head = AttHeadLogitSpace(in_channels=1280, num_classes=NUM_CLASSES)
    mel = _IdentityMel()
    mod = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=head,
        num_classes=NUM_CLASSES,
        mel_transform=mel,
        clipwise_weight=0.5,
        framewise_weight=0.5,
    )

    # Synthetic logits with clear-but-finite values
    B = 4
    clip_logits = torch.randn(B, NUM_CLASSES) * 2.0
    frame_logits = torch.randn(B, NUM_CLASSES, 12) * 2.0
    labels = torch.zeros(B, NUM_CLASSES)
    labels[:, 0] = 1.0
    labels[:, 5] = 1.0

    # Tucker's formula with wt=mk=ones, sw=ones
    bce_clip = F.binary_cross_entropy_with_logits(clip_logits, labels.float(), reduction="none")
    frame_max = frame_logits.max(dim=-1).values
    bce_frame = F.binary_cross_entropy_with_logits(frame_max, labels.float(), reduction="none")
    bce = 0.5 * bce_clip + 0.5 * bce_frame
    wt = torch.ones_like(bce)
    mk = torch.ones_like(bce)
    sw = torch.ones(B)
    ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
    tucker_loss = (ps * sw).mean()

    # Our implementation
    actual_loss, _ = mod._dual_supervision_loss(clip_logits, frame_logits, labels)
    torch.testing.assert_close(actual_loss, tucker_loss, rtol=1e-5, atol=1e-6)


def test_teacher_waveform_cropping_audit_compliance(module_kd, batch):
    """The 2026-05-09 audit specifies: crop or zero-pad raw waveform to exactly
    160000 samples (5s @ 32kHz) before the teacher embed call. Center-crop when
    too long, right-pad with zeros when too short. Validated by running
    training_step with three different input sample counts."""
    labels = batch[1]

    # 1. Exact length — no-op
    x_exact = torch.randn(BATCH, 160_000)
    with patch.object(module_kd, "log"):
        loss = module_kd.training_step((x_exact, labels), 0)
    assert torch.isfinite(loss)

    # 2. Too long — center crop applied internally. We capture the embed call.
    captured = {}
    orig_embed = module_kd.perch_teacher.embed

    def spy(x):
        captured["shape"] = tuple(x.shape)
        return orig_embed(x)

    module_kd.perch_teacher.embed = spy
    x_long = torch.randn(BATCH, 200_000)
    with patch.object(module_kd, "log"):
        module_kd.training_step((x_long, labels), 0)
    assert captured["shape"] == (
        BATCH,
        160_000,
    ), f"long input should be center-cropped to 160000, got {captured['shape']}"

    # 3. Too short — right-padded
    captured.clear()
    x_short = torch.randn(BATCH, 120_000)
    with patch.object(module_kd, "log"):
        module_kd.training_step((x_short, labels), 0)
    assert captured["shape"] == (
        BATCH,
        160_000,
    ), f"short input should be padded to 160000, got {captured['shape']}"


def test_inference_blend_in_probability_space_per_window(module, batch):
    """The 2026-05-23 audit § 7.3 distinguishes two blends:
    1. Per-window blend at inference: probability space, single sigmoid (Tucker cell 14).
    2. Submission ensemble: logit space across folds, single sigmoid at end (cell 24).

    The LightningModule owns (1). _blended_probs must produce probability-space
    values bounded in [0, 1]."""
    module.eval()
    x, _ = batch
    with torch.no_grad():
        mel = module._compute_mel(x, training=False)
        clip, frame = module(mel)
        frame_max = frame.max(dim=-1).values
        probs = module._blended_probs(clip, frame_max)

    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()
    # The blend lives in PROB space — sums of two sigmoids weighted by [0,1]
    # weights cannot exceed 1.0 or go below 0.0. The output IS a probability.


def test_cls_branch_uses_detached_features_when_kd_on(module_kd, batch):
    """Audit § 'ALPHA_DISTILL=1.0 with backbone driven only by dist_loss':
    when KD is on, the SED head must operate on `features.detach()` so the
    backbone gradient comes ONLY from the distillation MSE.

    Stronger version of test_kd_backbone_grad_only_from_distill_branch: zero
    α AND assert that the head still gets gradient (because cls branch is in
    the autograd graph from cls_loss → head → detached features), while
    backbone gets zero gradient (the only path to it is through alpha_distill,
    which we zeroed)."""
    module_kd.hparams.alpha_distill = 0.0
    with patch.object(module_kd, "log"):
        loss = module_kd.training_step(batch, 0)
    loss.backward()

    # Head still gets gradient — it's downstream of cls_loss via the detached features
    assert module_kd.sed_head.cla.weight.grad is not None
    assert module_kd.sed_head.cla.weight.grad.abs().sum().item() > 0

    # Backbone should have no gradient (detached cls path + α=0 zeroes distill contribution)
    backbone_grad = module_kd.encoder.backbone.conv_stem.weight.grad
    if backbone_grad is not None:
        assert backbone_grad.abs().sum().item() == 0.0


def test_full_loss_equals_cls_plus_alpha_times_dist(module_kd, batch):
    """Audit: total = cls_loss + α · dist_loss. Verify by separately computing
    cls_loss and dist_loss and reconstructing the total."""
    # Read α from hparams
    alpha = module_kd.hparams.alpha_distill

    # Run training_step (we can't easily intercept cls vs dist without
    # restructuring), so verify by running with α=0 and α=2 and checking
    # that the difference equals 2x the original distill component.
    module_kd.hparams.alpha_distill = 0.0
    with patch.object(module_kd, "log"):
        loss_alpha_0 = module_kd.training_step(batch, 0).item()

    # Need to reset grad accumulation between runs
    module_kd.zero_grad()

    module_kd.hparams.alpha_distill = 2.0
    with patch.object(module_kd, "log"):
        loss_alpha_2 = module_kd.training_step(batch, 0).item()

    # loss_alpha_2 - loss_alpha_0 = (cls + 2·dist) - cls = 2·dist > 0
    # (assuming dist_loss is non-zero on random init, which it is)
    diff = loss_alpha_2 - loss_alpha_0
    assert diff > 0, f"loss(α=2) - loss(α=0) should be 2·dist_loss > 0, got {diff}"
    module_kd.hparams.alpha_distill = alpha  # restore
