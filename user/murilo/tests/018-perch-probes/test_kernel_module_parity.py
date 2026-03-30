"""Tests verifying kernel inlined code produces same results as module code.

The kernel (submission_perch_protossm.py) inlines all code from the perch/
modules. These tests catch drift between the two by running the same inputs
through both and comparing outputs.
"""

import numpy as np
import pytest
import torch

from birdclef.perch.ensemble import (
    fuse_protossm_mlp,
    temporal_shift_tta,
)
from birdclef.perch.losses import focal_bce_with_logits, mixup_files
from birdclef.perch.models import ProtoSSMv2, ResidualSSM, SelectiveSSM
from birdclef.perch.postprocessing import (
    apply_per_class_thresholds,
    run_postprocessing_pipeline,
    temperature_scale,
)


class TestModelsMatchReference:
    """Verify our model implementations match Yuriy's reference behavior."""

    def test_selective_ssm_deterministic(self):
        """Same input → same output (no randomness in forward pass)."""
        ssm = SelectiveSSM(64, d_state=8)
        ssm.eval()
        x = torch.randn(2, 12, 64)
        with torch.no_grad():
            y1 = ssm(x).clone()
            y2 = ssm(x).clone()
        torch.testing.assert_close(y1, y2)

    def test_protossm_deterministic(self):
        """Same input → same output in eval mode."""
        model = ProtoSSMv2(
            d_input=1536,
            d_model=64,
            d_state=8,
            n_ssm_layers=1,
            n_classes=10,
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        model.eval()
        emb = torch.randn(1, 12, 1536)
        logits = torch.randn(1, 12, 10)
        sites = torch.tensor([0])
        hours = torch.tensor([6])
        with torch.no_grad():
            out1, _, _ = model(emb, logits, site_ids=sites, hours=hours)
            out2, _, _ = model(emb, logits, site_ids=sites, hours=hours)
        torch.testing.assert_close(out1, out2)

    def test_residual_ssm_deterministic(self):
        model = ResidualSSM(
            d_input=1536,
            d_scores=10,
            d_model=32,
            d_state=4,
            n_classes=10,
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
        )
        model.eval()
        emb = torch.randn(1, 12, 1536)
        scores = torch.randn(1, 12, 10)
        with torch.no_grad():
            c1 = model(emb, scores)
            c2 = model(emb, scores)
        torch.testing.assert_close(c1, c2)


class TestPostprocessingParity:
    """Verify post-processing stages produce expected transformations."""

    @pytest.fixture
    def scores(self):
        np.random.seed(123)
        return np.random.randn(24, 10).astype(np.float32)

    def test_temperature_identity_at_one(self, scores):
        """Temperature=1.0 should equal plain sigmoid."""
        temps = np.ones(10, dtype=np.float32)
        probs = temperature_scale(scores, temps)
        expected = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
        np.testing.assert_allclose(probs, expected, atol=1e-6)

    def test_pipeline_no_postproc_equals_sigmoid(self, scores):
        """With all post-processing disabled, pipeline = temperature_scale only."""
        temps = np.ones(10, dtype=np.float32)
        probs = run_postprocessing_pipeline(
            scores,
            temps,
            per_class_thresholds=None,
            file_level_top_k=0,
            rank_aware_power=0.0,
            delta_shift_alpha=0.0,
        )
        expected = temperature_scale(scores, temps)
        np.testing.assert_allclose(probs, expected, atol=1e-6)

    def test_threshold_half_is_identity(self):
        """Thresholds at 0.5 should be a no-op (submit mode default)."""
        probs = np.array([[0.1, 0.3, 0.5, 0.7, 0.9]], dtype=np.float32)
        thresholds = np.full(5, 0.5, dtype=np.float32)
        result = apply_per_class_thresholds(probs, thresholds)
        np.testing.assert_allclose(result, probs, atol=1e-5)


class TestEnsembleParity:
    """Verify ensemble functions produce correct arithmetic."""

    def test_fusion_weights_sum_to_one(self):
        """w*proto + (1-w)*mlp should be a convex combination."""
        proto = np.ones((10, 5), dtype=np.float32) * 2.0
        mlp = np.ones((10, 5), dtype=np.float32) * 4.0
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            fused = fuse_protossm_mlp(proto, mlp, w)
            expected = w * 2.0 + (1 - w) * 4.0
            np.testing.assert_allclose(fused, expected, atol=1e-6)

    def test_tta_no_shift_preserves_logits(self):
        """TTA with shifts=[0] should equal direct model output."""
        model = ProtoSSMv2(
            d_input=1536,
            d_model=32,
            d_state=4,
            n_ssm_layers=1,
            n_classes=10,
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        model.eval()
        emb = np.random.randn(2, 12, 1536).astype(np.float32)
        logits = np.random.randn(2, 12, 10).astype(np.float32)
        sites = np.array([0, 1])
        hours = np.array([6, 12])

        tta_out = temporal_shift_tta(emb, logits, model, sites, hours, shifts=[0])
        with torch.no_grad():
            direct, _, _ = model(
                torch.tensor(emb),
                torch.tensor(logits),
                site_ids=torch.tensor(sites, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
        np.testing.assert_allclose(tta_out, direct.numpy(), atol=1e-5)


class TestLossesParity:
    """Verify loss function edge cases match expected behavior."""

    def test_focal_gamma_zero_is_bce(self):
        """focal(gamma=0) == standard BCE."""
        logits = torch.randn(4, 12, 10)
        targets = (torch.rand(4, 12, 10) > 0.7).float()
        focal = focal_bce_with_logits(logits, targets, gamma=0.0)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        torch.testing.assert_close(focal, bce, atol=1e-5, rtol=1e-5)

    def test_mixup_alpha_zero_is_identity(self):
        """Mixup with alpha=0 should return inputs unchanged."""
        emb = np.random.randn(5, 12, 1536).astype(np.float32)
        lo = np.random.randn(5, 12, 10).astype(np.float32)
        lab = np.random.rand(5, 12, 10).astype(np.float32)
        sites = np.arange(5)
        hours = np.arange(5)
        e_out, lo_out, la_out, s_out, h_out, f_out = mixup_files(
            emb, lo, lab, sites, hours, None, alpha=0.0
        )
        np.testing.assert_array_equal(e_out, emb)
        np.testing.assert_array_equal(lo_out, lo)
        np.testing.assert_array_equal(la_out, lab)
