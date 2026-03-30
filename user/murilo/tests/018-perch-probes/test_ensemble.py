"""Tests for TTA, ensemble weight optimization, and score fusion.

Cross-ref: ref_yuriy_protossm_v5.py lines 562-589 (TTA), 2001-2015 (ensemble).
"""

import numpy as np
import pytest
import torch

from birdclef.perch.ensemble import (
    fuse_protossm_mlp,
    macro_auc_skip_empty,
    optimize_ensemble_weight,
    temporal_shift_tta,
)
from birdclef.perch.models import ProtoSSMv2


class TestMacroAUC:
    def test_perfect_prediction(self):
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
        auc = macro_auc_skip_empty(y_true, y_score)
        assert auc == 1.0

    def test_skip_empty_classes(self):
        """Classes with no positive samples should be skipped."""
        y_true = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0]])
        y_score = np.array([[0.9, 0.5, 0.5], [0.1, 0.5, 0.5], [0.8, 0.3, 0.3]])
        auc = macro_auc_skip_empty(y_true, y_score)
        # Only class 0 has positives (with both pos and neg samples), AUC should be 1.0
        assert auc == 1.0

    def test_all_empty_returns_zero(self):
        y_true = np.zeros((10, 5))
        y_score = np.random.rand(10, 5)
        auc = macro_auc_skip_empty(y_true, y_score)
        assert auc == 0.0


class TestTemporalShiftTTA:
    @pytest.fixture
    def model(self):
        m = ProtoSSMv2(
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
        m.eval()
        return m

    def test_no_shift_equals_direct(self, model):
        """TTA with shifts=[0] should equal a direct forward pass."""
        emb = np.random.randn(2, 12, 1536).astype(np.float32)
        logits = np.random.randn(2, 12, 10).astype(np.float32)
        sites = np.array([0, 1])
        hours = np.array([6, 12])

        tta_out = temporal_shift_tta(emb, logits, model, sites, hours, shifts=[0])

        with torch.no_grad():
            direct_out, _, _ = model(
                torch.tensor(emb),
                torch.tensor(logits),
                site_ids=torch.tensor(sites, dtype=torch.long),
                hours=torch.tensor(hours, dtype=torch.long),
            )
        np.testing.assert_allclose(tta_out, direct_out.numpy(), atol=1e-5)

    def test_tta_output_shape(self, model):
        emb = np.random.randn(3, 12, 1536).astype(np.float32)
        logits = np.random.randn(3, 12, 10).astype(np.float32)
        sites = np.array([0, 1, 2])
        hours = np.array([6, 12, 18])

        out = temporal_shift_tta(emb, logits, model, sites, hours, shifts=[0, 1, -1])
        assert out.shape == (3, 12, 10)

    def test_tta_shift_and_unshift(self, model):
        """Circular shift then unshift should preserve temporal position.

        Cross-ref: ref_yuriy_protossm_v5.py lines 572-585.
        """
        emb = np.random.randn(1, 12, 1536).astype(np.float32)
        # Shift +1 then unshift -1 should give same temporal positions
        shifted = np.roll(emb, 1, axis=1)
        unshifted = np.roll(shifted, -1, axis=1)
        np.testing.assert_array_equal(emb, unshifted)


class TestOptimizeEnsembleWeight:
    def test_basic_optimization(self):
        np.random.seed(42)
        n = 100
        y_true = np.zeros((n, 5))
        y_true[:30, 0] = 1.0
        y_true[30:60, 1] = 1.0

        # Make proto better than mlp
        proto = y_true + np.random.randn(n, 5) * 0.1
        mlp = y_true + np.random.randn(n, 5) * 0.5

        best_w, best_auc, results = optimize_ensemble_weight(proto, mlp, y_true)
        assert 0.0 <= best_w <= 1.0
        assert best_auc > 0.5
        assert len(results) == 21  # 0.0, 0.05, ..., 1.0

    def test_weight_one_pure_proto(self):
        proto = np.array([[1.0, 0.0]])
        mlp = np.array([[0.0, 1.0]])
        fused = fuse_protossm_mlp(proto, mlp, weight_proto=1.0)
        np.testing.assert_array_equal(fused, proto)

    def test_weight_zero_pure_mlp(self):
        proto = np.array([[1.0, 0.0]])
        mlp = np.array([[0.0, 1.0]])
        fused = fuse_protossm_mlp(proto, mlp, weight_proto=0.0)
        np.testing.assert_array_equal(fused, mlp)

    def test_fusion_formula(self):
        """Verify: final = w*proto + (1-w)*mlp."""
        proto = np.array([[2.0, 4.0]])
        mlp = np.array([[6.0, 8.0]])
        fused = fuse_protossm_mlp(proto, mlp, weight_proto=0.3)
        expected = 0.3 * proto + 0.7 * mlp
        np.testing.assert_allclose(fused, expected, atol=1e-6)
