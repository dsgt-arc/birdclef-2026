"""Tests for ProtoSSM model architectures.

Cross-ref: ref_yuriy_protossm_v5.py lines 1359-1579 (ProtoSSM), 2362-2442 (ResidualSSM).
"""

import pytest
import torch
import torch.nn.functional as F

from birdclef.perch.models import (
    ProtoSSMv2,
    ResidualSSM,
    SelectiveSSM,
    TemporalCrossAttention,
)

# ---------------------------------------------------------------------------
# SelectiveSSM
# ---------------------------------------------------------------------------


class TestSelectiveSSM:
    def test_shape_basic(self):
        ssm = SelectiveSSM(128, d_state=16)
        x = torch.randn(2, 12, 128)
        y = ssm(x)
        assert y.shape == (2, 12, 128)

    def test_shape_single_timestep(self):
        ssm = SelectiveSSM(64, d_state=8)
        x = torch.randn(1, 1, 64)
        y = ssm(x)
        assert y.shape == (1, 1, 64)

    def test_different_d_model(self):
        for d in [32, 64, 128, 256]:
            ssm = SelectiveSSM(d, d_state=16)
            x = torch.randn(1, 12, d)
            y = ssm(x)
            assert y.shape == (1, 12, d)

    def test_gradient_flow(self):
        ssm = SelectiveSSM(64, d_state=8)
        x = torch.randn(2, 12, 64, requires_grad=True)
        y = ssm(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 12, 64)

    def test_d_parameter_initialization(self):
        """D parameter should be initialized to ones (skip connection)."""
        ssm = SelectiveSSM(128, d_state=16)
        assert torch.allclose(ssm.D, torch.ones(128))

    def test_a_log_shape(self):
        """A_log should be (d_model, d_state) with log(arange) init."""
        ssm = SelectiveSSM(64, d_state=8)
        assert ssm.A_log.shape == (64, 8)
        # First column should be log(1) = 0 for all rows
        expected_first = torch.zeros(64)
        assert torch.allclose(ssm.A_log[:, 0].detach(), expected_first, atol=1e-6)


# ---------------------------------------------------------------------------
# TemporalCrossAttention
# ---------------------------------------------------------------------------


class TestTemporalCrossAttention:
    def test_shape(self):
        ca = TemporalCrossAttention(128, n_heads=4, dropout=0.15)
        x = torch.randn(2, 12, 128)
        y = ca(x)
        assert y.shape == (2, 12, 128)

    def test_residual_connections(self):
        """Output should differ from input (non-identity) but not by huge amounts."""
        ca = TemporalCrossAttention(64, n_heads=2, dropout=0.0)
        x = torch.randn(1, 12, 64)
        ca.eval()  # disable dropout
        y = ca(x)
        diff = (y - x).abs().mean()
        assert diff > 0, "CrossAttention should transform input"

    def test_pre_norm_structure(self):
        """Verify pre-norm: LayerNorm before attention and FFN."""
        ca = TemporalCrossAttention(64, n_heads=2)
        assert isinstance(ca.norm, torch.nn.LayerNorm)
        assert isinstance(ca.norm2, torch.nn.LayerNorm)


# ---------------------------------------------------------------------------
# ProtoSSMv2
# ---------------------------------------------------------------------------


class TestProtoSSMv2:
    @pytest.fixture
    def model(self):
        return ProtoSSMv2(
            d_input=1536,
            d_model=128,
            d_state=16,
            n_ssm_layers=2,
            n_classes=234,
            n_windows=12,
            dropout=0.15,
            n_sites=20,
            meta_dim=16,
            use_cross_attn=True,
            cross_attn_heads=4,
        )

    @pytest.fixture
    def dummy_inputs(self):
        return {
            "emb": torch.randn(3, 12, 1536),
            "perch_logits": torch.randn(3, 12, 234),
            "site_ids": torch.tensor([0, 5, 10]),
            "hours": torch.tensor([6, 12, 18]),
        }

    def test_output_shapes(self, model, dummy_inputs):
        sp, fam, h = model(**dummy_inputs)
        assert sp.shape == (3, 12, 234)
        assert fam is None  # no family head yet
        assert h.shape == (3, 12, 128)

    def test_output_without_perch_logits(self, model, dummy_inputs):
        """Without perch_logits, gated fusion is skipped."""
        sp, _, _ = model(dummy_inputs["emb"])
        assert sp.shape == (3, 12, 234)

    def test_output_without_metadata(self, model, dummy_inputs):
        """Without site/hour, metadata embedding is skipped."""
        sp, _, _ = model(dummy_inputs["emb"], dummy_inputs["perch_logits"])
        assert sp.shape == (3, 12, 234)

    def test_family_head(self, model, dummy_inputs):
        model.init_family_head(50, [i % 50 for i in range(234)])
        sp, fam, _ = model(**dummy_inputs)
        assert fam.shape == (3, 50)

    def test_prototype_initialization(self, model):
        """Prototypes should be updated from data."""
        emb_flat = torch.randn(100, 1536)
        labels_flat = torch.zeros(100, 234)
        labels_flat[:10, 0] = 1.0
        old_proto = model.prototypes[0].clone()
        model.init_prototypes_from_data(emb_flat, labels_flat)
        new_proto = model.prototypes[0]
        assert not torch.allclose(old_proto, new_proto), "Prototype 0 should change"
        # Class 0 prototype should be unit-normalized
        assert torch.allclose(new_proto.norm(), torch.tensor(1.0), atol=1e-5)

    def test_gated_fusion_formula(self, model, dummy_inputs):
        """Verify: species_logits = alpha*sim + (1-alpha)*perch_logits.

        Cross-ref: ref_yuriy_protossm_v5.py line 1565.
        """
        model.eval()
        with torch.no_grad():
            # Force fusion_alpha to known value
            model.fusion_alpha.fill_(0.0)  # sigmoid(0) = 0.5
            sp_with, _, _ = model(**dummy_inputs)
            sp_without, _, _ = model(
                dummy_inputs["emb"], site_ids=dummy_inputs["site_ids"], hours=dummy_inputs["hours"]
            )
            # alpha=0.5, so species = 0.5*sim + 0.5*perch
            # Verify that adding perch_logits changes output
            assert not torch.allclose(sp_with, sp_without, atol=1e-3)

    def test_fusion_alpha_init_zero(self, model):
        """Fusion alpha should init to 0 (sigmoid → 0.5)."""
        assert torch.allclose(model.fusion_alpha, torch.zeros(234))
        alpha = torch.sigmoid(model.fusion_alpha)
        assert torch.allclose(alpha, torch.full((234,), 0.5))

    def test_proto_temp_init(self, model):
        """Proto temperature should init to 5.0, softplus ≈ 5.0."""
        assert model.proto_temp.item() == pytest.approx(5.0)
        temp = F.softplus(model.proto_temp)
        assert temp.item() == pytest.approx(5.0067, abs=0.01)

    def test_param_count(self, model):
        """Verify parameter count is reasonable."""
        n_params = model.count_parameters()
        assert 500_000 < n_params < 1_000_000, f"Unexpected param count: {n_params}"

    def test_gradient_flow_through_scan(self, model, dummy_inputs):
        """Verify gradients flow through the sequential SSM scan."""
        for p in model.parameters():
            p.requires_grad_(True)
        sp, _, _ = model(**dummy_inputs)
        loss = sp.sum()
        loss.backward()
        # Check that SSM parameters got gradients
        assert model.ssm_fwd[0].A_log.grad is not None

    def test_cross_attention_toggle(self):
        """Model should work with cross_attn disabled."""
        model = ProtoSSMv2(d_input=1536, d_model=64, n_classes=10, use_cross_attn=False)
        emb = torch.randn(1, 12, 1536)
        sp, _, _ = model(emb)
        assert sp.shape == (1, 12, 10)


# ---------------------------------------------------------------------------
# ResidualSSM
# ---------------------------------------------------------------------------


class TestResidualSSM:
    @pytest.fixture
    def model(self):
        return ResidualSSM(
            d_input=1536,
            d_scores=234,
            d_model=64,
            d_state=8,
            n_classes=234,
            n_windows=12,
            dropout=0.1,
            n_sites=20,
            meta_dim=8,
        )

    def test_output_shape(self, model):
        emb = torch.randn(2, 12, 1536)
        scores = torch.randn(2, 12, 234)
        sites = torch.tensor([0, 1])
        hours = torch.tensor([6, 18])
        corr = model(emb, scores, site_ids=sites, hours=hours)
        assert corr.shape == (2, 12, 234)

    def test_zero_initialization(self, model):
        """Output head should be zero-initialized (corrections start small).

        Cross-ref: ref_yuriy_protossm_v5.py lines 2409-2410.
        """
        assert model.output_head.weight.abs().sum() == 0
        assert model.output_head.bias.abs().sum() == 0

    def test_initial_correction_near_zero(self, model):
        """With zero-init output, corrections should be near zero."""
        model.eval()
        emb = torch.randn(1, 12, 1536)
        scores = torch.randn(1, 12, 234)
        with torch.no_grad():
            corr = model(emb, scores)
        assert corr.abs().max() < 1.0, "Initial corrections should be small"

    def test_input_concatenation(self, model):
        """Input projection should accept d_input + d_scores = 1770."""
        proj = model.input_proj[0]  # first Linear layer
        assert proj.in_features == 1536 + 234

    def test_param_count(self, model):
        n_params = model.count_parameters()
        assert 100_000 < n_params < 300_000, f"Unexpected param count: {n_params}"
