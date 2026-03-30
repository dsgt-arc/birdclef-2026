"""Tests for configuration values.

Cross-ref: ref_yuriy_protossm_v5.py lines 162-253 (CFG dict).
Verifies all frozen hyperparameters match Yuriy's reference implementation.
"""

from birdclef.perch.config import (
    get_postprocessing_config,
    get_probe_config,
    get_protossm_config,
    get_protossm_train_config,
    get_residual_ssm_config,
)


class TestProtoSSMConfig:
    """Verify ProtoSSM config matches Yuriy's CFG['proto_ssm']."""

    def test_d_model(self):
        assert get_protossm_config()["d_model"] == 128

    def test_d_state(self):
        assert get_protossm_config()["d_state"] == 16

    def test_n_ssm_layers(self):
        assert get_protossm_config()["n_ssm_layers"] == 2

    def test_dropout(self):
        assert get_protossm_config()["dropout"] == 0.15

    def test_n_sites(self):
        assert get_protossm_config()["n_sites"] == 20

    def test_meta_dim(self):
        assert get_protossm_config()["meta_dim"] == 16

    def test_cross_attention_enabled(self):
        assert get_protossm_config()["use_cross_attn"] is True

    def test_cross_attn_heads(self):
        assert get_protossm_config()["cross_attn_heads"] == 4


class TestTrainConfig:
    """Verify training config matches Yuriy's CFG['proto_ssm_train']."""

    def test_n_epochs(self):
        assert get_protossm_train_config()["n_epochs"] == 60

    def test_lr(self):
        assert get_protossm_train_config()["lr"] == 1e-3

    def test_weight_decay(self):
        assert get_protossm_train_config()["weight_decay"] == 2e-3

    def test_patience(self):
        assert get_protossm_train_config()["patience"] == 15

    def test_pos_weight_cap(self):
        assert get_protossm_train_config()["pos_weight_cap"] == 30.0

    def test_distill_weight(self):
        assert get_protossm_train_config()["distill_weight"] == 0.1

    def test_label_smoothing(self):
        assert get_protossm_train_config()["label_smoothing"] == 0.02

    def test_oof_n_splits(self):
        assert get_protossm_train_config()["oof_n_splits"] == 3

    def test_mixup_alpha(self):
        assert get_protossm_train_config()["mixup_alpha"] == 0.3

    def test_focal_gamma(self):
        assert get_protossm_train_config()["focal_gamma"] == 2.0

    def test_swa_start_frac(self):
        assert get_protossm_train_config()["swa_start_frac"] == 0.7


class TestResidualSSMConfig:
    """Verify ResidualSSM config matches Yuriy's CFG['residual_ssm']."""

    def test_d_model(self):
        assert get_residual_ssm_config()["d_model"] == 64

    def test_d_state(self):
        assert get_residual_ssm_config()["d_state"] == 8

    def test_correction_weight(self):
        assert get_residual_ssm_config()["correction_weight"] == 0.3

    def test_n_epochs(self):
        assert get_residual_ssm_config()["n_epochs"] == 30

    def test_patience(self):
        assert get_residual_ssm_config()["patience"] == 8


class TestProbeConfig:
    """Verify probe config matches Yuriy's CFG['frozen_best_probe']."""

    def test_pca_dim(self):
        assert get_probe_config()["pca_dim"] == 64

    def test_min_pos(self):
        assert get_probe_config()["min_pos"] == 8

    def test_c(self):
        assert get_probe_config()["C"] == 0.50

    def test_alpha(self):
        assert get_probe_config()["alpha"] == 0.40

    def test_mlp_hidden(self):
        assert get_probe_config()["mlp_params"]["hidden_layer_sizes"] == (128,)


class TestPostprocessingConfig:
    """Verify post-processing config matches Yuriy's CFG."""

    def test_temperature_aves(self):
        assert get_postprocessing_config()["temperature_aves"] == 1.10

    def test_temperature_texture(self):
        assert get_postprocessing_config()["temperature_texture"] == 0.95

    def test_file_level_top_k(self):
        assert get_postprocessing_config()["file_level_top_k"] == 2

    def test_rank_aware_power(self):
        assert get_postprocessing_config()["rank_aware_power"] == 0.5

    def test_delta_shift_alpha(self):
        assert get_postprocessing_config()["delta_shift_alpha"] == 0.15

    def test_tta_shifts(self):
        assert get_postprocessing_config()["tta_shifts"] == [0, 1, -1]
