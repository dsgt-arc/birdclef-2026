"""Tests for `distilled_sed_control.yaml` and `distilled_sed_kd.yaml` (exp 032 task 0.9).

Locks in audit compliance: every hyperparameter value that's load-bearing
per the 2026-05-09 + 2026-05-23 audits is asserted here. Drift in the configs
should fail loudly.

Coverage:
- Backbone: tf_efficientnet_b0.ns_jft_in1k, use_post_head=True, drop_path=0.1
- Head: att_head_logit_space, hidden_dim=512, dropout=0.5, gem_p=3.0
- Mel: n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000, top_db=80
- SpecAugment: freq_p=10, time_p=10, n_freq=1, n_time=2
- Mixup: focal_focal=0.5, alpha=0.4, hard=True, focal_sc=0.5, aug_prob=0.5
- Schedule: lr=5e-4, wd=1e-4, max_epochs=25, warmup=2, warmup_start_factor=1/25,
  eta_min=1e-6, grad_clip=1.0
- Data: bs=64, focal_share=0.9, sc_share=0.1, n_folds=5, min_sample=20
- Loss / blend: 50/50 for both
- KD: alpha_distill=1.0, perch_embed_dim=1536
- Save: save_top_k=-1 (avoids the exp 031 best-vs-last pitfall)
- Paired-control invariant: control and kd differ ONLY in run_name +
  use_perch_distill + perch_onnx_path

Reference:
- analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md
- analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md (§§ 1-5)
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cells 4, 7, 10, 12, 14)
"""

from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = (
    Path(__file__).parent.parent.parent / "experiments" / "032-tucker-distilled-sed" / "configs"
)


@pytest.fixture(scope="module")
def control_cfg():
    with open(CONFIGS_DIR / "distilled_sed_control.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def kd_cfg():
    with open(CONFIGS_DIR / "distilled_sed_kd.yaml") as f:
        return yaml.safe_load(f)


# ============================================================================
# Backbone (audit § 7.1 + task 0.2)
# ============================================================================


def test_backbone_is_tf_effnet_b0_ns_jft(control_cfg, kd_cfg):
    """Tucker uses tf_efficientnet_b0.ns_jft_in1k (NS-JFT, not EffNetV2)."""
    for name, cfg in [("control", control_cfg), ("kd", kd_cfg)]:
        assert cfg["encoder"]["model_name"] == "tf_efficientnet_b0.ns_jft_in1k", name


def test_backbone_use_post_head_true(control_cfg, kd_cfg):
    """Tucker mode (1280-d post-conv_head, not 320-d features_only). Audit § 2."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["encoder"]["use_post_head"] is True


def test_backbone_drop_path_rate(control_cfg, kd_cfg):
    """Tucker's drop_path_rate=0.1."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["encoder"]["drop_path_rate"] == 0.1


def test_in_chans_1(control_cfg, kd_cfg):
    """1-channel mel input (Tucker's standard)."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["encoder"]["in_chans"] == 1


# ============================================================================
# Head (audit § Tucker's attention head)
# ============================================================================


def test_head_is_att_head_logit_space(control_cfg, kd_cfg):
    """AttHeadLogitSpace (task 0.1 — Tucker's logit-space aggregation)."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["sed_head"]["type"] == "att_head_logit_space"


def test_head_hidden_dim_512(control_cfg, kd_cfg):
    """Tucker's hidden_dim=512."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["sed_head"]["hidden_dim"] == 512


def test_head_dropout_05(control_cfg, kd_cfg):
    """dropout=0.5 produces Tucker's Dropout(0.25) → Dropout(0.5) sequence."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["sed_head"]["dropout"] == 0.5


def test_head_gem_p_3(control_cfg, kd_cfg):
    """Tucker's GeMFreqPool initial p=3.0."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["sed_head"]["gem_p"] == 3.0


# ============================================================================
# Mel pipeline (audit § 7.1)
# ============================================================================


def test_mel_params_match_tucker_cell_4(control_cfg, kd_cfg):
    """Tucker's mel: n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000, top_db=80."""
    for cfg in [control_cfg, kd_cfg]:
        m = cfg["mel"]
        assert m["n_fft"] == 2048
        assert m["hop_length"] == 512
        assert m["n_mels"] == 256
        assert m["fmin"] == 20
        assert m["fmax"] == 16000
        assert m["top_db"] == 80.0


# ============================================================================
# SpecAugment (audit § 2.4)
# ============================================================================


def test_spec_augment_params_match_tucker_cell_10(control_cfg, kd_cfg):
    """Tucker's SpecAugment: freq=10, time=10, n_freq=1, n_time=2."""
    for cfg in [control_cfg, kd_cfg]:
        sa = cfg["spec_augment"]
        assert sa["enabled"] is True
        assert sa["freq_mask_param"] == 10
        assert sa["time_mask_param"] == 10
        assert sa["num_freq_masks"] == 1
        assert sa["num_time_masks"] == 2


# ============================================================================
# Mixup (audit § 2.3)
# ============================================================================


def test_mixup_params_match_tucker_cell_12(control_cfg, kd_cfg):
    """Tucker's mixup: focal_focal_p=0.5, alpha=0.4, MIXUP_HARD=True,
    focal_sc_p=0.5, aug_prob=0.5."""
    for cfg in [control_cfg, kd_cfg]:
        m = cfg["mixup"]
        assert m["focal_focal_prob"] == 0.5
        assert m["alpha"] == 0.4
        assert m["hard"] is True
        assert m["focal_sc_prob"] == 0.5
        assert m["focal_sc_alpha"] == 0.4
        assert m["aug_prob"] == 0.5


# ============================================================================
# Schedule (audit § 5)
# ============================================================================


def test_schedule_params_match_tucker_cell_14(control_cfg, kd_cfg):
    """Tucker: AdamW(lr=5e-4, wd=1e-4), warmup=2 ep with start_factor=1/25,
    cosine to eta_min=1e-6, total 25 ep, grad_clip=1.0."""
    for cfg in [control_cfg, kd_cfg]:
        t = cfg["training"]
        assert t["lr"] == 5e-4
        assert t["weight_decay"] == 1e-4
        assert t["max_epochs"] == 25
        assert t["warmup_epochs"] == 2
        assert t["warmup_start_factor"] == 0.04  # 1/25
        assert t["eta_min"] == 1e-6
        assert t["grad_clip_val"] == 1.0
        assert t["precision"] == "16-mixed"


# ============================================================================
# Data + sampler (audit §§ 1, 3)
# ============================================================================


def test_data_params_match_tucker(control_cfg, kd_cfg):
    """Batch=64, MixSamp shares 0.9/0.1, 5-fold, min_sample=20, sr=32000."""
    for cfg in [control_cfg, kd_cfg]:
        d = cfg["data"]
        assert d["batch_size"] == 64
        assert d["focal_share"] == 0.9
        assert d["sc_share"] == 0.1
        assert d["n_folds"] == 5
        assert d["min_sample"] == 20
        assert d["sample_rate"] == 32000
        assert d["window_duration"] == 5.0
        assert d["seed"] == 42
        assert d["base_seed"] == 42


def test_data_path_exists_or_documented(control_cfg, kd_cfg):
    """data_dir path is the standard PACE shared data location."""
    for cfg in [control_cfg, kd_cfg]:
        d = cfg["data"]
        assert d["data_dir"].startswith("/storage/project/")


# ============================================================================
# Loss + inference blend (audit § Tucker's loss)
# ============================================================================


def test_loss_and_blend_weights_50_50(control_cfg, kd_cfg):
    """Tucker's dual-supervision = 0.5/0.5; inference blend = 0.5/0.5."""
    for cfg in [control_cfg, kd_cfg]:
        t = cfg["training"]
        assert t["clipwise_weight"] == 0.5
        assert t["framewise_weight"] == 0.5
        assert t["blend_clip"] == 0.5
        assert t["blend_frame"] == 0.5


def test_per_sample_mel_norm_enabled(control_cfg, kd_cfg):
    """Tucker normalizes per-sample (mean, std) on mel before SpecAugment (cell 14)."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["training"]["per_sample_mel_norm"] is True


# ============================================================================
# KD branch toggling — the paired-control invariant
# ============================================================================


def test_control_has_kd_off(control_cfg):
    """distilled_sed_control.yaml: KD must be disabled."""
    assert control_cfg["training"]["use_perch_distill"] is False


def test_kd_has_kd_on(kd_cfg):
    """distilled_sed_kd.yaml: KD must be enabled."""
    assert kd_cfg["training"]["use_perch_distill"] is True


def test_kd_perch_params(kd_cfg):
    """Tucker's KD: alpha_distill=1.0, perch_embed_dim=1536, valid onnx_path."""
    t = kd_cfg["training"]
    assert t["alpha_distill"] == 1.0
    assert t["perch_embed_dim"] == 1536
    assert "perch_onnx_path" in t
    assert t["perch_onnx_path"].endswith(".onnx")


def test_perch_onnx_file_exists(kd_cfg):
    """The Perch ONNX file referenced by distilled_sed_kd.yaml exists on disk."""
    # test_configs.py is at <repo>/user/murilo/tests/032-tucker-distilled-sed/
    # → 5 parents up = repo root
    repo_root = Path(__file__).parents[4]
    onnx_path = repo_root / kd_cfg["training"]["perch_onnx_path"]
    assert onnx_path.is_file(), f"Perch ONNX not found at {onnx_path}"


# ============================================================================
# Paired-control invariant: control and kd are bit-identical except for KD keys
# ============================================================================


def test_paired_invariant_only_kd_keys_differ(control_cfg, kd_cfg):
    """The two configs must be identical except for: run_name,
    training.use_perch_distill, training.perch_onnx_path. Any other drift is
    a methodology violation."""
    allowed_diffs = {
        "run_name",
        "training.use_perch_distill",
        "training.perch_onnx_path",
    }

    def collect_diffs(a, b, prefix=""):
        diffs = set()
        for key in set(a.keys()) | set(b.keys()):
            v_a = a.get(key)
            v_b = b.get(key)
            if isinstance(v_a, dict) and isinstance(v_b, dict):
                diffs |= collect_diffs(v_a, v_b, prefix + key + ".")
            elif v_a != v_b:
                diffs.add(prefix + key)
        return diffs

    actual_diffs = collect_diffs(control_cfg, kd_cfg)
    unexpected = actual_diffs - allowed_diffs
    assert not unexpected, (
        f"control vs kd diverge on unexpected keys: {unexpected}. "
        f"Only {allowed_diffs} are allowed to differ."
    )


# ============================================================================
# Save / checkpoint policy
# ============================================================================


def test_save_top_k_minus_one(control_cfg, kd_cfg):
    """save_top_k=-1 saves every epoch — avoids the exp 031 best-vs-last pitfall
    (`feedback_save_top_k_does_not_preserve_last`)."""
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["callbacks"]["model_checkpoint"]["save_top_k"] == -1


def test_early_stopping_disabled(control_cfg, kd_cfg):
    """Tucker trains the full 25 epochs — no early stopping."""
    for cfg in [control_cfg, kd_cfg]:
        # `patience=999` effectively disables ES (will never trigger in 25 ep)
        assert cfg["callbacks"]["early_stopping"]["patience"] >= 100


def test_logging_project_exp032(control_cfg, kd_cfg):
    for cfg in [control_cfg, kd_cfg]:
        assert cfg["logging"]["project"] == "birdclef-2026-exp032"


# ============================================================================
# Run names are distinct
# ============================================================================


def test_run_names_distinct(control_cfg, kd_cfg):
    assert control_cfg["run_name"] != kd_cfg["run_name"]
    assert "control" in control_cfg["run_name"]
    assert "kd" in kd_cfg["run_name"]
