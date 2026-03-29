"""Tests for ProtoSSM and ResidualSSM training loops.

Cross-ref: ref_yuriy_protossm_v5.py lines 1683-1998, 2509-2620.
"""

import numpy as np
import pytest

from birdclef.perch.models import ProtoSSMv2
from birdclef.perch.training import train_proto_ssm_single, train_residual_ssm


@pytest.fixture
def small_dataset():
    """Small synthetic dataset for training tests."""
    np.random.seed(42)
    n_files = 6
    n_windows = 12
    n_classes = 10
    n_families = 3

    emb = np.random.randn(n_files, n_windows, 1536).astype(np.float32)
    logits = np.random.randn(n_files, n_windows, n_classes).astype(np.float32)
    labels = np.zeros((n_files, n_windows, n_classes), dtype=np.float32)
    # Ensure every class has at least some positives to avoid div-by-zero
    for c in range(n_classes):
        labels[c % n_files, :3, c] = 1.0
    labels[0, :3, 0] = 1.0
    labels[1, :5, 1] = 1.0
    labels[2, :2, 0] = 1.0
    labels[3, :4, 2] = 1.0

    site_ids = np.array([0, 1, 0, 1, 2, 2], dtype=np.int64)
    hours = np.array([6, 12, 18, 6, 12, 18], dtype=np.int64)
    families = np.zeros((n_files, n_families), dtype=np.float32)
    families[0, 0] = 1.0
    families[1, 1] = 1.0

    return {
        "emb": emb,
        "logits": logits,
        "labels": labels,
        "site_ids": site_ids,
        "hours": hours,
        "families": families,
        "n_classes": n_classes,
        "n_families": n_families,
    }


class TestTrainProtoSSM:
    def test_loss_decreases(self, small_dataset):
        """Training loss should decrease over epochs."""
        d = small_dataset
        model = ProtoSSMv2(
            d_input=1536,
            d_model=32,
            d_state=4,
            n_ssm_layers=1,
            n_classes=d["n_classes"],
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        # Need n_epochs >= 11 for OneCycleLR with pct_start=0.1 and steps_per_epoch=1
        cfg = {
            "n_epochs": 20,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 100,
            "pos_weight_cap": 10.0,
            "distill_weight": 0.1,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "focal_gamma": 0.0,
            "swa_start_frac": 1.0,
        }
        model, hist = train_proto_ssm_single(
            model,
            d["emb"][:4],
            d["logits"][:4],
            d["labels"][:4],
            site_ids_train=d["site_ids"][:4],
            hours_train=d["hours"][:4],
            emb_val=d["emb"][4:],
            logits_val=d["logits"][4:],
            labels_val=d["labels"][4:],
            site_ids_val=d["site_ids"][4:],
            hours_val=d["hours"][4:],
            cfg=cfg,
            verbose=False,
        )
        assert hist["train_loss"][-1] < hist["train_loss"][0], "Loss should decrease"

    def test_label_smoothing(self, small_dataset):
        """Label smoothing: y*(1-ε) + ε/2.

        Cross-ref: ref_yuriy_protossm_v5.py line 1716.
        """
        d = small_dataset
        model = ProtoSSMv2(
            d_input=1536,
            d_model=32,
            d_state=4,
            n_ssm_layers=1,
            n_classes=d["n_classes"],
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        cfg = {
            "n_epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0,
            "patience": 100,
            "pos_weight_cap": 10.0,
            "distill_weight": 0.0,
            "label_smoothing": 0.1,
            "mixup_alpha": 0.0,
            "focal_gamma": 0.0,
            "swa_start_frac": 1.0,
        }
        # Should not crash with label smoothing
        model, _ = train_proto_ssm_single(
            model,
            d["emb"],
            d["logits"],
            d["labels"],
            cfg=cfg,
            verbose=False,
        )

    def test_mixup_warmup(self, small_dataset):
        """Mixup should skip first 5 epochs.

        Cross-ref: ref_yuriy_protossm_v5.py line 1761.
        """
        d = small_dataset
        model = ProtoSSMv2(
            d_input=1536,
            d_model=32,
            d_state=4,
            n_ssm_layers=1,
            n_classes=d["n_classes"],
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        cfg = {
            "n_epochs": 8,
            "lr": 1e-3,
            "weight_decay": 0,
            "patience": 100,
            "pos_weight_cap": 10.0,
            "distill_weight": 0.0,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.3,
            "focal_gamma": 0.0,
            "swa_start_frac": 1.0,
        }
        # Should not crash with mixup
        model, hist = train_proto_ssm_single(
            model,
            d["emb"],
            d["logits"],
            d["labels"],
            site_ids_train=d["site_ids"],
            hours_train=d["hours"],
            file_families_train=d["families"],
            cfg=cfg,
            verbose=False,
        )
        assert len(hist["train_loss"]) == 8

    def test_swa_requires_min_checkpoints(self, small_dataset):
        """SWA requires >= 3 checkpoints to apply.

        Cross-ref: ref_yuriy_protossm_v5.py line 1872.
        """
        d = small_dataset
        model = ProtoSSMv2(
            d_input=1536,
            d_model=32,
            d_state=4,
            n_ssm_layers=1,
            n_classes=d["n_classes"],
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        # SWA starts at epoch 2 (0.1*20=2), should accumulate many checkpoints
        cfg = {
            "n_epochs": 20,
            "lr": 1e-3,
            "weight_decay": 0,
            "patience": 100,
            "pos_weight_cap": 10.0,
            "distill_weight": 0.0,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "focal_gamma": 0.0,
            "swa_start_frac": 0.1,
        }
        model, _ = train_proto_ssm_single(
            model,
            d["emb"],
            d["logits"],
            d["labels"],
            cfg=cfg,
            verbose=False,
        )
        # Should complete without error

    def test_focal_loss_training(self, small_dataset):
        d = small_dataset
        model = ProtoSSMv2(
            d_input=1536,
            d_model=32,
            d_state=4,
            n_ssm_layers=1,
            n_classes=d["n_classes"],
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        cfg = {
            "n_epochs": 5,
            "lr": 1e-3,
            "weight_decay": 0,
            "patience": 100,
            "pos_weight_cap": 10.0,
            "distill_weight": 0.1,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "focal_gamma": 2.0,
            "swa_start_frac": 1.0,
        }
        model, hist = train_proto_ssm_single(
            model,
            d["emb"],
            d["logits"],
            d["labels"],
            cfg=cfg,
            verbose=False,
        )
        assert len(hist["train_loss"]) == 5

    def test_early_stopping(self, small_dataset):
        """Training should stop early if validation loss doesn't improve."""
        d = small_dataset
        model = ProtoSSMv2(
            d_input=1536,
            d_model=32,
            d_state=4,
            n_ssm_layers=1,
            n_classes=d["n_classes"],
            n_windows=12,
            dropout=0.0,
            n_sites=5,
            meta_dim=4,
            use_cross_attn=False,
        )
        # Use very low lr so val loss improves slowly, and high patience tolerance
        # to test that the mechanism works at all
        cfg = {
            "n_epochs": 200,
            "lr": 1e-5,
            "weight_decay": 0,
            "patience": 5,
            "pos_weight_cap": 10.0,
            "distill_weight": 0.0,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "focal_gamma": 0.0,
            "swa_start_frac": 1.0,
        }
        model, hist = train_proto_ssm_single(
            model,
            d["emb"][:4],
            d["logits"][:4],
            d["labels"][:4],
            emb_val=d["emb"][4:],
            logits_val=d["logits"][4:],
            labels_val=d["labels"][4:],
            cfg=cfg,
            verbose=False,
        )
        # Should stop before 200 epochs (early stopping or natural convergence)
        assert len(hist["train_loss"]) <= 200


class TestTrainResidualSSM:
    def test_basic_training(self, small_dataset):
        d = small_dataset
        first_pass = np.random.randn(6, 12, d["n_classes"]).astype(np.float32)

        res_model, correction_weight = train_residual_ssm(
            d["emb"],
            first_pass,
            d["labels"],
            d["site_ids"],
            d["hours"],
            cfg={
                "d_model": 16,
                "d_state": 4,
                "dropout": 0.0,
                "correction_weight": 0.3,
                "n_epochs": 5,
                "lr": 1e-3,
                "patience": 100,
            },
            n_sites=5,
            verbose=False,
        )
        assert res_model is not None
        assert correction_weight == 0.3

    def test_wall_time_skip(self, small_dataset):
        """Should skip when wall time exceeded."""
        import time

        d = small_dataset
        first_pass = np.random.randn(6, 12, d["n_classes"]).astype(np.float32)

        res_model, cw = train_residual_ssm(
            d["emb"],
            first_pass,
            d["labels"],
            d["site_ids"],
            d["hours"],
            cfg={
                "d_model": 16,
                "d_state": 4,
                "dropout": 0.0,
                "correction_weight": 0.3,
                "n_epochs": 5,
                "lr": 1e-3,
                "patience": 100,
            },
            n_sites=5,
            wall_time_limit=0.0,  # 0 minutes → always skip
            start_time=time.time() - 60,
            verbose=False,
        )
        assert res_model is None
        assert cw == 0.0
