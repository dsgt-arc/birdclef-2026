"""Frozen hyperparameters for ProtoSSM v5 pipeline.

All values verified against ref_yuriy_protossm_v5.py lines 162-253.
"""

import numpy as np


def get_protossm_config() -> dict:
    """ProtoSSM v4 architecture config."""
    return {
        "d_model": 128,
        "d_state": 16,
        "n_ssm_layers": 2,
        "dropout": 0.15,
        "n_sites": 20,
        "meta_dim": 16,
        "use_cross_attn": True,
        "cross_attn_heads": 4,
    }


def get_protossm_train_config() -> dict:
    """ProtoSSM training config."""
    return {
        "n_epochs": 60,
        "lr": 1e-3,
        "weight_decay": 2e-3,
        "val_ratio": 0.15,
        "patience": 15,
        "pos_weight_cap": 30.0,
        "distill_weight": 0.1,
        "label_smoothing": 0.02,
        "oof_n_splits": 3,
        "mixup_alpha": 0.3,
        "focal_gamma": 2.0,
        "swa_start_frac": 0.7,
    }


def get_residual_ssm_config() -> dict:
    """ResidualSSM architecture + training config."""
    return {
        "d_model": 64,
        "d_state": 8,
        "n_ssm_layers": 1,
        "dropout": 0.1,
        "correction_weight": 0.3,
        "n_epochs": 30,
        "lr": 1e-3,
        "patience": 8,
    }


def get_postprocessing_config() -> dict:
    """Post-processing pipeline config."""
    return {
        "temperature_aves": 1.10,
        "temperature_texture": 0.95,
        "file_level_top_k": 2,
        "rank_aware_power": 0.5,
        "delta_shift_alpha": 0.15,
        "tta_shifts": [0, 1, -1],
        "threshold_grid": np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70], dtype=np.float32),
    }


def get_probe_config() -> dict:
    """Frozen probe hyperparameters (Yuriy/jaejohn optimal)."""
    return {
        "pca_dim": 64,
        "min_pos": 8,
        "C": 0.50,
        "alpha": 0.40,
        "mlp_params": {
            "hidden_layer_sizes": (128,),
            "activation": "relu",
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 15,
            "random_state": 42,
            "learning_rate_init": 0.001,
            "alpha": 0.01,
        },
    }
