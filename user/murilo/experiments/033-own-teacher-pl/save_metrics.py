#!/usr/bin/env python3
"""Save structured metrics JSON after a training run.

Reads the raw test results from train.py and the (merged) config YAML, combines them
into a structured metrics.json in the artifacts directory.

Exp 033 difference vs exp 032: per-ablation YAMLs are diff-only and inherit from
configs/_common.yaml via deep merge. This script mirrors train.py's `load_config`
so the merged view is what gets serialized.

Usage:
    python save_metrics.py --config configs/a1_v5j.yaml --fold 0
    python save_metrics.py --config configs/a3_kd.yaml --fold 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins. Mirrors train.py:_deep_merge."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path) -> dict:
    """Load YAML + merge with sibling _common.yaml. Mirrors train.py:load_config."""
    with open(config_path) as f:
        override = yaml.safe_load(f) or {}
    common_p = config_path.parent / "_common.yaml"
    if common_p.exists() and config_path.name != "_common.yaml":
        with open(common_p) as f:
            base = yaml.safe_load(f) or {}
        return _deep_merge(base, override)
    return override


def main():
    parser = argparse.ArgumentParser(description="Save structured training metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/training")
    parser.add_argument("--fold", type=int, default=None, help="Fold ID (appends /foldN to paths)")
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent
    config_path = experiment_dir / args.config

    config = load_config(config_path)

    run_name = config["run_name"]

    if args.fold is not None:
        results_file = (
            experiment_dir / "outputs" / "test_results" / f"{run_name}-fold{args.fold}.json"
        )
    else:
        results_file = experiment_dir / "outputs" / "test_results" / f"{run_name}.json"
    if not results_file.exists():
        print(f"Error: test results not found at {results_file}", file=sys.stderr)
        sys.exit(1)

    with open(results_file) as f:
        raw = json.load(f)

    test_metrics = raw.get("test_metrics") or {}
    val_metrics = raw.get("val_metrics") or {}
    variant_name = Path(args.config).stem
    mel_cfg = config.get("mel", {})
    pl_cfg = config.get("pseudo_label", {})
    training_cfg = config.get("training", {})

    has_test = bool(test_metrics)
    metrics_source = "test" if has_test else "val"

    def get_metric(test_key, val_key):
        if has_test:
            return test_metrics.get(test_key)
        return val_metrics.get(val_key)

    metrics = {
        # Primary metrics (test if available, val otherwise)
        "auroc": get_metric("test/auroc", "val/auroc"),
        "cmap": get_metric("test/cmap", "val/cmap"),
        "f1_macro": get_metric("test/f1_macro", "val/f1_macro"),
        "f1_micro": get_metric("test/f1_micro", "val/f1_micro"),
        "top1_accuracy": get_metric("test/top1_acc", "val/top1_acc"),
        "top5_accuracy": get_metric("test/top5_acc", "val/top5_acc"),
        "loss": get_metric("test/loss", "val/loss"),
        "metrics_source": metrics_source,
        # Architecture
        "architecture": "sed",
        "encoder_type": config["encoder"]["type"],
        "encoder_model": config["encoder"].get("model_name", ""),
        "embed_dim": config["encoder"].get("embed_dim"),
        # SED head
        "sed_head": {
            "type": config["sed_head"].get("type"),
            "hidden_dim": config["sed_head"]["hidden_dim"],
            "dropout": config["sed_head"]["dropout"],
            "gem_p": config["sed_head"]["gem_p"],
        },
        # Mel spectrogram
        "mel": {
            "type": mel_cfg.get("type"),
            "n_fft": mel_cfg.get("n_fft"),
            "n_mels": mel_cfg.get("n_mels", 128),
            "hop_length": mel_cfg.get("hop_length"),
            "fmax": mel_cfg.get("fmax"),
            "sample_rate": config["data"]["sample_rate"],
        },
        # Dataset
        "dataset": "birdclef-2026",
        "num_classes": raw.get("num_classes"),
        # Training dynamics
        "best_val_score": raw.get("best_val_score"),
        "best_checkpoint_path": raw.get("best_checkpoint_path"),
        "stopped_epoch": raw.get("stopped_epoch"),
        # Training config
        "training": {
            "max_epochs": training_cfg["max_epochs"],
            "lr": training_cfg["lr"],
            "head_lr": training_cfg["head_lr"],
            "weight_decay": training_cfg["weight_decay"],
            "warmup_epochs": training_cfg["warmup_epochs"],
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "precision": training_cfg["precision"],
            "early_stopping_patience": config["callbacks"]["early_stopping"]["patience"],
            "grad_clip_val": training_cfg["grad_clip_val"],
            "accumulate_grad_batches": training_cfg.get("accumulate_grad_batches", 1),
            "clipwise_weight": training_cfg["clipwise_weight"],
            "framewise_weight": training_cfg["framewise_weight"],
            "blend_clip": training_cfg["blend_clip"],
            "blend_frame": training_cfg["blend_frame"],
            "window_duration": config["data"]["window_duration"],
            "sample_rate": config["data"]["sample_rate"],
            # KD / warm-start (exp 033 inherits from Tucker recipe)
            "use_perch_distill": training_cfg.get("use_perch_distill"),
            "alpha_distill": training_cfg.get("alpha_distill"),
            "warm_start_ckpt": training_cfg.get("warm_start_ckpt"),
        },
        # Exp 033-specific: pseudo-label setup
        "pseudo_label": {
            "path": pl_cfg.get("path"),
            "weight": pl_cfg.get("weight"),
            "sampler": pl_cfg.get("sampler"),
            "exclude_labeled_sc": pl_cfg.get("exclude_labeled_sc"),
            "focal_share": pl_cfg.get("focal_share"),
            "sc_share": pl_cfg.get("sc_share"),
            "pl_share": pl_cfg.get("pl_share"),
        },
        # Compute
        "compute": {
            "gpu": os.environ.get("SLURM_JOB_GPUS", "unknown"),
            "slurm_job_id": int(os.environ["SLURM_JOB_ID"])
            if "SLURM_JOB_ID" in os.environ
            else None,
            "node": os.environ.get("SLURM_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
    }

    output_dir = experiment_dir / args.artifacts_dir / variant_name
    if args.fold is not None:
        output_dir = output_dir / f"fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics.json"

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
