#!/usr/bin/env python3
"""Save structured metrics JSON after a training run.

Reads the raw test results from train.py and the config YAML, combines them
into a structured metrics.json in the artifacts directory.

Usage:
    python save_metrics.py --config configs/v10a_addmix_ce.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Save structured training metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/training")
    parser.add_argument("--fold", type=int, default=None, help="Fold ID (appends /foldN to paths)")
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent
    config_path = experiment_dir / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

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

    # Use test metrics if available (80/10/10 split), fall back to val metrics (90/10 split)
    has_test = bool(test_metrics)
    metrics_source = "test" if has_test else "val"

    def get_metric(test_key, val_key):
        """Get metric from test set first, fall back to val set."""
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
            "hidden_dim": config["sed_head"]["hidden_dim"],
            "dropout": config["sed_head"]["dropout"],
            "gem_p": config["sed_head"]["gem_p"],
        },
        # Mel spectrogram
        "mel": {
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
            "max_epochs": config["training"]["max_epochs"],
            "lr": config["training"]["lr"],
            "head_lr": config["training"]["head_lr"],
            "weight_decay": config["training"]["weight_decay"],
            "warmup_epochs": config["training"]["warmup_epochs"],
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "precision": config["training"]["precision"],
            "early_stopping_patience": config["callbacks"]["early_stopping"]["patience"],
            "grad_clip_val": config["training"]["grad_clip_val"],
            "accumulate_grad_batches": config["training"].get("accumulate_grad_batches", 1),
            "clipwise_weight": config["training"]["clipwise_weight"],
            "framewise_weight": config["training"]["framewise_weight"],
            "blend_clip": config["training"]["blend_clip"],
            "blend_frame": config["training"]["blend_frame"],
            "mixup_prob": config["training"].get("mixup_prob", 0.5),
            "mixup_alpha": config["training"].get("mixup_alpha", 0.5),
            "mixup_mode": config["training"].get("mixup_mode", "interpolate"),
            "loss_type": config["training"].get("loss_type", "bce"),
            "window_duration": config["data"]["window_duration"],
            "sample_rate": config["data"]["sample_rate"],
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
