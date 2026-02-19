#!/usr/bin/env python3
"""Save structured metrics JSON after a training run.

Reads the raw test results from train.py and the config YAML, combines them
with SLURM environment metadata, and writes a structured metrics.json to the
artifacts directory.

Usage:
    python save_metrics.py --config config/vertical_128x2.yaml
    python save_metrics.py --config config/baseline_16x16.yaml --artifacts-dir artifacts/training
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Save structured training metrics")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML used for training",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/training",
        help="Base directory for artifacts output (default: artifacts/training)",
    )
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent
    config_path = experiment_dir / args.config

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_name = config["run_name"]
    dataset_name = config["data"]["dataset_name"]

    # Load raw test results from train.py
    results_file = experiment_dir / "outputs" / "test_results" / f"{run_name}.json"
    if not results_file.exists():
        print(f"Error: test results not found at {results_file}", file=sys.stderr)
        print("Make sure train.py completed successfully first.", file=sys.stderr)
        sys.exit(1)

    with open(results_file) as f:
        raw = json.load(f)

    test_metrics = raw["test_metrics"]

    # Derive model variant name from config filename
    variant_name = Path(args.config).stem

    # Extract patch info
    patch_size = config["model"]["patch_size"]
    img_size = config["model"]["img_size"]
    freq_patches = img_size[0] // patch_size[0]
    time_patches = img_size[1] // patch_size[1]

    # Build structured metrics
    metrics = {
        # Primary metrics
        "cmap": test_metrics.get("test/cmap"),
        "auroc": test_metrics.get("test/auroc"),
        "f1_macro": test_metrics.get("test/f1_macro"),
        "f1_micro": test_metrics.get("test/f1_micro"),
        "top1_accuracy": test_metrics.get("test/top1_acc"),
        "top5_accuracy": test_metrics.get("test/top5_acc"),
        "loss": test_metrics.get("test/loss"),
        # Model info
        "model": config["model"]["model_name"].split(".")[0],
        "model_name_timm": config["model"]["model_name"],
        "pretrained": config["model"]["pretrained"],
        "num_params": raw["num_params"],
        # Tokenization info (exp 004 specific)
        "patch_size": patch_size,
        "patch_size_str": f"{patch_size[0]}x{patch_size[1]}",
        "img_size": img_size,
        "num_tokens": raw.get("num_tokens", freq_patches * time_patches),
        "patch_grid": f"{freq_patches}x{time_patches}",
        # Dataset info
        "dataset": dataset_name,
        "split": "test_5s",
        "num_classes": None,
        # Training dynamics
        "best_epoch": raw["best_epoch"],
        "stopped_epoch": raw["stopped_epoch"],
        "best_val_cmap": raw["best_val_cmap"],
        "best_checkpoint_path": raw["best_checkpoint_path"],
        # Training config
        "training": {
            "max_epochs": config["training"]["max_epochs"],
            "lr": config["training"]["lr"],
            "weight_decay": config["training"]["weight_decay"],
            "warmup_epochs": config["training"]["warmup_epochs"],
            "batch_size": config["data"]["batch_size"],
            "precision": config["training"]["precision"],
            "early_stopping_patience": config["callbacks"]["early_stopping"]["patience"],
            "drop_rate": config["model"]["drop_rate"],
            "grad_clip_val": config["training"]["grad_clip_val"],
            "cache_mel_specs": config["data"].get("cache_mel_specs", False),
        },
        # Compute info from SLURM environment
        "compute": {
            "gpu": os.environ.get("SLURM_JOB_GPUS", "unknown"),
            "slurm_job_id": int(os.environ["SLURM_JOB_ID"])
            if "SLURM_JOB_ID" in os.environ
            else None,
            "node": os.environ.get("SLURM_NODELIST"),
        },
    }

    # Try to get num_classes
    try:
        from birdset_benchmark.datasets import get_num_classes

        metrics["num_classes"] = get_num_classes(dataset_name)
    except Exception:
        pass

    # Write structured metrics
    output_dir = experiment_dir / args.artifacts_dir / variant_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics.json"

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
