#!/usr/bin/env python3
"""Save structured metrics JSON after a training run.

Reads the raw test results from train.py and the config YAML, combines them
with SLURM environment metadata, and writes a structured metrics.json to the
artifacts directory.

Usage:
    python save_metrics.py --config config/config_a_baseline.yaml
    python save_metrics.py --config config/config_c1a_frame_2x128.yaml --artifacts-dir artifacts/training
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

    # Load raw test results from train.py
    results_file = experiment_dir / "outputs" / "test_results" / f"{run_name}.json"
    if not results_file.exists():
        print(f"Error: test results not found at {results_file}", file=sys.stderr)
        print("Make sure train.py completed successfully first.", file=sys.stderr)
        sys.exit(1)

    with open(results_file) as f:
        raw = json.load(f)

    test_metrics = raw.get("test_metrics") or {}

    # Derive variant name from config filename
    variant_name = Path(args.config).stem

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
        # Encoder info
        "encoder_type": config["encoder"]["type"],
        "encoder_model": config["encoder"].get("model_name", ""),
        "embed_dim": config["encoder"].get("embed_dim"),
        # Patch embedding info
        "patch_embed": {
            "type": config["patch_embed"]["type"],
            "patch_size": config["patch_embed"]["patch_size"],
            "freeze": config["patch_embed"]["freeze"],
        },
        # LoRA info
        "lora": {
            "rank": config["lora"]["rank"],
            "alpha": config["lora"]["alpha"],
            "dropout": config["lora"]["dropout"],
            "target_modules": config["lora"]["target_modules"],
        },
        # Partial unfreeze
        "partial_unfreeze_blocks": config.get("model", {}).get("partial_unfreeze_blocks", []),
        # Dataset info
        "dataset": "birdclef-2025",
        "split": "test",
        "num_classes": raw.get("num_classes"),
        # Training dynamics
        "best_val_cmap": raw.get("best_val_cmap"),
        "best_checkpoint_path": raw.get("best_checkpoint_path"),
        "stopped_epoch": raw.get("stopped_epoch"),
        # Training config
        "training": {
            "max_epochs": config["training"]["max_epochs"],
            "lr": config["training"]["lr"],
            "weight_decay": config["training"]["weight_decay"],
            "warmup_epochs": config["training"]["warmup_epochs"],
            "batch_size": config["data"]["batch_size"],
            "num_workers": config["data"]["num_workers"],
            "precision": config["training"]["precision"],
            "early_stopping_patience": config["callbacks"]["early_stopping"]["patience"],
            "hidden_dim": config["training"].get("hidden_dim", 0),
            "dropout": config["training"].get("dropout", 0.0),
            "grad_clip_val": config["training"]["grad_clip_val"],
        },
        # Compute info from SLURM environment
        "compute": {
            "gpu": os.environ.get("SLURM_JOB_GPUS", "unknown"),
            "slurm_job_id": int(os.environ["SLURM_JOB_ID"])
            if "SLURM_JOB_ID" in os.environ
            else None,
            "node": os.environ.get("SLURM_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
    }

    # Write structured metrics
    output_dir = experiment_dir / args.artifacts_dir / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics.json"

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
