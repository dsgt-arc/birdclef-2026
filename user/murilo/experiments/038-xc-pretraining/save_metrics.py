#!/usr/bin/env python3
"""Save a structured metrics JSON after an exp 038 pretraining run (parity with exp 037).

Reads the experiment-local results written by train.py (`outputs/test_results/<run_name>.json`)
plus the merged config, and emits `artifacts/training/<variant>/metrics.json` with the final
val metrics + full hyperparameter provenance + compute info — the artifact used to build the
cross-cell results table.

Pretraining is val-only (no test split) and KD-off, so this drops exp 037's test/PL/head_lr/
early-stopping fields and adds the exp 038 levers (loss_type, label_smoothing, focal α/γ, mixup).

Metric semantics (matches exp 037 convention): the top-of-file `val/*` metrics are the
**final-epoch** values (exp 037's `val_metrics` comes from `trainer.callback_metrics` at the
end of `fit()` = the last validation epoch), while `best_val_auroc`/`best_checkpoint_path` track
the monitor-selected best epoch separately.

Sources, in priority order:
  1. `outputs/test_results/<run_name>.json` (written by train.py for normal runs).
  2. Fallback for runs launched before the metric-parity change (no test_results): the W&B run
     summary `outputs/wandb/wandb/<run>/files/wandb-summary.json` supplies the final-epoch `val/*`
     metrics (== callback_metrics), matched to this run via the sibling `config.yaml` run_name; and
     `checkpoints/<run>/pretrain_results.json` supplies best_val_auroc + checkpoint paths.

Usage:
    python save_metrics.py --config configs/pretrain_cellA_dualbce.yaml
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import yaml

# Final-epoch val metric keys we surface at the top of metrics.json (mirrors exp 037).
_VAL_KEYS = (
    "val/auroc",
    "val/cmap",
    "val/f1_macro",
    "val/f1_micro",
    "val/top1_acc",
    "val/top5_acc",
    "val/loss",
)


def _wandb_val_metrics(experiment_dir: Path, run_name: str) -> dict:
    """Reconstruct final-epoch `val/*` metrics from the local W&B run summary.

    Used only for runs launched before train.py wrote `outputs/test_results/` — the W&B summary
    holds the last logged value of each metric (= the final validation epoch, same semantics as
    `trainer.callback_metrics`). The run is matched by `run_name` via each run dir's `config.yaml`
    (`run_name.value`), so it is robust to the W&B run-id in the directory name.
    """
    wandb_root = experiment_dir / "outputs" / "wandb" / "wandb"
    if not wandb_root.is_dir():
        return {}
    for cfg_path in sorted(
        glob.glob(str(wandb_root / "run-*" / "files" / "config.yaml")), reverse=True
    ):
        try:
            with open(cfg_path) as f:
                wcfg = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            continue
        entry = wcfg.get("run_name")
        rn = entry.get("value") if isinstance(entry, dict) else entry
        if rn != run_name:
            continue
        summary_path = Path(cfg_path).parent / "wandb-summary.json"
        if not summary_path.exists():
            return {}
        with open(summary_path) as f:
            summary = json.load(f)
        return {k: summary[k] for k in _VAL_KEYS if k in summary}
    return {}


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


EXPERIMENT_NAME = "038-xc-pretraining"


def main():
    parser = argparse.ArgumentParser(description="Save structured exp 038 pretraining metrics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--project-name", type=str, default="ps-clef2026_birdclef-0")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/training")
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent
    config_path = experiment_dir / args.config
    config = load_config(config_path)
    run_name = config["run_name"]

    # Prefer the experiment-local results (new runs). Fall back to the run_dir copy that EVERY run
    # writes (incl. runs launched before the parity change — those lack the val_metrics dict, but
    # best_val_auroc is present and the per-epoch val curve lives in W&B/TB).
    results_file = experiment_dir / "outputs" / "test_results" / f"{run_name}.json"
    if not results_file.exists():
        fallback = (
            Path(f"/storage/project/{args.project_name}/mgustineli3")
            / "experiments"
            / EXPERIMENT_NAME
            / "checkpoints"
            / run_name
            / "pretrain_results.json"
        )
        if fallback.exists():
            print(f"Note: {results_file} missing; using fallback {fallback}", file=sys.stderr)
            results_file = fallback
        else:
            print(f"Error: results not found at {results_file} or {fallback}", file=sys.stderr)
            sys.exit(1)
    with open(results_file) as f:
        raw = json.load(f)

    # Final-epoch val metrics: from test_results when present, else reconstructed from the W&B
    # run summary (pre-parity runs whose train.py never wrote test_results / val_metrics).
    val_metrics = raw.get("val_metrics") or {}
    metrics_source = "val"
    if not val_metrics:
        val_metrics = _wandb_val_metrics(experiment_dir, run_name)
        if val_metrics:
            metrics_source = "val (W&B summary, final epoch)"
            print(
                f"Note: val_metrics reconstructed from W&B summary for {run_name}", file=sys.stderr
            )
        else:
            print(
                f"Warning: no val_metrics in {results_file} and no matching W&B summary for "
                f"{run_name}; top metrics will be null",
                file=sys.stderr,
            )
    mel_cfg = config.get("mel", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    mixup_cfg = config.get("mixup", {})
    variant_name = Path(args.config).stem

    metrics = {
        # Primary metrics — pretraining is val-only (held-out XC macro-AUC = the pretrain signal).
        "auroc": val_metrics.get("val/auroc"),
        "cmap": val_metrics.get("val/cmap"),
        "f1_macro": val_metrics.get("val/f1_macro"),
        "f1_micro": val_metrics.get("val/f1_micro"),
        "top1_accuracy": val_metrics.get("val/top1_acc"),
        "top5_accuracy": val_metrics.get("val/top5_acc"),
        "loss": val_metrics.get("val/loss"),
        "metrics_source": metrics_source,
        # Architecture
        "architecture": "sed-pretrain",
        "encoder_model": config["encoder"].get("model_name", ""),
        "sed_head": {
            "type": config["sed_head"].get("type"),
            "hidden_dim": config["sed_head"].get("hidden_dim"),
            "dropout": config["sed_head"].get("dropout"),
            "gem_p": config["sed_head"].get("gem_p"),
        },
        "mel": {
            "n_fft": mel_cfg.get("n_fft"),
            "n_mels": mel_cfg.get("n_mels"),
            "hop_length": mel_cfg.get("hop_length"),
            "fmax": mel_cfg.get("fmax"),
            "sample_rate": data_cfg.get("sample_rate"),
        },
        # Dataset
        "dataset": "birdset-neotropical",
        "num_classes": raw.get("num_classes"),
        # Training dynamics
        "best_val_auroc": raw.get("best_val_auroc"),
        "best_checkpoint_path": raw.get("best_checkpoint_path"),
        "last_checkpoint_path": raw.get("last_checkpoint_path"),
        "stopped_epoch": raw.get("stopped_epoch"),
        # Training config (exp 038 levers)
        "training": {
            "max_epochs": train_cfg.get("max_epochs"),
            "lr": train_cfg.get("lr"),
            "weight_decay": train_cfg.get("weight_decay"),
            "warmup_epochs": train_cfg.get("warmup_epochs"),
            "eta_min": train_cfg.get("eta_min"),
            "batch_size": data_cfg.get("batch_size"),
            "num_workers": data_cfg.get("num_workers"),
            "val_fraction": data_cfg.get("val_fraction"),
            "precision": train_cfg.get("precision"),
            "grad_clip_val": train_cfg.get("grad_clip_val"),
            "accumulate_grad_batches": train_cfg.get("accumulate_grad_batches", 1),
            "clipwise_weight": train_cfg.get("clipwise_weight"),
            "framewise_weight": train_cfg.get("framewise_weight"),
            "window_duration": data_cfg.get("window_duration"),
            "sample_rate": data_cfg.get("sample_rate"),
            # exp 038 cell levers
            "loss_type": train_cfg.get("loss_type"),
            "label_smoothing": train_cfg.get("label_smoothing"),
            "focal_alpha": train_cfg.get("focal_alpha"),
            "focal_gamma": train_cfg.get("focal_gamma"),
            "use_perch_distill": False,  # KD off for pretraining
        },
        "augmentation": {
            "mixup_enabled": mixup_cfg.get("enabled"),
            "mixup_prob": mixup_cfg.get("prob"),
            "mixup_alpha": mixup_cfg.get("alpha"),
            "aug_prob": mixup_cfg.get("aug_prob"),
            "spec_augment": config.get("spec_augment", {}).get("enabled"),
        },
        # Compute
        "compute": {
            "gpu": os.environ.get("SLURM_JOB_GPUS", "unknown"),
            "slurm_job_id": (
                int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
            ),
            "node": os.environ.get("SLURM_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
    }

    output_dir = experiment_dir / args.artifacts_dir / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
