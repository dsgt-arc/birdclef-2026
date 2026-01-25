#!/usr/bin/env python
"""Run direct inference with BirdSet-native models.

This script runs inference with models that were trained on BirdSet data
(BirdSetConvNeXT, BirdSetEfficientNetB1) and can directly predict BirdSet species.

Usage:
    # Basic inference on a single dataset
    python -m birdset_benchmark.run_inference --model birdset_convnext --dataset HSN

    # Custom batch size for better GPU utilization
    python -m birdset_benchmark.run_inference --model birdset_efficientnet_b1 --dataset PER --batch-size 32

    # Test on small subset before full run
    python -m birdset_benchmark.run_inference --model birdset_convnext --dataset HSN --subset-size 100

    # Custom output directory
    python -m birdset_benchmark.run_inference --model birdset_convnext --dataset HSN --output-dir ./my_results

    # Use different split (default is test_5s)
    python -m birdset_benchmark.run_inference --model birdset_convnext --dataset HSN --split test_10s

Available models: birdset_convnext, birdset_efficientnet_b1
Available datasets: HSN, PER, NES, UHH, POW, SSW, SNE, NBP

Outputs:
    - metrics.json: cmAP, F1 scores, Top-K accuracy, throughput
    - predictions.npz: y_pred and y_true arrays for further analysis
"""

import contextlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress TensorFlow warnings (must be set before importing TensorFlow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ERROR level only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom ops messages
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress all Python warnings

# Additional warning suppression for runtime
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Monkey-patch warnings.warn to suppress all warnings (last resort for clean output)
_original_warn = warnings.warn


def _suppress_warn(*args, **kwargs):
    pass


warnings.warn = _suppress_warn

# ruff: noqa: E402 - Imports after warning suppression is intentional
import bioacoustics_model_zoo as bmz
import numpy as np
import torch
import typer
from rich.console import Console
from sklearn.metrics import average_precision_score, f1_score

from birdset_benchmark.data import BIRDSET_DATASETS, get_num_classes, load_birdset_dataset

# Restore warnings.warn after import (optional - keep suppressed for cleaner output)
# warnings.warn = _original_warn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow and other module logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("birdset_benchmark.data").setLevel(logging.WARNING)

console = Console()

app = typer.Typer()


# Model mapping
MODELS = {
    "birdset_convnext": bmz.BirdSetConvNeXT,
    "birdset_efficientnet_b1": bmz.BirdSetEfficientNetB1,
}


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (for C++ warnings)."""
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth multi-hot labels [n_samples, n_classes]
        y_pred: Predicted probabilities [n_samples, n_classes]
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Class-mean Average Precision (cmAP) - BirdSet primary metric
    aps = []
    for i in range(num_classes):
        if y_true[:, i].sum() > 0:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            aps.append(ap)

    metrics["cmap"] = float(np.mean(aps)) if aps else 0.0
    metrics["num_classes_with_positives"] = len(aps)

    # F1 scores (using threshold of 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)
    metrics["f1_micro"] = float(f1_score(y_true, y_pred_binary, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0))

    # Top-K accuracy
    for k in [1, 5]:
        top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
        correct = 0
        for i in range(len(y_true)):
            true_labels = np.where(y_true[i] == 1)[0]
            if len(true_labels) > 0:
                if np.any(np.isin(top_k_preds[i], true_labels)):
                    correct += 1
        metrics[f"top{k}_accuracy"] = float(correct / len(y_true)) if len(y_true) > 0 else 0.0

    return metrics


@app.command()
def main(
    model: str = typer.Option(..., help="Model name (birdset_convnext, birdset_efficientnet_b1)"),
    dataset: str = typer.Option(..., help="Dataset code (HSN, PER, NES, UHH, POW, SSW, SNE, NBP)"),
    split: str = typer.Option("test_5s", help="Dataset split"),
    batch_size: int = typer.Option(16, help="Batch size for inference"),
    subset_size: int | None = typer.Option(None, help="Use subset for testing (e.g., 100)"),
    output_dir: Path | None = typer.Option(None, help="Output directory for results"),
):
    """Run inference with BirdSet-native models."""

    # Validate inputs
    if model not in MODELS:
        console.print(f"[red]Error: Unknown model '{model}'[/red]")
        console.print(f"Available models: {list(MODELS.keys())}")
        raise typer.Exit(1)

    if dataset not in BIRDSET_DATASETS:
        console.print(f"[red]Error: Unknown dataset '{dataset}'[/red]")
        console.print(f"Available datasets: {list(BIRDSET_DATASETS.keys())}")
        raise typer.Exit(1)

    # Setup output directory - default to experiment artifacts folder
    if output_dir is None:
        # Find the experiment directory relative to this script
        # Script is in: user/murilo/src/birdset_benchmark/run_inference.py
        # Experiment is in: user/murilo/experiments/001-birdset-bioacoustics-benchmark/
        script_dir = Path(__file__).parent  # src/birdset_benchmark
        user_murilo_dir = script_dir.parent.parent  # user/murilo
        experiment_dir = user_murilo_dir / "experiments" / "001-birdset-bioacoustics-benchmark"
        output_dir = experiment_dir / "artifacts" / "inference" / model / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]BirdSet Inference - Direct Prediction[/bold cyan]")
    console.print(f"Model: [green]{model}[/green]")
    console.print(f"Dataset: [green]{dataset}[/green] ({split})")
    console.print(f"Batch size: [green]{batch_size}[/green]")
    console.print(f"Output: [green]{output_dir}[/green]\n")

    # Check GPU
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n"
        )
    else:
        console.print("[yellow]No GPU available, using CPU[/yellow]\n")

    # Load dataset
    console.print("[bold]Loading dataset...[/bold]")
    hf_dataset = load_birdset_dataset(
        dataset_code=dataset,
        split=split,
        subset_size=subset_size,
    )
    num_classes = get_num_classes(dataset)
    console.print(f"✓ Loaded {len(hf_dataset)} samples with {num_classes} classes\n")

    # Get all audio paths and labels
    console.print("[bold]Extracting audio paths and labels...[/bold]")
    audio_paths = []
    y_true = np.zeros((len(hf_dataset), num_classes))

    for i in range(len(hf_dataset)):
        sample = hf_dataset[i]
        audio_paths.append(sample["audio"]["path"])

        # Build ground truth labels
        if "ebird_code_multilabel" in sample:
            for idx in sample["ebird_code_multilabel"]:
                if 0 <= idx < num_classes:
                    y_true[i, idx] = 1.0

    console.print(f"✓ Extracted {len(audio_paths)} audio paths\n")

    # Get species mapping from dataset
    console.print("[bold]Building species mapping...[/bold]")
    multilabel_feature = hf_dataset.features["ebird_code_multilabel"]
    species_names = multilabel_feature.feature.names
    species_to_idx = {species: idx for idx, species in enumerate(species_names)}
    console.print(f"✓ Mapped {len(species_to_idx)} species\n")

    # Load model
    console.print(f"[bold]Loading {model}...[/bold]")
    model_instance = MODELS[model]()
    console.print("✓ Model loaded\n")

    # Run inference - MODEL HANDLES BATCHING INTERNALLY
    console.print(f"[bold]Running inference on {len(audio_paths)} files...[/bold]")
    start_time = time.time()

    # Call .predict() with list of paths and batch_size (suppress C++ warnings from stderr)
    with suppress_stderr():
        pred_df = model_instance.predict(audio_paths, batch_size=batch_size)

    inference_time = time.time() - start_time
    console.print(f"✓ Inference complete in {inference_time:.2f}s")
    console.print(f"  Throughput: {len(audio_paths) / inference_time:.2f} samples/sec\n")

    # Convert predictions DataFrame to numpy array
    console.print("[bold]Converting predictions to numpy array...[/bold]")
    y_pred = np.zeros((len(audio_paths), num_classes))

    for species_code in pred_df.columns:
        if species_code in species_to_idx:
            class_idx = species_to_idx[species_code]
            # pred_df has shape [n_samples, n_species]
            y_pred[:, class_idx] = pred_df[species_code].values

    # Apply sigmoid to convert logits to probabilities
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    console.print(f"✓ Converted predictions shape: {y_pred.shape}\n")

    # Compute metrics
    console.print("[bold]Computing metrics...[/bold]")
    metrics = compute_metrics(y_true, y_pred, num_classes)

    # Add metadata
    metrics["model"] = model
    metrics["dataset"] = dataset
    metrics["split"] = split
    metrics["num_samples"] = len(audio_paths)
    metrics["num_classes"] = num_classes
    metrics["inference_time_sec"] = inference_time
    metrics["throughput_samples_per_sec"] = len(audio_paths) / inference_time
    metrics["batch_size"] = batch_size

    # Display results
    console.print("\n[bold cyan]Results:[/bold cyan]")
    console.print(f"cmAP:          [green]{metrics['cmap']:.4f}[/green]")
    console.print(f"F1 Micro:      {metrics['f1_micro']:.4f}")
    console.print(f"F1 Macro:      {metrics['f1_macro']:.4f}")
    console.print(f"Top-1 Acc:     {metrics['top1_accuracy']:.4f}")
    console.print(f"Top-5 Acc:     {metrics['top5_accuracy']:.4f}")

    # Save results
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"\n✓ Saved metrics to {metrics_path}")

    # Save predictions
    predictions_path = output_dir / "predictions.npz"
    np.savez(predictions_path, y_pred=y_pred, y_true=y_true)
    console.print(f"✓ Saved predictions to {predictions_path}\n")

    console.print("[bold green]Inference complete![/bold green]\n")


if __name__ == "__main__":
    app()
