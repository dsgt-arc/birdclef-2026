#!/bin/bash
#SBATCH --job-name=static-acc-002
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t120 -qembers
#SBATCH --output=logs/accuracy-eval-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Running static embeddings accuracy evaluation on ESC-50"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

uv run python << 'EOF'
"""Evaluate static embeddings accuracy on ESC-50 5-fold CV."""

import json
import sys
import time
from pathlib import Path

print("Loading dependencies...", flush=True)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "vendor/unilm/beats")

from birdclef.data import ESC50Dataset
from birdclef.static_embeddings import (
    StaticEmbeddingsPipelineFAISS,
    load_beats_encoder,
)
from birdclef.train import train_linear_probe

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
repo_dir = Path.home() / "arc" / "birdclef-2026"
onnx_frontend_path = scratch_dir / "outputs/exp002-benchmarks/onnx/beats_frontend.onnx"
static_dir = scratch_dir / "outputs/exp002-benchmarks/static_embeddings"
codebook_path = static_dir / "codebook.npy"
embeddings_768_path = static_dir / "static_embeddings.npy"
embeddings_256_path = static_dir / "static_embeddings_pca.npy"
weights_path = static_dir / "sif_weights.npy"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
output_dir = scratch_dir / "outputs/exp002-benchmarks"
esc50_root = repo_dir / "vendor" / "ESC-50"

print("=" * 60)
print("STATIC EMBEDDINGS ACCURACY EVALUATION")
print("=" * 60)


class StaticEmbeddingsEncoder:
    """Wrapper to use static embeddings pipeline as an encoder."""

    def __init__(self, pipeline, embed_dim):
        self._pipeline = pipeline
        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    def eval(self):
        pass  # No-op for numpy pipeline

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform batch."""
        # Convert to numpy (batch, samples) - remove channel dim if present
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        waveform_np = waveform.cpu().numpy()

        # Extract fbank features
        fbank = self._pipeline.extract_fbank(waveform_np)

        # Run pipeline
        embeddings = self._pipeline(fbank)

        return torch.from_numpy(embeddings).float()


class BEATsEncoderWrapper:
    """Wrapper for full BEATs encoder."""

    def __init__(self, encoder):
        self._encoder = encoder
        self._embed_dim = 768

    @property
    def embed_dim(self):
        return self._embed_dim

    def eval(self):
        self._encoder.eval()

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform batch."""
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        with torch.no_grad():
            features, _ = self._encoder.extract_features(waveform)
            # Mean pool over time
            return features.mean(dim=1)


def extract_features_from_dataloader(encoder, dataloader):
    """Extract features for entire dataset."""
    all_features = []
    all_labels = []

    encoder.eval()
    for waveforms, labels in tqdm(dataloader, desc="Extracting features"):
        features = encoder.extract_features(waveforms)
        all_features.append(features.cpu() if isinstance(features, torch.Tensor) else features)
        all_labels.append(labels)

    features = torch.cat(all_features) if isinstance(all_features[0], torch.Tensor) else torch.from_numpy(np.concatenate(all_features))
    return features, torch.cat(all_labels)


def evaluate_5fold_cv(encoder, dataset, epochs=100, lr=1e-3, batch_size=32):
    """Run 5-fold CV evaluation."""
    fold_accs = []

    for val_fold in range(1, 6):
        print(f"\n=== Fold {val_fold}/5 ===")

        train_ds, val_ds = dataset.get_fold_splits(val_fold)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Extract features
        print("Extracting training features...")
        train_features, train_labels = extract_features_from_dataloader(encoder, train_loader)
        print("Extracting validation features...")
        val_features, val_labels = extract_features_from_dataloader(encoder, val_loader)

        # Train linear probe
        print("Training linear probe...")
        probe, history = train_linear_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_classes=dataset.num_classes,
            epochs=epochs,
            lr=lr,
            device="cpu",
        )

        # Final evaluation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            val_preds = val_logits.argmax(dim=1)
            acc = (val_preds == val_labels).float().mean().item()

        fold_accs.append(acc)
        print(f"Fold {val_fold} accuracy: {acc:.4f}")

    mean_acc = sum(fold_accs) / len(fold_accs)
    print(f"\n=== Mean accuracy: {mean_acc:.4f} ===")

    return {"fold_accuracies": fold_accs, "mean_accuracy": mean_acc}


# Load dataset
print("\nLoading ESC-50 dataset...")
dataset = ESC50Dataset(esc50_root)
print(f"Dataset: {len(dataset)} samples, {dataset.num_classes} classes")

results = {}

# Initialize pipelines
print("\nInitializing pipelines...")

# 1. Static FAISS 768d (no weights)
print("  Loading Static FAISS 768d...")
pipeline_768 = StaticEmbeddingsPipelineFAISS(
    onnx_path=onnx_frontend_path,
    codebook_path=codebook_path,
    embeddings_path=embeddings_768_path,
    weights_path=None,
    num_threads=4,
)

# Add extract_fbank method to FAISS pipeline (it's missing)
import torchaudio.compliance.kaldi as ta_kaldi

def extract_fbank(self, waveform):
    """Extract fbank features from waveform."""
    fbank_mean = 15.41663
    fbank_std = 6.55582

    fbanks = []
    for wav in waveform:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0) * 2**15
        fbank = ta_kaldi.fbank(
            wav_tensor, num_mel_bins=128, sample_frequency=16000,
            frame_length=25, frame_shift=10
        )
        fbanks.append(fbank.numpy())

    fbank = np.stack(fbanks, axis=0)
    fbank = (fbank - fbank_mean) / (2 * fbank_std)
    fbank = fbank[:, np.newaxis, :, :]  # (batch, 1, time, 128)
    return fbank.astype(np.float32)

# Bind method to pipeline
import types
pipeline_768.extract_fbank = types.MethodType(extract_fbank, pipeline_768)

encoder_768 = StaticEmbeddingsEncoder(pipeline_768, embed_dim=768)

# 2. Static FAISS 768d + SIF
print("  Loading Static FAISS 768d + SIF...")
pipeline_768_sif = StaticEmbeddingsPipelineFAISS(
    onnx_path=onnx_frontend_path,
    codebook_path=codebook_path,
    embeddings_path=embeddings_768_path,
    weights_path=weights_path,
    num_threads=4,
)
pipeline_768_sif.extract_fbank = types.MethodType(extract_fbank, pipeline_768_sif)
encoder_768_sif = StaticEmbeddingsEncoder(pipeline_768_sif, embed_dim=768)

# 3. Static FAISS 256d + SIF
print("  Loading Static FAISS 256d + SIF...")
pipeline_256_sif = StaticEmbeddingsPipelineFAISS(
    onnx_path=onnx_frontend_path,
    codebook_path=codebook_path,
    embeddings_path=embeddings_256_path,
    weights_path=weights_path,
    num_threads=4,
)
pipeline_256_sif.extract_fbank = types.MethodType(extract_fbank, pipeline_256_sif)
encoder_256_sif = StaticEmbeddingsEncoder(pipeline_256_sif, embed_dim=256)

# 4. Full BEATs (baseline)
print("  Loading Full BEATs...")
beats_encoder = load_beats_encoder(encoder_path)
beats_encoder.eval()
encoder_beats = BEATsEncoderWrapper(beats_encoder)

# Run evaluations
print("\n" + "=" * 60)
print("EVALUATION: Full BEATs (baseline)")
print("=" * 60)
start_time = time.time()
results["full_beats"] = evaluate_5fold_cv(encoder_beats, dataset)
results["full_beats"]["time_s"] = time.time() - start_time

print("\n" + "=" * 60)
print("EVALUATION: Static FAISS 768d")
print("=" * 60)
start_time = time.time()
results["static_faiss_768"] = evaluate_5fold_cv(encoder_768, dataset)
results["static_faiss_768"]["time_s"] = time.time() - start_time

print("\n" + "=" * 60)
print("EVALUATION: Static FAISS 768d + SIF")
print("=" * 60)
start_time = time.time()
results["static_faiss_768_sif"] = evaluate_5fold_cv(encoder_768_sif, dataset)
results["static_faiss_768_sif"]["time_s"] = time.time() - start_time

print("\n" + "=" * 60)
print("EVALUATION: Static FAISS 256d + SIF")
print("=" * 60)
start_time = time.time()
results["static_faiss_256_sif"] = evaluate_5fold_cv(encoder_256_sif, dataset)
results["static_faiss_256_sif"]["time_s"] = time.time() - start_time

# Summary
print("\n" + "=" * 60)
print("ACCURACY SUMMARY")
print("=" * 60)
print(f"{'Method':<25} {'Mean Acc':>10} {'Time (s)':>10}")
print("-" * 50)
for name, res in results.items():
    print(f"{name:<25} {res['mean_accuracy']*100:>9.1f}% {res['time_s']:>10.1f}")
print("-" * 50)

# Compute accuracy drop vs baseline
beats_acc = results["full_beats"]["mean_accuracy"]
print("\nAccuracy drop vs Full BEATs:")
for name, res in results.items():
    if name != "full_beats":
        drop = (beats_acc - res["mean_accuracy"]) * 100
        print(f"  {name}: {drop:+.1f}%")

# Save results
output_path = output_dir / "static_accuracy_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

print("\nEvaluation complete!")
EOF

echo ""
echo "Accuracy evaluation complete!"
date
