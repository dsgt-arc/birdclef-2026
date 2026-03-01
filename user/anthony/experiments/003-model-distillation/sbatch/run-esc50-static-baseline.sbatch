#!/bin/bash
#SBATCH --job-name=esc50-static-base
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t30 -qinferno
#SBATCH --output=logs/esc50-static-baseline-%j.out
set -eu

# Usage: sbatch run-esc50-static-baseline.sbatch
#
# Evaluates static embeddings on ESC-50 without any student model.
# This is the "bag-of-words" baseline: token lookup + mean pooling + linear probe.

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "=============================================="
echo "ESC-50 Static Embeddings Baseline Evaluation"
echo "=============================================="
hostname
date

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
user_dir=$repo_dir/user/anthony

# Sync venv to local storage for faster package loading
export UV_PROJECT_ENVIRONMENT=${TMPDIR:-/tmp}/${SLURM_JOB_ID}/.venv
export HF_HOME=$scratch_dir/.cache/huggingface
cd $user_dir
uv sync

# Configuration - can override with environment variables
STATIC_EMB_DIR=${STATIC_EMB_DIR:-$scratch_dir/outputs/exp003-distillation/static_embeddings_500k_v2}
USE_PCA=${USE_PCA:-false}

echo ""
echo "Configuration:"
echo "  Static embeddings dir: $STATIC_EMB_DIR"
echo "  Use PCA embeddings: $USE_PCA"
echo ""

cd $repo_dir

${UV_PROJECT_ENVIRONMENT}/bin/python - "$STATIC_EMB_DIR" "$USE_PCA" << 'EOF'
"""Evaluate static embeddings baseline on ESC-50 5-fold CV.

This is the bag-of-words baseline:
1. Tokenize audio with BEATs tokenizer
2. Lookup static embeddings for each token
3. Mean pool across time
4. Train linear classifier

No student model is used - this measures the quality of static embeddings alone.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

static_emb_dir = Path(sys.argv[1])
use_pca = sys.argv[2].lower() == "true"

print(f"Static embeddings dir: {static_emb_dir}")
print(f"Use PCA embeddings: {use_pca}", flush=True)

sys.path.insert(0, "vendor/unilm/beats")

from birdclef.data import ESC50Dataset
from birdclef.distillation import load_teacher_and_tokenizer

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
repo_dir = Path.home() / "arc" / "birdclef-2026"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
tokenizer_path = scratch_dir / "checkpoints/beats/Tokenizers_iter3.pt"
output_dir = scratch_dir / "outputs/exp003-distillation"
esc50_root = repo_dir / "vendor" / "ESC-50"

# Select embedding file
if use_pca:
    static_emb_path = static_emb_dir / "static_embeddings_pca.npy"
else:
    static_emb_path = static_emb_dir / "static_embeddings.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("=" * 60)
print("ESC-50 STATIC EMBEDDINGS BASELINE")
print("=" * 60)

# Load tokenizer (for getting token IDs)
print("\nLoading tokenizer...")
_, tokenizer = load_teacher_and_tokenizer(encoder_path, tokenizer_path)
tokenizer = tokenizer.to(device)

# Load static embeddings
print(f"\nLoading static embeddings from {static_emb_path}...")
static_embeddings = torch.from_numpy(np.load(static_emb_path)).float().to(device)
print(f"  Static embeddings shape: {static_embeddings.shape}")
embed_dim = static_embeddings.shape[1]
print(f"  Embedding dimension: {embed_dim}")


def extract_static_embeddings(waveforms, tokenizer, static_embeds, device):
    """Extract embeddings using static lookup + mean pooling (bag-of-words)."""
    if waveforms.dim() == 3:
        waveforms = waveforms.squeeze(1)
    waveforms = waveforms.to(device)
    batch_size = waveforms.shape[0]

    with torch.no_grad():
        # Get token IDs (BEATs tokenizer returns flattened tensor)
        token_ids = tokenizer.extract_labels(waveforms)

        # Reshape to (batch, seq_len) if flattened
        if token_ids.dim() == 1:
            seq_len = token_ids.shape[0] // batch_size
            token_ids = token_ids.view(batch_size, seq_len)

        # Lookup static embeddings -> (batch, seq_len, embed_dim)
        inputs = static_embeds[token_ids]

        # Mean pool across time dimension (bag-of-words)
        pooled = inputs.mean(dim=1)

    return pooled


def extract_all_embeddings(dataloader, tokenizer, static_embeds, device):
    """Extract embeddings for entire dataset."""
    all_embeds = []
    all_labels = []

    for waveforms, labels in tqdm(dataloader, desc="Extracting"):
        embeds = extract_static_embeddings(waveforms, tokenizer, static_embeds, device)
        all_embeds.append(embeds.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeds), torch.cat(all_labels)


def train_linear_probe(train_embeds, train_labels, val_embeds, val_labels, num_classes, epochs=100):
    """Train a linear classifier on frozen embeddings."""
    embed_dim = train_embeds.shape[1]
    classifier = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_embeds = train_embeds.to(device)
    train_labels = train_labels.to(device)
    val_embeds = val_embeds.to(device)
    val_labels = val_labels.to(device)

    best_acc = 0.0

    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_embeds)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_embeds)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc

    return best_acc


# Load ESC-50 dataset
print("\nLoading ESC-50 dataset...")
dataset = ESC50Dataset(esc50_root)
print(f"  {len(dataset)} samples, {dataset.num_classes} classes")

# 5-fold cross-validation
print("\nRunning 5-fold CV...")
fold_accs = []

for val_fold in range(1, 6):
    print(f"\n=== Fold {val_fold}/5 ===")

    train_ds, val_ds = dataset.get_fold_splits(val_fold)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # Extract embeddings
    print("  Extracting train embeddings...")
    train_embeds, train_labels = extract_all_embeddings(
        train_loader, tokenizer, static_embeddings, device
    )
    print("  Extracting val embeddings...")
    val_embeds, val_labels = extract_all_embeddings(
        val_loader, tokenizer, static_embeddings, device
    )

    # Train linear probe
    print("  Training linear probe...")
    acc = train_linear_probe(
        train_embeds, train_labels, val_embeds, val_labels,
        num_classes=dataset.num_classes
    )
    fold_accs.append(acc)
    print(f"  Fold {val_fold} accuracy: {acc*100:.2f}%")

mean_acc = sum(fold_accs) / len(fold_accs)
std_acc = np.std(fold_accs)
print(f"\n=== Mean accuracy: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%) ===")

# Save results
emb_type = "pca" if use_pca else "full"
results = {
    "method": "static_embeddings_baseline",
    "embedding_type": emb_type,
    "embedding_dim": embed_dim,
    "static_emb_dir": str(static_emb_dir),
    "fold_accuracies": fold_accs,
    "mean_accuracy": mean_acc,
    "std_accuracy": std_acc,
}

results_path = output_dir / f"esc50_static_baseline_{emb_type}.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*60}")
print(f"ESC-50 STATIC EMBEDDINGS BASELINE COMPLETE")
print(f"  Embedding type: {emb_type} ({embed_dim}d)")
print(f"  Mean Accuracy: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")
print(f"{'='*60}")
EOF

echo ""
echo "ESC-50 static baseline evaluation complete!"
date
