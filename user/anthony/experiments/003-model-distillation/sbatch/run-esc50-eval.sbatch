#!/bin/bash
#SBATCH --job-name=eval-003-esc50
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t60 -qembers
#SBATCH --output=logs/esc50-eval-%j.out
set -eu

# Usage: sbatch --export=STUDENT=convnext-small,CHECKPOINT=pilot_convnext-small_epoch5.pt run-esc50-eval.sbatch

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

STUDENT=${STUDENT:-convnext-small}
CHECKPOINT=${CHECKPOINT:-pilot_${STUDENT}_epoch5.pt}

echo "Evaluating distilled student on ESC-50"
echo "Student: ${STUDENT}"
echo "Checkpoint: ${CHECKPOINT}"
hostname
date

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

uv run python - "$STUDENT" "$CHECKPOINT" << 'EOF'
"""Evaluate distilled student on ESC-50 5-fold CV."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

student_name = sys.argv[1]
checkpoint_name = sys.argv[2]
print(f"Student: {student_name}", flush=True)
print(f"Checkpoint: {checkpoint_name}", flush=True)

sys.path.insert(0, "vendor/unilm/beats")

from birdclef.data import ESC50Dataset
from birdclef.students import get_student_model, count_parameters
from birdclef.distillation import load_teacher_and_tokenizer

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
repo_dir = Path.home() / "arc" / "birdclef-2026"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
tokenizer_path = scratch_dir / "checkpoints/beats/Tokenizers_iter3.pt"
static_embeddings_path = scratch_dir / "outputs/exp002-benchmarks/static_embeddings/static_embeddings.npy"
checkpoint_path = scratch_dir / "outputs/exp003-distillation" / checkpoint_name
output_dir = scratch_dir / "outputs/exp003-distillation"
esc50_root = repo_dir / "vendor" / "ESC-50"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("=" * 60)
print("ESC-50 EVALUATION - EXPERIMENT 003")
print("=" * 60)

# Load tokenizer (for getting token IDs)
print("\nLoading tokenizer...")
_, tokenizer = load_teacher_and_tokenizer(encoder_path, tokenizer_path)
tokenizer = tokenizer.to(device)

# Load static embeddings
print("\nLoading static embeddings...")
static_embeddings = torch.from_numpy(np.load(static_embeddings_path)).float().to(device)
print(f"  Static embeddings shape: {static_embeddings.shape}")

# Load student model and checkpoint
print(f"\nLoading student model: {student_name}")
student = get_student_model(student_name)
n_params = count_parameters(student)
print(f"  Student parameters: {n_params:,}")

print(f"\nLoading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
student.load_state_dict(checkpoint["student_state_dict"])
student = student.to(device)
student.eval()
print(f"  Loaded from epoch {checkpoint['epoch']}, step {checkpoint['step']}")


def extract_student_embeddings(waveforms, tokenizer, static_embeds, student, device):
    """Extract embeddings from student model."""
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
        # Run through student
        outputs = student(inputs)
        # Mean pool
        pooled = outputs.mean(dim=1)

    return pooled


def extract_all_embeddings(dataloader, tokenizer, static_embeds, student, device):
    """Extract embeddings for entire dataset."""
    all_embeds = []
    all_labels = []

    for waveforms, labels in tqdm(dataloader, desc="Extracting"):
        embeds = extract_student_embeddings(
            waveforms, tokenizer, static_embeds, student, device
        )
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
    best_state = None

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
            best_state = classifier.state_dict().copy()

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
        train_loader, tokenizer, static_embeddings, student, device
    )
    print("  Extracting val embeddings...")
    val_embeds, val_labels = extract_all_embeddings(
        val_loader, tokenizer, static_embeddings, student, device
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
print(f"\n=== Mean accuracy: {mean_acc*100:.2f}% ===")

# Save results
results = {
    "student": student_name,
    "checkpoint": checkpoint_name,
    "n_params": n_params,
    "fold_accuracies": fold_accs,
    "mean_accuracy": mean_acc,
}

results_path = output_dir / f"esc50_eval_{student_name}.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*60}")
print(f"ESC-50 EVALUATION COMPLETE")
print(f"  Student: {student_name}")
print(f"  Mean Accuracy: {mean_acc*100:.2f}%")
print(f"{'='*60}")
EOF

echo ""
echo "ESC-50 evaluation complete for ${STUDENT}!"
date
