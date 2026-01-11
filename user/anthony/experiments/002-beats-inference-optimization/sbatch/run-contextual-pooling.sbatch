#!/bin/bash
#SBATCH --job-name=ctx-pool-002
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t180 -qembers
#SBATCH --output=logs/contextual-pooling-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Training contextual poolers on ESC-50"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

uv run python << 'EOF'
"""Train and evaluate contextual poolers on ESC-50 5-fold CV."""

import json
import sys
import time
from pathlib import Path

print("Loading dependencies...", flush=True)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, "vendor/unilm/beats")

from birdclef.data import ESC50Dataset
from birdclef.static_embeddings import (
    StaticEmbeddingsPipelineFAISS,
    AttentionPooler,
    CNN1DPooler,
    LightCNN1DPooler,
)

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
repo_dir = Path.home() / "arc" / "birdclef-2026"
onnx_frontend_path = scratch_dir / "outputs/exp002-benchmarks/onnx/beats_frontend.onnx"
static_dir = scratch_dir / "outputs/exp002-benchmarks/static_embeddings"
codebook_path = static_dir / "codebook.npy"
embeddings_path = static_dir / "static_embeddings.npy"
output_dir = scratch_dir / "outputs/exp002-benchmarks"
esc50_root = repo_dir / "vendor" / "ESC-50"

print("=" * 60)
print("CONTEXTUAL POOLER TRAINING")
print("=" * 60)


def extract_fbank(waveform, pipeline):
    """Extract fbank features from waveform batch."""
    import torchaudio.compliance.kaldi as ta_kaldi

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
    fbank = fbank[:, np.newaxis, :, :]
    return fbank.astype(np.float32)


def extract_static_embeddings_batch(pipeline, waveforms):
    """Extract static embeddings (before pooling) for a batch."""
    # Convert to numpy
    if isinstance(waveforms, torch.Tensor):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        waveforms = waveforms.numpy()

    # Extract fbank
    fbank = extract_fbank(waveforms, pipeline)

    # ONNX frontend: fbank → patch embeddings
    patches = pipeline.session.run(None, {pipeline.input_name: fbank})[0]

    batch_size, num_patches, embed_dim = patches.shape

    # Quantize with FAISS
    patches_flat = patches.reshape(-1, embed_dim).astype(np.float32)
    _, token_ids_flat = pipeline.index.search(patches_flat, 1)
    token_ids = token_ids_flat.reshape(batch_size, num_patches)

    # Lookup static embeddings (don't pool yet!)
    embeddings = pipeline.embeddings[token_ids]  # (batch, num_patches, embed_dim)

    return torch.from_numpy(embeddings).float()


def extract_all_embeddings(pipeline, dataloader):
    """Extract static embeddings for entire dataset."""
    all_embeddings = []
    all_labels = []

    for waveforms, labels in tqdm(dataloader, desc="Extracting embeddings"):
        embeddings = extract_static_embeddings_batch(pipeline, waveforms)
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


class PoolerWithClassifier(nn.Module):
    """Pooler + linear classifier for end-to-end training."""

    def __init__(self, pooler, embed_dim, num_classes):
        super().__init__()
        self.pooler = pooler
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, embeddings):
        pooled = self.pooler(embeddings)
        return self.classifier(pooled)


def train_pooler(
    pooler,
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    num_classes,
    epochs=100,
    lr=1e-3,
    batch_size=32,
    device="cpu",
):
    """Train pooler + classifier end-to-end."""
    model = PoolerWithClassifier(pooler, train_embeddings.shape[-1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)
    val_embeddings = val_embeddings.to(device)
    val_labels = val_labels.to(device)

    n_train = len(train_embeddings)
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_emb = train_embeddings[idx]
            batch_labels = train_labels[idx]

            optimizer.zero_grad()
            logits = model(batch_emb)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_embeddings)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}, val_acc={val_acc:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_acc


def evaluate_5fold_cv(pooler_class, pooler_kwargs, pipeline, dataset, epochs=100, lr=1e-3):
    """Run 5-fold CV for a pooler."""
    fold_accs = []

    for val_fold in range(1, 6):
        print(f"\n=== Fold {val_fold}/5 ===")

        train_ds, val_ds = dataset.get_fold_splits(val_fold)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # Extract embeddings
        print("  Extracting training embeddings...")
        train_embeddings, train_labels = extract_all_embeddings(pipeline, train_loader)
        print("  Extracting validation embeddings...")
        val_embeddings, val_labels = extract_all_embeddings(pipeline, val_loader)

        # Create fresh pooler for this fold
        pooler = pooler_class(**pooler_kwargs)

        # Train
        print("  Training pooler...")
        model, best_acc = train_pooler(
            pooler,
            train_embeddings,
            train_labels,
            val_embeddings,
            val_labels,
            num_classes=dataset.num_classes,
            epochs=epochs,
            lr=lr,
        )

        fold_accs.append(best_acc)
        print(f"  Fold {val_fold} accuracy: {best_acc:.4f}")

    mean_acc = sum(fold_accs) / len(fold_accs)
    print(f"\n=== Mean accuracy: {mean_acc:.4f} ===")

    return {"fold_accuracies": fold_accs, "mean_accuracy": mean_acc}


# Load dataset
print("\nLoading ESC-50 dataset...")
dataset = ESC50Dataset(esc50_root)
print(f"Dataset: {len(dataset)} samples, {dataset.num_classes} classes")

# Initialize FAISS pipeline
print("\nInitializing FAISS pipeline...")
pipeline = StaticEmbeddingsPipelineFAISS(
    onnx_path=onnx_frontend_path,
    codebook_path=codebook_path,
    embeddings_path=embeddings_path,
    weights_path=None,
    num_threads=4,
)
print(f"  embed_dim={pipeline.embed_dim}")

results = {}

# Pooler configurations to test
pooler_configs = [
    ("AttentionPooler", AttentionPooler, {"embed_dim": 768}),
    ("LightCNN1DPooler", LightCNN1DPooler, {"embed_dim": 768, "bottleneck_dim": 256}),
    ("CNN1DPooler", CNN1DPooler, {"embed_dim": 768, "hidden_dim": 768}),
]

for name, pooler_class, pooler_kwargs in pooler_configs:
    print("\n" + "=" * 60)
    print(f"TRAINING: {name}")
    print("=" * 60)

    # Count parameters
    temp_pooler = pooler_class(**pooler_kwargs)
    n_params = sum(p.numel() for p in temp_pooler.parameters())
    print(f"  Parameters: {n_params:,}")

    start_time = time.time()
    result = evaluate_5fold_cv(pooler_class, pooler_kwargs, pipeline, dataset)
    result["time_s"] = time.time() - start_time
    result["n_params"] = n_params
    results[name] = result

# Summary
print("\n" + "=" * 60)
print("ACCURACY SUMMARY")
print("=" * 60)
print(f"{'Method':<25} {'Mean Acc':>10} {'Params':>12} {'Time (s)':>10}")
print("-" * 60)

# Add baseline (mean pooling) from previous results
print(f"{'MeanPool (baseline)':<25} {'35.9%':>10} {'0':>12} {'-':>10}")

for name, res in results.items():
    print(f"{name:<25} {res['mean_accuracy']*100:>9.1f}% {res['n_params']:>12,} {res['time_s']:>10.1f}")
print("-" * 60)

# Save results
output_path = output_dir / "contextual_pooling_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

print("\nTraining complete!")
EOF

echo ""
echo "Contextual pooling training complete!"
date
