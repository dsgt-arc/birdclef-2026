#!/bin/bash
#SBATCH --job-name=pooler-002
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/pooler-%x-%j.out
set -eu

# Usage: sbatch --export=POOLER_NAME=AttentionPooler run-pooler.sbatch

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

POOLER_NAME=${POOLER_NAME:-AttentionPooler}

echo "Training ${POOLER_NAME} on ESC-50"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

uv run python - "$POOLER_NAME" << 'EOF'
"""Train and evaluate a single pooler on ESC-50 5-fold CV."""

import json
import sys
import time
from pathlib import Path

pooler_name = sys.argv[1]
print(f"Training pooler: {pooler_name}", flush=True)

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
    AttentionPooler,
    CNN1DPooler,
    LightCNN1DPooler,
    CNNAttentionPooler,
)

# Pooler configurations
POOLER_CONFIGS = {
    "AttentionPooler": (AttentionPooler, {"embed_dim": 768}),
    "LightCNN1DPooler": (LightCNN1DPooler, {"embed_dim": 768, "bottleneck_dim": 256}),
    "CNN1DPooler": (CNN1DPooler, {"embed_dim": 768, "hidden_dim": 768}),
    "CNNAttentionPooler": (CNNAttentionPooler, {"embed_dim": 768, "bottleneck_dim": 256}),
}

if pooler_name not in POOLER_CONFIGS:
    print(f"Unknown pooler: {pooler_name}")
    print(f"Available: {list(POOLER_CONFIGS.keys())}")
    sys.exit(1)

pooler_class, pooler_kwargs = POOLER_CONFIGS[pooler_name]

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
print(f"TRAINING: {pooler_name}")
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
    if isinstance(waveforms, torch.Tensor):
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
        waveforms = waveforms.numpy()

    fbank = extract_fbank(waveforms, pipeline)
    patches = pipeline.session.run(None, {pipeline.input_name: fbank})[0]

    batch_size, num_patches, embed_dim = patches.shape
    patches_flat = patches.reshape(-1, embed_dim).astype(np.float32)
    _, token_ids_flat = pipeline.index.search(patches_flat, 1)
    token_ids = token_ids_flat.reshape(batch_size, num_patches)
    embeddings = pipeline.embeddings[token_ids]

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
    def __init__(self, pooler, embed_dim, num_classes):
        super().__init__()
        self.pooler = pooler
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, embeddings):
        pooled = self.pooler(embeddings)
        return self.classifier(pooled)


def train_pooler(
    pooler, train_embeddings, train_labels, val_embeddings, val_labels,
    num_classes, epochs=100, lr=1e-3, batch_size=32, device="cpu",
):
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

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_acc


def evaluate_5fold_cv(pooler_class, pooler_kwargs, pipeline, dataset, epochs=100, lr=1e-3):
    fold_accs = []

    for val_fold in range(1, 6):
        print(f"\n=== Fold {val_fold}/5 ===")

        train_ds, val_ds = dataset.get_fold_splits(val_fold)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        print("  Extracting training embeddings...")
        train_embeddings, train_labels = extract_all_embeddings(pipeline, train_loader)
        print("  Extracting validation embeddings...")
        val_embeddings, val_labels = extract_all_embeddings(pipeline, val_loader)

        pooler = pooler_class(**pooler_kwargs)

        print("  Training pooler...")
        model, best_acc = train_pooler(
            pooler, train_embeddings, train_labels, val_embeddings, val_labels,
            num_classes=dataset.num_classes, epochs=epochs, lr=lr,
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

# Count parameters
temp_pooler = pooler_class(**pooler_kwargs)
n_params = sum(p.numel() for p in temp_pooler.parameters())
print(f"  Parameters: {n_params:,}")

# Train
start_time = time.time()
result = evaluate_5fold_cv(pooler_class, pooler_kwargs, pipeline, dataset)
result["time_s"] = time.time() - start_time
result["n_params"] = n_params
result["pooler_name"] = pooler_name

# Save results
output_path = output_dir / f"pooler_results_{pooler_name}.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {output_path}")

print(f"\n{'='*60}")
print(f"RESULT: {pooler_name}")
print(f"  Mean accuracy: {result['mean_accuracy']*100:.1f}%")
print(f"  Parameters: {n_params:,}")
print(f"  Time: {result['time_s']:.1f}s")
print(f"{'='*60}")
EOF

echo ""
echo "Training complete for ${POOLER_NAME}!"
date
