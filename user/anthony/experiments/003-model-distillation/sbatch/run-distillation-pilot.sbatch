#!/bin/bash
#SBATCH --job-name=distill-003-pilot
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH --output=logs/distillation-pilot-%j.out
set -eu

# Usage: sbatch --export=STUDENT=convnext-small run-distillation-pilot.sbatch

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

STUDENT=${STUDENT:-convnext-small}

echo "Running Pilot Distillation - Experiment 003"
echo "Student model: ${STUDENT}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Check for required files
encoder_path=$scratch_dir/checkpoints/beats/BEATs_iter3.pt
tokenizer_path=$scratch_dir/checkpoints/beats/Tokenizers_iter3.pt
static_embeddings_path=$scratch_dir/outputs/exp002-benchmarks/static_embeddings/static_embeddings.npy

if [ ! -f "$encoder_path" ]; then
    echo "ERROR: BEATs encoder not found at $encoder_path"
    echo "Run experiment 002 download-checkpoints.sbatch first"
    exit 1
fi

if [ ! -f "$static_embeddings_path" ]; then
    echo "ERROR: Static embeddings not found at $static_embeddings_path"
    echo "Run experiment 002 distillation step first"
    exit 1
fi

# Download AudioSet-20k if not present
audioset_dir=$scratch_dir/data/audioset-wds/20k
if [ ! -d "$audioset_dir/train" ]; then
    echo "Downloading AudioSet-20k..."
    cd $repo_dir
    uv run huggingface-cli download confit/audioset-16khz-wds \
      --include "20k/train/*.tar" "20k/test/*.tar" \
      --repo-type=dataset \
      --local-dir $scratch_dir/data/audioset-wds
fi

cd $repo_dir

uv run python - "$STUDENT" << 'EOF'
"""Pilot distillation training on AudioSet-20k."""

import json
import sys
import time
from pathlib import Path

student_name = sys.argv[1]
print(f"Student model: {student_name}", flush=True)

print("Loading dependencies...", flush=True)
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "vendor/unilm/beats")

from birdclef.students import get_student_model, count_parameters
from birdclef.distillation import DistillationTrainer, load_teacher_and_tokenizer

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
tokenizer_path = scratch_dir / "checkpoints/beats/Tokenizers_iter3.pt"
static_embeddings_path = scratch_dir / "outputs/exp002-benchmarks/static_embeddings/static_embeddings.npy"
audioset_dir = scratch_dir / "data/audioset-wds/20k"
output_dir = scratch_dir / "outputs/exp003-distillation"
output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("=" * 60)
print("PILOT DISTILLATION - EXPERIMENT 003")
print("=" * 60)

# Load teacher and tokenizer
print("\nLoading teacher and tokenizer...")
teacher, tokenizer = load_teacher_and_tokenizer(encoder_path, tokenizer_path)
print(f"  Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")

# Load static embeddings
print("\nLoading static embeddings...")
static_embeddings = torch.from_numpy(np.load(static_embeddings_path)).float()
print(f"  Static embeddings shape: {static_embeddings.shape}")

# Create student model
print(f"\nCreating student model: {student_name}")
student = get_student_model(student_name)
n_params = count_parameters(student)
print(f"  Student parameters: {n_params:,}")

# Create trainer
print("\nInitializing trainer...")
trainer = DistillationTrainer(
    student=student,
    teacher_encoder=teacher,
    tokenizer=tokenizer,
    static_embeddings=static_embeddings,
    device=device,
    lr=1e-4,
    weight_decay=0.01,
    mse_weight=0.5,
    cosine_weight=0.5,
)

# Create simple dataloader from AudioSet shards
print("\nLoading AudioSet-20k...")
import glob
import io
import tarfile
import soundfile as sf

class SimpleTarDataset(torch.utils.data.Dataset):
    """Simple dataset that loads from tar files."""
    def __init__(self, tar_pattern, target_length=160000):
        self.samples = []
        self.target_length = target_length

        tar_files = sorted(glob.glob(tar_pattern))
        print(f"  Found {len(tar_files)} tar files")

        for tar_path in tar_files:
            with tarfile.open(tar_path, 'r') as tar:
                members = [m for m in tar.getmembers() if m.name.endswith(('.wav', '.flac', '.mp3'))]
                for member in members:
                    self.samples.append((tar_path, member.name))

        print(f"  Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tar_path, member_name = self.samples[idx]

        with tarfile.open(tar_path, 'r') as tar:
            f = tar.extractfile(member_name)
            audio_data, sr = sf.read(io.BytesIO(f.read()))

        waveform = torch.from_numpy(audio_data.astype(np.float32))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=-1)

        # Pad or trim
        if len(waveform) < self.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - len(waveform)))
        else:
            waveform = waveform[:self.target_length]

        return waveform

train_pattern = str(audioset_dir / "train" / "*.tar")
dataset = SimpleTarDataset(train_pattern, target_length=160000)  # 10s at 16kHz
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Verify shapes with first batch
print("\nVerifying shapes with first batch...")
test_batch = next(iter(dataloader))
print(f"  Waveform batch shape: {test_batch.shape}")
with torch.no_grad():
    test_batch = test_batch.to(device)
    test_tokens = tokenizer.extract_labels(test_batch)
    print(f"  Token IDs raw shape: {test_tokens.shape}")
    # Reshape if flattened
    if test_tokens.dim() == 1:
        seq_len = test_tokens.shape[0] // test_batch.shape[0]
        test_tokens = test_tokens.view(test_batch.shape[0], seq_len)
    print(f"  Token IDs reshaped: {test_tokens.shape}")
    test_embeds = static_embeddings[test_tokens.cpu()]
    print(f"  Static embeddings shape: {test_embeds.shape}")
    test_teacher = teacher.extract_features(test_batch, padding_mask=None)[0]
    print(f"  Teacher output shape: {test_teacher.shape}")

# Training loop
print("\nStarting training...")
epochs = 5
results = {"student": student_name, "n_params": n_params, "epochs": []}

start_time = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    metrics = trainer.train_epoch(dataloader, log_interval=50)
    epoch_time = time.time() - epoch_start

    epoch_result = {
        "epoch": epoch + 1,
        "loss": metrics["loss"],
        "mse": metrics["mse"],
        "cosine": metrics["cosine"],
        "time_s": epoch_time,
    }
    results["epochs"].append(epoch_result)

    print(f"\nEpoch {epoch+1}/{epochs}:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  Cosine: {metrics['cosine']:.4f}")
    print(f"  Time: {epoch_time:.1f}s")

    # Save checkpoint
    checkpoint_path = output_dir / f"pilot_{student_name}_epoch{epoch+1}.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint: {checkpoint_path}")

total_time = time.time() - start_time
results["total_time_s"] = total_time

# Save results
results_path = output_dir / f"pilot_{student_name}_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*60}")
print(f"PILOT DISTILLATION COMPLETE")
print(f"  Student: {student_name}")
print(f"  Parameters: {n_params:,}")
print(f"  Final Loss: {results['epochs'][-1]['loss']:.4f}")
print(f"  Total Time: {total_time:.1f}s")
print(f"{'='*60}")
EOF

echo ""
echo "Pilot distillation complete for ${STUDENT}!"
date
