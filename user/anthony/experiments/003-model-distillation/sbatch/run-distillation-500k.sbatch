#!/bin/bash
#SBATCH --job-name=distill-500k
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH -p gpu-rtx6000
#SBATCH --output=logs/distill-500k-%j.out
set -eu

# Usage: sbatch --export=STUDENT=convnext-small,MAX_EPOCHS=20,PATIENCE=3 run-distillation-500k.sbatch
#
# Distillation on AudioSet-500k using ShardAffinityDataset
# - Per-worker shard affinity with aggressive prefetching
# - 143 shards, ~500k samples, 687GB cache
# - Comparison test with StreamingCacheDataset baseline

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

STUDENT=${STUDENT:-convnext-small}
MAX_EPOCHS=${MAX_EPOCHS:-20}
PATIENCE=${PATIENCE:-3}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
COPY_TO_LOCAL=${COPY_TO_LOCAL:-false}

echo "=============================================="
echo "ShardAffinityDataset - AudioSet-500k"
echo "=============================================="
echo "Student model: ${STUDENT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Early stopping patience: ${PATIENCE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LR}"
echo "Copy to local: ${COPY_TO_LOCAL}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

# Sync venv to local NVMe for faster package loading
local_venv=${TMPDIR}/venv
echo "Syncing venv to local NVMe..."
time rsync -a --info=progress2 $scratch_dir/.venv/ $local_venv/

export UV_PROJECT_ENVIRONMENT=$local_venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

# Start GPU monitoring in background
$repo_dir/scripts/monitor-gpu.sh 5 logs/gpu-${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

# Cleanup function for background processes
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "${GPU_MONITOR_PID:-}" ]; then
        echo "  Stopping GPU monitor (PID: $GPU_MONITOR_PID)"
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Setup cache directories
source_cache_dir=$scratch_dir/outputs/exp003-distillation/teacher_cache_500k
local_cache_dir=${TMPDIR}/teacher_cache_500k

echo ""
echo "Cache setup:"
echo "  Source cache: $source_cache_dir ($(du -sh $source_cache_dir 2>/dev/null | cut -f1))"
echo "  Local cache: $local_cache_dir"
echo "  TMPDIR: ${TMPDIR}"

# Optionally copy cached data to local NVMe for faster I/O
if [ "$COPY_TO_LOCAL" = "true" ]; then
    echo ""
    echo "Copying cached data to local NVMe (using parallel cp)..."
    mkdir -p $local_cache_dir
    echo "Starting parallel copy..."
    time (ls $source_cache_dir/*.safetensors | xargs -P 8 -I {} cp {} $local_cache_dir/ 2>/dev/null || true)

    echo "Copy complete. Local cache:"
    ls $local_cache_dir/*.safetensors 2>/dev/null | wc -l || echo "0"
    du -sh $local_cache_dir 2>/dev/null || echo "0B"
else
    echo ""
    echo "Skipping copy to local (COPY_TO_LOCAL=false)"
    echo "Will use source cache directly with fallback support"
fi

uv run python - "$STUDENT" "$MAX_EPOCHS" "$PATIENCE" "$BATCH_SIZE" "$LR" "$source_cache_dir" "$local_cache_dir" << 'EOF'
"""Distillation on AudioSet-500k using ShardAffinityDataset."""

import json
import os
import sys
import time
from pathlib import Path

student_name = sys.argv[1]
max_epochs = int(sys.argv[2])
patience = int(sys.argv[3])
batch_size = int(sys.argv[4])
lr = float(sys.argv[5])
source_cache_dir = sys.argv[6]
local_cache_dir = sys.argv[7]

print(f"Student model: {student_name}", flush=True)
print(f"Max epochs: {max_epochs}", flush=True)
print(f"Early stopping patience: {patience}", flush=True)
print(f"Batch size: {batch_size}", flush=True)
print(f"Learning rate: {lr}", flush=True)
print(f"Source cache dir: {source_cache_dir}", flush=True)
print(f"Local cache dir: {local_cache_dir}", flush=True)

print("\nLoading dependencies...", flush=True)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import load_file
from safetensors import safe_open
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import threading
import random

from birdclef.students import get_student_model, count_parameters
from birdclef.distillation import DistillationLoss
from birdclef.data.shard_affinity_dataset import ShardAffinityDataset, create_dataloader, worker_init_fn, collate_fn_skip_none

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
source_cache = Path(source_cache_dir)
local_cache = Path(local_cache_dir)
output_dir = scratch_dir / "outputs/exp003-distillation"
output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("=" * 60)
print("DISTILLATION - AudioSet-500k (ShardAffinityDataset)")
print("=" * 60)


class EarlyStoppingTrainer:
    """Trainer with early stopping based on validation loss."""

    def __init__(
        self,
        student: nn.Module,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        mse_weight: float = 0.5,
        cosine_weight: float = 0.5,
        max_epochs: int = 20,
    ):
        self.student = student.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-6
        )
        self.criterion = DistillationLoss(mse_weight, cosine_weight)

        self.step = 0
        self.epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        n_steps = 0

        pbar = tqdm(dataloader, desc=f"Train Epoch {self.epoch}")
        for batch in pbar:
            if batch is None:  # Skip empty batches due to filtering
                continue

            static_embeds, teacher_embeds = batch
            static_embeds = static_embeds.to(self.device)
            teacher_embeds = teacher_embeds.to(self.device)

            # Forward
            student_embeds = self.student(static_embeds)
            loss_dict = self.criterion(student_embeds, teacher_embeds)
            loss = loss_dict["loss"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss_dict["loss"].item()
            total_mse += loss_dict["mse"].item()
            total_cosine += loss_dict["cosine"].item()
            n_steps += 1
            self.step += 1

            if self.step % 100 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.scheduler.step()
        self.epoch += 1

        if n_steps == 0:
            # No valid samples in epoch
            return {
                "loss": 0.0,
                "mse": 0.0,
                "cosine": 0.0,
            }

        return {
            "loss": total_loss / n_steps,
            "mse": total_mse / n_steps,
            "cosine": total_cosine / n_steps,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Compute validation loss."""
        self.student.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_cosine = 0.0
        n_steps = 0

        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:  # Skip empty batches due to filtering
                continue

            static_embeds, teacher_embeds = batch
            static_embeds = static_embeds.to(self.device)
            teacher_embeds = teacher_embeds.to(self.device)

            student_embeds = self.student(static_embeds)
            loss_dict = self.criterion(student_embeds, teacher_embeds)

            total_loss += loss_dict["loss"].item()
            total_mse += loss_dict["mse"].item()
            total_cosine += loss_dict["cosine"].item()
            n_steps += 1

        if n_steps == 0:
            # No valid samples in validation
            return {
                "loss": 0.0,
                "mse": 0.0,
                "cosine": 0.0,
            }

        return {
            "loss": total_loss / n_steps,
            "mse": total_mse / n_steps,
            "cosine": total_cosine / n_steps,
        }

    def save_checkpoint(self, path: Path, val_loss: float, is_best: bool = False):
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "epoch": self.epoch,
            "val_loss": val_loss,
            "is_best": is_best,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("val_loss", float("inf"))


# Load cached data
print("\nLoading cached teacher embeddings...")
shard_files = sorted(source_cache.glob("*.safetensors"))
print(f"  Found {len(shard_files)} shard files in source cache")

if len(shard_files) == 0:
    raise ValueError(f"No cached shards found in {source_cache}")

# Split: 90% train, 10% val (last ~14 shards for validation)
n_val_shards = max(1, len(shard_files) // 10)
train_shard_files = shard_files[:-n_val_shards]
val_shard_files = shard_files[-n_val_shards:]

print(f"\nCreating ShardAffinityDataset instances...")
print(f"  Train shards: {len(train_shard_files)}")
print(f"  Val shards: {len(val_shard_files)}")

# Create datasets with ShardAffinityDataset
train_dataset = ShardAffinityDataset(
    source_cache_dir=source_cache,
    local_cache_dir=local_cache if local_cache.exists() or local_cache.parent.exists() else None,
    shards_per_worker=3,
    prefetch_ahead=2,
    copy_to_local=COPY_TO_LOCAL == "true" if isinstance(COPY_TO_LOCAL, str) else COPY_TO_LOCAL,
    seed=42,
)

val_dataset = ShardAffinityDataset(
    source_cache_dir=source_cache,
    local_cache_dir=local_cache if local_cache.exists() or local_cache.parent.exists() else None,
    shards_per_worker=3,
    prefetch_ahead=2,
    copy_to_local=COPY_TO_LOCAL == "true" if isinstance(COPY_TO_LOCAL, str) else COPY_TO_LOCAL,
    seed=42,
)

# Create dataloaders using the helper function
train_loader = create_dataloader(
    train_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    drop_last=True,
)

val_loader = create_dataloader(
    val_dataset,
    batch_size=batch_size,
    num_workers=8,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    drop_last=False,
)

print(f"  Train samples: {len(train_dataset):,}")
print(f"  Val samples: {len(val_dataset):,}")
print(f"  Train batches/epoch: {len(train_loader):,}")

# Create student model
print(f"\nCreating student model: {student_name}")
student = get_student_model(student_name)
n_params = count_parameters(student)
print(f"  Student parameters: {n_params:,}")

# Check for existing checkpoint to resume from
checkpoint_path = output_dir / f"500k_{student_name}_shard_affinity_latest.pt"
best_checkpoint_path = output_dir / f"500k_{student_name}_shard_affinity_best.pt"

# Create trainer
print("\nInitializing trainer...")
trainer = EarlyStoppingTrainer(
    student=student,
    device=device,
    lr=lr,
    weight_decay=0.01,
    mse_weight=0.5,
    cosine_weight=0.5,
    max_epochs=max_epochs,
)

# Resume from checkpoint if exists
start_epoch = 0
best_val_loss = float("inf")
if checkpoint_path.exists():
    print(f"\nResuming from checkpoint: {checkpoint_path}")
    best_val_loss = trainer.load_checkpoint(checkpoint_path)
    start_epoch = trainer.epoch
    print(f"  Resumed at epoch {start_epoch}, best val_loss: {best_val_loss:.4f}")

# Training loop with early stopping
print(f"\nStarting training (max {max_epochs} epochs, patience {patience})...")
results = {
    "student": student_name,
    "n_params": n_params,
    "batch_size": batch_size,
    "lr": lr,
    "max_epochs": max_epochs,
    "patience": patience,
    "dataset_type": "ShardAffinityDataset",
    "shards_per_worker": 3,
    "prefetch_ahead": 2,
    "num_workers": 8,
    "prefetch_factor": 4,
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "epochs": [],
}

epochs_without_improvement = 0
start_time = time.time()

for epoch in range(start_epoch, max_epochs):
    epoch_start = time.time()

    # Reshuffle for this epoch
    train_dataset.reset_epoch(epoch)

    # Train
    train_metrics = trainer.train_epoch(train_loader)

    # Validate
    val_metrics = trainer.validate(val_loader)

    epoch_time = time.time() - epoch_start
    current_lr = trainer.scheduler.get_last_lr()[0]

    # Get worker statistics (from first worker if available)
    worker_stats = train_dataset.get_worker_stats()

    epoch_result = {
        "epoch": epoch + 1,
        "train_loss": train_metrics["loss"],
        "train_mse": train_metrics["mse"],
        "train_cosine": train_metrics["cosine"],
        "val_loss": val_metrics["loss"],
        "val_mse": val_metrics["mse"],
        "val_cosine": val_metrics["cosine"],
        "lr": current_lr,
        "time_s": epoch_time,
        "worker_stats": worker_stats,
    }
    results["epochs"].append(epoch_result)

    print(f"\nEpoch {epoch+1}/{max_epochs}:")
    print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Train MSE:  {train_metrics['mse']:.4f} | Val MSE:  {val_metrics['mse']:.4f}")
    print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
    if worker_stats != {"mode": "single_process"}:
        print(f"  Worker Stats: {worker_stats}")

    # Check for improvement
    is_best = val_metrics["loss"] < best_val_loss
    if is_best:
        best_val_loss = val_metrics["loss"]
        epochs_without_improvement = 0
        print(f"  >> New best validation loss: {best_val_loss:.4f}")
        trainer.save_checkpoint(best_checkpoint_path, best_val_loss, is_best=True)
    else:
        epochs_without_improvement += 1
        print(f"  No improvement for {epochs_without_improvement}/{patience} epochs")

    # Save latest checkpoint
    trainer.save_checkpoint(checkpoint_path, val_metrics["loss"])

    # Early stopping check
    if epochs_without_improvement >= patience:
        print(f"\n>> Early stopping triggered after {epoch+1} epochs")
        results["early_stopped"] = True
        results["stopped_at_epoch"] = epoch + 1
        break

total_time = time.time() - start_time
results["total_time_s"] = total_time
results["best_val_loss"] = best_val_loss
results["final_epoch"] = trainer.epoch

# Save results
results_path = output_dir / f"500k_{student_name}_shard_affinity_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

# Cleanup
train_dataset.cleanup()
val_dataset.cleanup()

print(f"\n{'='*60}")
print(f"DISTILLATION COMPLETE (ShardAffinityDataset)")
print(f"  Student: {student_name}")
print(f"  Parameters: {n_params:,}")
print(f"  Train samples: {len(train_dataset):,}")
print(f"  Best Val Loss: {best_val_loss:.4f}")
print(f"  Epochs: {trainer.epoch}")
print(f"  Total Time: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"  Best checkpoint: {best_checkpoint_path}")
print(f"{'='*60}")
EOF

echo ""
echo "ShardAffinityDataset distillation complete for ${STUDENT}!"
echo "GPU utilization log: logs/gpu-${SLURM_JOB_ID}.csv"
date
