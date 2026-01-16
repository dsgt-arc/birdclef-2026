#!/bin/bash
#SBATCH --job-name=distill-500k
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH --output=logs/distill-500k-%j.out
set -eu

# Usage: sbatch --export=STUDENT=convnext-small,MAX_EPOCHS=20,PATIENCE=3 run-cached-distillation-500k.sbatch
#
# Full-scale distillation on AudioSet-500k cached embeddings
# - 143 shards, ~500k samples, 687GB cache
# - Uses local NVMe for fast IO

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

STUDENT=${STUDENT:-convnext-small}
MAX_EPOCHS=${MAX_EPOCHS:-20}
PATIENCE=${PATIENCE:-3}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-1e-4}
COPY_TO_LOCAL=${COPY_TO_LOCAL:-false}

echo "=============================================="
echo "Full-Scale Distillation - AudioSet-500k"
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

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
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
"""Full-scale distillation on AudioSet-500k cached embeddings."""

import json
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

from birdclef.students import get_student_model, count_parameters
from birdclef.distillation import DistillationLoss

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
source_cache = Path(source_cache_dir)
local_cache = Path(local_cache_dir)
output_dir = scratch_dir / "outputs/exp003-distillation"
output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("=" * 60)
print("FULL-SCALE DISTILLATION - AudioSet-500k")
print("=" * 60)


class StreamingCacheDataset(torch.utils.data.Dataset):
    """Dataset with smart prefetching strategy for efficient streaming from cache.

    Features:
    - Background ThreadPoolExecutor copies shards predictively
    - Maintains warming set of next N shards (e.g., 5-10 shards ahead)
    - LRU eviction if local cache space gets full
    - Tracks shard status: warm (copied), warming (copying), cold (not started)
    - Falls back to source if copy hasn't finished yet
    - No blocking waits - GPU never starves
    """

    from concurrent.futures import ThreadPoolExecutor
    from collections import OrderedDict
    import threading

    def __init__(
        self,
        shard_files: list[Path],
        source_cache_dir: Path,
        local_cache_dir: Path,
        max_local_shards: int = 20,
        prefetch_shards: int = 5,
        num_copy_workers: int = 4,
    ):
        self.shard_files = shard_files
        self.source_cache_dir = source_cache_dir
        self.local_cache_dir = local_cache_dir
        self.max_local_shards = max_local_shards
        self.prefetch_shards = prefetch_shards

        # Background copying infrastructure
        self.executor = ThreadPoolExecutor(max_workers=num_copy_workers)
        self.lock = threading.Lock()

        # Track shard states: warm (in local cache), warming (currently copying), cold (not started)
        self.warm_shards = set()  # Successfully copied to local
        self.warming_futures = {}  # shard_idx -> Future
        self.access_order = OrderedDict()  # For LRU: shard_idx -> timestamp

        # Statistics
        self.stats = {
            "local_hits": 0,
            "source_fallback": 0,
            "shard_copies_total": 0,
            "lru_evictions": 0,
        }

        # Build index: list of (shard_idx, sample_idx_within_shard)
        self.index = []
        self.shard_lengths = []
        self.skipped_shards = []
        self.skipped_samples = 0

        print(f"  Loading {len(shard_files)} shards...")
        for shard_idx, shard_path in enumerate(tqdm(shard_files, desc="Indexing")):
            try:
                tensors = load_file(shard_path)
                n_samples = tensors["teacher_embeds"].shape[0]
                self.shard_lengths.append(n_samples)
                for i in range(n_samples):
                    self.index.append((shard_idx, i))
            except FileNotFoundError:
                self.skipped_shards.append(shard_path.name)
                self.skipped_samples += 1

        print(f"  Total samples: {len(self.index)}")
        if self.skipped_shards:
            print(f"  WARNING: {len(self.skipped_shards)} shards not found and will be skipped")

        # Discover which shards already exist in local cache
        if self.local_cache_dir.exists():
            for shard_path in self.local_cache_dir.glob("*.safetensors"):
                shard_name = shard_path.name
                # Find corresponding shard_idx
                for idx, shard_file in enumerate(self.shard_files):
                    if shard_file.name == shard_name:
                        with self.lock:
                            self.warm_shards.add(idx)
                        break

        print(f"  Pre-existing warm shards in local cache: {len(self.warm_shards)}")

        # Cache for current shard
        self._cached_shard_idx = None
        self._cached_tensors = None
        self.last_accessed_shard = -1

    def _get_shard_path(self, shard_idx: int) -> Path | None:
        """Try local first, then source. Returns path or None if not found."""
        shard_name = self.shard_files[shard_idx].name

        # Try local cache first
        local_path = self.local_cache_dir / shard_name
        if local_path.exists():
            return local_path

        # Fall back to source cache
        source_path = self.source_cache_dir / shard_name
        if source_path.exists():
            return source_path

        # Neither exists
        return None

    def _copy_shard_worker(self, shard_idx: int) -> bool:
        """Worker function: copy a single shard from source to local."""
        try:
            shard_name = self.shard_files[shard_idx].name
            source_path = self.source_cache_dir / shard_name
            local_path = self.local_cache_dir / shard_name

            if not source_path.exists():
                return False

            # Copy file
            import shutil
            shutil.copy2(source_path, local_path)

            with self.lock:
                self.warm_shards.add(shard_idx)
                self.stats["shard_copies_total"] += 1

            return True
        except Exception as e:
            print(f"WARNING: Failed to copy shard {shard_idx}: {e}")
            return False

    def _evict_lru_shard(self):
        """Remove the least recently used shard from local cache to make space."""
        if not self.access_order:
            return

        # Get oldest shard (first in OrderedDict)
        oldest_shard_idx = next(iter(self.access_order))
        shard_name = self.shard_files[oldest_shard_idx].name
        local_path = self.local_cache_dir / shard_name

        try:
            if local_path.exists():
                local_path.unlink()
                with self.lock:
                    self.warm_shards.discard(oldest_shard_idx)
                    del self.access_order[oldest_shard_idx]
                    self.stats["lru_evictions"] += 1
        except Exception as e:
            print(f"WARNING: Failed to evict shard {oldest_shard_idx}: {e}")

    def _schedule_prefetch(self, current_shard_idx: int):
        """Schedule background copying of next N shards."""
        prefetch_end = min(current_shard_idx + self.prefetch_shards, len(self.shard_files))

        for shard_idx in range(current_shard_idx, prefetch_end):
            with self.lock:
                # Skip if already warm or warming
                if shard_idx in self.warm_shards or shard_idx in self.warming_futures:
                    continue

                # Skip if exceeding local cache limit and would need eviction
                if len(self.warm_shards) >= self.max_local_shards:
                    self._evict_lru_shard()

                # Submit copy task
                future = self.executor.submit(self._copy_shard_worker, shard_idx)
                self.warming_futures[shard_idx] = future

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor] | None:
        """Load shard from best available location. Returns None if not found."""
        if self._cached_shard_idx == shard_idx and self._cached_tensors is not None:
            return self._cached_tensors

        shard_path = self._get_shard_path(shard_idx)
        if shard_path is None:
            return None

        try:
            self._cached_tensors = load_file(shard_path)
            self._cached_shard_idx = shard_idx

            # Track access for LRU
            with self.lock:
                self.access_order[shard_idx] = True
                self.access_order.move_to_end(shard_idx)

                if shard_path.parent == self.local_cache_dir:
                    self.stats["local_hits"] += 1
                else:
                    self.stats["source_fallback"] += 1

            return self._cached_tensors
        except Exception as e:
            print(f"ERROR loading {shard_path}: {e}")
            return None

    def get_cache_stats(self) -> dict:
        """Return current cache statistics."""
        with self.lock:
            return {
                "warm_shards": len(self.warm_shards),
                "warming_shards": len(self.warming_futures),
                "local_hits": self.stats["local_hits"],
                "source_fallback": self.stats["source_fallback"],
                "total_copies": self.stats["shard_copies_total"],
                "lru_evictions": self.stats["lru_evictions"],
            }

    def validate_cache(self) -> dict[str, any]:
        """Check which shards exist in each location before training starts."""
        source_shards = set()
        local_shards = set()
        missing_shards = set()

        if self.source_cache_dir.exists():
            source_shards = set(f.name for f in self.source_cache_dir.glob("*.safetensors"))

        if self.local_cache_dir.exists():
            local_shards = set(f.name for f in self.local_cache_dir.glob("*.safetensors"))

        # Check all shard files
        for shard_path in self.shard_files:
            if shard_path.name not in source_shards and shard_path.name not in local_shards:
                missing_shards.add(shard_path.name)

        return {
            "source_count": len(source_shards),
            "local_count": len(local_shards),
            "missing_count": len(missing_shards),
            "local_shards": sorted(local_shards),
            "source_shards": sorted(source_shards),
            "missing_shards": sorted(missing_shards),
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            shard_idx, sample_idx = self.index[idx]

            # Schedule prefetch of next shards on first access to new shard
            if shard_idx != self.last_accessed_shard:
                self._schedule_prefetch(shard_idx)
                self.last_accessed_shard = shard_idx

            tensors = self._load_shard(shard_idx)

            if tensors is None:
                # Shard not available, skip this sample
                print(f"WARNING: Skipping sample {idx} - shard {shard_idx} not found")
                return None

            static_embeds = tensors["static_embeds"][sample_idx].float()
            teacher_embeds = tensors["teacher_embeds"][sample_idx].float()

            return static_embeds, teacher_embeds
        except Exception as e:
            print(f"WARNING: Error loading sample {idx}: {e}")
            return None

    def __del__(self):
        """Cleanup: shutdown executor on deletion."""
        try:
            self.executor.shutdown(wait=True, timeout=30)
        except Exception as e:
            print(f"WARNING: Error shutting down executor: {e}")


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
train_shards = shard_files[:-n_val_shards]
val_shards = shard_files[-n_val_shards:]

print(f"\nCreating datasets with smart prefetching...")
print(f"  Train shards: {len(train_shards)}")
print(f"  Val shards: {len(val_shards)}")

train_dataset = StreamingCacheDataset(
    train_shards,
    source_cache,
    local_cache,
    max_local_shards=20,
    prefetch_shards=5,
    num_copy_workers=4,
)
val_dataset = StreamingCacheDataset(
    val_shards,
    source_cache,
    local_cache,
    max_local_shards=20,
    prefetch_shards=5,
    num_copy_workers=2,
)

# Validate cache availability before training
print("\nValidating cache availability...")
print("  Train dataset cache status:")
train_cache_status = train_dataset.validate_cache()
print(f"    Source cache: {train_cache_status['source_count']} shards")
print(f"    Local cache: {train_cache_status['local_count']} shards")
if train_cache_status['missing_count'] > 0:
    print(f"    WARNING: {train_cache_status['missing_count']} shards missing!")

print("  Val dataset cache status:")
val_cache_status = val_dataset.validate_cache()
print(f"    Source cache: {val_cache_status['source_count']} shards")
print(f"    Local cache: {val_cache_status['local_count']} shards")
if val_cache_status['missing_count'] > 0:
    print(f"    WARNING: {val_cache_status['missing_count']} shards missing!")

# Custom collate function to skip None values (samples that couldn't be loaded)
def collate_fn_skip_none(batch):
    """Filter out None entries and collate the rest."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    static_embeds = torch.stack([item[0] for item in batch])
    teacher_embeds = torch.stack([item[1] for item in batch])
    return static_embeds, teacher_embeds

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=collate_fn_skip_none,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    collate_fn=collate_fn_skip_none,
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
checkpoint_path = output_dir / f"500k_{student_name}_latest.pt"
best_checkpoint_path = output_dir / f"500k_{student_name}_best.pt"

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
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "epochs": [],
}

epochs_without_improvement = 0
start_time = time.time()

for epoch in range(start_epoch, max_epochs):
    epoch_start = time.time()

    # Train
    train_metrics = trainer.train_epoch(train_loader)

    # Validate
    val_metrics = trainer.validate(val_loader)

    epoch_time = time.time() - epoch_start
    current_lr = trainer.scheduler.get_last_lr()[0]

    # Get cache statistics
    train_cache_stats = train_dataset.get_cache_stats()
    local_hits = train_cache_stats["local_hits"]
    source_fallback = train_cache_stats["source_fallback"]
    total_accesses = local_hits + source_fallback
    cache_hit_rate = (local_hits / total_accesses * 100) if total_accesses > 0 else 0

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
        "cache_hit_rate": cache_hit_rate,
        "cache_stats": train_cache_stats,
    }
    results["epochs"].append(epoch_result)

    print(f"\nEpoch {epoch+1}/{max_epochs}:")
    print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
    print(f"  Train MSE:  {train_metrics['mse']:.4f} | Val MSE:  {val_metrics['mse']:.4f}")
    print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
    print(f"  Cache Hit Rate: {cache_hit_rate:.1f}% ({local_hits} local hits, {source_fallback} source fallback)")
    print(f"  Cache State: {train_cache_stats['warm_shards']} warm, {train_cache_stats['warming_shards']} warming, {train_cache_stats['lru_evictions']} evictions")

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
results_path = output_dir / f"500k_{student_name}_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*60}")
print(f"FULL-SCALE DISTILLATION COMPLETE")
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
echo "Full-scale distillation complete for ${STUDENT}!"
echo "GPU utilization log: logs/gpu-${SLURM_JOB_ID}.csv"
date
