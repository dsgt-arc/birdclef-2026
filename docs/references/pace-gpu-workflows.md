# PACE GPU Workflow Best Practices

Reference for running efficient GPU training jobs on Georgia Tech's PACE cluster.

## Storage Tiers

PACE has four storage tiers with dramatically different I/O performance:

| Storage | Path | Size | Speed | Persistent? |
|---------|------|------|-------|-------------|
| Home | `~/` | **20 GB** | Slow NFS | Yes |
| Scratch | `~/scratch/` | 15 TB | Network NFS | Yes (60-day purge) |
| Project | `/storage/project/<alloc>/` | Varies | Network NFS | Yes |
| **Local NVMe** | **`$TMPDIR`** (SLURM) or `/tmp/` | **~1.4 TB SSD** | **10-100x faster** | **No** (wiped after job) |

The key insight: **Home, Scratch, and Project storage are all network-attached (NFS).**
Every file read goes `compute node → network → storage server → disk → network → compute node`.
Local NVMe (`$TMPDIR`) is a physical SSD on the compute node itself — no network hop.

## Why This Matters for Training

A typical training loop does thousands of disk reads per epoch. Where those files live
determines a huge chunk of training time.

**Example from Exp 006** (batch_size=32, 5,411 steps/epoch, 30 epochs):

| Data location | Avg read time | Data loading % | Est. wasted time |
|---------------|---------------|----------------|------------------|
| Network NFS (cold node) | 68 ms/step | 35.5% | ~2.9 hours |
| Local NVMe / page cache | 3 ms/step | 1.9% | ~8 minutes |

That's ~2.9 hours of the 9-hour training spent waiting for NFS file reads.

## Best Practice: Use `$TMPDIR` for Everything Ephemeral

### 1. Virtual environments

Python venvs involve thousands of small file reads (imports). Always create them on
local NVMe:

```bash
# In sbatch job scripts
JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
```

**Also set `UV_CACHE_DIR` on the same filesystem** to avoid the cross-filesystem hardlink
bug (uv silently creates empty directory stubs when cache and venv are on different
filesystems):

```bash
export UV_CACHE_DIR="$TMPDIR/uv-cache"
```

See `user/murilo/create_venv.sh` and `user/murilo/experiments/006-lora-finetune/sbatch/_job.sh`
for working examples.

### 2. Training data

Copy training data to `$TMPDIR` at job start. This is the single biggest speedup for
I/O-bound training:

```bash
# Copy dataset to local NVMe (~7GB, takes 1-3 min)
LOCAL_DATA="$TMPDIR/data/birdclef-2025"
if [[ ! -d "$LOCAL_DATA" ]]; then
    echo "Copying training data to local NVMe..."
    mkdir -p "$TMPDIR/data"
    cp -r ~/ps-clef2026_birdclef-0/data/birdclef-2025 "$LOCAL_DATA"
    echo "  Done ($(du -sh "$LOCAL_DATA" | cut -f1))"
fi
```

Then point your training config or CLI at `$LOCAL_DATA` instead of the NFS path.

**When to do this:**
- Dataset fits in `$TMPDIR` (~1.4 TB available)
- Training runs multiple epochs (cost of copy is amortized)
- You see >20% data loading time in GPU checks

**When NOT to do this:**
- Datasets larger than available `$TMPDIR` space
- Single-epoch or few-step jobs (copy overhead > time saved)
- Data is already being read from memory (e.g., HuggingFace `datasets` with memory mapping)

### 3. Model caches (HuggingFace, torch hub)

Pre-trained model weights are downloaded once and reused. Keep these on **project storage**
(shared across jobs/users) rather than `$TMPDIR` (wiped each job) or home (tiny quota):

```bash
export HF_HOME="/storage/project/<alloc>/shared/birdclef-2026/cache/huggingface"
```

## Anatomy of a Well-Optimized sbatch Job

```bash
#!/bin/bash
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# === 1. Local NVMe setup ===
export UV_CACHE_DIR="$TMPDIR/uv-cache"

# === 2. Venv on local NVMe ===
JOB_VENV="$TMPDIR/birdclef-$SLURM_JOB_ID/.venv"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
uv pip install -e "$EXPERIMENT_DIR[peft]"
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# === 3. Copy training data to local NVMe ===
LOCAL_DATA="$TMPDIR/data/birdclef-2025"
mkdir -p "$TMPDIR/data"
cp -r ~/ps-clef2026_birdclef-0/data/birdclef-2025 "$LOCAL_DATA"

# === 4. Model cache on project storage (persistent, shared) ===
export HF_HOME="/storage/project/$PROJECT_NAME/shared/birdclef-2026/cache/huggingface"

# === 5. Train (data reads from local NVMe, model from shared cache) ===
srun python train.py --config config/my_config.yaml --data-dir "$LOCAL_DATA"
```

## Diagnosing I/O Bottlenecks

### GPU utilization check

Run a 1-2 epoch GPU check before submitting full training:

```bash
python train.py --config config/my_config.yaml --gpu-check 1
```

This produces a summary in `outputs/gpu_check/` with data time, compute time, and
memory usage. Key things to look for:

| Metric | Healthy | Bottlenecked |
|--------|---------|--------------|
| Data loading % | <5% | >20% |
| GPU utilization | >85% | <50% |
| Memory % | 20-80% | <5% (batch too small) or OOM |

### Common bottleneck patterns

**High data loading % (>20%)**
- Data on NFS → copy to `$TMPDIR`
- Too many small file reads → precompute features as tensors
- num_workers too low → increase (max 12 per GPU on PACE)

**Low GPU utilization (<50%)**
- Batch size too small → increase batch_size (check memory headroom)
- CPU transform bottleneck → move transforms to GPU or precompute
- Python overhead → reduce number of steps (larger batches)

**Low memory usage (<5%)**
- Batch size is much smaller than what the GPU can handle
- Note: `memory_allocated` is a snapshot, not peak. Check `memory_reserved` for a
  better estimate of actual peak usage. Activation memory scales linearly with batch
  size during backprop.

**OOM despite "low" memory reported**
- Peak activation memory during backprop is much higher than the average snapshot
- LoRA still backprops through all encoder layers — activation memory ≈ full fine-tuning
- Use gradient accumulation: keep batch_size small, accumulate N steps for larger
  effective batch

## Gradient Accumulation vs Batch Size

Both achieve larger effective batches, but they are NOT equivalent for performance:

```
batch_size=32, accumulate=8  → effective_batch=256, but 8× forward+backward passes
batch_size=256, accumulate=1 → effective_batch=256, but 1× forward+backward pass
```

Gradient accumulation does NOT save compute — the GPU runs the same number of
forward+backward passes per epoch. It only saves memory (smaller activations per step).

**Prefer larger batch_size with less accumulation** when memory allows. This reduces:
- DataLoader fetch overhead (fewer calls)
- Python loop overhead (fewer iterations)
- CPU-GPU synchronization points

**Use accumulation only when batch_size is limited by memory.** Find the largest
batch_size that fits, then accumulate to reach the target effective batch.
