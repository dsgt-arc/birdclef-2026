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

A typical training loop does thousands of disk reads per epoch (loading audio, images,
etc.). Where those files live determines a huge chunk of wall-clock training time.

On a cold compute node (no OS page cache), reading data from NFS can consume 20-40% of
each training step. The OS caches recently-read files in RAM, so repeated reads on the
same node are fast — but the first epoch on a fresh SLURM node always hits NFS.

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

### 2. Training data

Copy training data to `$TMPDIR` at job start. This is the single biggest speedup for
I/O-bound training:

```bash
# Copy dataset to local NVMe (takes a few minutes, saves hours)
LOCAL_DATA="$TMPDIR/data/my-dataset"
if [[ ! -d "$LOCAL_DATA" ]]; then
    mkdir -p "$TMPDIR/data"
    cp -r /path/to/nfs/dataset "$LOCAL_DATA"
fi
```

Then point your training script at `$LOCAL_DATA` instead of the NFS path.

**When to do this:**
- Dataset fits in `$TMPDIR` (~1.4 TB available)
- Training runs multiple epochs (cost of copy is amortized)
- You observe >20% data loading time in profiling

**When NOT to do this:**
- Datasets larger than available `$TMPDIR` space
- Single-epoch or few-step jobs (copy overhead > time saved)
- Data is already memory-mapped (e.g., HuggingFace `datasets` with Arrow)

### 3. Model caches (HuggingFace, torch hub)

Pre-trained model weights are downloaded once and reused. Keep these on **project storage**
(shared across jobs/users) rather than `$TMPDIR` (wiped each job) or home (tiny quota):

```bash
export HF_HOME="/storage/project/<alloc>/shared/cache/huggingface"
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
JOB_VENV="$TMPDIR/my-project-$SLURM_JOB_ID/.venv"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"
uv pip install -e "$EXPERIMENT_DIR"
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# === 3. Copy training data to local NVMe ===
LOCAL_DATA="$TMPDIR/data/my-dataset"
mkdir -p "$TMPDIR/data"
cp -r /path/to/nfs/dataset "$LOCAL_DATA"

# === 4. Model cache on project storage (persistent, shared) ===
export HF_HOME="/storage/project/$PROJECT_NAME/shared/cache/huggingface"

# === 5. Train ===
srun python train.py --data-dir "$LOCAL_DATA"
```

## Diagnosing I/O Bottlenecks

### What to look for

| Metric | Healthy | Bottlenecked |
|--------|---------|--------------|
| Data loading % of step time | <5% | >20% |
| GPU compute utilization | >85% | <50% |
| GPU memory usage | 20-80% | <5% (batch too small) or OOM |

### Common bottleneck patterns

**High data loading % (>20%)**
- Data on NFS → copy to `$TMPDIR`
- Too many small file reads → precompute features as tensors
- num_workers too low → increase (max 12 per GPU on RTX 6000 / V100 partitions)

**Low GPU utilization (<50%)**
- Batch size too small → increase batch_size (check memory headroom)
- CPU transform bottleneck → move transforms to GPU or precompute
- Python loop overhead → reduce number of steps (larger batches)

**Low memory usage (<5%)**
- Batch size is much smaller than what the GPU can handle
- Note: `memory_allocated` is a point-in-time snapshot, not peak. Check
  `memory_reserved` for a better estimate. Activation memory during backprop
  scales linearly with batch size.

**OOM despite "low" reported memory**
- Peak activation memory during backprop is much higher than the average snapshot
- Use gradient accumulation: keep batch_size small, accumulate N steps for a larger
  effective batch

## Gradient Accumulation vs Batch Size

Both achieve larger effective batches, but they are NOT equivalent for performance:

```
batch_size=32, accumulate=8  → effective_batch=256, but 8x forward+backward per epoch
batch_size=256, accumulate=1 → effective_batch=256, but 1x forward+backward per epoch
```

Gradient accumulation does NOT save compute — the GPU runs the same number of
forward+backward passes. It only saves memory (smaller activations per step).

**Prefer larger batch_size with less accumulation** when memory allows. This reduces:
- DataLoader fetch overhead (fewer calls)
- Python loop overhead (fewer iterations)
- CPU-GPU synchronization points

**Use accumulation only when batch_size is limited by memory.** Find the largest
batch_size that fits, then accumulate to reach the target effective batch.
