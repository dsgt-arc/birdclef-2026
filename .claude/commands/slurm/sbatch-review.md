---
name: SLURM SBATCH Review
description: Write and review SLURM sbatch scripts following HPC best practices
category: HPC
tags: [slurm, sbatch, hpc, gpu]
---

**Purpose**

Create or review SLURM sbatch scripts that follow idiomatic patterns for reproducible, robust HPC job execution on PACE clusters.

**Guardrails**

- Always use `set -eu` for strict error handling
- Resource requests must match actual usage (CPUs, memory, GPUs)
- Jobs should be idempotent where possible (skip already-completed work)
- Include comprehensive logging and monitoring
- Handle cleanup properly (temp files, background processes)

**Idiomatic SBATCH Patterns**

## 1. Header and Metadata

```bash
#!/bin/bash
#SBATCH --job-name=descriptive-name
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH --output=logs/job-name-%j.out
# Optional for array jobs:
#SBATCH --array=0-14

# Usage comments
# Usage: sbatch --export=VAR1=value1,VAR2=value2 script.sbatch
#
# Description of what this job does
# - Key features
# - Expected runtime
# - Resource requirements
```

**Resource Guidelines:**
- Match `--cpus-per-task` to actual parallelism (DataLoader workers, copy threads)
- Use `--mem-per-cpu` (not `--mem`) for better scheduling
- Time limits: `-t` in minutes, common values: 120 (2h), 360 (6h), 1440 (24h)
- QOS: `-qembers` for GPU jobs, `-qinferno` for CPU-only

## 2. Error Handling and Setup

```bash
set -eu  # Exit on error, undefined variables

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Configuration with defaults
STUDENT=${STUDENT:-convnext-small}
MAX_EPOCHS=${MAX_EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}

echo "=============================================="
echo "Job: descriptive-name"
echo "=============================================="
echo "Config:"
echo "  Student: ${STUDENT}"
echo "  Max epochs: ${MAX_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
hostname
date
nvidia-smi  # For GPU jobs
```

## 3. Environment Setup

```bash
scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir
```

**Key points:**
- Use `scratch_dir` for large data, outputs, caches
- Use `repo_dir` for code
- Export environment variables before running Python

## 4. Local NVMe Staging

```bash
# Use TMPDIR for local NVMe (much faster than network storage)
source_data=$scratch_dir/data/large_dataset
local_cache=${TMPDIR}/dataset_cache

echo "Copying data to local NVMe..."
mkdir -p $local_cache

# Parallel copy for large files (match CPU count)
time (ls $source_data/*.tar | xargs -P 8 -I {} cp {} $local_cache/)

echo "Copy complete:"
ls $local_cache | wc -l
du -sh $local_cache
```

**Best practices:**
- Always use `${TMPDIR}` for temporary files (auto-cleaned after job)
- Copy large data to local NVMe before processing
- Use parallel operations (`xargs -P`) to match CPU count
- Show copy progress (time, file count, disk usage)

## 5. GPU Monitoring (Background)

```bash
# Start GPU monitoring in background
$repo_dir/scripts/monitor-gpu.sh 5 logs/gpu-${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    if [ ! -z "${GPU_MONITOR_PID:-}" ]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Now run training...
```

**Key points:**
- Run monitoring as background process (`&`)
- Save PID for cleanup
- Use trap to ensure cleanup on exit (success or failure)
- Log to job-specific file with `${SLURM_JOB_ID}`

## 6. Python Execution

**Inline Python (for simple scripts):**
```bash
uv run python - "$VAR1" "$VAR2" << 'EOF'
import sys
var1 = sys.argv[1]
var2 = int(sys.argv[2])

# Python code here...
EOF
```

**External script (for complex jobs):**
```bash
uv run python scripts/train.py \
    --model "$STUDENT" \
    --epochs $MAX_EPOCHS \
    --batch-size $BATCH_SIZE \
    --output-dir $scratch_dir/outputs/exp003
```

**Key points:**
- Use `uv run` for dependency management
- Pass variables as arguments, not environment variables
- Use quotes for string variables that might have spaces
- Use heredoc (`<< 'EOF'`) for inline Python (single quotes prevent shell expansion)

## 7. Idempotency for Array Jobs

```bash
#SBATCH --array=0-14

# Calculate work assignment
SHARDS_PER_TASK=10
START_SHARD=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
END_SHARD=$((START_SHARD + SHARDS_PER_TASK - 1))

for i in $(seq $START_SHARD $END_SHARD); do
    output_file="outputs/shard-$(printf "%05d" $i).safetensors"

    # Skip if already processed
    if [ -f "$output_file" ]; then
        echo "Skipping shard $i (already processed)"
        continue
    fi

    # Process shard...
done
```

**Key points:**
- Use `SLURM_ARRAY_TASK_ID` for work distribution
- Check for existing outputs before processing
- Use `printf` for zero-padded filenames
- Log skipped vs processed items

## 8. Error Reporting and Logging

```bash
echo ""
echo "Job complete!"
date

# Report key metrics
echo ""
echo "Summary:"
echo "  Items processed: ${processed_count}"
echo "  Runtime: ${runtime_seconds}s"
echo "  Output size: $(du -sh $output_dir | cut -f1)"
echo "  GPU utilization: see logs/gpu-${SLURM_JOB_ID}.csv"

# Check for errors
if [ $processed_count -eq 0 ]; then
    echo "ERROR: No items processed!"
    exit 1
fi
```

## 9. DataLoader Configuration (PyTorch)

```bash
# In Python code, match workers to CPUs
num_workers = 8  # Match --cpus-per-task=8
prefetch_factor = 2  # Prefetch 2 batches per worker

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # For GPU training
    prefetch_factor=prefetch_factor,
)
```

**Key points:**
- `num_workers` should match `--cpus-per-task`
- Use `pin_memory=True` for GPU jobs
- `prefetch_factor=2` is a good default for GPU saturation

## 10. Common Pitfalls to Avoid

❌ **Bad:**
```bash
num_workers=4  # Hardcoded, wastes available CPUs
cd ~/arc/birdclef-2026  # Hardcoded path, not portable
cp $source_data/* $local_cache/  # Slow, sequential copy
python train.py  # No dependency management
```

✅ **Good:**
```bash
num_workers=8  # Match --cpus-per-task=8
cd $repo_dir  # Use variable
ls $source_data/* | xargs -P 8 -I {} cp {} $local_cache/  # Parallel
uv run python train.py  # Managed environment
```

**Review Checklist**

When reviewing an sbatch script, check:

- [ ] `set -eu` at the top for error handling
- [ ] Resource requests match actual usage (CPUs, memory, GPU)
- [ ] Clear usage comment with examples
- [ ] Echo job config at start (hostname, date, nvidia-smi)
- [ ] Use `${SLURM_SUBMIT_DIR}` or repo_dir variable
- [ ] Environment variables exported (UV_PROJECT_ENVIRONMENT, HF_HOME)
- [ ] Local NVMe staging for large data (`${TMPDIR}`)
- [ ] Parallel operations match CPU count
- [ ] GPU monitoring as background process with cleanup trap
- [ ] Python execution uses `uv run`
- [ ] Idempotent operations (skip already-completed work)
- [ ] Error checking and summary at the end
- [ ] Logs saved to identifiable files (`logs/job-${SLURM_JOB_ID}.out`)

**PACE-Specific Notes**

- **TMPDIR**: Automatically set to local NVMe on each node
- **Storage hierarchy**: NVMe (fastest) > Scratch (slow, large) > Home (slowest, smallest)
- **QOS options**: `embers` (GPU, 6hr), `inferno` (CPU, 24hr)
- **Array job limit**: Max 1000 tasks per array
- **Job dependencies**: Use `--dependency=afterok:jobid` for pipelines

**Example: Complete Training Job**

```bash
#!/bin/bash
#SBATCH --job-name=train-model
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH --output=logs/train-%j.out

# Usage: sbatch --export=MODEL=resnet50,EPOCHS=20 train.sbatch

set -eu
cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

MODEL=${MODEL:-resnet50}
EPOCHS=${EPOCHS:-10}

echo "=============================="
echo "Training: $MODEL"
echo "=============================="
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/project)
repo_dir=$(realpath ~/arc/project)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
cd $repo_dir

# Start GPU monitoring
$repo_dir/scripts/monitor-gpu.sh 5 logs/gpu-${SLURM_JOB_ID}.csv &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null || true" EXIT

# Run training
uv run python scripts/train.py \
    --model $MODEL \
    --epochs $EPOCHS \
    --num-workers 8 \
    --output-dir $scratch_dir/outputs

echo "Training complete!"
date
```
