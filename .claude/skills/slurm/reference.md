# SLURM Reference Guide

Detailed patterns, examples, and troubleshooting for PACE cluster sbatch scripts.

## Detailed Patterns

### 1. Local NVMe Staging

Use `${TMPDIR}` for local NVMe (much faster than network storage):

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

### 2. GPU Monitoring (Background)

Start GPU monitoring in background with proper cleanup:

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

### 3. Array Jobs with Idempotency

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
- Max 1000 tasks per array

### 4. Error Reporting and Logging

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

### 5. Inline Python Execution

```bash
uv run python - "$VAR1" "$VAR2" << 'EOF'
import sys
var1 = sys.argv[1]
var2 = int(sys.argv[2])

# Python code here...
EOF
```

**Key points:**
- Use heredoc (`<< 'EOF'`) for inline Python
- Single quotes prevent shell expansion
- Arguments passed as sys.argv

## DataLoader Configuration (PyTorch)

Match workers to CPU allocation:

```python
# Match workers to CPUs
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

## Complete Example

Here's a complete training script that demonstrates all best practices:

```bash
#!/bin/bash
#SBATCH --job-name=train-model
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
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

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
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

## Advanced Topics

### Multi-GPU Training

For jobs requiring multiple GPUs:

```bash
#SBATCH --gres=gpu:2
#SBATCH -p gpu-a100

# In Python code, use:
# torch.distributed.init_process_group(backend='nccl')
```

### CPU-Only Jobs

For preprocessing, inference, or CPU-heavy tasks:

```bash
#SBATCH -N1 -n1 --cpus-per-task=16 --mem-per-cpu=8G
#SBATCH -p cpu
#SBATCH -t1440 -qinferno
```

### Job Dependency

Chain jobs together:

```bash
# Submit first job
JOB1=$(sbatch job1.sbatch | awk '{print $4}')

# Submit second job that depends on first
sbatch --dependency=afterok:$JOB1 job2.sbatch
```

### Canceling Jobs

```bash
# Cancel specific job
scancel JOB_ID

# Cancel all jobs
scancel -u $USER

# Cancel job array
scancel JOB_ID_*
```

## Common Environment Variables

| Variable | Value | Use |
|----------|-------|-----|
| `${SLURM_JOB_ID}` | Job ID (integer) | Log filenames, monitoring |
| `${SLURM_ARRAY_TASK_ID}` | Array task ID (0-indexed) | Distributing work in array jobs |
| `${SLURM_SUBMIT_DIR}` | Submission directory | Changing to correct working directory |
| `${TMPDIR}` | Node-local NVMe path | Fast temporary storage |
| `${SLURM_NTASKS}` | Number of tasks | Parallel execution count |
| `${SLURM_CPUS_PER_TASK}` | CPUs per task | Worker/thread configuration |

## Monitoring and Debugging

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View detailed job info
scontrol show job JOB_ID

# View job history (completed jobs)
sacct -u $USER --starttime=2024-01-01
```

### View Job Output

```bash
# Follow job output (tail -f)
tail -f logs/job-${SLURM_JOB_ID}.out

# Show GPU monitoring output
cat logs/gpu-${SLURM_JOB_ID}.csv
```

### Resource Utilization

```bash
# Check cluster load
pace-check-queue

# Check your account balance
pace-mybalance

# Check storage quotas
pace-quota
```

## Performance Tips

1. **Use Local NVMe for I/O Heavy Tasks**: Copy data to `${TMPDIR}` for ~10x speedup
2. **Match Worker Count**: Set `num_workers` and xargs parallelism to `--cpus-per-task`
3. **Pin Memory**: Use `pin_memory=True` for GPU training
4. **Batch Operations**: Group multiple small operations into single job
5. **Monitor GPU**: Use background monitoring to identify bottlenecks

## NFS Lock Issues

If you encounter "NFS lock" errors:

```bash
# May occur with `.claude/commands/slurm/` directory
# Workaround: Use different storage location if needed
# Or wait for 10-15 minutes for lock to clear
```

## Additional Resources

- PACE Documentation: https://docs.pace.gatech.edu/
- SLURM Documentation: https://slurm.schedmd.com/
- Georgia Tech PACE Wiki: https://gtwiki.gatech.edu/wiki/index.php/PACE_Cluster_Introduction
