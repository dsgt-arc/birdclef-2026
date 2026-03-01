#!/bin/bash
#SBATCH --job-name=ssamba-onnx-bench
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem=32G
#SBATCH -t60 -qinferno
#SBATCH --array=0-10
#SBATCH --output=logs/onnx-bench-%A_%a.out

# CPU-only ONNX benchmark using mamba-onnx custom ops.
# No GPU needed — benchmarks ONNX Runtime inference on CPU.
#
# Usage:
#   cd user/anthony/experiments/005-ssamba-token-context
#   mkdir -p logs
#
#   # Test single shard first:
#   sbatch --array=0 sbatch/run-onnx-benchmark.sbatch
#
#   # Then full array:
#   sbatch sbatch/run-onnx-benchmark.sbatch
#
#   # Collect results:
#   $PYTHON scripts/benchmark_onnx_token_shapes.py --collect --results-dir onnx-results/

set -eu

echo "=== SSAMBA ONNX Benchmark — Config $SLURM_ARRAY_TASK_ID ==="
echo "Array Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
date

repo_dir=~/arc/birdclef-2026
project_dir=$repo_dir/user/anthony
exp_dir=$repo_dir/user/anthony/experiments/005-ssamba-token-context
PYTHON=$project_dir/.venv/bin/python

# Use $venv/bin/python directly — NOT uv run — because parallel array tasks
# sharing the same venv will race to sync and corrupt each other's packages.

# Verify mamba-onnx is installed
echo ""
$PYTHON -c "
import mamba_onnx
backends = mamba_onnx.available_backends()
assert backends, 'mamba-onnx has no backends — reinstall from wheel'
print(f'mamba-onnx backends: {backends}')
"

echo ""
echo "=== Running ONNX benchmark ==="
mkdir -p $exp_dir/onnx-results $exp_dir/logs

$PYTHON $exp_dir/scripts/benchmark_onnx_token_shapes.py \
    --config-index $SLURM_ARRAY_TASK_ID \
    --results-dir $exp_dir/onnx-results \
    --ort-threads $SLURM_CPUS_PER_TASK

echo ""
echo "=== Task $SLURM_ARRAY_TASK_ID Complete ==="
date
