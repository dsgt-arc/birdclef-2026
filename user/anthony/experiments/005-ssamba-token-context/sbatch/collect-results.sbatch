#!/bin/bash
#SBATCH --job-name=ssamba-token-collect
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=2 --mem=4G
#SBATCH -t5 -qinferno
#SBATCH --output=logs/token-collect-%j.out

# Submit this AFTER the array job completes:
#   ARRAY_JOB_ID=$(sbatch --parsable sbatch/run-benchmark.sbatch)
#   sbatch --dependency=afterany:$ARRAY_JOB_ID sbatch/collect-results.sbatch

set -eu

echo "=== Collecting Benchmark Results ==="
date

repo_dir=~/arc/birdclef-2026
project_dir=$repo_dir/user/anthony
exp_dir=$repo_dir/user/anthony/experiments/005-ssamba-token-context
PYTHON=$project_dir/.venv/bin/python

echo ""
echo "Result files found:"
ls -la $exp_dir/results/config-*.json 2>/dev/null || echo "  (none)"

echo ""
$PYTHON $exp_dir/scripts/benchmark_token_shapes.py \
    --collect \
    --results-dir $exp_dir/results \
    --output $exp_dir/results.json

echo ""
echo "=== Collection Complete ==="
date
