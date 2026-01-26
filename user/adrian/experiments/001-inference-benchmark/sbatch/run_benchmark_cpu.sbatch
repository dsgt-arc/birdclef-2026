#!/bin/bash
#SBATCH --job-name=infer-bench-cpu
#SBATCH --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t120 -qembers
#SBATCH --output=logs/benchmark-cpu-%j.out
set -eu

echo "Running inference benchmark (CPU only)"
echo "Job ID: $SLURM_JOB_ID"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

source $scratch_dir/.venv/bin/activate
# uv sync --active

cd $repo_dir
# uv pip install -e user/adrian
# uv pip install -e user/anthony

# Use scratch venv
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

# Set HuggingFace cache to scratch
export HF_HOME=$scratch_dir/.cache/huggingface
export TORCH_HOME=$scratch_dir/.cache/torch

# Disable CUDA to force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Run benchmark (CPU only)
# Models: BirdNET, Perch, beats
uv run python -m birdclef_adrian.benchmark_inference \
    # --models ${MODELS:-bmz beats} \
    --models ${MODELS:-beats BirdNET BirdSetConvNeXT BirdSetEfficientNetB1 HawkEars Perch Perch2 RanaSierraeCNN YAMNet} \
    --duration ${DURATION:-60.0} \
    --num-runs ${NUM_RUNS:-10} \
    --warmup-runs ${WARMUP_RUNS:-1} \
    --checkpoint-dir $scratch_dir/checkpoints \
    --output-dir $scratch_dir/outputs/001-inference-benchmark-cpu \
    --device cpu

echo ""
echo "Done!"
date
