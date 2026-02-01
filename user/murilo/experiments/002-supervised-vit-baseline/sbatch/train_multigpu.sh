#!/bin/bash
#SBATCH --job-name=exp002-vit-multigpu
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:2              # Request 2 GPUs (or 4 for even faster)
#SBATCH --cpus-per-task=32        # 16 per GPU
#SBATCH --mem=128G                # 64GB per GPU
#SBATCH --time=24:00:00           # Can now fit in 24 hours with 2 GPUs
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu

# Multi-GPU Training for Experiment 002
# Using PyTorch Lightning's built-in DDP support

set -e

echo "========================================="
echo "Multi-GPU Training"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================================="

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load modules
module load anaconda3/2024.06
module load cuda/12.4

# Activate environment
source activate birdclef-2026

# Verify GPUs
echo "GPU Information:"
nvidia-smi

# Create output directories
mkdir -p outputs/slurm outputs/checkpoints outputs/wandb

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# CRITICAL: Set DDP environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

echo "========================================="
echo "Configuration:"
cat config.yaml
echo "========================================="

# Run training with multi-GPU support
# PyTorch Lightning handles all DDP setup automatically!
echo "Starting multi-GPU training..."
python train.py \
    --config config.yaml \
    --devices 2 \
    --strategy ddp

echo "========================================="
echo "End Time: $(date)"
echo "Training speedup: ~2x with 2 GPUs"
echo "========================================="
