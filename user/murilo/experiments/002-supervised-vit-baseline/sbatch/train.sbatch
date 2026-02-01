#!/bin/bash
#SBATCH --job-name=exp002-vit-baseline
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu

# Experiment 002: Supervised ViT Baseline
# Purpose: Build robust PyTorch Lightning pipeline, stress-test GPU data loading

set -e  # Exit on error

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================================="

# Navigate to experiment directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load required modules
module load anaconda3/2024.06
module load cuda/12.4

# Activate conda environment (adjust to your environment name)
source activate birdclef-2026

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Create output directories
mkdir -p outputs/slurm
mkdir -p outputs/checkpoints
mkdir -p outputs/wandb

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Log configuration
echo "========================================="
echo "Configuration:"
cat config.yaml
echo "========================================="

# Run training
echo "Starting training..."
python train.py --config config.yaml

echo "========================================="
echo "End Time: $(date)"
echo "Job completed successfully!"
echo "========================================="
