#!/bin/bash
#SBATCH --job-name=ssamba-verify
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t30 -qinferno
#SBATCH --output=logs/run-verification-%j.out

set -eu

# Load CUDA 12.1 to match the venv
module load cuda/12.1.1

scratch_dir=~/scratch/birdclef-2026
exp_dir=$scratch_dir/exp004-ssamba-hf
repo_dir=~/arc/birdclef-2026
exp_scripts=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/scripts
ssamba_hf_dir=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/ssamba_hf

mkdir -p $exp_dir/verification

# Step 1: Extract embeddings with original SSAMBA (GPU)
echo "=== Step 1: Extracting original SSAMBA embeddings ==="
source $exp_dir/original-venv/bin/activate
export PYTHONPATH=$exp_dir/Vim:$exp_dir/Vim/vim:$repo_dir/vendor/ssamba/src
python $exp_scripts/extract_original_embeddings.py \
    --checkpoint $scratch_dir/checkpoints/ssamba/ssamba_small_400.pth \
    --output $exp_dir/verification/original_embeddings.pt
deactivate

# Step 2: Extract embeddings with HF port (GPU for fair comparison)
echo "=== Step 2: Extracting HF port embeddings ==="
cd $ssamba_hf_dir
uv run python $exp_scripts/extract_hf_embeddings.py \
    --checkpoint $scratch_dir/checkpoints/ssamba/ssamba_small_400.pth \
    --output $exp_dir/verification/hf_embeddings.pt \
    --device cuda

# Step 3: Compare
echo "=== Step 3: Comparing embeddings ==="
uv run python $exp_scripts/verify_numerical_equivalence.py \
    --original $exp_dir/verification/original_embeddings.pt \
    --ported $exp_dir/verification/hf_embeddings.pt \
    --output $exp_dir/verification/comparison_report.json

echo "=== Results ==="
cat $exp_dir/verification/comparison_report.json
