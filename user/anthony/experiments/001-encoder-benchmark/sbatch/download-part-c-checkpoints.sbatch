#!/bin/bash
#SBATCH --job-name=dl-ckpts-c
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t120 -qinferno
#SBATCH --output=logs/download-part-c-%j.out
set -eu

echo "Downloading Part C checkpoints (SSAST, SSAMBA)"
echo "Job ID: $SLURM_JOB_ID"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026
checkpoint_dir=$scratch_dir/checkpoints

# Use scratch venv
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir

# Create checkpoint directories
mkdir -p $checkpoint_dir/ssast
mkdir -p $checkpoint_dir/ssamba

echo ""
echo "=== Downloading SSAST checkpoints ==="
# SSAST checkpoints from Dropbox
# Base model with 400 masked patches (patch-based)
if [ ! -f "$checkpoint_dir/ssast/SSAST-Base-Patch-400.pth" ]; then
    echo "Downloading SSAST-Base-Patch-400.pth..."
    wget -q --show-progress -O "$checkpoint_dir/ssast/SSAST-Base-Patch-400.pth" \
        "https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1"
else
    echo "SSAST-Base-Patch-400.pth already exists, skipping"
fi

echo ""
echo "=== Downloading SSAMBA checkpoints from HuggingFace ==="
# SSAMBA checkpoints from HuggingFace: attentionisallyouneed369/ssamba
HF_BASE="https://huggingface.co/attentionisallyouneed369/ssamba/resolve/main"

# ssamba_tiny_400.pth (29.3 MB)
if [ ! -f "$checkpoint_dir/ssamba/ssamba_tiny_400.pth" ]; then
    echo "Downloading ssamba_tiny_400.pth (29.3 MB)..."
    wget -q --show-progress -O "$checkpoint_dir/ssamba/ssamba_tiny_400.pth" \
        "$HF_BASE/ssamba_tiny_400.pth"
else
    echo "ssamba_tiny_400.pth already exists, skipping"
fi

# ssamba_small_400.pth (105 MB)
if [ ! -f "$checkpoint_dir/ssamba/ssamba_small_400.pth" ]; then
    echo "Downloading ssamba_small_400.pth (105 MB)..."
    wget -q --show-progress -O "$checkpoint_dir/ssamba/ssamba_small_400.pth" \
        "$HF_BASE/ssamba_small_400.pth"
else
    echo "ssamba_small_400.pth already exists, skipping"
fi

# ssamba_base_400.pth (396 MB)
if [ ! -f "$checkpoint_dir/ssamba/ssamba_base_400.pth" ]; then
    echo "Downloading ssamba_base_400.pth (396 MB)..."
    wget -q --show-progress -O "$checkpoint_dir/ssamba/ssamba_base_400.pth" \
        "$HF_BASE/ssamba_base_400.pth"
else
    echo "ssamba_base_400.pth already exists, skipping"
fi

echo ""
echo "=== Setting up Vim (Vision Mamba) repository ==="
# Clone Vim repo into vendor/ssamba if not exists
vim_dir=$repo_dir/vendor/ssamba/Vim
if [ ! -d "$vim_dir" ]; then
    echo "Cloning Vision Mamba repository..."
    cd $repo_dir/vendor/ssamba
    git clone https://github.com/hustvl/Vim.git
else
    echo "Vim directory already exists, skipping clone"
fi

echo ""
echo "=== Checkpoint download summary ==="
echo "SSAST checkpoints:"
ls -lh $checkpoint_dir/ssast/ 2>/dev/null || echo "  (none)"
echo ""
echo "SSAMBA checkpoints:"
ls -lh $checkpoint_dir/ssamba/ 2>/dev/null || echo "  (none)"
echo ""
echo "Vim repo:"
ls -la $vim_dir 2>/dev/null | head -5 || echo "  (not found)"

echo ""
echo "Done!"
date
