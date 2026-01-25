#!/bin/bash
#SBATCH --job-name=ssamba-setup
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t60 -qinferno
#SBATCH --output=logs/setup-original-ssamba-%j.out

set -eu

# Load CUDA 12.1 to match PyTorch 2.2.2+cu121 (GCC 12 compatible)
module load cuda/12.1.1
echo "CUDA version: $(nvcc --version | grep release)"

scratch_dir=~/scratch/birdclef-2026
exp_dir=$scratch_dir/exp004-ssamba-hf
venv_dir=$exp_dir/original-venv
vim_dir=$exp_dir/Vim
repo_dir=~/arc/birdclef-2026
exp_repo_dir=$repo_dir/user/anthony/experiments/004-ssamba-hf-port
# Use relaxed requirements (original has conflicting version pins)
requirements_file=$exp_repo_dir/requirements-ssamba.txt

mkdir -p $exp_dir

# Create isolated venv with Python 3.11 (torch 2.2.2 compatible)
# Remove existing venv if corrupted
if [ -d "$venv_dir" ]; then
    echo "=== Removing existing venv ==="
    rm -rf $venv_dir
fi
echo "=== Creating isolated venv ==="
uv venv --python 3.11 $venv_dir

source $venv_dir/bin/activate

# Compile and install all dependencies from requirements.txt
echo "=== Compiling dependencies from vendor/ssamba/requirements.txt ==="
echo "Requirements file: $requirements_file"
cat $requirements_file

# Use uv pip compile to resolve transitive dependencies
# Using CUDA 12.1 (GCC 12 compatible)
compiled_requirements=$exp_dir/requirements-compiled.txt
uv pip compile $requirements_file -o $compiled_requirements \
    --extra-index-url https://download.pytorch.org/whl/cu121

echo "=== Compiled requirements ==="
cat $compiled_requirements

# Clean problematic cache entries
echo "=== Cleaning cache ==="
uv cache clean scikit-learn || true

# Install build tools first (needed for mamba_ssm compilation)
echo "=== Installing build dependencies ==="
uv pip install setuptools wheel packaging ninja

# Install from compiled requirements (CUDA 12.1)
echo "=== Installing dependencies ==="
uv pip install --no-build-isolation -r $compiled_requirements \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Clone Vision Mamba
if [ ! -d "$vim_dir" ]; then
    echo "=== Cloning Vision Mamba ==="
    git clone https://github.com/hustvl/Vim.git $vim_dir
fi

# Install causal_conv1d from Vim (matching CUDA/PyTorch version)
# Use --no-deps to prevent dependency resolution from overwriting torch
echo "=== Installing causal_conv1d from Vim ==="
uv cache clean causal-conv1d || true
uv pip install --no-build-isolation --no-deps --reinstall $vim_dir/causal-conv1d

# Install custom mamba_ssm from Vim (with bidirectional support)
# Force building from source to get bimamba_inner_fn (not in pre-built wheels)
# Use --no-deps to prevent dependency resolution from overwriting torch
echo "=== Installing custom mamba_ssm from Vim (with bidirectional support) ==="
uv cache clean mamba-ssm || true
MAMBA_FORCE_BUILD=TRUE uv pip install --no-build-isolation --no-deps --reinstall $vim_dir/mamba-1p1p1

# Verify installation
echo "=== Verifying installation ==="
python -c "import mamba_ssm; print('mamba_ssm OK:', mamba_ssm.__version__)"
python -c "from mamba_ssm.ops.selective_scan_interface import bimamba_inner_fn; print('bimamba_inner_fn OK:', bimamba_inner_fn)"
python -c "import causal_conv1d; print('causal_conv1d OK')"
python -c "import torch; print('PyTorch OK:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import numpy; print('NumPy OK:', numpy.__version__)"
python -c "
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
    print('RMSNorm from triton OK:', RMSNorm)
except ImportError as e:
    print('RMSNorm import failed:', e)
"

# Test importing VisionMamba
echo "=== Testing VisionMamba import ==="
PYTHONPATH=$vim_dir:$vim_dir/vim python -c "
import sys
sys.path.insert(0, '$vim_dir/vim')
from models_mamba import VisionMamba
print('VisionMamba import OK')
"

echo "=== Setup complete ==="
echo "Venv location: $venv_dir"
echo "Vim location: $vim_dir"
