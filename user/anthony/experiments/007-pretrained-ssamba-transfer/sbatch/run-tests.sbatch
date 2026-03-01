#!/bin/bash
#SBATCH --job-name=007-pytest
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000
#SBATCH -t30 -qembers
#SBATCH --output=logs/pytest-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

exp_dir=$repo_dir/user/anthony/experiments/007-pretrained-ssamba-transfer
cd $exp_dir

module load cuda/12.1.1

# Set env vars for test fixtures
export SSAMBA_CKPT=$scratch_dir/checkpoints/ssamba/ssamba_tiny_400.pth
export MEL_CACHE_DIR=$scratch_dir/cache/007-hsn-events

echo "=== Environment ==="
uv run --project $repo_dir/user/anthony python -c "
import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
try:
    import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__}')
except: print('mamba-ssm: NOT INSTALLED')
try:
    import causal_conv1d; print(f'causal-conv1d {causal_conv1d.__version__}')
except: print('causal-conv1d: NOT INSTALLED')
"

echo ""
echo "=== Running tests ==="
uv run --project $repo_dir/user/anthony --extra dev python -m pytest tests/ -v --tb=short -x 2>&1

echo ""
echo "=== Done ==="
