#!/bin/bash
#SBATCH --job-name=kaggle-net-test
#SBATCH --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=1 --mem-per-cpu=1G
#SBATCH -t5 -qembers
#SBATCH --output=logs/network-test-%j.out
set -eu

# Test network connectivity from compute node
# This helps determine if pip download can run on compute nodes

# Source common functions (absolute path since SLURM copies scripts)
source ~/arc/birdclef-2026/user/adrian/experiments/002-kaggle-submission/sbatch/common.sh
print_job_info "Network Connectivity Test"

echo "Testing outbound network connectivity from compute node..."
echo ""

# Test PyPI
echo -n "PyPI (pypi.org): "
if curl -s --connect-timeout 10 https://pypi.org > /dev/null 2>&1; then
    echo "✓ OK"
    PYPI_OK=1
else
    echo "✗ FAILED"
    PYPI_OK=0
fi

# Test Kaggle
echo -n "Kaggle (kaggle.com): "
if curl -s --connect-timeout 10 https://kaggle.com > /dev/null 2>&1; then
    echo "✓ OK"
    KAGGLE_OK=1
else
    echo "✗ FAILED"
    KAGGLE_OK=0
fi

# Test HuggingFace
echo -n "HuggingFace (huggingface.co): "
if curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "✓ OK"
    HF_OK=1
else
    echo "✗ FAILED"
    HF_OK=0
fi

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="

if [[ $PYPI_OK -eq 1 ]]; then
    echo "✓ Compute nodes CAN download dependencies via pip"
    echo "  You can use: sbatch sbatch/build_dependencies.sbatch"
else
    echo "✗ Compute nodes CANNOT access PyPI"
    echo "  Run dependency builds on LOGIN NODE instead:"
    echo "  bash sbatch/build_dependencies.sbatch"
fi

if [[ $KAGGLE_OK -eq 1 ]]; then
    echo "✓ Compute nodes CAN upload to Kaggle"
else
    echo "✗ Kaggle uploads must be done from LOGIN NODE"
fi

echo ""
echo "Done!"
date
