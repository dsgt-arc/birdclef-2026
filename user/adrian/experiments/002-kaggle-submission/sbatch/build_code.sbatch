#!/bin/bash
#SBATCH --job-name=kaggle-build-code
#SBATCH --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=2G
#SBATCH -t15 -qembers
#SBATCH --output=logs/build-code-%j.out
set -eu

# Build wheel files for Kaggle submission

# Source common functions (absolute path since SLURM copies scripts)
source ~/arc/birdclef-2026/user/adrian/experiments/002-kaggle-submission/sbatch/common.sh
print_job_info "Build Code Wheels"
setup_environment

ensure_dist_dirs
clean_dist_dir "code"

CODE_DIR="${DIST_DIR}/code"

echo "Building wheel files..."
cd "${REPO_DIR}"

# Build user/adrian wheel
if [[ -f "user/adrian/pyproject.toml" ]]; then
    echo ""
    echo "Building user/adrian wheel..."
    cd user/adrian
    uv build --wheel --out-dir "${CODE_DIR}"
    cd "${REPO_DIR}"
fi

# Build user/anthony wheel
if [[ -f "user/anthony/pyproject.toml" ]]; then
    echo ""
    echo "Building user/anthony wheel..."
    cd user/anthony
    uv build --wheel --out-dir "${CODE_DIR}"
    cd "${REPO_DIR}"
fi

echo ""
echo "Wheel files saved to: ${CODE_DIR}"
ls -la "${CODE_DIR}"

echo ""
echo "Done!"
date

