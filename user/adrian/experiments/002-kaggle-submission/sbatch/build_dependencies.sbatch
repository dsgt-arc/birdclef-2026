#!/bin/bash
#SBATCH --job-name=kaggle-build-deps
#SBATCH --account=paceship-dsgt_clef2025
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60 -qembers
#SBATCH --output=logs/build-deps-%j.out
set -eu

# Build Python wheel dependencies for offline Kaggle install
# NOTE: Requires network access - may need to run on login node

# Source common functions (absolute path since SLURM copies scripts)
source ~/arc/birdclef-2026/user/adrian/experiments/002-kaggle-submission/sbatch/common.sh
print_job_info "Build Dependencies (Wheels)"
setup_environment

# Check network first
if ! check_network; then
    echo ""
    echo "ERROR: No network access from this node."
    echo "Run this script directly on login node instead:"
    echo "  bash sbatch/build_dependencies.sbatch"
    exit 1
fi

ensure_dist_dirs
clean_dist_dir "dependencies"

DEPS_DIR="${DIST_DIR}/dependencies"

echo "Step 1: Exporting uv lock to requirements.txt..."
cd "${REPO_DIR}"

# Export requirements from user/adrian pyproject.toml
uv pip compile user/adrian/pyproject.toml -o "${DEPS_DIR}/requirements.txt"

echo ""
echo "Step 2: Downloading wheels for Kaggle platform..."
echo "  Platform: manylinux2014_x86_64"
echo "  Python: 3.10"

# Download wheels for Kaggle's Linux environment
pip download \
    -r "${DEPS_DIR}/requirements.txt" \
    --dest "${DEPS_DIR}" \
    --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version 3.10 \
    --no-deps \
    2>&1 | tee "${DEPS_DIR}/download.log"

echo ""
echo "Step 3: Verifying downloads..."
wheel_count=$(find "${DEPS_DIR}" -name "*.whl" | wc -l)
echo "Downloaded ${wheel_count} wheel files"

echo ""
echo "Wheels saved to: ${DEPS_DIR}"
ls -la "${DEPS_DIR}"

echo ""
echo "Done!"
date
