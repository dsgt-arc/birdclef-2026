#!/bin/bash
#SBATCH --job-name=dl-audioset
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t240 -qembers
#SBATCH --output=logs/download-audioset-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Downloading AudioSet from HuggingFace"
hostname
date

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Ensure huggingface-cli is available
cd $repo_dir

# Download AudioSet-20k (pilot version, ~2GB)
echo ""
echo "Downloading AudioSet-20k (pilot)..."
uv run huggingface-cli download confit/audioset-16khz-wds \
  --include "20k/train/*.tar" "20k/test/*.tar" \
  --repo-type=dataset \
  --local-dir $scratch_dir/data/audioset-wds

echo ""
echo "Download complete!"
ls -la $scratch_dir/data/audioset-wds/20k/

date
