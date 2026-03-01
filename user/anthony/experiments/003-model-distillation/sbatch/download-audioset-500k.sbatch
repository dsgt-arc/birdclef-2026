#!/bin/bash
#SBATCH --job-name=dl-audioset-500k
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t360 -qembers
#SBATCH --output=logs/download-audioset-500k-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Downloading AudioSet-500k from HuggingFace"
hostname
date

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

# Environment setup
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

# Download AudioSet-500k (~50GB)
echo ""
echo "Downloading AudioSet-500k (main)..."
echo "Expected size: ~50GB, ~170 shards"
uv run huggingface-cli download confit/audioset-16khz-wds \
  --include "500k/train/*.tar" \
  --repo-type=dataset \
  --local-dir $scratch_dir/data/audioset-wds

echo ""
echo "Download complete!"
echo ""
echo "Directory contents:"
ls -la $scratch_dir/data/audioset-wds/500k/train/ | head -20
echo "..."
ls -la $scratch_dir/data/audioset-wds/500k/train/ | tail -5
echo ""
echo "Total shards:"
ls $scratch_dir/data/audioset-wds/500k/train/*.tar | wc -l
echo ""
echo "Total size:"
du -sh $scratch_dir/data/audioset-wds/500k/

date
