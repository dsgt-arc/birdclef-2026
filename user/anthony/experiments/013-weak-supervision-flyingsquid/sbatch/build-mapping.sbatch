#!/bin/bash
#SBATCH --job-name=013-build-mapping
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/build-mapping-%j.out

# Build BirdNET/Perch → BirdCLEF species index mappings.
# Quick CPU job (~1 min). Downloads BirdNET labels from GitHub.
# Usage:
#   sbatch sbatch/build-mapping.sbatch

set -euo pipefail

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"
mkdir -p logs

echo "=== Building species mapping ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

source "$venv/bin/activate"
uv pip install -e "$exp_dir" --quiet

"$venv/bin/weaksup-build-mapping" \
    --taxonomy-csv "$scratch_dir/data/birdclef-2025/taxonomy.csv" \
    --perch-label-csv /tmp/tfhub_modules/50cbe9e746f7b3e38819b8f3dc1f73e682ceca90/assets/label.csv \
    --output-dir "$scratch_dir/outputs/exp013/mappings"

echo ""
echo "=== Done: $(date) ==="
