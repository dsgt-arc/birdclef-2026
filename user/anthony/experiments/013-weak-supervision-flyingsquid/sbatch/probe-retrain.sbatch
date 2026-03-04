#!/bin/bash
#SBATCH --job-name=013-probe-retrain
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t240 -qinferno
#SBATCH --output=logs/probe-retrain-%j.out

# Retrain BirdNET and Perch probes with energy-attenuated labels.
# Energy attenuation uses bioacoustic_index per-recording normalization
# so that silent clips get weight ≈ 0 and vocalization clips get weight ≈ 1.
# Usage:
#   sbatch sbatch/probe-retrain.sbatch birdnet
#   sbatch sbatch/probe-retrain.sbatch perch

set -eu

source="${1:?Usage: sbatch probe-retrain.sbatch <birdnet|perch>}"

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"
mkdir -p logs

echo "=== Probe retraining (energy-attenuated): $source ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

source "$venv/bin/activate"
uv pip install -e "$exp_dir" --quiet

"$venv/bin/weaksup-probe" species "$source" \
    --gamma 2.0 \
    --lr 1e-3 \
    --epochs 50 \
    --batch-size 16384 \
    --energy-attenuate \
    --data-dir "$scratch_dir/data/birdclef-2025" \
    --output-dir "$scratch_dir/outputs/exp013/probes-energy"

echo ""
echo "=== Done: $(date) ==="
