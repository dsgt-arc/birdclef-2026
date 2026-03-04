#!/bin/bash
#SBATCH --job-name=013-visualize
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/visualize-%j.out

# Generate soft label spectrogram visualizations for focal and soundscape.
# CPU-only, needs audio I/O (librosa, soundfile).
# Usage:
#   sbatch sbatch/visualize.sbatch

set -euo pipefail

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"
mkdir -p logs

echo "=== Soft Label Visualization ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo ""

source "$venv/bin/activate"
uv pip install -e "$exp_dir[viz]" --quiet

fused_dir="$scratch_dir/outputs/exp013/fused"
data_dir="$scratch_dir/data/birdclef-2025"
eval_dir="$scratch_dir/outputs/exp013/eval"

echo "=== Visualizing focal ==="
"$venv/bin/weaksup-visualize" \
    --split focal \
    --fused-dir "$fused_dir" \
    --data-dir "$data_dir" \
    --output "$eval_dir/focal/soft_label_spectrograms.png" \
    --n-rows 5 \
    --n-candidates 20 \
    --threshold 0.3 \
    --seed 42

echo ""

echo "=== Visualizing soundscape ==="
"$venv/bin/weaksup-visualize" \
    --split soundscape \
    --fused-dir "$fused_dir" \
    --data-dir "$data_dir" \
    --output "$eval_dir/soundscape/soft_label_spectrograms.png" \
    --n-rows 5 \
    --n-candidates 20 \
    --threshold 0.3 \
    --seed 42

echo ""
echo "=== Done: $(date) ==="
ls -la "$eval_dir"/focal/soft_label_spectrograms.png "$eval_dir"/soundscape/soft_label_spectrograms.png
