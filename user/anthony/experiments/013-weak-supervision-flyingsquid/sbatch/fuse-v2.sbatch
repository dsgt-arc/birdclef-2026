#!/bin/bash
#SBATCH --job-name=013-fuse-v2
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=16G
#SBATCH -t360 -qinferno
#SBATCH --array=0-1
#SBATCH --output=logs/fuse-v2-%A_%a.out

# Fuse labels with raw logit LFs (v2): BirdNET + Perch raw logits for birds.
# Requires species_mapping.json from build-mapping.sbatch.
# Higher memory than v1 due to loading predict parquets (~6522/10932 columns).
# Usage:
#   sbatch sbatch/fuse-v2.sbatch
#
# Submission workflow with dependencies:
#   JOB_MAP=$(sbatch sbatch/build-mapping.sbatch | awk '{print $4}')
#   JOB_FUSE=$(sbatch --dependency=afterok:$JOB_MAP sbatch/fuse-v2.sbatch | awk '{print $4}')
#   sbatch --dependency=afterok:$JOB_FUSE sbatch/evaluate.sbatch

set -euo pipefail

SPLITS=(focal soundscape)
SPLIT="${SPLITS[$SLURM_ARRAY_TASK_ID]}"

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"
mkdir -p logs

echo "=== Fusing $SPLIT (v2 with raw logit LFs) ==="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Time: $(date)"
echo ""

source "$venv/bin/activate"
uv pip install -e "$exp_dir" --quiet

"$venv/bin/weaksup-fuse" \
    --split "$SPLIT" \
    --probe-dir "$scratch_dir/outputs/exp013/probes" \
    --data-dir "$scratch_dir/data/birdclef-2025" \
    --exp010-dir "$scratch_dir/outputs/exp010/labels" \
    --output-dir "$scratch_dir/outputs/exp013/fused-v2" \
    --mapping-path "$scratch_dir/outputs/exp013/mappings/species_mapping.json" \
    --n-workers 8

echo ""
echo "=== Done $SPLIT: $(date) ==="
