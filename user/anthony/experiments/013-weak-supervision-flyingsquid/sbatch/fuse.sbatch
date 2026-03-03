#!/bin/bash
#SBATCH --job-name=013-fuse
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t360 -qinferno
#SBATCH --output=logs/fuse-%j.out

# CPU-only: Train indices probe, build label matrices for focal + soundscape,
# run FlyingSquid per species, and evaluate results.
# Usage:
#   sbatch sbatch/fuse.sbatch

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir="$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
venv="$scratch_dir/.venv"

cd "$exp_dir"

echo "=== Phase 2: Label Fusion ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Time: $(date)"
echo ""

# Install/update the package
"$venv/bin/uv" pip install -e "$exp_dir" --quiet

probe_dir="$scratch_dir/outputs/exp013/probes"
data_dir="$scratch_dir/data/birdclef-2025"
exp010_dir="$scratch_dir/outputs/exp010/labels"
output_dir="$scratch_dir/outputs/exp013/fused"
eval_dir="$scratch_dir/outputs/exp013/eval"

# 1. Train indices probe (fast, ~minutes on CPU)
echo "=== Training indices probe ==="
"$venv/bin/weaksup-probe" species indices \
    --gamma 2.0 \
    --epochs 50 \
    --batch-size 16384 \
    --data-dir "$data_dir" \
    --output-dir "$probe_dir"

echo ""

# 2. Fuse focal (4 species-specific LFs: BirdNET, Perch, indices, recording label)
echo "=== Fusing focal ==="
"$venv/bin/weaksup-fuse" \
    --split focal \
    --probe-dir "$probe_dir" \
    --data-dir "$data_dir" \
    --exp010-dir "$exp010_dir" \
    --output-dir "$output_dir" \
    --n-workers 4

echo ""

# 3. Fuse soundscape (3 species-specific LFs: BirdNET, Perch, indices)
echo "=== Fusing soundscape ==="
"$venv/bin/weaksup-fuse" \
    --split soundscape \
    --probe-dir "$probe_dir" \
    --data-dir "$data_dir" \
    --exp010-dir "$exp010_dir" \
    --output-dir "$output_dir" \
    --n-workers 4

echo ""

# 4. Evaluate focal
echo "=== Evaluating focal ==="
"$venv/bin/weaksup-evaluate" \
    --split focal \
    --fused-dir "$output_dir" \
    --data-dir "$data_dir" \
    --output-dir "$eval_dir"

echo ""

# 5. Evaluate soundscape
echo "=== Evaluating soundscape ==="
"$venv/bin/weaksup-evaluate" \
    --split soundscape \
    --fused-dir "$output_dir" \
    --data-dir "$data_dir" \
    --output-dir "$eval_dir"

echo ""
echo "=== Done: $(date) ==="
