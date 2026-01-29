#!/bin/bash
#SBATCH --job-name=ssamba-verify-suite
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=6 --mem=32G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t120 -qinferno
#SBATCH --output=logs/verification-suite-%j.out

set -eu

# Load CUDA 12.1 to match the venv
module load cuda/12.1.1

echo "=========================================="
echo "SSAMBA-HF VERIFICATION SUITE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

# Paths
repo_dir=~/arc/birdclef-2026
scratch_dir=~/scratch/birdclef-2026
user_dir=$repo_dir/user/anthony
exp_dir=$scratch_dir/exp004-ssamba-hf
exp_scripts=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/scripts
ssamba_hf_dir=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/ssamba_hf
checkpoint=$scratch_dir/checkpoints/ssamba/ssamba_small_400.pth
output_dir=$exp_dir/verification-suite
esc50_root=$repo_dir/vendor/ESC-50

mkdir -p $output_dir

# Track overall results
FAILURES=0

# ============================================================
# Step 0: Sync dependencies (uses pre-built wheels from pyproject.toml)
# ============================================================
echo ""
echo "=========================================="
echo "STEP 0: Sync Dependencies"
echo "=========================================="

cd $user_dir
uv sync --extra cuda

# ============================================================
# Step 1: Pytest — HF-only verification tests
# (preprocessing, CUDA-vs-PyTorch kernels, multi-input robustness)
# ============================================================
echo ""
echo "=========================================="
echo "STEP 1: Pytest Verification Suite (HF-only)"
echo "=========================================="

cd $ssamba_hf_dir
export SSAMBA_CHECKPOINT=$checkpoint
export ESC50_ROOT=$esc50_root

uv run --extra dev --extra cuda \
    pytest tests/test_verification.py -v --tb=short -x \
    || FAILURES=$((FAILURES + 1))

# ============================================================
# Step 2: Multi-Input Verification (requires original SSAMBA)
# ============================================================
echo ""
echo "=========================================="
echo "STEP 2: Multi-Input Original-vs-HF Comparison"
echo "=========================================="

ESC50_FLAG=""
esc50_audio=$esc50_root/audio
if [ -d "$esc50_audio" ]; then
    ESC50_FLAG="--esc50-dir $esc50_audio"
    echo "ESC-50 audio found: $esc50_audio"
else
    echo "ESC-50 audio not found, skipping real audio tests"
fi

if [ -d "$exp_dir/original-venv" ]; then
    echo "Setting up PYTHONPATH for original SSAMBA..."
    export PYTHONPATH="${exp_dir}/Vim/vim:${exp_dir}/Vim/vim/mamba-1p1p1:${repo_dir}/vendor/ssamba/src:${ssamba_hf_dir}:${exp_scripts}:${PYTHONPATH:-}"

    source $exp_dir/original-venv/bin/activate

    python $exp_scripts/verify_multi_input.py \
        --checkpoint $checkpoint \
        --device cuda \
        $ESC50_FLAG \
        --output $output_dir/multi_input_report.json \
        || FAILURES=$((FAILURES + 1))

    deactivate || true
else
    echo "WARNING: Original SSAMBA venv not found at $exp_dir/original-venv"
    echo "Skipping (requires original model). Run setup-original-ssamba.sbatch first."
    FAILURES=$((FAILURES + 1))
fi

# ============================================================
# Step 3: Layer-by-Layer Activation Comparison (requires original)
# ============================================================
echo ""
echo "=========================================="
echo "STEP 3: Layer-by-Layer Activation Comparison"
echo "=========================================="

if [ -d "$exp_dir/original-venv" ]; then
    export PYTHONPATH="${exp_dir}/Vim/vim:${exp_dir}/Vim/vim/mamba-1p1p1:${repo_dir}/vendor/ssamba/src:${ssamba_hf_dir}:${exp_scripts}:${PYTHONPATH:-}"

    source $exp_dir/original-venv/bin/activate

    python $exp_scripts/verify_layer_activations.py \
        --checkpoint $checkpoint \
        --device cuda \
        --output $output_dir/layer_activations_report.json \
        || FAILURES=$((FAILURES + 1))

    deactivate || true
else
    echo "WARNING: Skipping (requires original venv)"
    FAILURES=$((FAILURES + 1))
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "VERIFICATION SUITE SUMMARY"
echo "=========================================="
echo "Results saved to: $output_dir/"
ls -la $output_dir/ 2>/dev/null || echo "(no files written)"

if [ $FAILURES -eq 0 ]; then
    echo ""
    echo "ALL STEPS PASSED"
else
    echo ""
    echo "FAILURES: $FAILURES step(s) failed"
    echo "Check individual reports in $output_dir/ for details"
fi

echo ""
date
echo "Done."

exit $FAILURES
