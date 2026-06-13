#!/bin/bash
#SBATCH --job-name=301-smoke-dapt
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/smoke-dapt-%j.out

# Smoke gate for the DAPT track (the chosen soundscape-adaptation path).
# 2-step continued-pretraining of both DAPT encoders. Validates the never-run
# DAPT-specific paths:
#   - SoundscapeDataset glob over the 10,658-file pool (one 5s window/file)
#   - _time_mask (zero random spans of the waveform)
#   - copy.deepcopy(student) -> EMA teacher + _ema_update
#   - latent_bootstrap_loss (1 - cosine) + backward through FULL encoder weights
#   - state_dict.pt save (what exp 300 loads via --state-dict)
# (pooled_features for eat/ssamba is shared with the LoRA smoke.)
#
# Reuses the venv built by smoke.sbatch — run that (or any 301 GPU job) first so
# this takes the "Using existing venv" path. Fault-tolerant loop: both encoders
# run even if one fails.
#
# Usage:  sbatch sbatch/smoke-dapt.sbatch

set -uo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh   # ends by cd-ing into the exp 301 dir

echo "===================== SMOKE GATE: 2x DAPT --max-steps 2 ====================="
declare -A status
for cfg in configs/dapt-eat.yaml configs/dapt-ssamba.yaml; do
    enc=$(basename "$cfg" .yaml)
    echo "----------------------------------------------------------------------------"
    echo ">>> SMOKE $enc"
    if birdclef-301 dapt --config "$cfg" --max-steps 2 \
            --output-dir "/tmp/smoke-$enc-${SLURM_JOB_ID:-local}"; then
        status[$enc]=PASS
    else
        status[$enc]="FAIL(rc=$?)"
    fi
done

echo "============================== SMOKE SUMMARY ==============================="
rc=0
for enc in dapt-eat dapt-ssamba; do
    echo "  $enc : ${status[$enc]:-MISSING}"
    [ "${status[$enc]:-}" = "PASS" ] || rc=1
done
echo "==========================================================================="
[ "$rc" -eq 0 ] && echo "GATE: GREEN — proceed to Phase 1 (full DAPT)" || echo "GATE: RED — fix DAPT path before full runs"
exit "$rc"
