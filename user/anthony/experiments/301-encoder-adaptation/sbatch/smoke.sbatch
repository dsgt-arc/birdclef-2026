#!/bin/bash
#SBATCH --job-name=301-smoke
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/smoke-%j.out

# Smoke gate for the LoRA track: 2-step adaptation of all three encoders.
# Validates (cheaply, ~10 min incl. one-time venv build) the never-run paths:
#   - .venv-301 build (torch/cu126 + mamba-ssm + peft)
#   - FocalDataset load from train.csv / train_audio
#   - LoRA attach -> the `target_modules` risk (ssamba validated in exp 004;
#     eat [qkv,proj] + beats [q,k,v,out_proj] are UNVALIDATED -> peft raises if no match)
#   - grad forward through encoder_forward.pooled_features
#   - focal loss + backward + adapter save
#
# Does NOT use `set -e` around the per-encoder loop: we want all three results
# even if one fails, so the failing target_modules set is obvious in one run.
#
# Usage:  sbatch sbatch/smoke.sbatch

set -uo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh   # ends by cd-ing into the exp 301 dir

echo "===================== SMOKE GATE: 3x LoRA --max-steps 2 ====================="
declare -A status
for cfg in configs/lora-beats.yaml configs/lora-eat.yaml configs/lora-ssamba.yaml; do
    enc=$(basename "$cfg" .yaml)
    echo "----------------------------------------------------------------------------"
    echo ">>> SMOKE $enc"
    if birdclef-301 lora --config "$cfg" --max-steps 2 \
            --output-dir "/tmp/smoke-$enc-${SLURM_JOB_ID:-local}"; then
        status[$enc]=PASS
    else
        status[$enc]="FAIL(rc=$?)"
    fi
done

echo "============================== SMOKE SUMMARY ==============================="
rc=0
for enc in lora-beats lora-eat lora-ssamba; do
    echo "  $enc : ${status[$enc]:-MISSING}"
    [ "${status[$enc]:-}" = "PASS" ] || rc=1
done
echo "==========================================================================="
[ "$rc" -eq 0 ] && echo "GATE: GREEN — proceed to Phase 1" || echo "GATE: RED — fix target_modules / forward before full runs"
exit "$rc"
