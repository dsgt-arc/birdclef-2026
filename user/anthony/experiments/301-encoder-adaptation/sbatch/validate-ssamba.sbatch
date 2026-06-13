#!/bin/bash
#SBATCH --job-name=301-validate-ssamba
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/validate-ssamba-%j.out

# Lighter go/no-go gate for SSAMBA native DAPT. Front-loads the two SSAMBA risks:
#   risk #1 pos_embed geometry — the hand-rolled masked forward adds pos_embed[:,1:]
#           to the patch grid; this asserts pos_embed length == num_patches (+CLS) at
#           the wrapper's eval geometry (prints live shapes).
#   risk #2 grad through the Mamba CUDA kernel — backward must reach the backbone.
# Plus frontend parity, masked-forward finiteness, state_dict round-trip, tiny-overfit.
# MUST be green before sbatch/dapt.sbatch on dapt-ssamba-native.yaml.
#
# Usage:  sbatch sbatch/validate-ssamba.sbatch

set -uo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh
set +e                         # _setup-venv.sh leaves `set -e` on; we capture rc ourselves

uv pip install -q pytest

echo "===================== SSAMBA DAPT VALIDATION GATE ====================="
pytest tests/test_ssamba_dapt.py -v -s
rc=$?
echo "======================================================================="
[ "$rc" -eq 0 ] && echo "GATE: GREEN — SSAMBA DAPT verified, proceed to full DAPT" \
                || echo "GATE: RED — fix SSAMBA DAPT before any full run"
exit "$rc"
