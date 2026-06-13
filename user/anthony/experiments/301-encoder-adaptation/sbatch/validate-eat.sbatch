#!/bin/bash
#SBATCH --job-name=301-validate-eat
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 30 -q inferno
#SBATCH --output=logs/validate-eat-%j.out

# Hard go/no-go gate for the EAT data2vec port (the from-scratch, highest-risk path).
# Runs tests/test_eat_dapt.py on a GPU node against the real HF EAT model:
#   numerical stability — FFN-target semantics (hand-recompute), target shape/finite,
#                         anti-collapse variance, non-degeneracy at init, finite grads,
#                         state_dict round-trip (== exp300 overlay), LoRA merge.
#   directional learning — tiny-overfit drop, beats trivial baselines, bounded move.
# MUST be green before sbatch/dapt.sbatch on dapt-eat-native.yaml.
#
# Usage:  sbatch sbatch/validate-eat.sbatch

set -uo pipefail
mkdir -p logs

module load cuda/12.6.1
source sbatch/_setup-venv.sh   # ends by cd-ing into the exp 301 dir
set +e                         # _setup-venv.sh leaves `set -e` on; we capture rc ourselves

uv pip install -q pytest       # idempotent on the persistent venv

echo "===================== EAT PORT VALIDATION GATE ====================="
pytest tests/test_eat_dapt.py -v -s
rc=$?
echo "===================================================================="
[ "$rc" -eq 0 ] && echo "GATE: GREEN — EAT port verified, proceed to full DAPT" \
                || echo "GATE: RED — fix the EAT port before any full run"
exit "$rc"
