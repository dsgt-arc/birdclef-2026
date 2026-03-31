#!/bin/bash
#SBATCH --job-name=104-tokenize
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=10G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -p gpu-rtx6000
#SBATCH -t 120
#SBATCH -q inferno
#SBATCH --output=logs/tokenize-%j.out

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

echo "=== Step 1: Select species subset ==="
exp104-select --config configs/exp104.yaml

echo ""
echo "=== Step 2: Tokenize (encode + decode) ==="
exp104-tokenize --config configs/exp104.yaml

echo ""
echo "=== Done ==="
