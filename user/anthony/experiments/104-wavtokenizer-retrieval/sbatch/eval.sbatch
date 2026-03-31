#!/bin/bash
#SBATCH --job-name=104-eval
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 60
#SBATCH -q inferno
#SBATCH --output=logs/eval-%j.out

set -eu
mkdir -p logs

source sbatch/_setup-venv.sh

echo "=== Phase 1: Round-trip evaluation (Perch v2 ONNX) ==="
exp104-roundtrip --config configs/exp104.yaml

echo ""
echo "=== Phase 2a: Codebook quantization ==="
exp104-codebook --config configs/exp104.yaml

echo ""
echo "=== Phase 2b: k-mer retrieval ==="
exp104-retrieval --config configs/exp104.yaml

echo ""
echo "=== Phase 3: Soundscape search ==="
exp104-soundscape --config configs/exp104.yaml

echo ""
echo "=== All phases complete ==="
