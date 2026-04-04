#!/bin/bash
#SBATCH --job-name=105-inspect-onnx
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 30
#SBATCH -q inferno
#SBATCH --output=logs/inspect-onnx-%j.out

# Download the Perch v2 ONNX model and dump a structural report.
# Identifies the pre-pool feature-map tap point we need for exp 105.
#
# CPU-only: ONNX graph inspection is light; we don't need GPU for this.
# Submit from the experiment directory:
#   cd user/anthony/experiments/105-perch-frame-sed
#   sbatch sbatch/inspect-onnx.sbatch

set -eu
mkdir -p logs artifacts/onnx_inspection

source sbatch/_setup-venv.sh

echo ""
echo "=== Inspect Perch v2 ONNX ==="
python scripts/inspect_onnx.py

echo ""
echo "=== Done ==="
