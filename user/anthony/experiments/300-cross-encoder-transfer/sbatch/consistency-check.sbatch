#!/bin/bash
#SBATCH --job-name=300-consistency
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=6 --mem-per-cpu=8G
#SBATCH -p cpu-small
#SBATCH -t 45 -q inferno
#SBATCH --output=logs/consistency-%j.out

# Risk #1: in-kernel encode must reproduce the training embedding (cosine > 0.999).
#   ENCODER=beats   sbatch sbatch/consistency-check.sbatch   # torch venv
#   ENCODER=birdnet sbatch sbatch/consistency-check.sbatch   # bmz venv
#   ENCODER=ast     sbatch sbatch/consistency-check.sbatch   # torch venv + transformers

set -euo pipefail
mkdir -p logs
export REPO_DIR="${REPO_DIR:-$(realpath "${SLURM_SUBMIT_DIR:-$PWD}/../../../..")}"
echo "REPO_DIR=$REPO_DIR ENCODER=${ENCODER:?set ENCODER=beats|birdnet|ast}"

# NB: the check script loads vendored BEATs directly from ~/arc (temp model_dir),
# so no $REPO_DIR/vendor/unilm self-heal is needed here.

if [ "$ENCODER" = "birdnet" ]; then
    source sbatch/_setup-venv-birdnet.sh
else
    source sbatch/_setup-venv.sh
fi
# AST runs HF transformers (mel) + onnxruntime (graph) in the torch venv; ensure both.
[ "$ENCODER" = "ast" ] && uv pip install transformers onnxruntime >/dev/null 2>&1 || true
# EAT runs torchaudio kaldi-fbank (mel) + onnxruntime (graph) — torchaudio is in the venv.
[ "$ENCODER" = "eat" ] && uv pip install onnxruntime >/dev/null 2>&1 || true
# SSAMBA runs torchaudio kaldi-fbank (mel) + onnxruntime (pure-PyTorch Mamba graph). The
# gate here is CPU-ONNX vs the GPU-CUDA focal cache (the deploy-critical parity).
[ "$ENCODER" = "ssamba_hf" ] && uv pip install onnxruntime >/dev/null 2>&1 || true

python scripts/consistency_check.py "$ENCODER"
