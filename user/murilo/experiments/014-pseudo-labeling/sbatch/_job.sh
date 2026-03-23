#!/bin/bash
#SBATCH --job-name=exp014-pseudo-label
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -qinferno
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Experiment 014: Teacher pseudo-labeling with GPU ONNX inference.
# Submitted by sbatch/run.sh — do not run directly.

set -eu

CONFIG="${CONFIG:-config/direct.yaml}"
SPLIT="${SPLIT:-labeled}"
MAX_FILES="${MAX_FILES:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ============================================================================
# Determine paths
# ============================================================================

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
fi

# ============================================================================
# Print job info
# ============================================================================

echo "=============================================="
echo "Job: Experiment 014 — Pseudo-Labeling"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Worker: ${WORKER_ID}/${NUM_WORKERS}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Config: ${CONFIG}"
echo "Split: ${SPLIT}"
echo "Max files: ${MAX_FILES:-all}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo ""
nvidia-smi
echo ""

# ============================================================================
# Setup per-job virtual environment in TMPDIR (fast local NVMe)
# ============================================================================

echo "Setting up per-job virtual environment..."

export UV_CACHE_DIR="$TMPDIR/uv-cache"

JOB_VENV_DIR="$TMPDIR/birdclef-$SLURM_JOB_ID"
JOB_VENV="$JOB_VENV_DIR/.venv"

echo "  TMPDIR: $TMPDIR"
echo "  Venv: $JOB_VENV"

mkdir -p "$JOB_VENV_DIR"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

echo "  Installing experiment dependencies..."
uv pip install -e "$EXPERIMENT_DIR"
# NOTE: torch/torchaudio NOT installed — only needed for kaldi_fbank (EAT),
# which is excluded from this config. Installing torch cu121 downgrades
# nvidia-cublas-cu12 from 12.9→12.1, breaking onnxruntime-gpu CUDA provider.

echo "  Virtual environment ready!"
echo ""

# Verify ONNX + CUDA libs
python -c "
import onnxruntime as ort
import importlib.metadata

provs = ort.get_available_providers()
print(f'onnxruntime: {ort.__version__}, providers: {provs}')

cublas = importlib.metadata.version('nvidia-cublas-cu12')
cudnn = importlib.metadata.version('nvidia-cudnn-cu12')
print(f'nvidia-cublas-cu12: {cublas}, nvidia-cudnn-cu12: {cudnn}')

assert 'CUDAExecutionProvider' in provs, 'ERROR: CUDAExecutionProvider not available — GPU inference will fail'
print('CUDA provider: OK')
"
echo ""

# ============================================================================
# Run pseudo-labeling with GPU monitoring
# ============================================================================

cd "$EXPERIMENT_DIR"
mkdir -p logs

# Start GPU monitoring in background (every 10s)
GPU_LOG="logs/gpu_${SPLIT}_${SLURM_JOB_ID}.csv"
echo "timestamp,gpu_name,pci_bus,driver,pstate,temp_c,gpu_util_%,mem_util_%,mem_total_mb,mem_used_mb,mem_free_mb" > "$GPU_LOG"
while true; do
    nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv,noheader,nounits >> "$GPU_LOG"
    sleep 10
done &
GPU_MONITOR_PID=$!
echo "GPU monitoring started (PID=$GPU_MONITOR_PID, log=$GPU_LOG)"
echo ""

EXTRA_FLAGS=""
if [[ -n "${MAX_FILES}" ]]; then
    EXTRA_FLAGS="--max-files ${MAX_FILES}"
fi
if [[ "${NUM_WORKERS}" -gt 1 ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --worker-id ${WORKER_ID} --num-workers ${NUM_WORKERS}"
fi

echo "Running pseudo-labeling..."
python pseudo_label.py --config "${CONFIG}" --split "${SPLIT}" ${EXTRA_FLAGS}

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
echo ""
echo "GPU log: $GPU_LOG"

# Print GPU utilization summary
echo "GPU utilization summary:"
awk -F',' 'NR>1 {sum+=$7; n++} END {if(n>0) printf "  Mean GPU util: %.1f%% (%d samples)\n", sum/n, n}' "$GPU_LOG"
awk -F',' 'NR>1 {if($10>max) max=$10} END {printf "  Peak VRAM: %s MB\n", max}' "$GPU_LOG"

echo ""
echo "=============================================="
echo "Complete! $(date)"
echo "=============================================="
