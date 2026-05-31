#!/bin/bash
#SBATCH --job-name=exp038-dl-birdset
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -qinferno
#SBATCH --output=logs/slurm_download_birdset_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgustineli3@gatech.edu
#
# Exp 038 Stage 0b — stream-extract the curated BirdSet-XCL audio (CPU job, big I/O, NO GPU).
# Runs scripts/download_birdset_audio.py: per XCL_shard download -> extract our manifest basenames
# -> delete tarball. Transient shards land on $TMPDIR (fast local NVMe, auto-cleaned); only the
# extracted audio (~150 GB) is written to project storage. RESUMABLE — if the 8h walltime is hit,
# just re-submit (per-shard done-set + skip-if-dest-exists continue where it left off).
#
# Reuses the prebuilt interactive venv (deps already present: huggingface_hub, pandas, pyarrow) —
# NO install, so no home-quota risk (CLAUDE.md). HF token loaded from user/murilo/.env (gitignored).
#
# Usage (self-submitting — run with bash, not sbatch):
#   cd user/murilo/experiments/038-xc-pretraining
#   bash sbatch/download_birdset_audio.sh --limit-shards 2   # smoke: 2 shards
#   bash sbatch/download_birdset_audio.sh                     # full 98 (resumable)
set -eu

# Self-submit if not already under SLURM.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
    mkdir -p "$EXPERIMENT_DIR/logs"
    cd "$EXPERIMENT_DIR"
    echo "Submitting BirdSet download job (args: $*)..."
    sbatch "$SCRIPT_DIR/download_birdset_audio.sh" "$@"
    exit 0
fi

EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
cd "$EXPERIMENT_DIR"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"

# Prebuilt interactive venv (no install → no home-quota risk).
source "$HOME/clef/birdclef-2026/.venv/bin/activate"
python -c "import huggingface_hub, pandas; print('hf_hub', huggingface_hub.__version__, '| pandas', pandas.__version__)"

# HF token from the gitignored .env; keep HF cache off home (20 GB quota) on local scratch.
set -a
[ -f "$USER_DIR/.env" ] && source "$USER_DIR/.env"
set +a
export HF_HOME="$TMPDIR/hf-cache"
mkdir -p "$HF_HOME"
echo "HF_TOKEN set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo NO)"

echo "Starting download (transient shards -> \$TMPDIR, audio -> project)..."
srun python scripts/download_birdset_audio.py --shard-dir "$TMPDIR/birdset_shards" "$@"

echo "=============================================="
echo "End: $(date)"
echo "=============================================="
