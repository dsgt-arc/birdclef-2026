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
#
# *** SUBMIT WITH `sbatch`, NOT `bash`. ***
# `bash sbatch/...` runs the script in your CURRENT shell (and inside an interactive `interact`
# session that means it runs there, tying it up). `sbatch` hands the script to the scheduler as a
# DETACHED batch job and returns immediately — the correct PACE way (this is exactly how run.sh
# launches _job.sh in exp 037). Run from the experiment dir so $SLURM_SUBMIT_DIR + the relative
# --output path resolve correctly:
#   cd user/murilo/experiments/038-xc-pretraining
#   mkdir -p logs                                              # one-time: sbatch opens --output before the script runs
#   sbatch sbatch/download_birdset_audio.sh --limit-shards 2   # smoke: first 2 shards
#   sbatch sbatch/download_birdset_audio.sh                    # full 98 shards (resumable)
#   sbatch sbatch/download_birdset_audio.sh --dry-run          # list shards only, download nothing
# Monitor:  squeue -u $USER ; tail -f logs/slurm_download_birdset_<jobid>.log
#
# Mechanics: per XCL_shard -> hf_hub_download -> extract only our manifest basenames -> delete the
# tarball. Transient shards land on $TMPDIR (fast local NVMe, auto-cleaned); only the extracted
# audio (~150 GB) is written to project storage. RESUMABLE (per-shard done-set + skip-if-dest-
# exists): if the 8h walltime is hit, just re-submit and it continues where it left off.
#
# Reuses Murilo's prebuilt venv (deps already present: huggingface_hub, pandas, pyarrow) — NO
# install, so no home-quota risk (CLAUDE.md). HF token loaded from user/murilo/.env (gitignored).
set -eu

# $SLURM_SUBMIT_DIR = where `sbatch` was invoked (the experiment dir). NOT $BASH_SOURCE — sbatch
# spools a copy of this script to a temp path ([[feedback_slurm_script_conventions]]). The
# $BASH_SOURCE fallback only applies if someone runs it with bash for a quick local check.
EXPERIMENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USER_DIR="$(dirname "$(dirname "$EXPERIMENT_DIR")")"
cd "$EXPERIMENT_DIR"

echo "=============================================="
echo "Exp 038 Stage 0b — BirdSet-XCL audio download"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local} | Node: $(hostname) | $(date)"
echo "Submit dir: $EXPERIMENT_DIR"
echo "Args:   $*"
echo ""

# Murilo's prebuilt venv (deps already present -> no install -> no home-quota risk; CLAUDE.md).
source "$HOME/clef/birdclef-2026/.venv/bin/activate"
python -c "import huggingface_hub, pandas; print('hf_hub', huggingface_hub.__version__, '| pandas', pandas.__version__)"

# HF token from the gitignored .env; keep the HF cache off home (20 GB quota) on local scratch.
set -a
[ -f "$USER_DIR/.env" ] && source "$USER_DIR/.env"
set +a
export HF_HOME="${TMPDIR:-/tmp}/hf-cache"
mkdir -p "$HF_HOME"
echo "HF_TOKEN set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo NO)"
echo ""

echo "Starting download (transient shards -> \$TMPDIR, audio -> project)..."
srun python scripts/download_birdset_audio.py --shard-dir "${TMPDIR:-/tmp}/birdset_shards" "$@"

echo ""
echo "=============================================="
echo "End: $(date)"
echo "=============================================="
