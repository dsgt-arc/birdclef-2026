#!/bin/bash
#SBATCH --job-name=101-debug
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH -t10 -qinferno
#SBATCH --output=logs/debug-%j.out

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/101-soundscape-calibration

cd $exp_dir

venv=~/scratch/birdclef-2026/.venv-010
source $venv/bin/activate

python scripts/debug_alignment.py
