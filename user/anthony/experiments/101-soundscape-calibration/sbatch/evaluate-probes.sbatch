#!/bin/bash
#SBATCH --job-name=101-eval-probes
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/evaluate-probes-%j.out

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/101-soundscape-calibration

cd $exp_dir

# Use exp 010 venv (has numpy, pandas, sklearn, pyarrow)
venv=~/scratch/birdclef-2026/.venv-010
source $venv/bin/activate

python scripts/evaluate_probes.py \
    --output artifacts/ \
    --eda-artifacts $repo_dir/user/anthony/experiments/100-eda/artifacts/
