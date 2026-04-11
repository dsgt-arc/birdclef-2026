#!/bin/bash
#SBATCH --job-name=exp107-coverage
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -o %x-%j.out

set -euo pipefail

cd "$HOME/arc/birdclef-2026"
python3 user/anthony/experiments/107-hopfield-prototypes/scripts/analyze_coverage.py
