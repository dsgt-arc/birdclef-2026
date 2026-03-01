#!/bin/bash
#SBATCH --job-name=009-download
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t120 -qinferno
#SBATCH --output=logs/download-data-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
data_dir=$scratch_dir/data/birdclef-2025

mkdir -p $data_dir

echo "Downloading BirdCLEF 2025 data to $data_dir"

# Requires kaggle CLI and API key (~/.kaggle/kaggle.json)
kaggle competitions download -c birdclef-2025 -p $data_dir

echo "Extracting..."
cd $data_dir
unzip -o birdclef-2025.zip
rm -f birdclef-2025.zip

echo "Download complete. Contents:"
ls -la $data_dir
