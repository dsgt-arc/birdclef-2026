#!/bin/bash
#SBATCH --job-name=009-build-kaggle
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60 -qinferno
#SBATCH --output=logs/build-kaggle-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)
export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv

cd $repo_dir/user/anthony/experiments/009-birdclef-pipeline

echo "=== Step 1: Download dependency wheels ==="
bash scripts/build_deps.sh

echo ""
echo "=== Step 2: Verify weight artifacts ==="
if [ ! -f artifacts/weights/ssamba_classifier.onnx ]; then
    echo "ERROR: artifacts/weights/ssamba_classifier.onnx not found."
    echo "Run export-onnx.sbatch first."
    exit 1
fi

for f in artifacts/weights/ssamba_classifier.onnx artifacts/weights/species_list.json artifacts/weights/mel_config.json; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(du -h "$f" | cut -f1))"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Step 3: Upload dry-run ==="
uv run --project $repo_dir/user/anthony python scripts/upload_kaggle.py --dry-run

echo ""
echo "=== Build complete ==="
echo "To upload for real, run:"
echo "  uv run --project $repo_dir/user/anthony python scripts/upload_kaggle.py"
