#!/bin/bash
#SBATCH --job-name=101-train-probes
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH -t120 -qinferno
#SBATCH --output=logs/train-probes-%j.out

set -eu

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

# 2026 paths
DATA_DIR=/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026
EMBED_2026=$scratch_dir/outputs/exp010-2026/labels
PROBE_DIR=$scratch_dir/outputs/exp101/probes

# Exp 013 code hardcodes DEFAULT_OUTPUT_ROOT to exp010/labels (2025 embeddings).
# Symlink so it finds 2026 embeddings at the expected path structure.
EMBED_OVERRIDE=$TMPDIR/exp010-labels
ln -sfn $EMBED_2026 $EMBED_OVERRIDE

# Install exp 013 into temp venv
JOB_VENV="$TMPDIR/birdclef-$SLURM_JOB_ID/.venv"
mkdir -p "$(dirname $JOB_VENV)"
uv venv "$JOB_VENV" --python 3.11
source "$JOB_VENV/bin/activate"

uv pip install -e "$repo_dir/user/anthony/experiments/013-weak-supervision-flyingsquid"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Patch the default path at runtime via a tiny wrapper
cat > $TMPDIR/patch_defaults.py << 'PYEOF'
import weaksup.linear_probe as lp
from pathlib import Path
import os

override = os.environ.get("EMBED_OVERRIDE")
if override:
    lp.DEFAULT_OUTPUT_ROOT = Path(override)
    # Also update _embed_dir to use the new root
    original_embed_dir = lp._embed_dir
    def _patched_embed_dir(source):
        if source == "indices":
            return lp.DEFAULT_OUTPUT_ROOT / "indices" / "train_audio" / "parts"
        return lp.DEFAULT_OUTPUT_ROOT / source / "train_audio" / "parts" / "embed"
    lp._embed_dir = _patched_embed_dir
    print(f"Patched DEFAULT_OUTPUT_ROOT -> {override}")

lp.app()
PYEOF

export EMBED_OVERRIDE=$EMBED_OVERRIDE

echo "=== Training BirdNET species probe on 2026 focal data ==="
python $TMPDIR/patch_defaults.py species birdnet \
    --gamma 2.0 --epochs 50 --batch-size 16384 \
    --data-dir $DATA_DIR \
    --output-dir $PROBE_DIR

echo ""
echo "=== Training Perch species probe on 2026 focal data ==="
python $TMPDIR/patch_defaults.py species perch \
    --gamma 2.0 --epochs 50 --batch-size 16384 \
    --data-dir $DATA_DIR \
    --output-dir $PROBE_DIR

echo ""
echo "=== Training Indices species probe on 2026 focal data ==="
python $TMPDIR/patch_defaults.py species indices \
    --gamma 2.0 --epochs 50 --batch-size 16384 \
    --data-dir $DATA_DIR \
    --output-dir $PROBE_DIR

echo ""
echo "=== Done. Probes saved to $PROBE_DIR ==="
ls -la $PROBE_DIR/*/species_probe.npz 2>/dev/null || echo "No probe files found"
