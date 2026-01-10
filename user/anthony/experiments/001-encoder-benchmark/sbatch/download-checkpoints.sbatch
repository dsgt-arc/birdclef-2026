#!/bin/bash
#SBATCH --job-name=download-ckpt
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t60 -qembers
#SBATCH --output=logs/download-checkpoints-%j.out
set -eu

echo "Downloading model checkpoints to scratch"
hostname
date

scratch_dir=~/scratch/birdclef-2026
repo_dir=~/arc/birdclef-2026

# Create ephemeral venv in TMPDIR (fast local disk)
LOCAL_VENV=$TMPDIR/venv
uv venv $LOCAL_VENV
source $LOCAL_VENV/bin/activate

# Set HuggingFace cache to scratch (persistent)
export HF_HOME=$scratch_dir/.cache/huggingface
mkdir -p $HF_HOME

# Install minimal dependencies for download
uv pip install huggingface_hub

cd $scratch_dir/checkpoints

# ============================================================
# EAT Checkpoints (via HuggingFace)
# ============================================================
echo ""
echo "=== Downloading EAT checkpoints via HuggingFace ==="

python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

scratch = os.path.expanduser("~/scratch/birdclef-2026")
eat_dir = f"{scratch}/checkpoints/eat"

# Download EAT-base (epoch 30, fine-tuned on AS-2M) - best for classification
print("Downloading EAT-base (epoch30, AS-2M fine-tuned)...")
snapshot_download(
    repo_id="worstchan/EAT-base_epoch30_finetune_AS2M",
    local_dir=f"{eat_dir}/EAT-base_epoch30_finetune_AS2M",
    local_dir_use_symlinks=False
)
print("EAT-base (finetuned) downloaded!")

# Also download pretrained version for feature extraction
print("Downloading EAT-base (epoch30, pretrained)...")
snapshot_download(
    repo_id="worstchan/EAT-base_epoch30_pretrain",
    local_dir=f"{eat_dir}/EAT-base_epoch30_pretrain",
    local_dir_use_symlinks=False
)
print("EAT-base (pretrained) downloaded!")
EOF

echo "EAT checkpoints downloaded to $scratch_dir/checkpoints/eat/"

# ============================================================
# BEATs Checkpoints (via HuggingFace mirror)
# ============================================================
echo ""
echo "=== Downloading BEATs checkpoints via HuggingFace ==="

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

scratch = os.path.expanduser("~/scratch/birdclef-2026")
beats_dir = f"{scratch}/checkpoints/beats"
os.makedirs(beats_dir, exist_ok=True)

# Download BEATs_iter3_plus_AS2M.pt from HuggingFace mirror
print("Downloading BEATs_iter3_plus_AS2M.pt...")
hf_hub_download(
    repo_id="lpepino/beats_ckpts",
    filename="BEATs_iter3_plus_AS2M.pt",
    local_dir=beats_dir,
)
print("BEATs checkpoint downloaded!")
EOF

echo "BEATs checkpoints downloaded to $scratch_dir/checkpoints/beats/"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== Checkpoint Status ==="
echo ""
echo "EAT:"
ls -la $scratch_dir/checkpoints/eat/ 2>/dev/null || echo "  (not found)"
echo ""
echo "BEATs:"
ls -la $scratch_dir/checkpoints/beats/ 2>/dev/null || echo "  (not found)"

echo ""
echo "Done!"
date
