#!/bin/bash
#SBATCH --job-name=ssamba-dl
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=2 --mem=4G
#SBATCH -t30 -qinferno
#SBATCH --output=logs/download-weights-%j.out

set -eu

scratch_dir=~/scratch/birdclef-2026
checkpoint_dir=$scratch_dir/checkpoints/ssamba
mkdir -p $checkpoint_dir

# Create temp venv for download tools
cd $TMPDIR
uv venv .venv && source .venv/bin/activate
uv pip install huggingface_hub gdown

# Try HuggingFace Hub first, fall back to gdown
python -c "
from huggingface_hub import hf_hub_download, list_repo_files
import os

checkpoint_dir = os.path.expanduser('$checkpoint_dir')
target_file = os.path.join(checkpoint_dir, 'amba_small_400.pth')

if os.path.exists(target_file):
    print(f'Checkpoint already exists: {target_file}')
    exit(0)

try:
    files = list_repo_files('attentionisallyouneed369/ssamba')
    print('HF files:', files)
    # Download if amba_small_400.pth exists
    if 'amba_small_400.pth' in files:
        hf_hub_download('attentionisallyouneed369/ssamba', 'amba_small_400.pth',
                        local_dir=checkpoint_dir)
        print('Downloaded from HuggingFace')
        exit(0)
    else:
        print('amba_small_400.pth not found in HuggingFace repo')
        exit(1)
except Exception as e:
    print(f'HuggingFace failed: {e}')
    exit(1)
" || {
    echo "HuggingFace download failed, trying gdown..."
    # Fallback: gdown from Google Drive
    # Google Drive folder: 1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX
    # The actual file ID needs to be found from the Drive folder
    gdown --folder "https://drive.google.com/drive/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX" -O $checkpoint_dir/ --remaining-ok
}

echo "=== Checkpoint directory contents ==="
ls -la $checkpoint_dir/
