#!/bin/bash
#SBATCH --job-name=ssamba-verify
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem=16G
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH -t30 -qinferno
#SBATCH --output=logs/run-verification-%j.out

set -eu

# Load CUDA 12.1 to match the venv
module load cuda/12.1.1

echo "=== SSAMBA HF Port Verification ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
date

# Paths
scratch_dir=~/scratch/birdclef-2026
exp_dir=$scratch_dir/exp004-ssamba-hf
repo_dir=~/arc/birdclef-2026
exp_scripts=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/scripts
ssamba_hf_dir=$repo_dir/user/anthony/experiments/004-ssamba-hf-port/ssamba_hf
checkpoint=$scratch_dir/checkpoints/ssamba/ssamba_small_400.pth
output_dir=$exp_dir/verification

mkdir -p $output_dir

# Step 1: Extract embeddings with original SSAMBA (GPU)
echo ""
echo "=== Step 1: Extracting original SSAMBA embeddings ==="
source $exp_dir/original-venv/bin/activate
export PYTHONPATH="${exp_dir}/Vim/vim:${exp_dir}/Vim/vim/mamba-1p1p1:${repo_dir}/vendor/ssamba/src:${PYTHONPATH:-}"
python $exp_scripts/extract_original_embeddings.py \
    --checkpoint $checkpoint \
    --output $output_dir/original_embeddings.pt \
    --device cuda
deactivate || true

# Step 2: Extract HF embeddings and compare inline
echo ""
echo "=== Step 2: Extract HF embeddings and compare ==="
cd $ssamba_hf_dir

uv run python -c "
import torch
import numpy as np

from ssamba_hf import load_from_ssamba_checkpoint

# Load HF model
checkpoint_path = '$checkpoint'
model, config = load_from_ssamba_checkpoint(checkpoint_path, device='cpu', verbose=True)
model.eval()

# Create test spectrogram (same as extract_original_embeddings.py)
rng = np.random.default_rng(42)
spec = rng.standard_normal((1, 1024, 128)).astype(np.float32)
spectrogram = torch.from_numpy(spec)

# Extract HF embeddings
with torch.no_grad():
    hf_embeddings = model.get_embeddings(spectrogram)

print(f'HF embeddings shape: {hf_embeddings.shape}')
print(f'HF embeddings norm: {torch.linalg.norm(hf_embeddings).item():.4f}')

# Save HF embeddings for later analysis
torch.save({
    'embeddings': hf_embeddings.cpu(),
    'spectrogram': spectrogram,
    'checkpoint': checkpoint_path,
}, '$output_dir/hf_embeddings.pt')
print(f'Saved HF embeddings to $output_dir/hf_embeddings.pt')

# Load original (correct) reference embeddings
original_path = '$output_dir/original_embeddings.pt'
original_data = torch.load(original_path, map_location='cpu', weights_only=False)
original_embeddings = original_data['embeddings']

print(f'Original embeddings shape: {original_embeddings.shape}')
print(f'Original embeddings norm: {torch.linalg.norm(original_embeddings).item():.4f}')

# Compute similarity metrics
cos_sim = torch.nn.functional.cosine_similarity(hf_embeddings, original_embeddings).item()
max_diff = (hf_embeddings - original_embeddings).abs().max().item()
mean_diff = (hf_embeddings - original_embeddings).abs().mean().item()

print()
print('=== Comparison Results ===')
print(f'  Cosine Similarity: {cos_sim:.6f}')
print(f'  Max Absolute Diff: {max_diff:.2e}')
print(f'  Mean Absolute Diff: {mean_diff:.2e}')

# Save comparison report
import json
report = {
    'cosine_similarity': cos_sim,
    'max_abs_diff': max_diff,
    'mean_abs_diff': mean_diff,
    'hf_norm': torch.linalg.norm(hf_embeddings).item(),
    'original_norm': torch.linalg.norm(original_embeddings).item(),
}
with open('$output_dir/comparison_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f'Saved comparison report to $output_dir/comparison_report.json')

# Verdict
print()
if cos_sim > 0.99:
    print('SUCCESS: HF port matches original! (cosine > 0.99)')
elif cos_sim > 0.95:
    print('WARNING: Good match but not perfect (0.95 < cosine < 0.99)')
else:
    print('FAILURE: Low similarity - needs investigation (cosine < 0.95)')
"

echo ""
echo "=== Verification Complete ==="
date
