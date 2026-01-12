#!/bin/bash
#SBATCH --job-name=cache-teacher-003
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/cache-teacher-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Caching Teacher Outputs - Experiment 003"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

cd $repo_dir

uv run python << 'EOF'
"""Cache teacher outputs for fast distillation training."""

import glob
import io
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, "vendor/unilm/beats")

from birdclef.distillation import load_teacher_and_tokenizer

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
tokenizer_path = scratch_dir / "checkpoints/beats/Tokenizers_iter3.pt"
static_embeddings_path = scratch_dir / "outputs/exp002-benchmarks/static_embeddings/static_embeddings.npy"
audioset_dir = scratch_dir / "data/audioset-wds/20k"
cache_dir = scratch_dir / "outputs/exp003-distillation/teacher_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
target_length = 160000  # 10s at 16kHz
batch_size = 32  # Larger batch for efficiency

print("=" * 60)
print("CACHING TEACHER OUTPUTS")
print("=" * 60)

# Load models
print("\nLoading teacher and tokenizer...")
teacher, tokenizer = load_teacher_and_tokenizer(encoder_path, tokenizer_path)
teacher = teacher.to(device)
tokenizer = tokenizer.to(device)
print(f"  Device: {device}")

# Load static embeddings
print("\nLoading static embeddings...")
static_embeddings = torch.from_numpy(np.load(static_embeddings_path)).float()
print(f"  Shape: {static_embeddings.shape}")

# Collect all samples from tar files
print("\nScanning tar files...")
tar_files = sorted(glob.glob(str(audioset_dir / "train" / "*.tar")))
print(f"  Found {len(tar_files)} tar files")

all_samples = []
for tar_path in tar_files:
    with tarfile.open(tar_path, 'r') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(('.wav', '.flac', '.mp3'))]
        for member in members:
            all_samples.append((tar_path, member.name))
print(f"  Total samples: {len(all_samples)}")

def load_audio(tar_path, member_name):
    """Load and preprocess audio from tar."""
    with tarfile.open(tar_path, 'r') as tar:
        f = tar.extractfile(member_name)
        audio_data, sr = sf.read(io.BytesIO(f.read()))

    waveform = torch.from_numpy(audio_data.astype(np.float32))
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=-1)

    # Pad or trim
    if len(waveform) < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    return waveform

# Process in batches
print("\nCaching teacher outputs...")
start_time = time.time()

all_teacher_embeds = []
all_token_ids = []
all_static_embeds = []

for i in tqdm(range(0, len(all_samples), batch_size), desc="Processing"):
    batch_samples = all_samples[i:i + batch_size]

    # Load batch
    waveforms = torch.stack([load_audio(p, m) for p, m in batch_samples])
    waveforms = waveforms.to(device)

    with torch.no_grad():
        # Get teacher embeddings
        teacher_embeds, _ = teacher.extract_features(waveforms, padding_mask=None)

        # Get token IDs and reshape
        token_ids = tokenizer.extract_labels(waveforms)
        if token_ids.dim() == 1:
            seq_len = token_ids.shape[0] // waveforms.shape[0]
            token_ids = token_ids.view(waveforms.shape[0], seq_len)

        # Get static embeddings
        static_embeds = static_embeddings[token_ids.cpu()]

    all_teacher_embeds.append(teacher_embeds.cpu().half())  # FP16 to save space
    all_token_ids.append(token_ids.cpu().short())  # int16 for token IDs
    all_static_embeds.append(static_embeds.half())

# Concatenate all
print("\nConcatenating and saving...")
teacher_embeds = torch.cat(all_teacher_embeds, dim=0)
token_ids = torch.cat(all_token_ids, dim=0)
static_embeds = torch.cat(all_static_embeds, dim=0)

print(f"  Teacher embeddings: {teacher_embeds.shape}")
print(f"  Token IDs: {token_ids.shape}")
print(f"  Static embeddings: {static_embeds.shape}")

# Save to disk
torch.save({
    'teacher_embeds': teacher_embeds,
    'static_embeds': static_embeds,
    'token_ids': token_ids,
    'n_samples': len(all_samples),
    'seq_len': teacher_embeds.shape[1],
    'embed_dim': teacher_embeds.shape[2],
}, cache_dir / "audioset_20k_cache.pt")

elapsed = time.time() - start_time
print(f"\nCaching complete!")
print(f"  Time: {elapsed:.1f}s ({elapsed/len(all_samples)*1000:.1f}ms per sample)")
print(f"  Saved to: {cache_dir / 'audioset_20k_cache.pt'}")

# Calculate sizes
cache_file = cache_dir / "audioset_20k_cache.pt"
size_mb = cache_file.stat().st_size / (1024 * 1024)
print(f"  Cache size: {size_mb:.1f} MB")
EOF

echo ""
echo "Teacher caching complete!"
date
