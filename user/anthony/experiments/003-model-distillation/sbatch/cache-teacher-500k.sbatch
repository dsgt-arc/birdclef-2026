#!/bin/bash
#SBATCH --job-name=cache-teacher-500k
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t360 -qembers
#SBATCH --output=logs/cache-teacher-500k-%j.out
set -eu

# Usage: sbatch cache-teacher-500k.sbatch
# Can be rerun - skips already cached shards

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Caching Teacher Outputs for AudioSet-500k"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Start GPU monitoring in background
(
    while true; do
        echo ""
        echo "=== GPU Status $(date '+%H:%M:%S') ==="
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU: %s%% | Mem: %s%% (%sMiB/%sMiB) | Temp: %s°C\n", $1, $2, $3, $4, $5}'
        sleep 60
    done
) &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

cd $repo_dir

uv run python << 'EOF'
"""Cache teacher outputs for AudioSet-500k.

Produces one safetensors file per input tar shard.
Idempotent - skips already cached shards.
"""

import io
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from tqdm import tqdm

import sys
sys.path.insert(0, "vendor/unilm/beats")

from birdclef.distillation import load_teacher_and_tokenizer

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
tokenizer_path = scratch_dir / "checkpoints/beats/Tokenizers_iter3.pt"
static_embeddings_path = scratch_dir / "outputs/exp002-benchmarks/static_embeddings/static_embeddings.npy"
audioset_dir = scratch_dir / "data/audioset-wds/500k"
cache_dir = scratch_dir / "outputs/exp003-distillation/teacher_cache_500k"
cache_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
target_length = 160000  # 10s at 16kHz
batch_size = 32

print("=" * 60)
print("CACHING TEACHER OUTPUTS FOR AUDIOSET-500K")
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

# Find all tar shards
tar_files = sorted(audioset_dir.glob("train/*.tar"))
print(f"\nFound {len(tar_files)} tar shards")

# Check existing cache
existing = set(f.stem for f in cache_dir.glob("*.safetensors"))
to_process = [f for f in tar_files if f.stem not in existing]
print(f"Already cached: {len(existing)} shards")
print(f"To process: {len(to_process)} shards")

if not to_process:
    print("\nAll shards already cached!")
    sys.exit(0)


def load_audio_from_tar(tar: tarfile.TarFile, member_name: str) -> torch.Tensor:
    """Load and preprocess audio from open tar file."""
    f = tar.extractfile(member_name)
    audio_data, sr = sf.read(io.BytesIO(f.read()))

    waveform = torch.from_numpy(audio_data.astype(np.float32))
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=-1)

    # Pad or trim to target length
    if len(waveform) < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    return waveform


def process_shard(tar_path: Path) -> dict[str, torch.Tensor]:
    """Process all samples in a tar shard, return tensors."""
    shard_teacher_embeds = []
    shard_token_ids = []
    shard_static_embeds = []

    with tarfile.open(tar_path, 'r') as tar:
        audio_members = [m for m in tar.getmembers()
                        if m.name.endswith(('.wav', '.flac', '.mp3'))]

        for i in range(0, len(audio_members), batch_size):
            batch_members = audio_members[i:i + batch_size]

            waveforms = torch.stack([
                load_audio_from_tar(tar, m.name) for m in batch_members
            ])
            waveforms = waveforms.to(device)

            with torch.no_grad():
                teacher_embeds, _ = teacher.extract_features(waveforms, padding_mask=None)
                token_ids = tokenizer.extract_labels(waveforms)
                if token_ids.dim() == 1:
                    seq_len = token_ids.shape[0] // waveforms.shape[0]
                    token_ids = token_ids.view(waveforms.shape[0], seq_len)
                static_embeds = static_embeddings[token_ids.cpu()]

            shard_teacher_embeds.append(teacher_embeds.cpu().half())
            shard_token_ids.append(token_ids.cpu().short())
            shard_static_embeds.append(static_embeds.half())

    return {
        'teacher_embeds': torch.cat(shard_teacher_embeds, dim=0),
        'token_ids': torch.cat(shard_token_ids, dim=0),
        'static_embeds': torch.cat(shard_static_embeds, dim=0),
    }


# Process shards
print("\nProcessing shards...")
start_time = time.time()
total_samples = 0
processed = 0

for tar_path in tqdm(to_process, desc="Shards"):
    shard_name = tar_path.stem
    output_path = cache_dir / f"{shard_name}.safetensors"
    temp_path = cache_dir / f"{shard_name}.safetensors.tmp"

    # Clean up any leftover temp file
    if temp_path.exists():
        temp_path.unlink()

    try:
        tensors = process_shard(tar_path)
        n_samples = tensors['teacher_embeds'].shape[0]
        total_samples += n_samples

        save_file(tensors, str(temp_path))
        temp_path.rename(output_path)
        processed += 1

        tqdm.write(f"  {shard_name}: {n_samples} samples")
    except Exception as e:
        tqdm.write(f"  ERROR {shard_name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        continue

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("CACHING PROGRESS")
print("=" * 60)
print(f"  Processed this run: {processed} shards ({total_samples} samples)")
print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
if total_samples > 0:
    print(f"  Speed: {elapsed/total_samples*1000:.1f}ms per sample")

# Report total cache
all_cached = list(cache_dir.glob("*.safetensors"))
total_size = sum(f.stat().st_size for f in all_cached)
print(f"\nTotal cache: {len(all_cached)} shards, {total_size / (1024**3):.2f} GB")
print(f"Remaining: {len(tar_files) - len(all_cached)} shards")
EOF

echo ""
echo "Teacher caching progress saved!"
date
