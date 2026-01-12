#!/bin/bash
#SBATCH --job-name=cache-teacher-003
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/cache-teacher-%j.out
set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

echo "Caching Teacher Outputs (Sharded) - Experiment 003"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Start GPU monitoring in background (every 30s)
(
    while true; do
        echo ""
        echo "=== GPU Status $(date '+%H:%M:%S') ==="
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
            awk -F', ' '{printf "GPU: %s%% | Mem: %s%% (%sMiB/%sMiB) | Temp: %s°C\n", $1, $2, $3, $4, $5}'
        sleep 30
    done
) &
GPU_MONITOR_PID=$!
trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT

cd $repo_dir

uv run python << 'EOF'
"""Cache teacher outputs for fast distillation training.

Produces one safetensors file per input tar shard for:
- Streaming/parallel loading during training
- Idempotent reruns (skips existing shards)
- Fault tolerance (partial progress preserved)
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
audioset_dir = scratch_dir / "data/audioset-wds/20k"
cache_dir = scratch_dir / "outputs/exp003-distillation/teacher_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
target_length = 160000  # 10s at 16kHz
batch_size = 32

print("=" * 60)
print("CACHING TEACHER OUTPUTS (SHARDED)")
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
        # Get audio members
        audio_members = [m for m in tar.getmembers()
                        if m.name.endswith(('.wav', '.flac', '.mp3'))]

        # Process in batches
        for i in range(0, len(audio_members), batch_size):
            batch_members = audio_members[i:i + batch_size]

            # Load batch of waveforms
            waveforms = torch.stack([
                load_audio_from_tar(tar, m.name) for m in batch_members
            ])
            waveforms = waveforms.to(device)

            with torch.no_grad():
                # Get teacher embeddings
                teacher_embeds, _ = teacher.extract_features(waveforms, padding_mask=None)

                # Get token IDs and reshape if needed
                token_ids = tokenizer.extract_labels(waveforms)
                if token_ids.dim() == 1:
                    seq_len = token_ids.shape[0] // waveforms.shape[0]
                    token_ids = token_ids.view(waveforms.shape[0], seq_len)

                # Get static embeddings via lookup
                static_embeds = static_embeddings[token_ids.cpu()]

            # Store as FP16/int16 to save space
            shard_teacher_embeds.append(teacher_embeds.cpu().half())
            shard_token_ids.append(token_ids.cpu().short())
            shard_static_embeds.append(static_embeds.half())

    # Concatenate all batches in this shard
    return {
        'teacher_embeds': torch.cat(shard_teacher_embeds, dim=0),
        'token_ids': torch.cat(shard_token_ids, dim=0),
        'static_embeds': torch.cat(shard_static_embeds, dim=0),
    }


# Process each shard
print("\nProcessing shards...")
start_time = time.time()
total_samples = 0
skipped = 0
processed = 0

for tar_path in tqdm(tar_files, desc="Shards"):
    # Output path mirrors input structure
    shard_name = tar_path.stem  # e.g., "shard-000000"
    output_path = cache_dir / f"{shard_name}.safetensors"
    temp_path = cache_dir / f"{shard_name}.safetensors.tmp"

    # Skip if already exists (idempotent)
    if output_path.exists():
        skipped += 1
        continue

    # Clean up any leftover temp file from interrupted run
    if temp_path.exists():
        temp_path.unlink()

    # Process shard
    tensors = process_shard(tar_path)
    n_samples = tensors['teacher_embeds'].shape[0]
    total_samples += n_samples

    # Atomic write: save to temp, then rename
    # If interrupted, temp file won't block rerun
    save_file(tensors, temp_path)
    temp_path.rename(output_path)  # atomic on POSIX
    processed += 1

    tqdm.write(f"  {shard_name}: {n_samples} samples -> {output_path.name}")

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("CACHING COMPLETE")
print("=" * 60)
print(f"  Processed: {processed} shards ({total_samples} samples)")
print(f"  Skipped:   {skipped} shards (already cached)")
print(f"  Time:      {elapsed:.1f}s")
if total_samples > 0:
    print(f"  Speed:     {elapsed/total_samples*1000:.1f}ms per sample")

# Report cache size
cache_files = list(cache_dir.glob("*.safetensors"))
total_size = sum(f.stat().st_size for f in cache_files)
print(f"\nCache directory: {cache_dir}")
print(f"  Files: {len(cache_files)}")
print(f"  Total size: {total_size / (1024**3):.2f} GB")
EOF

echo ""
echo "Teacher caching complete!"
date
