#!/bin/bash
#SBATCH --job-name=cache-teacher-arr
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=8 --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -t120 -qembers
#SBATCH --output=logs/cache-teacher-array-%A_%a.out
#SBATCH --array=0-14

# Usage: sbatch cache-teacher-array.sbatch
#
# Processes AudioSet-500k shards in parallel using SLURM array jobs.
# Each array task handles ~10 shards (145 total / 15 tasks = ~10 shards each).
# Idempotent: skips already cached shards.
#
# To process all 145 shards with 15 parallel jobs:
#   sbatch --array=0-14 cache-teacher-array.sbatch
#
# To rerun failed tasks (e.g., tasks 3 and 7):
#   sbatch --array=3,7 cache-teacher-array.sbatch

set -eu

cd ${SLURM_SUBMIT_DIR:-$(dirname "$0")}

# Configuration
SHARDS_PER_TASK=10
TOTAL_SHARDS=145
DATASET_VERSION=${DATASET_VERSION:-500k}

echo "=============================================="
echo "Teacher Caching Array Job"
echo "=============================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Shards per task: ${SHARDS_PER_TASK}"
echo "Dataset: ${DATASET_VERSION}"
hostname
date
nvidia-smi

scratch_dir=$(realpath ~/scratch/birdclef-2026)
repo_dir=$(realpath ~/arc/birdclef-2026)

export UV_PROJECT_ENVIRONMENT=$scratch_dir/.venv
export HF_HOME=$scratch_dir/.cache/huggingface

# Copy tar files for this task to local NVMe for faster IO
audioset_dir=$scratch_dir/data/audioset-wds/${DATASET_VERSION}/train
cache_dir=$scratch_dir/outputs/exp003-distillation/teacher_cache_${DATASET_VERSION}
local_tar_dir=${TMPDIR:-/tmp}/audioset_shards_${SLURM_ARRAY_TASK_ID}

# Calculate shard range for this task
START_SHARD=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
END_SHARD=$((START_SHARD + SHARDS_PER_TASK - 1))
if [ $END_SHARD -ge $TOTAL_SHARDS ]; then
    END_SHARD=$((TOTAL_SHARDS - 1))
fi

echo ""
echo "This task processes shards ${START_SHARD} to ${END_SHARD}"

# Copy relevant shards to local NVMe
echo ""
echo "Copying shards to local NVMe..."
mkdir -p $local_tar_dir

for i in $(seq $START_SHARD $END_SHARD); do
    shard_name=$(printf "shard-%05d.tar" $i)
    if [ -f "$audioset_dir/$shard_name" ]; then
        # Check if already cached
        cache_file="$cache_dir/$(printf "shard-%05d.safetensors" $i)"
        if [ -f "$cache_file" ]; then
            echo "  Skipping $shard_name (already cached)"
        else
            echo "  Copying $shard_name"
            cp "$audioset_dir/$shard_name" "$local_tar_dir/"
        fi
    fi
done

echo ""
echo "Local tar directory:"
ls -lh $local_tar_dir/ 2>/dev/null | head -5 || echo "  (empty - all shards already cached)"

cd $repo_dir

uv run python - "$local_tar_dir" "$cache_dir" "$START_SHARD" "$END_SHARD" << 'EOF'
"""Cache teacher outputs for a range of shards."""

import io
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from tqdm import tqdm

local_tar_dir = Path(sys.argv[1])
cache_dir = Path(sys.argv[2])
start_shard = int(sys.argv[3])
end_shard = int(sys.argv[4])

cache_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "vendor/unilm/beats")
from birdclef.distillation import load_teacher_and_tokenizer

# Paths
scratch_dir = Path.home() / "scratch" / "birdclef-2026"
encoder_path = scratch_dir / "checkpoints/beats/BEATs_iter3.pt"
tokenizer_path = scratch_dir / "checkpoints/beats/Tokenizers_iter3.pt"
static_embeddings_path = scratch_dir / "outputs/exp002-benchmarks/static_embeddings/static_embeddings.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
target_length = 160000
batch_size = 32

print("=" * 60)
print(f"CACHING SHARDS {start_shard} to {end_shard}")
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

# Find tar files to process (from local NVMe)
tar_files = sorted(local_tar_dir.glob("shard-*.tar"))
print(f"\nFound {len(tar_files)} tar files to process")

if len(tar_files) == 0:
    print("All shards already cached - nothing to do!")
    sys.exit(0)


def load_audio_from_tar(tar: tarfile.TarFile, member_name: str) -> torch.Tensor:
    f = tar.extractfile(member_name)
    audio_data, sr = sf.read(io.BytesIO(f.read()))
    waveform = torch.from_numpy(audio_data.astype(np.float32))
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=-1)
    if len(waveform) < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]
    return waveform


def process_shard(tar_path: Path) -> dict[str, torch.Tensor]:
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

for tar_path in tqdm(tar_files, desc="Shards"):
    shard_name = tar_path.stem
    output_path = cache_dir / f"{shard_name}.safetensors"
    temp_path = cache_dir / f"{shard_name}.safetensors.tmp"

    # Skip if already cached (double-check)
    if output_path.exists():
        tqdm.write(f"  {shard_name}: already cached, skipping")
        continue

    if temp_path.exists():
        temp_path.unlink()

    try:
        tensors = process_shard(tar_path)
        n_samples = tensors['teacher_embeds'].shape[0]
        total_samples += n_samples

        save_file(tensors, str(temp_path))
        temp_path.rename(output_path)
        processed += 1

        tqdm.write(f"  {shard_name}: {n_samples} samples cached")
    except Exception as e:
        tqdm.write(f"  ERROR {shard_name}: {e}")
        if temp_path.exists():
            temp_path.unlink()

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("TASK COMPLETE")
print("=" * 60)
print(f"  Processed: {processed} shards ({total_samples} samples)")
print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
if total_samples > 0:
    print(f"  Speed: {elapsed/total_samples*1000:.1f}ms per sample")
    print(f"  Throughput: {total_samples/elapsed:.1f} samples/sec")
EOF

echo ""
echo "Array task ${SLURM_ARRAY_TASK_ID} complete!"
date
