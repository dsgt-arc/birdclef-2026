#!/bin/bash
#SBATCH --job-name=103-ssl-batchtest
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=4 --mem-per-cpu=10G
#SBATCH --gres=gpu:l40s:1
#SBATCH -p gpu-l40s
#SBATCH -t60 -qinferno
#SBATCH --output=logs/pretrain-dev-l40s-%j.out

# L40S bf16 batch size sweep:
#   Tests batch sizes 8, 16, 24, 32, 48 with fast_dev_run (1 train + 1 val batch)
#   to find max batch size before OOM, then overfit test at the winning size.

set -eu

repo_dir=$(realpath ~/arc/birdclef-2026)
exp_dir=$repo_dir/user/anthony/experiments/103-ssamba-ssl

# Persistent venv on scratch
source "$exp_dir/sbatch/_setup-venv.sh"

CONFIG_TEMPLATE="$exp_dir/configs/pretrain-103a-a100.yaml"
TMP_CONFIG="/tmp/pretrain-batchtest-$SLURM_JOB_ID.yaml"

echo "=== L40S bf16 batch size sweep ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

best_bs=0
for bs in 8 16 24 32 48; do
    echo "--- Testing batch_size=$bs ---"
    sed "s/__BATCH_SIZE__/$bs/" "$CONFIG_TEMPLATE" > "$TMP_CONFIG"

    if birdclef-ssl-pretrain --config "$TMP_CONFIG" --fast-dev-run 2>&1; then
        echo "  batch_size=$bs: OK"
        best_bs=$bs
        # Print peak GPU memory
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    else
        echo "  batch_size=$bs: FAILED (likely OOM)"
        break
    fi
    echo ""
done

echo ""
echo "=== Best batch size: $best_bs ==="

if [ "$best_bs" -gt 0 ]; then
    echo ""
    echo "=== Overfit test with batch_size=$best_bs ==="
    sed "s/__BATCH_SIZE__/$best_bs/" "$CONFIG_TEMPLATE" > "$TMP_CONFIG"
    birdclef-ssl-pretrain --config "$TMP_CONFIG" --overfit-batches 4 \
        --ckpt-dir logs/overfit-l40s/checkpoints
fi
