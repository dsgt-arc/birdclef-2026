#!/bin/bash
#SBATCH --job-name=010-verify-2026
#SBATCH --account=paceship-clef2026_birdclef
#SBATCH -N1 -n1 --cpus-per-task=2 --mem-per-cpu=4G
#SBATCH -t10 -qinferno
#SBATCH --output=logs/verify-2026-%j.out

set -eu

output_root=~/scratch/birdclef-2026/outputs/exp010-2026/labels

echo "=== Output directory structure ==="
find $output_root -type d | sort

echo ""
echo "=== Parquet file counts ==="
for model in birdnet perch indices; do
    for split in train_audio train_soundscapes; do
        count=$(find $output_root/$model/$split -name "*.parquet" 2>/dev/null | wc -l)
        echo "  $model/$split: $count parquet files"
    done
done

echo ""
echo "=== Total size ==="
du -sh $output_root

echo ""
echo "=== Sample parquet check ==="
venv=~/scratch/birdclef-2026/.venv-010
$venv/bin/python3 -c "
import pandas as pd
import os

root = os.path.expanduser('$output_root')
for model in ['birdnet', 'perch', 'indices']:
    for split in ['train_audio', 'train_soundscapes']:
        parts_dir = os.path.join(root, model, split, 'parts')
        if not os.path.exists(parts_dir):
            parts_dir = os.path.join(root, model, split)
        parquets = sorted([f for f in os.listdir(parts_dir) if f.endswith('.parquet')]) if os.path.exists(parts_dir) else []
        if not parquets:
            # check subdirs
            for subdir in ['embed', 'logits']:
                sub = os.path.join(parts_dir, subdir)
                if os.path.exists(sub):
                    parquets = sorted([f for f in os.listdir(sub) if f.endswith('.parquet')])
                    if parquets:
                        parts_dir = sub
                        break
        if parquets:
            first = os.path.join(parts_dir, parquets[0])
            df = pd.read_parquet(first)
            print(f'{model}/{split}: {len(parquets)} files, first has {len(df)} rows, {len(df.columns)} cols')
            print(f'  columns: {list(df.columns[:5])}...')
            print(f'  dtypes: {dict(list(df.dtypes.items())[:3])}')
        else:
            print(f'{model}/{split}: NO PARQUET FILES FOUND')
"
echo ""
echo "=== Done ==="
