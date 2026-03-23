"""Debug the seg_keys alignment issue."""
import numpy as np
import pandas as pd
from pathlib import Path

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
EDA = Path(__file__).resolve().parents[2] / "100-eda/artifacts"

# Load ground truth
labels = pd.read_parquet(EDA / "soundscape_labels.parquet")
gt_filenames = sorted(labels["filename"].unique())
print(f"Ground truth: {len(gt_filenames)} unique filenames")
print(f"  samples: {gt_filenames[:3]}")

# Load npz seg_keys
data = np.load(
    SHARED / "inference/birdclef-2026/protoclr-2026-soundscape/labeled.npz",
    allow_pickle=True,
)
seg_keys = data["seg_keys"]
print(f"\nseg_keys shape: {seg_keys.shape}, dtype: {seg_keys.dtype}")
print(f"  first 5 rows:")
for i in range(min(5, len(seg_keys))):
    print(f"    {seg_keys[i]}")

# Check unique filenames in seg_keys
npz_filenames = sorted(set(str(seg_keys[i, 0]) for i in range(len(seg_keys))))
print(f"\nnpz: {len(npz_filenames)} unique filenames")
print(f"  samples: {npz_filenames[:3]}")

# Check overlap
gt_set = set(gt_filenames)
npz_set = set(npz_filenames)
print(f"\nOverlap: {len(gt_set & npz_set)}")
print(f"In GT only: {len(gt_set - npz_set)}")
print(f"In NPZ only: {len(npz_set - gt_set)}")

if len(gt_set & npz_set) == 0:
    # Try stripping extensions or paths
    gt_stems = {Path(f).stem for f in gt_filenames}
    npz_stems = {Path(f).stem for f in npz_filenames}
    print(f"\nStem overlap: {len(gt_stems & npz_stems)}")
    print(f"  GT stems: {sorted(gt_stems)[:3]}")
    print(f"  NPZ stems: {sorted(npz_stems)[:3]}")

# Check gt columns
gt_windows = labels[["filename", "start_sec", "end_sec"]].drop_duplicates()
print(f"\nGT windows: {len(gt_windows)}")
print(f"GT columns: {list(labels.columns[:10])}")

# Check window_ids range
window_ids = [seg_keys[i, 1] for i in range(len(seg_keys))]
print(f"\nWindow IDs: min={min(window_ids)}, max={max(window_ids)}")
print(f"  unique: {sorted(set(window_ids))[:15]}")
