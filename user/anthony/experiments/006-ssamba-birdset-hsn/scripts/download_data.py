"""Pre-download BirdSet HSN dataset.

Run on login node: uv run python scripts/download_data.py
Caches to HF_HOME (~/scratch/.cache/huggingface by default).
"""

from datasets import load_dataset

DATASET_CODE = "HSN"

print(f"Downloading BirdSet {DATASET_CODE} train split...")
train = load_dataset(
    "DBD-research-group/BirdSet",
    DATASET_CODE,
    split="train",
    trust_remote_code=True,
)
print(f"  Train: {len(train)} samples")

print(f"Downloading BirdSet {DATASET_CODE} test_5s split...")
test = load_dataset(
    "DBD-research-group/BirdSet",
    DATASET_CODE,
    split="test_5s",
    trust_remote_code=True,
)
print(f"  Test_5s: {len(test)} samples")

print("Done. Data cached for offline use.")
