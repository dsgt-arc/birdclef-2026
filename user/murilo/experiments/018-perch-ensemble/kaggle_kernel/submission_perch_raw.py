#!/usr/bin/env python3
"""BirdCLEF 2026 — Raw Perch v2 Baseline (no post-processing).

Minimal kernel matching the kdmitrie/birdclef26-google-perch-starter approach:
  1. Install TF 2.20 from wheel
  2. Load Perch v2 CPU model
  3. For each soundscape: 12 x 5s windows → raw logits → map to 234 species
  4. Submit raw logits (no sigmoid, no probes, no priors, no smoothing)

Expected LB: ~0.825 (matching starter notebook)
"""

import os
import subprocess
import sys
import time

START = time.time()

KAGGLE_USER = "murilogustineli"


def _find_input(slug: str) -> str:
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


# Debug
if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")

# Install TF 2.20 from our wheel dataset
TF_WHEEL_DIR = _find_input("birdclef-tf220-wheel")
if os.path.exists(TF_WHEEL_DIR):
    print(f"Installing TensorFlow 2.20 from {TF_WHEEL_DIR}")
    for whl in sorted(os.listdir(TF_WHEEL_DIR)):
        if whl.endswith(".whl"):
            whl_path = os.path.join(TF_WHEEL_DIR, whl)
            print(f"  Installing {whl}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", whl_path])
else:
    print(f"WARNING: TF wheel not found at {TF_WHEEL_DIR}")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import glob  # noqa: E402
import re  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.experimental.numpy.experimental_enable_numpy_behavior()
print(f"TensorFlow: {tf.__version__}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
PERCH_MODEL_CANDIDATES = [
    "/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
    "/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1",
]

perch_model_dir = None
for candidate in PERCH_MODEL_CANDIDATES:
    if os.path.exists(candidate) and os.path.exists(os.path.join(candidate, "saved_model.pb")):
        perch_model_dir = candidate
        break

if perch_model_dir is None:
    # Walk /kaggle/input to find it
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "saved_model.pb" in files and "perch" in root.lower():
            perch_model_dir = root
            break

assert perch_model_dir is not None, "Perch model not found!"
print(f"Perch model: {perch_model_dir}")

birdclassifier = tf.saved_model.load(perch_model_dir)

# ---------------------------------------------------------------------------
# Species mapping (same approach as starter notebook)
# ---------------------------------------------------------------------------
# Read labels.csv — first line becomes header, so bc_index 0 = first real species
bc_labels = pd.read_csv(os.path.join(perch_model_dir, "assets", "labels.csv"))
bc_labels = (
    bc_labels.reset_index()
    .rename({"inat2024_fsd50k": "scientific_name", "index": "bc_index"}, axis=1)
    .set_index("scientific_name")
)

# Load competition taxonomy
COMP_DIR = (
    "/kaggle/input/competitions/birdclef-2026"
    if os.path.exists("/kaggle/input/competitions/birdclef-2026")
    else "/kaggle/input/birdclef-2026"
)
taxonomy = pd.read_csv(os.path.join(COMP_DIR, "taxonomy.csv"))

# Map BirdCLEF species to Perch indices
mapping = taxonomy.join(bc_labels, on="scientific_name", how="left")
mapping.bc_index = mapping.bc_index.fillna(len(bc_labels)).astype(int)
mapping = mapping[["primary_label", "bc_index"]].set_index("primary_label")

# Get species order from sample_submission
sample_sub = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))
primary_labels = sample_sub.columns[1:].to_list()
birdclassifier_indices = [int(mapping.loc[pl].iloc[0]) for pl in primary_labels]

total_predicted = len(set(birdclassifier_indices) - {len(bc_labels)})
print(f"Can predict {total_predicted} out of {len(primary_labels)} species")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
SR = 32_000
TERMINATE_TIME = START + 5300  # Safety cutoff


def get_soundscapes():
    test_dir = os.path.join(COMP_DIR, "test_soundscapes")
    oggs = sorted(glob.glob(os.path.join(test_dir, "*.ogg")))
    if not oggs:
        train_dir = os.path.join(COMP_DIR, "train_soundscapes")
        oggs = sorted(glob.glob(os.path.join(train_dir, "*.ogg")))[:2]
        print(f"No test soundscapes, using {len(oggs)} train files")
    return oggs


def process_file(ogg_path):
    ss_id = re.search(r"/([^/]+)\.ogg$", ogg_path).group(1)
    row_ids = [f"{ss_id}_{n}" for n in range(5, 65, 5)]

    if time.time() > TERMINATE_TIME:
        return row_ids, -1000 * np.ones((12, len(primary_labels)))

    data, _ = librosa.load(ogg_path, sr=SR)

    # Ensure exactly 60s
    expected = 60 * SR
    if len(data) < expected:
        data = np.pad(data, (0, expected - len(data)))
    elif len(data) > expected:
        data = data[:expected]

    # Run Perch: 12 x 5s windows
    waveform = data.reshape((-1, 5 * SR))
    model_outputs = birdclassifier.signatures["serving_default"](inputs=waveform)["label"]

    # Pad with one zero column for unmapped species (index = len(bc_labels))
    model_outputs = tf.pad(model_outputs, tf.constant([[0, 0], [0, 1]]))

    # Map to BirdCLEF species — raw logits, no sigmoid
    result = model_outputs[:, birdclassifier_indices].numpy()

    return row_ids, result


# Process all files with threading (like starter notebook)
oggs = get_soundscapes()
print(f"\nProcessing {len(oggs)} soundscapes...")

row_ids = []
results = []

with ThreadPoolExecutor(max_workers=4) as executor:
    for ogg_row_ids, ogg_result in executor.map(process_file, oggs):
        row_ids += ogg_row_ids
        results.append(ogg_result)

# Build submission
submission = pd.DataFrame(np.concatenate(results), columns=primary_labels)
submission["row_id"] = row_ids
submission = submission[["row_id"] + primary_labels]

output_path = "/kaggle/working/submission.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
submission.to_csv(output_path, index=False)

elapsed = time.time() - START
print(f"\nSubmission saved: {output_path} ({len(submission)} rows)")
print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
