"""
BirdCLEF 2026 Inference Template for Kaggle

This script runs inference offline using pre-uploaded datasets:
- birdclef-2026-deps: Python wheel files for offline pip install
- birdclef-2026-code: Project wheel files (birdclef_adrian, birdclef_anthony)
- birdclef-2026-weights: Model checkpoints

Usage on Kaggle:
1. Create a new notebook
2. Attach the three datasets above
3. Copy this script content and run
"""

import os
from pathlib import Path

# ============================================================================
# Setup (Offline Installation)
# ============================================================================

# Kaggle input paths
INPUT_DIR = Path("/kaggle/input")
DEPS_DIR = INPUT_DIR / "birdclef-2026-deps"
CODE_DIR = INPUT_DIR / "birdclef-2026-code"
WEIGHTS_DIR = INPUT_DIR / "birdclef-2026-weights"
TEST_DIR = INPUT_DIR / "birdclef-2025" / "test_soundscapes"  # Update for 2026

# Install dependencies offline
print("Installing dependencies offline...")
os.system(
    f"pip install --no-index --find-links={DEPS_DIR} "
    f"-r {DEPS_DIR}/requirements.txt --quiet"
)

# Install project wheels (birdclef_adrian, birdclef_anthony)
print("Installing project packages...")
for wheel_file in CODE_DIR.glob("*.whl"):
    print(f"  Installing: {wheel_file.name}")
    os.system(f"pip install --no-index {wheel_file} --quiet")

# ============================================================================
# Imports (after offline install)
# ============================================================================

import pandas as pd
import torch
import torchaudio

# Import your inference code
# from birdclef_adrian.inference import predict
# from birdclef_anthony.models import load_model

# ============================================================================
# Model Loading
# ============================================================================

print("Loading model...")

# Find model checkpoint
MODEL_PATH = list(WEIGHTS_DIR.glob("*.pt"))[0]
print(f"  Model: {MODEL_PATH}")

# Load model
# model = load_model(MODEL_PATH)
# model.eval()

# Placeholder for actual model loading
model = None
print("  WARNING: Using placeholder model (update with real inference code)")

# ============================================================================
# Inference
# ============================================================================

def run_inference(model, audio_dir: Path) -> pd.DataFrame:
    """Run inference on test soundscapes.
    
    Args:
        model: Loaded model
        audio_dir: Directory containing test audio files
        
    Returns:
        DataFrame with predictions in submission format
    """
    results = []
    
    audio_files = list(audio_dir.glob("*.ogg"))
    print(f"Processing {len(audio_files)} audio files...")
    
    for audio_path in audio_files:
        filename = audio_path.stem
        
        # TODO: Replace with actual inference
        # audio, sr = torchaudio.load(audio_path)
        # predictions = model(audio)
        
        # Placeholder: Generate dummy predictions
        # Format: row_id = "{filename}_{end_time}", target = "{species1} {species2}"
        for end_time in range(5, 605, 5):  # 5-second windows up to 600s
            row_id = f"{filename}_{end_time}"
            target = "nocall"  # Placeholder
            results.append({"row_id": row_id, "target": target})
    
    return pd.DataFrame(results)


# Run inference
print("\nRunning inference...")
predictions = run_inference(model, TEST_DIR)

# ============================================================================
# Save Submission
# ============================================================================

submission_path = WORK_DIR / "submission.csv"
predictions.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")
print(f"Shape: {predictions.shape}")
print(predictions.head())
