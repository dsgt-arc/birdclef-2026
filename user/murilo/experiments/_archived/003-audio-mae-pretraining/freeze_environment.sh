#!/bin/bash
# Freeze exact dependencies for Experiment 003 reproducibility
# Run this when experiment is complete and ready to publish

echo "==========================================="
echo "Freezing dependencies for Experiment 003"
echo "==========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No virtual environment activated!"
    echo "Please activate the environment used for Experiment 003:"
    echo "  source /tmp/birdclef-*/. venv/bin/activate"
    exit 1
fi

echo "Virtual environment: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Create frozen requirements
echo "Creating frozen requirements..."
pip freeze > requirements-frozen.txt

echo "✓ Frozen to requirements-frozen.txt"
echo ""

# Create a clean version (only direct dependencies with versions)
echo "Creating clean requirements (direct dependencies only)..."
cat > requirements-exp003.txt << 'EOF'
# Experiment 003: Audio-MAE Pre-training
# Frozen on: FROZEN_DATE
# Python: PYTHON_VERSION
#
# Core dependencies for self-supervised Audio-MAE pre-training on BirdSet
# This file captures the exact versions used for the published results.

# ============================================================================
# Core ML/DL
# ============================================================================
torch==2.10.0
torchaudio==2.10.0
torchvision==0.20.0
pytorch-lightning==2.6.1
torchmetrics==1.6.2
transformers==4.48.1
numpy==2.2.2

# ============================================================================
# Vision Transformers (from Experiment 002)
# ============================================================================
timm==1.0.24                 # Vision Transformer models from HuggingFace

# ============================================================================
# Tensor Operations (NEW in Experiment 003)
# ============================================================================
einops==0.8.2                # Rearrange, reduce, repeat for MAE patchify/unpatchify

# ============================================================================
# Logging and Monitoring (from Experiment 002)
# ============================================================================
wandb==0.24.1                # Weights & Biases for experiment tracking

# ============================================================================
# Data handling
# ============================================================================
datasets==3.6.0              # BirdSet data loading
pyarrow==19.0.0
polars==1.23.0
pandas==2.2.3

# ============================================================================
# Audio processing
# ============================================================================
librosa==0.10.2.post1
resampy==0.4.3
soundfile==0.12.1
audioread==3.0.1

# ============================================================================
# Bioacoustics models (from Experiment 001)
# ============================================================================
# NOTE: bioacoustics-model-zoo installed from git:
#   git+https://github.com/kitzeslab/bioacoustics-model-zoo.git@main
opensoundscape==0.10.4
ai-edge-litert==1.1.1

# ============================================================================
# Machine learning
# ============================================================================
scikit-learn==1.6.1

# ============================================================================
# Utilities
# ============================================================================
tqdm==4.67.1
pyyaml==6.0.2
typer==0.15.2
rich==13.9.4

# ============================================================================
# Testing
# ============================================================================
pytest==8.3.4
pytest-cov==6.0.0

# ============================================================================
# Visualization
# ============================================================================
matplotlib==3.10.0
seaborn==0.13.2

# ============================================================================
# Notes
# ============================================================================
#
# Progressive dependency additions:
#   Experiment 001: Base dependencies (bioacoustics-model-zoo, BirdSet)
#   Experiment 002: + timm, wandb (supervised ViT)
#   Experiment 003: + einops (Audio-MAE self-supervised pre-training)
#
# Key difference from Experiment 002:
#   + einops==0.8.2 (for efficient tensor rearrangements in MAE)
#
# To install this environment:
#   uv venv .venv --python 3.11
#   source .venv/bin/activate
#   uv pip install -r requirements-exp003.txt
#   uv pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo.git@main
#
# To reproduce exact environment:
#   uv pip install -r requirements-frozen.txt
EOF

# Replace placeholders with actual values
sed -i "s/FROZEN_DATE/$(date +%Y-%m-%d)/" requirements-exp003.txt
sed -i "s/PYTHON_VERSION/$(python --version 2>&1)/" requirements-exp003.txt

echo "✓ Created requirements-exp003.txt (clean, direct dependencies)"
echo ""

# Summary
echo "==========================================="
echo "✓ Freeze complete!"
echo "==========================================="
echo ""
echo "Created files:"
echo "  1. requirements-frozen.txt   - Full pip freeze (all packages)"
echo "  2. requirements-exp003.txt   - Clean direct dependencies only"
echo ""
echo "To reproduce this environment:"
echo "  uv venv .venv --python 3.11"
echo "  source .venv/bin/activate"
echo "  uv pip install -r requirements-exp003.txt"
echo ""
echo "Or for exact reproduction:"
echo "  uv pip install -r requirements-frozen.txt"
echo ""
echo "Next steps:"
echo "  1. Review requirements-exp003.txt"
echo "  2. Git commit both files:"
echo "     git add requirements-*.txt"
echo "     git commit -m 'Freeze Experiment 003 dependencies'"
echo ""
