#!/bin/bash
# Freeze exact dependencies for Experiment 001 reproducibility
# Run this when experiment is complete and ready to publish

echo "==========================================="
echo "Freezing dependencies for Experiment 001"
echo "==========================================="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No virtual environment activated!"
    echo "Please activate the environment used for Experiment 001:"
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
cat > requirements-exp001.txt << 'EOF'
# Experiment 001: BirdSet Bioacoustics Benchmark
# Frozen on: $(date +%Y-%m-%d)
# Python: $(python --version)
#
# Core dependencies for Phase 2A (Direct Inference) and Phase 2B (Transfer Learning)
# This file captures the exact versions used for the published results.

# Core ML/DL
torch==2.10.0
torchaudio==2.10.0
torchvision==0.20.0
pytorch-lightning==2.6.1
torchmetrics==1.6.2
transformers==4.48.1
tensorflow==2.17.0
tensorflow-hub==0.16.1
numpy==2.2.2

# Data handling
datasets==3.6.0
pyarrow==19.0.0
polars==1.23.0
pandas==2.2.3

# Audio processing
librosa==0.10.2.post1
resampy==0.4.3
soundfile==0.12.1
audioread==3.0.1

# Bioacoustics models
# bioacoustics-model-zoo @ git+https://github.com/kitzeslab/bioacoustics-model-zoo.git@main
opensoundscape==0.10.4
ai-edge-litert==1.1.1

# Machine learning
scikit-learn==1.6.1

# Utilities
tqdm==4.67.1
pyyaml==6.0.2
typer==0.15.2
rich==13.9.4

# Testing
pytest==8.3.4
pytest-cov==6.0.0

# Visualization
matplotlib==3.10.0
seaborn==0.13.2
EOF

# Replace placeholders with actual values
sed -i "s/\$(date +%Y-%m-%d)/$(date +%Y-%m-%d)/" requirements-exp001.txt
sed -i "s/\$(python --version)/$(python --version 2>&1)/" requirements-exp001.txt

echo "✓ Created requirements-exp001.txt (clean, direct dependencies)"
echo ""

# Summary
echo "==========================================="
echo "✓ Freeze complete!"
echo "==========================================="
echo ""
echo "Created files:"
echo "  1. requirements-frozen.txt   - Full pip freeze (all packages)"
echo "  2. requirements-exp001.txt   - Clean direct dependencies only"
echo ""
echo "To reproduce this environment:"
echo "  uv venv .venv"
echo "  source .venv/bin/activate"
echo "  uv pip install -r requirements-exp001.txt"
echo ""
echo "Or for exact reproduction:"
echo "  uv pip install -r requirements-frozen.txt"
echo ""
echo "Next steps:"
echo "  1. Review requirements-exp001.txt"
echo "  2. Git commit both files:"
echo "     git add requirements-*.txt"
echo "     git commit -m 'Freeze Experiment 001 dependencies'"
echo ""
