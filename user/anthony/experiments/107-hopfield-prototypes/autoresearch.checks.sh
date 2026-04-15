#!/bin/bash
set -euo pipefail

source "$HOME/scratch/birdclef-2026/.venv-107/bin/activate"
pytest -q \
  tests/test_model_shapes.py \
  tests/test_cross_attention.py \
  tests/test_gradient_routing.py \
  tests/test_affinity.py >/dev/null
