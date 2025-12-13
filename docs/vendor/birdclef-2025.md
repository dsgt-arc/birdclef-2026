---
name: "birdclef-2025"
source: "https://github.com/dsgt-kaggle-clef/birdclef-2025"
license: "Unspecified (internal project)"
category: "utility"
pipeline_stage:
  - "tokenization"
  - "embedding"
  - "inference"
relevance: "Reference implementation of discrete token + Word2Vec pipeline for BirdCLEF competition"
status: "active"
---

# birdclef-2025

## Overview

This repository contains the BirdCLEF 2025 competition solution that demonstrates discrete audio tokenization combined with Word2Vec-style embeddings for efficient bird sound classification. The approach achieves competitive performance with significantly reduced inference cost, making it suitable for Kaggle's CPU-constrained evaluation environment.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `mel2vec.workflow` | Main tokenization and embedding pipeline | `birdclef/mel2vec/workflow.py` |
| `mel2vec.student_teacher` | Knowledge distillation for embeddings | `birdclef/mel2vec/student_teacher.py` |
| `mfcc` | MFCC feature extraction | `birdclef/mfcc/` |
| `classifier` | Classification head training | `birdclef/classifier/` |
| `infer` | Inference pipeline for Kaggle | `birdclef/infer/` |
| `config` | Centralized configuration | `birdclef/config.py` |

## Relevance to Pipeline

- **Tokenization**: Uses k-means clustering on MFCCs to create discrete audio tokens; this is our Phase 1 baseline to improve upon
- **Embedding**: Word2Vec-style training on token sequences to learn semantic embeddings
- **Inference**: Demonstrates efficient CPU inference suitable for Kaggle constraints

## Usage

```bash
# Install with uv
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run CLI
birdclef --help

# Run tests
pytest -vv tests
```

## Dependencies

- Python 3.x
- PyTorch
- gensim (for Word2Vec)
- librosa (for audio processing)
- scikit-learn (for k-means)
- Luigi (for workflow orchestration)

## Notes

- Core approach: MFCC -> k-means tokens -> Word2Vec embeddings -> classifier
- Pipeline proposal aims to improve each stage:
  1. Better tokenization (learned codebooks vs k-means)
  2. Transformer instead of Word2Vec for contextual representations
  3. model2vec-style distillation for static embeddings
- PACE cluster scripts available in `scripts/` and `sbatch/`
- Notebooks in `notebooks/` contain experiments and analysis
