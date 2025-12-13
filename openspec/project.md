# Project Context

## Purpose

BirdCLEF 2026 competition entry focused on efficient bird species identification from audio recordings. This repository is a research/meta project exploring discrete tokenization and self-supervised learning approaches for bioacoustics, building on the birdclef-2025 work.

**Goals:**
1. Develop birdcall-specific discrete audio tokens
2. Train self-supervised transformers on unlabeled soundscape data
3. Distill to static embeddings via model2vec methodology
4. Build efficient linear classifiers that maximize Kaggle CPU budget

## Tech Stack

- **Python 3.11+** with `uv` for package management
- **PyTorch / Lightning** for model training
- **faiss** for efficient nearest neighbor and k-means
- **Polars** for data processing (preferred over pandas)
- **Luigi** for workflow orchestration (inherited from birdclef-2025)
- **numpy** for numerical operations

## Project Conventions

### Code Style
- Use `ruff` for formatting and linting
- Type hints required for public functions
- Docstrings for modules and classes

### Architecture Patterns
- Modular pipeline: tokenization → representation learning → embedding → inference
- Luigi tasks for reproducible workflows
- CLI via `typer` for entry points

### Testing Strategy
- `pytest` for unit tests
- Integration tests on small data subsets
- Benchmark tests for inference timing

### Git Workflow
- Main branch for stable code
- Feature branches for experiments
- Commit messages: imperative mood, reference tasks

## Domain Context

- **BirdCLEF**: Annual Kaggle competition for bird species identification
- **Soundscape data**: Long field recordings (often 60+ seconds) with multiple overlapping species
- **Training data**: Labeled clips of individual species calls
- **Evaluation**: Macro F1 or similar multi-label metric
- **Constraint**: CPU-only inference on Kaggle (9-hour limit typical)

## Important Constraints

1. **Inference must be CPU-only** - Kaggle submission environment
2. **Time budget** - Must process full test set within allowed time
3. **Memory limit** - Typically 16GB RAM on Kaggle
4. **No internet** - All models/data must be packaged in submission

## External Dependencies

- **vendor/birdclef-2025**: Previous year's codebase (mel2vec, tokenization)
- **vendor/BirdSet**: Bioacoustics dataset toolkit
- **vendor/bioacoustics-model-zoo**: Pre-trained models (BirdNET, Perch, etc.)
- **vendor/opensoundscape**: Audio processing utilities
