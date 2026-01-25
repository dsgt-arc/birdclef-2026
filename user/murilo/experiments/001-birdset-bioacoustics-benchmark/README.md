# Experiment 001: BirdSet Bioacoustics Model Benchmark

This experiment evaluates pre-trained bioacoustics models on the full BirdSet benchmark (8 datasets) using two evaluation strategies: direct inference for BirdSet-native models and transfer learning (linear probing) for external models. This complements the original BirdSet paper by providing a standardized comparison of how well external, pre-trained models transfer to the BirdSet ecosystems.

**Related Docs**:
- [Proposal](proposal.md) - Hypothesis and methodology
- [Design](design.md) - Technical decisions and architecture
- [Tasks](tasks.md) - Implementation checklist and status

## Quick Start

### 1. Interactive Testing (Phase 1 ✅)

An interactive test script, `test_setup.sh`, has been used to validate the environment, dependencies, and data access. This phase is complete. The script verifies:
- GPU availability and PyTorch CUDA support
- Installation of all dependencies via `uv`
- Correct loading of `bioacoustics-model-zoo` models (BirdNET, Perch)
- Accessibility of all 8 BirdSet datasets from HuggingFace
- A comprehensive `pytest` suite (28 tests) passes successfully.

### 2. Full Experiment Execution (Phase 2 🔄)

The experiment is run via a modular SLURM architecture.

```bash
# Navigate to the experiment directory
cd user/murilo/experiments/001-birdset-bioacoustics-benchmark

# PART A: Direct Inference for BirdSet-native models
# (BirdSetConvNeXT, BirdSetEfficientNetB1)
bash sbatch/inference/run.sh

# PART B: Transfer Learning for external models
# (BirdNET, Perch, etc.)
# Step 1: Extract embeddings for all models and datasets
bash sbatch/extract_embeddings/run.sh

# Step 2: Train linear probes on the extracted embeddings
bash sbatch/train_classifier/run.sh
```

## Methodology

The evaluation is split into two parts based on the model's origin.

### Part A: Direct Inference (BirdSet-Native Models)

- **Models**: `BirdSetConvNeXT`, `BirdSetEfficientNetB1`.
- **Rationale**: These models were trained directly on BirdSet data, so they can produce species predictions for the BirdSet taxonomy out-of-the-box.
- **Process**:
    1. Run inference on the 8 BirdSet test sets.
    2. Compute cmAP and other metrics directly from the model's predictions.
    3. No training is required.

### Part B: Transfer Learning (External Models)

- **Models**: `BirdNET`, `Perch`, `YAMNet`, `HawkEars`, `RanaSierraeCNN`.
- **Rationale**: These models were trained on different datasets with different taxonomies. A linear probe is used to adapt them to the BirdSet species labels and fairly evaluate the quality of their learned representations.
- **Process**:
    1. **Embedding Extraction**: For each model, extract feature embeddings for the train and test splits of all 8 BirdSet datasets.
    2. **Linear Probing**: Train a `MultiLabelLinearProbe` (a 2-layer MLP implemented in PyTorch Lightning) on the extracted training embeddings.
    3. **Evaluation**: Evaluate the trained linear probe on the test embeddings to compute cmAP and other metrics.

## Models Under Test

| Model | Evaluation | Window | Embed Size | Notes |
|-------|------------|--------|------------|-------|
| `BirdSetConvNeXT` | Direct | 5.0s | 1024 | BirdSet-native |
| `BirdSetEfficientNetB1` | Direct | 5.0s | 1280 | BirdSet-native |
| `BirdNET` | Linear Probe | 3.0s | 1024 | External (6.5k species) |
| `Perch` | Linear Probe | 5.0s | 1280 | External (Google) |
| `YAMNet` | Linear Probe | 0.96s | 1024 | External (General audio) |
| `HawkEars` | Linear Probe | 3.0s | 2048 | External (Raptor-focused) |
| `RanaSierraeCNN` | Linear Probe | 2.0s | 512 | External (Frog-focused) |

## Project Structure

```
001-birdset-bioacoustics-benchmark/
├── README.md              # This file
├── proposal.md            # Hypothesis and methodology
├── design.md              # Technical decisions
├── tasks.md               # Implementation checklist
├── config.yaml            # Experiment configuration
├── test_setup.sh          # Interactive environment test script
├── sbatch/                # SLURM job scripts
│   ├── common.sh
│   ├── inference/
│   ├── extract_embeddings/
│   └── train_classifier/
└── artifacts/             # Outputs (embeddings, models, results)
```

## Status

**Current Phase**: Phase 2A.5 (Production Runs) - 7/8 datasets complete for birdset_convnext

- [x] **Phase 0: Planning**: Proposal, Design, and Tasks are complete.
- [x] **Phase 1: Environment Setup**: All dependencies installed and validated via `test_setup.sh` and a comprehensive `pytest` suite.
- [x] **Phase 2A.1-2A.3**: Data pipeline, inference script, and SLURM scripts implemented and tested.
- [🔄] **Phase 2A.4-2A.5**: Production inference runs in progress (7/8 datasets complete, SSW running).
- [ ] **Phase 2B: Transfer Learning**: Implement embedding extraction and linear probe training for external models.
- [ ] **Phase 3: Results Aggregation**: Collect all results and generate final analysis.

**Latest Update (2026-01-25)**:
- ✅ Completed 7/8 datasets for `birdset_convnext`: HSN, NBP, NES, PER, POW, SNE, UHH
- 🔄 SSW dataset running (205,200 samples - largest test set)
- ⚠️ **AUROC metric fix implemented** - requires re-running all datasets to include AUROC in results
- Current results show cmAP range: 0.17 (PER) to 0.63 (NBP)

See [tasks.md](tasks.md) for a detailed checklist.

## Next Steps

1.  **Wait for SSW completion** and verify all 8 datasets complete for `birdset_convnext`
2.  **Re-run Phase 2A with AUROC fix**: Re-run all datasets with `--force` to include AUROC metric
3.  **Run second model**: Execute `birdset_efficientnet_b1` on all 8 datasets
4.  **Implement Phase 2B**: Complete the `extract_embeddings.py` and `train_classifier.py` scripts and execute them for external models.
5.  **Implement Phase 3**: Write the `aggregate_results.py` script to produce the final results matrix and visualizations.