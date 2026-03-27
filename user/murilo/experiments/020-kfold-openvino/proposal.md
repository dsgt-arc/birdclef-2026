---
experiment: "020-kfold-openvino"
status: proposed
date: 2026-03-27
author: Murilo Gustineli
---

# Proposal: K-Fold SED Ensemble with OpenVINO

**Status**: proposed
**Created**: 2026-03-27
**Author**: Murilo Gustineli

## Hypothesis

OpenVINO conversion will provide 2-3x CPU inference speedup, enabling a 4-fold SED ensemble that improves LB by +0.01-0.02 over a single v4b model while fitting within Kaggle's 90-minute budget.

## Background

Every BirdCLEF winner (2023-2025) uses OpenVINO for CPU inference. Tawara achieves 0.859 with a 4-fold HGNetV2-B0 ensemble using OpenVINO. Our v4b single model takes ~30min on Kaggle CPU — too slow for multi-model ensembles. OpenVINO would cut this to ~15min, making 4 SED models + Perch feasible within the 90-minute budget.

## Method

### Part A: OpenVINO Conversion

- Convert ONNX to OpenVINO IR (XML + bin)
- Expected speedup: 2-3x on Kaggle CPU (v4b from ~30min to ~15min)
- Key: `mo --input_model model.onnx --output_dir openvino/` + runtime with `openvino.Core()`
- Benefits: enables fitting 4 SED models + Perch in 90-min budget
- Reference: `docs/references/kaggle/birdclef-2026/discussions/hgnetv2-b0-baseline-tawara.md` (Tawara uses OpenVINO)
- Reference: `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` (2025 1st uses OpenVINO INT8)

### Part B: K-Fold SED Ensemble

- 4-fold StratifiedGroupKFold (group by primary_label to keep same species in same fold)
- Or Multi-Label Stratified Group K-Fold (Tawara's approach — minimizes per-class label count variance)
- Train 4 independent v4b models (or best from exp 019)
- Rank-averaged ensemble at inference (Tawara: per-class rank / n_samples -> scale-invariant [0,1])
- Training cost: 4 x ~4h = 16h on GPU
- Inference budget: 4 x 15min (OpenVINO) + 15min (Perch) = 75min -> fits in 90min

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 training set, 4-fold stratified group splits |
| Compute | 1x RTX 6000, ~4h per fold (16h total training), Kaggle CPU for inference |
| Dependencies | Exp 017 v4b recipe, exp 018 Perch infrastructure, openvino-dev |
| Code | `user/murilo/experiments/017-mel-overhaul/` (training), new inference kernel |

### Procedure

1. Convert v4b ONNX model to OpenVINO IR format
2. Benchmark OpenVINO inference speed on Kaggle CPU
3. Implement StratifiedGroupKFold or Multi-Label Stratified Group K-Fold splitting
4. Train 4 independent v4b models (one per fold)
5. Convert all 4 models to OpenVINO
6. Implement rank-averaged ensemble in inference kernel
7. Submit combined SED ensemble (+ Perch from exp 018) to Kaggle LB

### Variables

- **Independent**: Number of folds (4), inference runtime (ONNX vs OpenVINO), ensemble method (rank-average)
- **Dependent**: Kaggle LB score (AUROC), inference time
- **Controlled**: Architecture (EfficientNet-B0), training recipe (v4b or best from 019), mel config

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Kaggle LB (AUROC) | Primary competition metric |
| Inference time | Must fit within 90-minute Kaggle budget |
| OpenVINO speedup | Target 2-3x over ONNX |

### Baseline

v4b single model = 0.859 LB, ~30min inference on Kaggle CPU.

### Success Criteria

- **Confirm if**: 4-fold ensemble achieves >= 0.870 LB AND total inference fits in 90 minutes
- **Reject if**: OpenVINO speedup < 1.5x (makes multi-model infeasible) OR ensemble <= 0.862 LB

## Limitations

- Gate condition: OpenVINO must be working before K-fold is useful (otherwise 4 x 30min = 120min > 90min budget)
- 16h total training time requires 4 sequential GPU jobs
- Fold splits must ensure no species leakage between train/val
- OpenVINO quantization (INT8) could provide further speedup but adds calibration complexity
