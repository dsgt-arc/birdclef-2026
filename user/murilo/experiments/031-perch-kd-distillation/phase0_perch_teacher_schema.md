# Phase 0.1 — Perch v2 Teacher Schema and Verification

**Date**: 2026-05-09
**Source**: Tucker Arrants's Kaggle dataset
`tuckerarrants/perch-v2-no-dft-onnx`, vendored to
`vendor/kaggle/datasets/perch-v2-no-dft-onnx/`.

## File

- Path: `vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx`
- Size: 394.2 MB

## I/O schema

### Inputs

| Index | Name | Shape | Type |
|---|---|---|---|
| 0 | inputs | ['batch', 160000] | tensor(float) |

### Outputs

| Index | Name | Shape | Type | Notes |
|---|---|---|---|---|
| 0 | embedding | ['batch', 1536] | tensor(float) | ⭐ 1536-d EMBEDDING |
| 1 | spatial_embedding | ['batch', 16, 4, 1536] | tensor(float) | ⭐ 1536-d EMBEDDING |
| 2 | spectrogram | ['batch', 500, 128] | tensor(float) |  |
| 3 | label | ['batch', 14795] | tensor(float) |  |

**Embedding output index**: 0

## Verification results

### Random waveform (B=4, σ=0.1 Gaussian)

- Output shape: `(4, 1536)`
- std: 0.0923
- mean: -0.0021
- range: [-0.2319, 0.4456]
- ✅ Non-degenerate

### Real training waveform (B=4 deterministic from `train.csv`, seed=42)

- Output shape: `(4, 1536)`
- std: 0.1074
- mean: -0.0051
- range: [-0.2366, 0.6612]
- Cross-sample off-diagonal cos sim: mean=0.151, min=0.054, max=0.281
- ✅ Distinct (different waveforms produce different embeddings)

## Compute

- **CPU (CPUExecutionProvider)**: ~1828ms / batch of 4
- **CUDA (CUDAExecutionProvider)**: ~18ms / batch of 4 (5-run mean)

## Implications for Phase 1

- The teacher takes (B, 160000) float32 raw waveforms at 32 kHz, returns
  `(B, 1536)` clip embeddings as output index 0.
- Our `PerchTeacher` class in `sed_finetune.py` (Phase 0.2) wraps this.
- Per-batch teacher inference cost on the training partition's GPU
  determines whether we can use ONNX at training time or need to
  precompute embeddings. CUDA works → use directly in training loop.

## Files

- This doc: `phase0_perch_teacher_schema.md`
- Verification script: `user/murilo/analysis/verify_perch_teacher.py`
- ONNX (vendored): `vendor/kaggle/datasets/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx`
