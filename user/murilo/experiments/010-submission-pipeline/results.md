# Results: BirdCLEF 2026 Submission Pipeline

**Date Started**: 2026-03-14
**Author**: Murilo Gustineli

## Summary

End-to-end Kaggle submission pipeline validated. Models trained on both BirdCLEF 2025 and 2026
focal data are exported to ONNX, uploaded to Kaggle datasets, and scored via unified CPU kernel.

## Kaggle Leaderboard Results

Competition metric: **macro-averaged ROC-AUC** (skips classes with no true positive labels).

### Models Trained on BirdCLEF 2025

These models cover only 41/234 BirdCLEF 2026 species. Low scores are expected — the goal was
pipeline validation, not competitive scores.

| Model | Training Data | Overlapping Species | Public LB Score | Kernel Version | Date |
|-------|--------------|--------------------:|----------------:|----------------|------|
| ProtoCLR LoRA attn | BirdCLEF 2025 | 41/234 | **0.531** | v12 | 2026-03-15 |
| EAT LoRA attn+MLP head | BirdCLEF 2025 | 41/234 | — | — | — |
| ConvNeXt LoRA MLP+MLP head | BirdCLEF 2025 | 41/234 | — | — | — |

### Models Trained on BirdCLEF 2026

These models cover all 206 focal species. The 28 soundscape-only species (insect sonotypes)
have no focal training data and get 0.0 predictions.

| Model | Training Data | Mapped Species | Public LB Score | Kernel Version | Date |
|-------|--------------|---------------:|----------------:|----------------|------|
| ProtoCLR LoRA attn | BirdCLEF 2026 (exp 011) | 206/234 | — | — | — |
| EAT LoRA attn+MLP head | BirdCLEF 2026 (exp 011) | 206/234 | — | — | — |
| ConvNeXt LoRA MLP+MLP head | BirdCLEF 2026 (exp 011) | 206/234 | — | — | — |

## Observations

- **ProtoCLR 2025 scored 0.531 macro-AUROC** on the public leaderboard despite only covering
  41/234 species. The metric skips classes with no true positives, so the score reflects
  quality on the 41 overlapping species rather than being diluted by 193 zero-probability species.

- **ONNX export works reliably for all 3 architectures**: EAT and ConvNeXt export cleanly,
  ProtoCLR required einops-compatible tracing but succeeded with opset 17.

- **CPU timing is well within Kaggle limits**: All models complete inference on ~600 soundscapes
  within the 90-minute budget.

## Pipeline Issues Resolved

| Issue | Root Cause | Fix | Versions Affected |
|-------|-----------|-----|-------------------|
| Notebook Threw Exception | ONNX output shape (1,206) vs (206,) | `logits.flatten()` before sigmoid | v3–v10 |
| Empty test_soundscapes | Expected Kaggle behavior during public run | train_soundscapes fallback (2 files) | v11 |
| onnxruntime not found | No internet on Kaggle | Bundled wheels in `birdclef-2025-deps` dataset | v4–v6 |
| Kernel ID mismatch | `id` field must match slugified `title` | Fixed metadata generation | v2 |

## Artifacts

| Type | Location |
|------|----------|
| ONNX weights (2025) | `artifacts/weights/{eat,convnext,protoclr}-2025/model.onnx` |
| Species mapping (2025→2026) | `artifacts/weights/species_mapping.json` (41 mapped) |
| Kaggle kernel | `kaggle_kernel/submission.py` |
| Kernel metadata | `kaggle_kernel/kernel-metadata-{model}-{2025,2026}.json` |
| Export scripts | `scripts/export_{eat,convnext,protoclr}.py` |
| Offline deps | `artifacts/dist/dependencies/` |
| SLURM logs | `logs/slurm_{export,validate}_*.log` |
| Code | Branch `mg/010-submission-pipeline`, PR #16 |
