# Experiment 010: BirdCLEF 2026 Submission Pipeline

## Objective

Submit BirdCLEF 2025 models (EAT, ConvNeXt, ProtoCLR) to BirdCLEF 2026 to:
1. Test the submission pipeline end-to-end
2. Test CPU inference timing (90-min Kaggle limit)
3. Get baseline scores (expected low: only 41/234 species overlap)

## Approach

ONNX export + librosa mel + ONNX Runtime on Kaggle CPU.
Follows Anthony's exp 012 (EfficientNet linear probe) pattern.

## Models

| Model | Checkpoint | AUROC | LoRA Backend |
|-------|-----------|-------|-------------|
| EAT | `006-lora-finetune/checkpoints/eat-lora-attn-mlphead/best-epochepoch=16.ckpt` | 0.942 | PEFT |
| ConvNeXt | `007-convnext-finetune/checkpoints/convnext-lora-mlp-mlphead/best-epochepoch=12.ckpt` | 0.941 | PEFT |
| ProtoCLR | `009-protoclr-finetune/checkpoints/protoclr-lora-attn/best-epochepoch=20.ckpt` | 0.942 | PEFT |

## Pipeline

1. Export each model to ONNX (merge LoRA → wrap encoder+head → export)
2. Build species mapping (BirdCLEF 2025→2026, 41 species overlap)
3. Unified Kaggle kernel: load ONNX, compute mel, inference, map species
4. Local validation on train_soundscapes
5. Upload to Kaggle and submit

## Expected Results

- Low leaderboard scores due to species mismatch (41/234 overlap)
- Validated submission pipeline for future models
- CPU timing data for each architecture
