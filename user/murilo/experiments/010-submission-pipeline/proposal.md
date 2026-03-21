# Experiment 010: BirdCLEF 2026 Submission Pipeline

## Objective

Submit LoRA fine-tuned models (EAT, ConvNeXt, ProtoCLR) to BirdCLEF 2026 via ONNX export
and a unified Kaggle CPU kernel. Supports two model sources:

1. **BirdCLEF 2025-trained** (exps 006/007/009) — pipeline validation with 41/234 species overlap
2. **BirdCLEF 2026-trained** (exp 011) — competitive baselines with 206/234 species mapped

## Approach

ONNX export + librosa mel + ONNX Runtime on Kaggle CPU.
Follows Anthony's exp 012 (EfficientNet linear probe) pattern.

## Models

### 2025-Trained (pipeline validation)

| Model | Checkpoint | AUROC | LoRA Backend |
|-------|-----------|-------|-------------|
| EAT | `006-lora-finetune/checkpoints/eat-lora-attn-mlphead/best-epochepoch=16.ckpt` | 0.942 | PEFT |
| ConvNeXt | `007-convnext-finetune/checkpoints/convnext-lora-mlp-mlphead/best-epochepoch=12.ckpt` | 0.941 | PEFT |
| ProtoCLR | `009-protoclr-finetune/checkpoints/protoclr-lora-attn/best-epochepoch=20.ckpt` | 0.942 | PEFT |

### 2026-Trained (exp 011 baselines)

| Model | Checkpoint | AUROC | LoRA Backend |
|-------|-----------|-------|-------------|
| ConvNeXt | `011-lora-baseline-2026/checkpoints/2026-convnext-lora-mlp-mlphead/best-epochepoch=09.ckpt` | 0.938 | PEFT |
| ProtoCLR | `011-lora-baseline-2026/checkpoints/2026-protoclr-lora-attn/best-epochepoch=21.ckpt` | 0.937 | PEFT |
| EAT | `011-lora-baseline-2026/checkpoints/2026-eat-lora-attn-mlphead/best-epochepoch=17.ckpt` | 0.929 | PEFT |

## Pipeline

1. Export each model to ONNX (merge LoRA → wrap encoder+head → export)
2. Build species mapping (model output indices → 2026 submission columns)
3. Unified Kaggle kernel: load ONNX, compute mel, inference, map species
4. Local validation on train_soundscapes
5. Upload to Kaggle and submit

## Expected Results

- **2025-trained**: Low LB scores due to species mismatch (41/234 overlap). Primary goal is pipeline validation.
- **2026-trained**: Competitive baselines with 206/234 species mapped. The 28 soundscape-only species (insect sonotypes) have no focal training data and get 0.0 predictions.
- Validated submission pipeline and CPU timing data for each architecture.
