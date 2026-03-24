# Model Runs

Single source of truth for all trained models, their checkpoints, training data, and scores.
Referenced by experiment proposals, pseudo-labeling configs, and submission pipelines.

## Naming Convention

Model slugs follow: `{encoder}-{year}-{variant}`

- `{encoder}`: `protoclr`, `convnext`, `eat`
- `{year}`: `2025`, `2026`
- `{variant}`: training data / method (e.g., `soundscape`, `fulldata`, `pseudo-v1`)

## BirdCLEF 2025 Models (exps 006–009)

Trained on BirdCLEF 2025 focal recordings (28,564 files, 206 species).

| Slug | Encoder | Experiment | Training Data | Classes | Test AUROC | LB | Checkpoint |
|------|---------|------------|--------------|---------|-----------|-----|------------|
| `protoclr-2025` | ProtoCLR/CvT-13 | 009 | 2025 focal 80/10/10 | 206 | 0.942 | 0.531 | `009-protoclr-finetune/.../best-epochepoch=20.ckpt` |
| `convnext-2025` | ConvNeXt-Base | 007 | 2025 focal 80/10/10 | 206 | 0.941 | — | `007-convnext-finetune/.../best-epochepoch=12.ckpt` |
| `eat-2025` | EAT-base | 006 | 2025 focal 80/10/10 | 206 | 0.945 | — | `006-lora-finetune/.../best-epochepoch=16.ckpt` |

## BirdCLEF 2026 Models — Focal Only (exp 011)

Trained on BirdCLEF 2026 focal recordings (35,549 files, 206 species).

| Slug | Encoder | Variant | Training Data | Classes | Test AUROC | LB | Checkpoint |
|------|---------|---------|--------------|---------|-----------|-----|------------|
| `protoclr-2026` | ProtoCLR/CvT-13 | 80/10/10 | 2026 focal | 206 | 0.937 | 0.721 | `011-.../2026-protoclr-lora-attn/best-epochepoch=21.ckpt` |
| `convnext-2026` | ConvNeXt-Base | 80/10/10 | 2026 focal | 206 | 0.938 | 0.720 | `011-.../2026-convnext-lora-mlp-mlphead/best-epochepoch=09.ckpt` |
| `eat-2026` | EAT-base | 80/10/10 | 2026 focal | 206 | 0.929 | 0.701 | `011-.../2026-eat-lora-attn-mlphead/best-epochepoch=17.ckpt` |
| `protoclr-2026-fulldata` | ProtoCLR/CvT-13 | 100% focal | 2026 focal (all) | 206 | — | 0.736 | `011-.../2026-protoclr-lora-attn-fulldata/last-v1.ckpt` |
| `convnext-2026-fulldata` | ConvNeXt-Base | 100% focal | 2026 focal (all) | 206 | — | 0.695 | `011-.../2026-convnext-lora-mlp-mlphead-fulldata/last.ckpt` |
| `eat-2026-fulldata` | EAT-base | 100% focal | 2026 focal (all) | 206 | — | 0.698 | `011-.../2026-eat-lora-attn-mlphead-fulldata/last.ckpt` |

## BirdCLEF 2026 Models — Soundscape Fine-Tuned (exp 012)

Fine-tuned from exp 011 (80/10/10) checkpoints on focal + 739 labeled soundscape segments.
Head expanded from 206 → 234 species.

| Slug | Encoder | Training Data | Classes | Test AUROC | LB | Checkpoint |
|------|---------|--------------|---------|-----------|-----|------------|
| `protoclr-2026-soundscape` | ProtoCLR/CvT-13 | 2026 focal + soundscape | 234 | 0.845 | **0.767** | `012-.../2026-protoclr-soundscape/best-epochepoch=11.ckpt` |
| `convnext-2026-soundscape` | ConvNeXt-Base | 2026 focal + soundscape | 234 | 0.844 | **0.755** | `012-.../2026-convnext-soundscape/best-epochepoch=02.ckpt` |
| `eat-2026-soundscape` | EAT-base | 2026 focal + soundscape | 234 | 0.838 | **0.720** | `012-.../2026-eat-soundscape/best-epochepoch=03.ckpt` |

### ONNX Exports

All exp 012 models are exported to ONNX via exp 010 pipeline.

| Slug | ONNX Path | Mel Type | fmax |
|------|-----------|----------|------|
| `protoclr-2026-soundscape` | `010-.../artifacts/weights/protoclr-2026-soundscape/model.onnx` | protoclr_mel | 8,000 Hz |
| `convnext-2026-soundscape` | `010-.../artifacts/weights/convnext-2026-soundscape/model.onnx` | mel | 16,000 Hz |
| `eat-2026-soundscape` | `010-.../artifacts/weights/eat-2026-soundscape/model.onnx` | kaldi_fbank | 8,000 Hz |

### Known Limitations (from error analysis)

| Model | Insect AUROC | Issue |
|-------|-------------|-------|
| `eat-2026-soundscape` | **0.673** | kaldi fbank fmax=8kHz misses insect stridulation |
| `protoclr-2026-soundscape` | 0.856 | fmax=8kHz but handles insects better |
| `convnext-2026-soundscape` | 0.848 | fmax=16kHz, best spectral coverage |

EAT excluded from pseudo-labeling teacher ensemble (exp 014) due to insect failure.

## Pseudo-Label Runs (exp 014)

| Run Name | Teachers | Method | Split | Files | Segments | Path |
|----------|---------|--------|-------|-------|----------|------|
| `pcr-cnx-soundscape-v1` | protoclr + convnext (exp 012) | Direct average | labeled | 66 | 792 | `shared/pseudo-labels/birdclef-2026/pcr-cnx-soundscape-v1/labeled/` |
| `pcr-cnx-soundscape-v1` | protoclr + convnext (exp 012) | Direct average | unlabeled | 10,592 | 127,104 | `shared/pseudo-labels/birdclef-2026/pcr-cnx-soundscape-v1/unlabeled/` |

### Labeled Split Validation (ensemble vs ground truth, 75/234 species)

| Model | Macro AUROC | cmAP |
|-------|-----------|------|
| **Ensemble (mean)** | **0.8851** | **0.3876** |
| protoclr-2026-soundscape | 0.8543 | 0.3583 |
| convnext-2026-soundscape | 0.8224 | 0.3085 |

Per-taxon ensemble AUROC: Insecta 0.904, Aves 0.888, Mammalia 0.864, Amphibia 0.862, Reptilia 0.794.

## BirdCLEF 2026 Models — Student Training (exp 015, round 0)

Fine-tuned from exp 012 checkpoints on focal + labeled soundscape + 127K pseudo-labeled
soundscape segments (soft labels from exp 014 `pcr-cnx-soundscape-v1`). Per-sample weighted
BCE loss: w=1.0 for ground truth, w=0.5 for pseudo-labels. Temperature T=1.0 (no scaling).

| Slug | Encoder | Training Data | Classes | Test AUROC | Test cmAP | Best Epoch | LB | Checkpoint |
|------|---------|--------------|---------|-----------|----------|------------|-----|------------|
| `protoclr-2026-student-v1` | ProtoCLR/CvT-13 | focal + soundscape + pseudo (340K) | 234 | 0.846 | 0.485 | 5 | **0.762** | `015-.../2026-protoclr-student-v1/best-epochepoch=05.ckpt` |
| `convnext-2026-student-v1` | ConvNeXt-Base | focal + soundscape + pseudo (340K) | 234 | 0.845 | 0.479 | 1 | pending | `015-.../2026-convnext-student-v1/best-epochepoch=01.ckpt` |
| `eat-2026-student-v1` | EAT-base | focal + soundscape + pseudo (340K) | 234 | 0.842 | 0.465 | 2 | **0.696** | `015-.../2026-eat-student-v1/best-epochepoch=02.ckpt` |

### Comparison vs Exp 012 Baseline

| Model | Exp 012 Test AUROC | Exp 015 Test AUROC | Delta | Exp 012 LB | Exp 015 LB | LB Delta |
|-------|-------------------|-------------------|-------|-----------|-----------|----------|
| ProtoCLR | 0.845 | 0.846 | +0.001 | 0.767 | 0.762 | **−0.005** |
| ConvNeXt | 0.844 | 0.845 | +0.001 | 0.755 | pending | — |
| EAT | 0.838 | 0.842 | +0.004 | 0.720 | 0.696 | **−0.024** |

ConvNeXt student completed: test AUROC 0.845 (+0.001), LB pending.

**Conclusion (round 0)**: Pseudo-labels at w=0.5, T=1.0 **hurt** LB performance.
Both ProtoCLR (−0.005) and EAT (−0.024) regressed despite flat/improved offline
test metrics. EAT's larger drop may reflect the pseudo-label quality mismatch —
EAT was excluded from the teacher ensemble (exp 014) due to poor insect AUROC,
yet the pseudo-labels encode the ProtoCLR+ConvNeXt teachers' behavior which may
conflict with EAT's learned representations. The clip-level architecture is the
primary bottleneck — exp 016 (SED) addresses this directly.

## BirdCLEF 2026 Models — SED Architecture (exp 016)

EfficientNet-B0 with SED attention head (GeM freq pooling + attention + framewise max).
Trained from ImageNet-pretrained backbone (no prior BirdCLEF checkpoint). 10s windows,
spectrogram mixup, 32kHz. v1: soft CE dual loss, standard ImageNet init.
v2: standard CE with label_smoothing=0.05, NS-JFT pretrained (`tf_efficientnet_b0.ns_jft_in1k`).
**v2 regressed** on all offline metrics vs v1 (−0.023 AUROC, −0.054 cmAP).

| Slug | Encoder | Architecture | Classes | Test AUROC | Test cmAP | Best Epoch | LB | Checkpoint |
|------|---------|-------------|---------|-----------|----------|------------|-----|------------|
| `effnet-b0-sed-v1` | EfficientNet-B0 | SED (GeM + AttHead) | 234 | **0.860** | **0.593** | 24 | 0.795 / **0.823** (+ pp) | `016-.../2026-effnet-b0-sed-v1/best-epochepoch=24.ckpt` |
| `effnet-b0-sed-v2` | EfficientNet-B0 (NS-JFT) | SED (GeM + AttHead) + CE | 234 | 0.837 | 0.539 | 24 | **0.755** (+pp) | `016-.../2026-effnet-b0-sed-v2/best-epochepoch=24.ckpt` |
| `effnet-b0-sed-v3b` | EfficientNet-B0 | SED + raw waveform mixup | 234 | **0.862** | **0.607** | 25 | pending | `016-.../2026-effnet-b0-sed-v3b/best-epochepoch=25.ckpt` |

### Comparison: SED vs Clip-Level (offline test set)

| Model | Architecture | Test AUROC | Test cmAP | LB |
|-------|-------------|-----------|----------|-----|
| **EfficientNet-B0 SED v3b** | **SED + raw waveform mixup** | **0.862** | **0.607** | pending |
| EfficientNet-B0 SED v1 | SED (GeM + AttHead) | 0.860 | 0.593 | 0.795 / **0.823** (+pp) |
| EfficientNet-B0 SED v2 | SED + NS-JFT + CE | 0.837 | 0.539 | **0.755** (+pp) |
| ProtoCLR (exp 012) | Clip (global avg pool) | 0.845 | 0.481 | 0.767 |
| ConvNeXt (exp 012) | Clip (global avg pool) | 0.844 | 0.470 | 0.755 |
| EAT (exp 012) | Clip (global avg pool) | 0.838 | 0.452 | 0.720 |

SED architecture achieves +0.015 AUROC and +0.112 cmAP over the best clip-level model,
using a smaller backbone (3.6M params vs 19-90M for our LoRA encoders). The cmAP
improvement is particularly notable — SED's framewise detection captures brief calls
that clip-level averaging misses.

**Post-processing**: TTA (2.5s shifts) + temporal smoothing (Gaussian) + TopN (file-level
max multiplication) added +0.028 LB (0.795 → 0.823) with zero retraining.

**Competition reference**: EfficientNet-B0 SED = 0.908 LB (competitor). Remaining gap
0.085 — likely from: standard ImageNet init (vs NS-JFT), spectrogram mixup (vs raw
waveform), soft CE (vs CE/ASL), and 5s inference windows (trained on 10s).
See exp 016 results.md for gap analysis.

### Training Dynamics

Strong learning curve from random init:
- Epoch 0: val/auroc 0.517 (random)
- Epoch 8: val/auroc 0.833 (rapid improvement)
- Epoch 18: val/auroc 0.844 (plateauing)
- Epoch 24: val/auroc 0.845 (best, early stopped at epoch 34)

## Checkpoint Base Path

All checkpoints under: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/`
