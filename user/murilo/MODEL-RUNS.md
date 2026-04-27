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
| `effnet-b0-sed-v1` | EfficientNet-B0 | SED (GeM + AttHead) | 234 | **0.860** | **0.593** | 24 | 0.795 / 0.823 (+pp) / 0.826 (+pp +10s) / **0.827** (+pp2 +10s) | `016-.../2026-effnet-b0-sed-v1/best-epochepoch=24.ckpt` |
| `effnet-b0-sed-v2` | EfficientNet-B0 (NS-JFT) | SED (GeM + AttHead) + CE | 234 | 0.837 | 0.539 | 24 | **0.755** (+pp) | `016-.../2026-effnet-b0-sed-v2/best-epochepoch=24.ckpt` |
| `effnet-b0-sed-v3a` | EfficientNet-B0 (TF) | SED (TF-ported weights) | 234 | 0.857 | 0.599 | 25 | **0.813** (+pp) | `016-.../2026-effnet-b0-sed-v3a/best-epochepoch=25.ckpt` |
| `effnet-b0-sed-v3b` | EfficientNet-B0 | SED + raw waveform mixup | 234 | **0.862** | **0.607** | 25 | **0.802** (+pp) | `016-.../2026-effnet-b0-sed-v3b/best-epochepoch=25.ckpt` |
| `effnet-b0-sed-v3c` | EfficientNet-B0 | SED + CE loss | 234 | 0.850 | 0.552 | 25 | pending | `016-.../2026-effnet-b0-sed-v3c/best-epochepoch=25.ckpt` |
| `effnet-b0-sed-v3d` | EfficientNetV2-B0 | SED (V2 architecture) | 234 | 0.841 | 0.523 | 13 | pending | `016-.../2026-effnet-b0-sed-v3d/best-epochepoch=13.ckpt` |

### Comparison: SED vs Clip-Level (offline test set)

| Model | Architecture | Test AUROC | Test cmAP | LB |
|-------|-------------|-----------|----------|-----|
| EfficientNet-B0 SED v3b | SED + raw waveform mixup | **0.862** | **0.607** | **0.802** (+pp) |
| **EfficientNet-B0 SED v1** | **SED (GeM + AttHead)** | **0.860** | **0.593** | 0.795 / 0.823 (+pp) / **0.826** (+pp +10s) |
| EfficientNet-B0 SED v3a | SED (TF-ported weights) | 0.857 | 0.599 | **0.813** (+pp) |
| EfficientNet-B0 SED v3c | SED + CE loss | 0.850 | 0.552 | pending |
| EfficientNetV2-B0 SED v3d | SED (V2 architecture) | 0.841 | 0.523 | pending |
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
0.082 — likely from: mel resolution (n_fft=1024 vs 4096, 128 vs 224 mels),
standard ImageNet init, spectrogram mixup (vs raw waveform), and head design.
10s inference alignment added +0.003, pp2 (temp scaling + taxa smoothing + metadata priors) added +0.001. See exp 016 summary.md for gap analysis;
next steps in [concept: strong-baseline-mel-overhaul](docs/concepts/strong-baseline-mel-overhaul.md).

### Training Dynamics

Strong learning curve from random init:
- Epoch 0: val/auroc 0.517 (random)
- Epoch 8: val/auroc 0.833 (rapid improvement)
- Epoch 18: val/auroc 0.844 (plateauing)
- Epoch 24: val/auroc 0.845 (best, early stopped at epoch 34)

## BirdCLEF 2026 Models — Mel Overhaul (exp 017)

EfficientNet-B0 SED with high-resolution mel: n_fft=4096, n_mels=224 (vs v1's 1024/128).
All other parameters identical to exp 016 v1 to isolate the mel resolution variable.
Mel computed on GPU (n_fft=4096 too slow on CPU DataLoader workers).

| Slug | Encoder | Architecture | Mel | Classes | Test AUROC | Test cmAP | Best Epoch | LB | Checkpoint |
|------|---------|-------------|-----|---------|-----------|----------|------------|-----|------------|
| `effnet-b0-sed-v4a` | EfficientNet-B0 | SED (GeM + AttHead) | n_fft=4096, n_mels=224 | 234 | **0.862** | **0.615** | 24 | 0.825 / **0.830** (+pp2) | `017-.../2026-effnet-b0-sed-v4a/best-epochepoch=24.ckpt` |

### Comparison: v4a vs v1

| Metric | v1 (exp 016) | v4a (exp 017) | Delta |
|--------|-------------|---------------|-------|
| test/auroc | 0.860 | **0.862** | +0.002 |
| test/cmap | 0.593 | **0.615** | **+0.022** |
| test/top1_acc | 0.762 | 0.772 | +0.010 |
| test/top5_acc | 0.887 | 0.894 | +0.007 |
| LB (raw) | 0.795 | **0.825** | **+0.030** |
| LB (+pp2) | 0.827 | **0.830** | **+0.003** |

**Mel resolution is a major lever.** v4a raw (0.825) nearly matches v1+pp2 (0.827) with
zero post-processing. PP2 adds only +0.005 on v4a (vs +0.032 on v1) — diminishing returns
as the model already captures distinctions that post-processing was compensating for.
Decision gate passed → continue to Step 2.

Inference optimized with full-signal mel (compute STFT once per 60s file, slice frames)
to stay within the 90-minute Kaggle CPU budget at n_fft=4096.

### Step 2: Parallel Ablations (v4b-v4e)

Single-variable ablations from v4a. 90/10 split (no test set), 50 epochs.
No test metrics available — only val/auroc from checkpoint callback.

| Slug | Change | Mel Norm | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4b` | AmplitudeToDB norm | amplitude_to_db | AttHead | soft_ce | **0.865** | 30 | **0.859** / **0.860** (+pp2) | `017-.../2026-effnet-b0-sed-v4b/best-epochepoch=30.ckpt` |
| `effnet-b0-sed-v4c` | AttBlockV2 head | audioset | AttBlockV2 | soft_ce | 0.815 | 15 | skipped | `017-.../2026-effnet-b0-sed-v4c/best-epochepoch=15.ckpt` |
| `effnet-b0-sed-v4d` | ASL loss | audioset | AttHead | asl | 0.858 | 16 | 0.808 | `017-.../2026-effnet-b0-sed-v4d/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v4e` | StochasticDepth | audioset | AttHead | soft_ce | **0.868** | 29 | 0.840 | `017-.../2026-effnet-b0-sed-v4e/best-epochepoch=29.ckpt` |

### Step 3: Combined winners + re-ablations

| Slug | Change | Mel Norm | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4f` | v4b + StochasticDepth | amplitude_to_db | AttHead | soft_ce | 0.862 | 17 | 0.832 | `017-.../2026-effnet-b0-sed-v4f/best-epochepoch=17.ckpt` |
| `effnet-b0-sed-v4g` | v4b + AttBlockV2 + BCE | amplitude_to_db | AttBlockV2 | bce | **0.868** | 28 | 0.819 | `017-.../2026-effnet-b0-sed-v4g/best-epochepoch=28.ckpt` |
| `effnet-b0-sed-v4h` | v4b + AttBlockV2 + ASL | amplitude_to_db | AttBlockV2 | asl | 0.858 | 12 | 0.800 | `017-.../2026-effnet-b0-sed-v4h/best-epochepoch=12.ckpt` |

### Step 4: Fulldata Training

| Slug | Change | Mel Norm | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4b-fulldata` | v4b on 100% data | amplitude_to_db | AttHead | soft_ce | — (no val) | 44 (last) | 0.853 | `017-.../2026-effnet-b0-sed-v4b-fulldata/last.ckpt` |

**v4b (AmplitudeToDB)** is the current best at 0.859 LB (0.860 with pp2).
Fulldata training regressed −0.006 (0.853 vs 0.859) — likely overtrained without val monitoring.

### Step 5: Loss & Augmentation Ablations

Motivated by Salman Ahmed's 0.908 LB (single EfficientNet-B0 SED with CE + raw waveform MixUp).

| Slug | Change | Mel Norm | Head | Loss | Waveform MixUp | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|----------|------|------|---------------|---------------|------------|-----|------------|
| `effnet-b0-sed-v4i` | Standard CE loss (primary_label) | amplitude_to_db | AttHead | ce | No | 0.835 | 12 | skip | `017-.../2026-effnet-b0-sed-v4i/best-epochepoch=12.ckpt` |
| **`effnet-b0-sed-v4j`** | **Raw waveform MixUp** | amplitude_to_db | AttHead | soft_ce | **Yes** | **0.868** | 34 | 0.853 | `017-.../2026-effnet-b0-sed-v4j/best-epochepoch=34.ckpt` |
| `effnet-b0-sed-v4k` | CE + waveform MixUp | amplitude_to_db | AttHead | ce | Yes | 0.835 | 18 | skip | `017-.../2026-effnet-b0-sed-v4k/best-epochepoch=18.ckpt` |

**v4j (waveform MixUp)** had best offline (+0.003 val/auroc) but regressed −0.006 on LB (0.853 vs 0.859).
Offline val/auroc continues to be a poor LB predictor (v4j: 0.868 val → 0.853 LB).
Standard CE loss (v4i, v4k) regressed −0.030 val/auroc even with the primary_label fix
(using actual primary label index instead of argmax of multi-hot). CE loss with our soft CE
setup consistently underperforms — the gap to Salman's 0.908 is from other factors.

**Note**: v4i and v4k used the fixed CE implementation (primary_label_idx from dataloader,
not argmax of multi-hot). The original broken CE (argmax) gave similar results (v4k
first run: 0.846), confirming the low mismatch rate (0.2% for focal recordings).

## BirdCLEF 2026 Models — SED Ablations (exp 019)

Single-variable ablations from v4b (0.860 LB). 90/10 split, 50 epochs, same data pipeline.
Goal: push SED from 0.860 → 0.885+ LB before pseudo-labeling.

### Step 1: Quick wins ablation

| Slug | Change | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|--------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v5a` | BCEWithLogitsLoss | AttHead | bce | **0.875** | 29 | **0.851** | `017-.../2026-effnet-b0-sed-v5a/best-epochepoch=29.ckpt` |
| `effnet-b0-sed-v5b` | Stronger MixUp (α=1.0, θ=0.8, warmup=5) | AttHead | soft_ce | **0.869** | 31 | **0.840** | `017-.../2026-effnet-b0-sed-v5b/best-epochepoch=31.ckpt` |
| `effnet-b0-sed-v5c` | weight_decay=1e-4 | AttHead | soft_ce | **0.867** | 35 | pending | `017-.../2026-effnet-b0-sed-v5c/best-epochepoch=35.ckpt` |
| `effnet-b0-sed-v5d` | LSEHead (GeM+LSE pooling) | **LSEHead** | soft_ce | **0.861** | 21 | pending | `017-.../2026-effnet-b0-sed-v5d/best-epochepoch=21.ckpt` |
| `effnet-b0-sed-v5e` | HGNetV2-B0 backbone (embed=1024) | AttHead | soft_ce | 0.861 | 23 | pending | `017-.../2026-hgnet-b0-sed-v5e/best-epochepoch=23.ckpt` |

**v5d (LSEHead)**: −0.004 val/auroc vs v4b baseline (0.861 vs 0.865). LSE pooling slightly
regresses compared to attention head on EfficientNet-B0.

### Step 2: ttahara full recipe reproduction

Faithful reproduction of ttahara's 0.884 LB pipeline: LSEHead(mean freq), BCE, per-sample
MixUp(α=1.0, θ=0.8), OneCycleLR(lr=5e-4, wd=1e-4), torchaudio mel(n_fft=2048, resize
256x256, min-max), 5s segments, 20 epochs. Ref: `user/murilo/notes/ttahara-hgnetv2-b0-annotated.md`.

| Slug | Backbone | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|----------|------|------|---------------|------------|-----|------------|
| `effnet-b0-sed-v5f` | EfficientNet-B0 | LSEHead(mean) | bce | **0.866** | 14 | **0.855** | `017-.../2026-effnet-b0-sed-v5f/best-epochepoch=14.ckpt` |
| `hgnet-b0-sed-v5g` | HGNetV2-B0 | LSEHead(mean) | bce | **0.872** | 18 | **0.892** | `017-.../2026-hgnet-b0-sed-v5g/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v5g-4fold` | HGNetV2-B0 ×4 | LSEHead(mean) | bce | 0.939/0.842/0.826/0.802 | 14/19/18/16 | **0.901** | `017-.../2026-hgnet-b0-sed-v5g/fold{0-3}/` |

### Step 4: v5g Recipe Ablations

Single-variable ablations from v5g (0.892 LB). All use HGNetV2-B0 + ttahara recipe, 20 epochs.

| Slug | Change from v5g | Head | Loss | Best val/auroc | Best Epoch | LB | Checkpoint |
|------|----------------|------|------|---------------|------------|-----|------------|
| `hgnet-b0-sed-v5h` | n_fft 2048→4096 | LSEHead(mean) | bce | 0.869 | 16 | **0.893** | `017-.../2026-hgnet-b0-sed-v5h/best-epochepoch=16.ckpt` |
| `hgnet-b0-sed-v5i` | window 5s→10s | LSEHead(mean) | bce | **0.878** | 17 | **0.880** | `017-.../2026-hgnet-b0-sed-v5i/best-epochepoch=17.ckpt` |
| **`hgnet-b0-sed-v5j`** | **AmplitudeToDB norm†** | LSEHead(mean) | bce | 0.874 | 19 | **0.906** | `017-.../2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt` |
| `hgnet-b0-sed-v5k` | 3-channel mel (in_chans=3) | LSEHead(mean) | bce | 0.868 | 16 | **0.886** | `017-.../2026-hgnet-b0-sed-v5k/best-epochepoch=16.ckpt` |

| `hgnet-b0-sed-v5l` | v5j + n_fft=4096 | LSEHead(mean) | bce | 0.874 | 16 | **0.904** | `017-.../2026-hgnet-b0-sed-v5l/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v5m` | EffNet-B0 + proj 320→1024‡ | LSEHead(mean) | bce | **0.873** | 16 | **0.898** | `017-.../2026-effnet-b0-sed-v5m/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v5n` | EffNet-B0, no proj (v5j recipe)‡‡ | LSEHead(mean) | bce | 0.871 | 19 | **0.875** | `017-.../2026-effnet-b0-sed-v5n/best-epochepoch=19.ckpt` |
| **`hgnet-b0-sed-v5j-4fold`** | **AmplitudeToDB norm† ×4** | LSEHead(mean) | bce | 0.951/0.843/0.823/0.808 | 15/19/19/18 | **0.913** | `017-.../2026-hgnet-b0-sed-v5j/fold{0-3}/` |

† v5j confounded: also changes win_length (626→2048), removes resize (256×256→256×511), uses NNAudio.
‡ v5m uses EfficientNet-B0 (320 native embed_dim) + 1x1 Conv2d projection to 1024. Tests whether HGNet's advantage is from embed_dim. All other params match v5j.
‡‡ v5n uses EfficientNet-B0 with v5j recipe (NNAudio+AmplitudeToDB) but NO projection (native 320 embed_dim). Control for v5m.

**Note on inference post-processing**: All LB scores from v5g onward include
inference-time TTA (±2.5s shifts, 50/25/25 weighting), temporal smoothing
(Gaussian), and TopN filtering, enabled globally in `submission.py`
(`ENABLE_TTA=True`, `ENABLE_TEMPORAL_SMOOTHING=True`, `ENABLE_TOPN=True`).
These are baked into every score — there is no "raw" v5j LB without them.

## BirdCLEF 2026 — Perch Probes (exp 018)

Frozen Perch v2 as feature extractor + probe classifiers. No GPU training —
probes train in ~2s on CPU. All inference on Kaggle CPU kernel (~85 min for ~600 test files).

| Slug | Pipeline | Probe Features | OOF | LB | Delta | Kaggle Kernel |
|------|----------|---------------|-----|-----|-------|---------------|
| `perch-raw` | Perch logits only | — | No | 0.825 | baseline | `submission_perch_raw.py` |
| `perch-cache-v1` | + genus proxy mapping | — | No | 0.836 | +0.011 | `submission_perch_cache.py` |
| `perch-probes-v1` | + LogReg probes, logit-space | 39-dim (PCA32 + 7) | No | 0.893 | +0.057 | `submission_perch_probes.py` v1 |
| `perch-probes-v2` | + OOF fold-safe priors | 39-dim (PCA32 + 7) | 5-fold | 0.898 | +0.005 | `submission_perch_probes.py` v2 (buggy) |
| **`perch-probes-v2-fix`** | **+ 6-cat fuse, sc_clean priors** | **39-dim (PCA32 + 7)** | **5-fold** | **0.911** | **+0.013** | **`submission_perch_probes.py` v2-fix** |
| `perch-protossm-v4` | + BiSSM×2 + cross-attn + TTA + ResidualSSM + 6-stage PP | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.918** | +0.007 | `submission_perch_protossm.py` v1 |
| `perch-protossm-v4.1` | v4 + OOF prior by filename, per-stage clipping, all-data train | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.918** | +0.000 | `submission_perch_protossm.py` v4 |
| `perch-protossm-v4.2` | v4.1 + taxonomy 5-way, fixed ensemble 0.5, MLP seed, 1-based sites | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.899** | **−0.019** | `submission_perch_protossm.py` v5 |

### Perch Probe Progression Analysis

| Transition | Key Change | LB Delta | Why it helped |
|-----------|-----------|----------|---------------|
| raw → cache | Genus proxy mapping for unmapped species | +0.011 | 31 unmapped species get non-zero predictions |
| cache → probes-v1 | LogReg probes on PCA-compressed embeddings | +0.057 | Learned species-specific decision boundaries |
| v1 → v2-fix | OOF fold-safe priors, 6-category fusion, sc_clean | +0.013 | Proper prior estimation, bug fixes (K_HOUR, labeled subset) |
| v2-fix → v4 | ProtoSSM temporal modeling + ResidualSSM + TTA | +0.007 | Captures temporal patterns across 12 windows |
| v4 → v4.2 | 4 simultaneous changes | **−0.019** | Batched changes; cannot isolate which one broke it |

## BirdCLEF 2026 — Perch ProtoSSM v5 (exp 020)

Faithful port of pantanal-distill V17 (0.930 LB reference). V17 config: d_model=256,
3 BiSSM layers, cross-attention, focal BCE + distill + mixup, 40 epochs. Uses
jaejohn/perch-meta cache for labeled embeddings. All training on Kaggle CPU kernel.

| Slug | Pipeline | Config | LB | Kaggle Kernel |
|------|----------|--------|-----|---------------|
| `perch-protossm-v5-v2` | pantanal port (V18 config, no cache) | d=320, 4L, 80ep | timeout | `020-.../submission_perch_protossm.py` v2 |
| `perch-protossm-v5-v3` | pantanal port (V18 config, cache) | d=320, 4L, 80ep | timeout | `020-.../submission_perch_protossm.py` v3 |
| `perch-protossm-v5-v4` | pantanal port (V17 config, cache) | d=256, 3L, 40ep | **0.922** | `020-.../submission_perch_protossm.py` v4 |
| `perch-protossm-v5-v5` | V17+ONNX, MLP-only (w=0.0) | d=256, 3L, 40ep | 0.911 | `020-.../submission_perch_protossm.py` v5 |
| `perch-protossm-v5-v7` | V18+ONNX | d=320, 4L, 80ep | 0.920 | `020-.../submission_perch_protossm.py` v7 |
| **`perch-protossm-v18`** | **two-kernel V18+TF** | **d=320, 4L, 80ep, TTA×5** | **0.925** | **`020-.../submission_perch_protossm.py` (infer_only)** |

V18 pipeline (two-kernel split): Training kernel saves pretrained artifacts to Kaggle dataset.
Inference kernel loads artifacts AFTER Perch TF inference (memory optimization: loading 75MB
before TF caused 2x slowdown). Pipeline: Perch v2 TF → MLP(256,128) probes (PCA-128) →
ProtoSSM v5 (d=320, 4 BiSSM, cross-attn, 80ep, 5-fold, focal γ=2.5, cosine restart, SWA) →
ResidualSSM (d=128) → TTA [0,±1,±2] → ensemble (0.5 proto + 0.5 mlp) →
6-stage PP (temp, confidence, rank-aware, delta-shift, V18 per-class thresholds).

Gap vs pantanal reference (0.930): −0.005. V18+TF gained +0.003 over V17+TF (0.922).

### v2-fix Details (0.911 LB)

Pipeline: Perch v2 frozen → StandardScaler + PCA(32) → 5-fold OOF GroupKFold by site →
52 LogReg probes (39-dim, C=0.25, class_weight='balanced') → Gaussian smoothing → sigmoid.

Key fixes over buggy v2 (0.898):
- K_HOUR = 8 (was 4)
- Prior tables from ALL 739 soundscape annotations (was labeled subset only)
- 6-category fuse_scores: proxy λ=0.8, prior-only REPLACE, inactive = -8.0
- Filtered to 59 fully-labeled files / 708 windows (was 66 files / 792 windows)

## BirdCLEF 2026 — SED Training Improvements (exp 021)

Post-processing and training recipe improvements targeting 0.920+ single-model LB.
Motivated by competitor insights from Kaggle discussion #683791.

**All exp 021 LB scores include TTA + temporal smoothing + TopN** (inherited from
`submission.py` global flags). See note under exp 019 Step 4.

### Phase A: Site/Hour Priors (no retraining)

Multiplicative site×hour×species priors from 66 labeled soundscapes (1478 segments,
9 sites, 13 hours). Taxa-aware lambda: Aves=0.3, Amphibia/Insecta=0.5, Mammalia/Reptilia=0.2.

| Slug | Base Model | Priors | LB | Delta | Notes |
|------|-----------|--------|-----|-------|-------|
| `hgnet-b0-sed-v5j` | HGNetV2-B0 single | No | **0.906** | — | Baseline (exp 019) |
| **`hgnet-b0-sed-v5j-pp2`** | HGNetV2-B0 single | **Yes** | **0.909** | **+0.003** | Site/hour priors, 11 min runtime |
| `hgnet-b0-sed-v5j-4fold` | HGNetV2-B0 ×4 | No | **0.913** | — | Baseline (exp 019) |
| **`hgnet-b0-sed-v5j-4fold-pp2`** | **HGNetV2-B0 ×4** | **Yes** | **0.917** | **+0.004** | **New best SED** |

Priors gave +0.003 single / +0.004 4-fold (not the +0.006 Salman reported). Possible
reasons: different prior implementation (multiplicative vs logit-space), different lambda
tuning, or model already captures site/time patterns that priors are redundant with.

### Phase B: Training Recipe Ablations

Single-variable changes from v5j (HGNetV2-B0, val/auroc 0.874, LB 0.906).
All trained with seed=42, deterministic=True, on PACE GPU.

| Slug | Change from v5j | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------------|-----------|----------|------------|-----|------------|
| `hgnet-b0-sed-v6a` | CE loss (primary_label) | 0.854 | 0.580 | 18 | pending | `021-.../2026-hgnet-b0-sed-v6a/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v6b` | Waveform MixUp | 0.871 | 0.625 | 16 | **0.887** | `021-.../2026-hgnet-b0-sed-v6b/best-epochepoch=16.ckpt` |
| `hgnet-b0-sed-v6c` | Waveform MixUp + noise + stretch | 0.861 | 0.608 | 16 | pending | `021-.../2026-hgnet-b0-sed-v6c/best-epochepoch=16.ckpt` |
| `hgnet-b0-sed-v6d` | 25 epochs (vs 20) | **0.873** | **0.651** | 22 | pending | `021-.../2026-hgnet-b0-sed-v6d/best-epochepoch=22.ckpt` |

CE loss (v6a) regressed −0.020 — single-label assumption hurts on multi-label soundscapes.
Waveform MixUp (v6b) confirmed hurts on LB: 0.887 (−0.019 vs v5j's 0.906). Spectrogram MixUp remains better.
25 epochs (v6d) essentially flat on AUROC but improved cmap (+0.026).

### Phase C: AttHead + Backbone Scaling

All use **AttHead with dual supervision** (clipwise_weight=0.5, framewise_weight=0.5,
blend 75% clip + 25% frame max). Replaces LSEHead from v5j. 20 epochs, seed=42.

| Slug | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|------|-----------|----------|------------|-----|------------|
| `hgnet-b2-sed-v6h` | HGNetV2-B2 | 12M | AttHead | BCE | **0.876** | **0.658** | 18 | **0.889** | `021-.../2026-hgnet-b2-sed-v6h/best-epochepoch=18.ckpt` |
| `convnextv2-nano-sed-v6i` | ConvNeXt-V2 Nano | 15M | AttHead | BCE | 0.874 | **0.663** | 16 | — | `021-.../2026-convnextv2-nano-sed-v6i/best-epochepoch=16.ckpt` |
| **`effnet-b0-ns-sed-v6j`** | **EffNet-B0 NS** | **5.3M** | AttHead | BCE | 0.874 | 0.626 | 15 | **0.893** | `021-.../2026-effnet-b0-ns-sed-v6j/best-epochepoch=15.ckpt` |
| `hgnet-b0-sed-v6e` | HGNetV2-B0 | 6M | AttHead | BCE | 0.870 | 0.611 | 18 | skip | `021-.../2026-hgnet-b0-sed-v6e/best-epochepoch=18.ckpt` |
| `effnet-b3-ns-sed-v6g` | EffNet-B3 NS | 12M | AttHead | CE | 0.859 | 0.611 | 19 | skip | `021-.../2026-effnet-b3-ns-sed-v6g/best-epochepoch=19.ckpt` |
| `hgnet-b0-sed-v6f` | HGNetV2-B0 | 6M | AttHead | CE | 0.850 | 0.577 | 18 | skip | `021-.../2026-hgnet-b0-sed-v6f/best-epochepoch=18.ckpt` |

### Phase F: Competition-Standard AttHead Fix (v6k)

Fixed AttHead to match Nikita 1st place 2025 / Tattaka exactly: pre-dropout(p/2),
softmax(tanh(x)) attention, probability-space clipwise. Dual supervision (50/50).
hidden_dim=1024 (matching v5j). Everything else identical to v5j.

| Slug | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|------|-----------|----------|------------|-----|------------|
| `hgnet-b0-atthead-v6k` | HGNetV2-B0 | 6M | AttHead (fixed) | BCE | 0.871 | 0.596 | 19 | **0.888** | `021-.../2026-hgnet-b0-atthead-v6k/best-epochepoch=19.ckpt` |

**Fixed AttHead regresses** −0.018 on LB (0.888 vs 0.906) and −0.003 offline.
Barely differs from old buggy AttHead v6e (0.871 vs 0.870). The three fixes
(pre-dropout, tanh, probability-space) are not the differentiator. AttHead
consistently underperforms LSEHead on HGNetV2-B0 on both offline and LB.

**Both Phase C AttHead models regressed on LB** vs v5j (LSEHead, 0.906): v6j 0.893
(−0.013), v6h 0.889 (−0.017). Offline metrics are anti-correlated with LB.

### Phase D: Multi-Context Training (AttHeadMultiContext)

Train on 15-20s windows with center 5s inference crop. `AttHeadMultiContext` head with
competition-standard attention (softmax(tanh(att)), probability-space clipwise).

| Slug | Backbone | Params | Head | Context | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|---------|-----------|----------|------------|-----|------------|
| `effnet-b0-ns-mc-sed-v7c` | EffNet-B0 NS | 5.3M | AttHeadMC | 3×5s (15s) | 0.851 | 0.605 | 19 | — | `021-.../2026-effnet-b0-ns-mc-sed-v7c/best-epochepoch=19.ckpt` |
| `hgnet-b0-mc-sed-v7a` | HGNetV2-B0 | 6M | AttHeadMC | 3×5s (15s) | 0.820 | 0.579 | 16 | — | `021-.../2026-hgnet-b0-mc-sed-v7a/best-epochepoch=16.ckpt` |
| `hgnet-b0-mc-sed-v7b` | HGNetV2-B0 | 6M | AttHeadMC | 4×5s (20s) | 0.799 | 0.542 | 16 | — | `021-.../2026-hgnet-b0-mc-sed-v7b/best-epochepoch=16.ckpt` |

**All v7 models massively regressed** vs v5j (0.874): v7c −0.023, v7a −0.054, v7b −0.075.
Longer context = worse. Changed too many variables from v5j (head type + window length +
batch size). Cannot isolate multi-context effect. hengck23 uses LSEHead with multi-context,
not AttHead — the head change likely caused most of the regression, not the context itself.

### Phase E: LSEHead + Temporal Split (correct multi-context)

Keep LSEHead exactly as v5j, add `n_chunks` to split temporal features into N chunks
before LSE pooling. Backbone sees full 15-20s context, each chunk pools at 5s resolution.
Inspired by hengck23 (0.896 LB, CNN-only, 3×5s). See `multi-context-data-flow.md`.

| Slug | Backbone | Head | Context | n_chunks | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|------|---------|----------|-----------|----------|------------|-----|------------|
| `hgnet-b0-mc-lse-sed-v8a` | HGNetV2-B0 | LSEHead | 3×5s (15s) | 3 | **0.882** | **0.719** | 13 | 0.882 (avg) / 0.883 (center) | `021-.../2026-hgnet-b0-mc-lse-sed-v8a/best-epochepoch=13.ckpt` |
| **`hgnet-b0-mc-lse-sed-v8b`** | **HGNetV2-B0** | **LSEHead** | **4×5s (20s)** | **4** | **0.881** | **0.727** | 9 | **0.896 (avg)** | `021-.../2026-hgnet-b0-mc-lse-sed-v8b/best-epochepoch=09.ckpt` |

Best offline metrics ever (+0.007-0.008 AUROC, +0.094-0.102 cmap vs v5j). LB results:
v8a 0.882 (avg) / 0.883 (center-chunk, +0.001 negligible), v8b **0.896** (avg). Center-chunk
inference didn't help — the offline-to-LB gap is not an inference strategy issue. v8b's
20s context outperforms v8a's 15s on LB (0.896 vs 0.883), suggesting longer backbone
context is valuable but the approach still regresses vs v5j's 0.906 single-model LB.

### Phase G: Manifold Mixup (v6l)

Manifold mixup inspired by Mix2 (Moummad et al. 2024). Each batch randomly alternates
between input-space spectrogram mixup (50%) and embedding-space manifold mixup (50%).
Single-variable change from v5j — same backbone, head, loss, schedule.

| Slug | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|------|-----------|----------|------------|-----|------------|
| `hgnet-b0-sed-v6l` | HGNetV2-B0 | 6M | LSEHead | BCE | 0.874 | 0.630 | 15 | **0.899** | `021-.../2026-hgnet-b0-sed-v6l/best-epochepoch=15.ckpt` |

**Manifold mixup regresses −0.007 on LB** (0.899 vs 0.906) despite matching v5j offline
(0.874). Embedding-space mixing disrupts soundscape generalization. Another training-time
change that fails to improve on v5j.

## BirdCLEF 2026 Models — PCEN Normalization (exp 022)

PCEN (Per-Channel Energy Normalization) replacing AmplitudeToDB in the mel pipeline.
All configs identical to v5j except normalization. Targets domain gap between focal
recordings and Pantanal soundscapes.

| Slug | Backbone | Params | Head | Normalization | val/auroc | val/cmap | Best Epoch | LB | Checkpoint |
|------|----------|--------|------|---------------|-----------|----------|------------|-----|------------|
| `hgnet-b0-sed-v10a` | HGNetV2-B0 | 6M | LSEHead | Learnable PCEN | 0.871 | 0.620 | 18 | **0.880** | `022-.../2026-hgnet-b0-sed-v10a/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v10b` | HGNetV2-B0 | 6M | LSEHead | Fixed PCEN | 0.870 | 0.610 | 18 | — | `022-.../2026-hgnet-b0-sed-v10b/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v10c` | HGNetV2-B0 | 6M | LSEHead | PCEN + AmplitudeToDB | 0.870 | 0.602 | 17 | — | `022-.../2026-hgnet-b0-sed-v10c/best-epochepoch=17.ckpt` |
| `hgnet-b0-sed-v5j-repro` | HGNetV2-B0 | 3.1M | LSEHead | AmplitudeToDB | 0.873 | — | 18 | **0.903** | `022-.../2026-hgnet-b0-sed-v5j-repro/best-epochepoch=18.ckpt` |

**v10a regressed −0.026 on LB** (0.880 vs 0.906). PCEN normalization hurts soundscape
generalization. v10b/v10c not submitted (worse offline than v10a). **v5j-repro scored
0.903 LB** (−0.003 vs original v5j 0.906), confirming pipeline is correct and PCEN
is the cause of regression. **Experiment rejected.**

## BirdCLEF 2026 — Perch + SED Ensemble (exp 023)

Score-level fusion of Perch ProtoSSM v18 (ONNX) and SED v5j-4fold-pp2 in a single
Kaggle CPU kernel. Each pipeline runs its own full post-processing, then weighted
average produces the final submission. Estimated runtime ~64 min for ~600 test files.

| Slug | W_PERCH | Perch Backend | SED Model | LB | Kaggle Kernel |
|------|---------|--------------|-----------|-----|---------------|
| `perch-sed-ensemble-w05` | 0.50 | ONNX | v5j-4fold-pp2 | 0.927 | `023-.../submission_ensemble.py` |
| `perch-sed-ensemble-w03` | 0.30 | ONNX | v5j-4fold-pp2 | 0.927 | `023-.../submission_ensemble.py` |
| `perch-sed-ensemble-w07` | 0.70 | ONNX | v5j-4fold-pp2 | 0.926 | `023-.../submission_ensemble.py` |
| **`perch-sed-ensemble-mlp-only`** | **0.50** | **ONNX** | **v5j-4fold-pp2** | **0.929** | **`023-.../submission_ensemble.py` (proto_weight=0.0)** |
| `perch-sed-ensemble-no-pp` | 0.50 | ONNX | v5j-4fold-pp2 | 0.922 | `023-.../submission_ensemble.py` (PP disabled) |

**0.929 LB is the new best overall** (MLP-only, +0.004 vs Perch TF 0.925).
Key findings:
- Dropping ProtoSSM (MLP-only) gains +0.002 — ProtoSSM trained on TF embeddings hurts
  under ONNX inference mismatch. Matches pantanal-distill 0.930 approach.
- Post-processing is worth +0.005 (0.927 with PP vs 0.922 without).
- Weight curve flat (0.926-0.927) — both models contribute equally.
- Next: TF Perch + single SED (recover 0.005 ONNX penalty, could reach 0.932+).

## Experiment 026: Domain-Shift Training

Dual validation (separate focal/soundscape metrics) + window length + AttHead ablations.
Goal: close the single-model gap (0.903 → 0.922+).

### Phase 2: v5j-dualval baseline

| Slug | Window | Head | val/auroc | val_sc/auroc | LB | Checkpoint |
|------|--------|------|-----------|-------------|-----|------------|
| `v5j-dualval` | 5s | LSEHead | 0.849 | 0.956 | 0.897 | `026-.../2026-exp026-v5j-dualval/best-epochepoch=17.ckpt` |

*val_sc/auroc is broken — macro AUROC over 234 species when only ~35 appear in
86 soundscape val segments. See `026-domain-shift-training/results.md`.

### Phase 3: AttHead + window ablation

All Phase 3 jobs ran with old broken val_sc/auroc (~0.093). Compare using val/auroc.

| Slug | Window | Head | val/auroc | val/cmap | val/top1 | LB | Checkpoint |
|------|--------|------|-----------|----------|----------|-----|------------|
| `v9a-10s-att` | 10s | AttHead | 0.859 | 0.645 | 0.827 | **0.868** | `026-.../last.ckpt` (ep19) |
| `v9b-15s-att` | 15s | AttHead | 0.858 | 0.664 | 0.847 | **0.877** | `026-.../last.ckpt` (ep19) |
| `v9c-20s-att` | 20s | AttHead | 0.861 | 0.681 | 0.862 | **0.878** | `026-.../last.ckpt` (ep19) |

All AttHead models regressed on LB despite best-ever offline metrics. Train/inference
window mismatch (train on 10-20s, infer on 5s) likely causes attention calibration issues.

## Experiment 027: Model Training Fixes

Fix two implementation errors identified by deep research: additive waveform MixUp
(x1+x2 with max labels) and CE loss on primary label (sigmoid at inference).

| Slug | Loss | MixUp | val/auroc | val/cmap | val_sc/auroc | LB | Checkpoint |
|------|------|-------|-----------|----------|-------------|-----|------------|
| `v10a-addmix-ce` | CE | additive wfm | 0.835 | 0.571 | 0.863 | **0.494** | `027-.../last.ckpt` (ep19) |
| `v10b-addmix-bce` | BCE | additive wfm | 0.844 | 0.592 | 0.966 | **0.875** | `027-.../best-epochepoch=17.ckpt` |

**v10a = 0.494 LB (catastrophic).** CE logits have ~1 unit spread centered at zero;
sigmoid maps all 234 species to ~0.50 → random on soundscapes. CE is now confirmed
dead (7 tests, 0 successes). Root cause: CE optimizes relative ranking (softmax),
not absolute magnitudes; logits lack safety margin for focal→soundscape domain shift.
See `027-model-training-fixes/results.md` for full analysis.

**v10b = 0.875 LB (-0.031).** Additive waveform MixUp regresses even with BCE.
Worse than interpolative waveform MixUp (v6b = 0.887). All waveform MixUp variants
regress vs spectrogram MixUp. Offline val_sc/auroc improved +0.010 but LB regressed
-0.031 — offline metrics anti-correlated with LB again.

### I-1: v5j-10s (overlapping window inference, no retraining)

Tested 10s inference window on v5j (2.5s context each side). LB = **0.906** — flat vs
v5j baseline (0.906). Existing TTA (±2.5s shifts) already provides similar overlap.
Inference-only change, no new model.

### Phase 4: Salman-derived structural changes

| Slug | Change | val/auroc | val/cmap | val_sc/auroc | LB | Checkpoint |
|------|--------|-----------|----------|-------------|-----|------------|
| `v5j-control` | v5j baseline (W&B control) | 0.849 | 0.596 | 0.955 | — | `027-.../best-epochepoch=14.ckpt` |
| `v11a-maxtrick` | max trick + RMS sampling | 0.804 | 0.516 | 0.953 | — | `027-.../best-epochepoch=18.ckpt` |
| `v11b-melnorm` | simple mel norm (X+80)/80 | 0.847 | 0.599 | 0.960 | **0.895** | `027-.../best-epochepoch=18.ckpt` |

v5j-control matches v5j-dualval (exp 026): val/auroc 0.849 vs 0.849. Pipeline verified.

## Experiment 025: Iterative Pseudo-Labeling

### Phase B: Row-addition (FAILED)

Added PLs as separate dataset rows with stratified batch sampling (50/50 GT:PL).

| Slug | Method | val/auroc | val/cmap | LB | Checkpoint |
|------|--------|-----------|----------|-----|------------|
| `v5j-pl-v1` | Row-addition (w=0.5, fold 0) | 0.872 | 0.620 | **0.897** | `025-.../best-epochepoch=18.ckpt` |

### Phase C: Waveform mixing (FAILED)

Mario-style PL waveform mixing: 30% of focal samples overlaid with random PL soundscape
segment (amplitude-controlled). Labels: max(focal, PL). Standard 3-tuple, v5j recipe unchanged.

| Slug | Method | Epochs | val/auroc | val/cmap | val_sc/auroc | Best Epoch | LB | Checkpoint |
|------|--------|--------|-----------|----------|-------------|------------|-----|------------|
| `v3-waveform-mix-20ep` | Waveform mixing (30%, 90/10) | 20 | 0.866 | 0.603 | — | 16 | — | `025-.../best-epochepoch=16.ckpt` |
| `v3-waveform-mix-40ep` | Waveform mixing (30%, 90/10) | 40 | 0.851 | 0.612 | 0.991 | 38 | **0.894** | `025-.../best-epochepoch=38.ckpt` |

**Result: 0.894 LB — regressed -0.012 vs v5j baseline (0.906).** Worse than Phase B
row-addition (0.897). Both pseudo-label approaches failed.

## Experiment 029: 2025 Pretrain → 2026 Fine-tune (Ali Ozan path)

Sequential transfer learning. Stage 1 pretrains v5j on BirdCLEF 2025 focal data
(28,564 rows, 206 species) from ImageNet init. Stage 2 (pending) fine-tunes on
2026 with the 41 (2025 ∩ 2026) overlap species' classifier rows transferred;
remaining 193 rows (165 2026-only + 28 soundscape-only) kept at random init.
Single-variable change from v5j: 2025-pretrained init vs ImageNet init.

Motivated by 2026-04-25 per-frame analysis (v5j is structurally honest on hard
crops; further training-recipe tweaks won't move LB; adding more *information*
via 2025 data is the credible remaining lever). Ali Ozan (0.941 LB) used a
similar pretrain → finetune pattern with filtered Xeno-Canto.

### Stage 1: Pretrain on 2025 — DONE

| Slug | Backbone | Head | Loss | val/auroc | val/cmap | val/top1 | Best Epoch | Walltime | Checkpoint |
|------|----------|------|------|-----------|----------|----------|------------|----------|------------|
| `hgnet-b0-sed-v5j-pretrain-2025` | HGNetV2-B0 | LSEHead | BCE | **0.972** | 0.649 | 0.761 | 15 | 30m16s | `029-.../checkpoints/2026-exp029-v5j-pretrain-2025/best-epochepoch=15.ckpt` |

Trajectory: 0.543 (ep 0) → 0.794 (ep 2) → 0.945 (ep 5) → 0.968 (ep 11) →
**0.972 (ep 15)** → plateau through ep 19. Clean convergence under
OneCycleLR(20 ep), no overfitting, no instability.

### Stage 2: Fine-tune on 2026 — DONE (REGRESSED)

| Slug | Init | LR | val/auroc | val/cmap | val/top1 | Best Epoch | Walltime | LB | Δ vs v5j | Checkpoint |
|------|------|----|-----------|----------|----------|------------|----------|-----|----------|------------|
| `hgnet-b0-sed-v5j-finetune-2026` | 2025-pretrained (Stage 1) | 2e-4 | **0.850** | 0.590 | 0.757 | 12 | 38m40s | **0.885** | **−0.021** | `029-.../checkpoints/2026-exp029-v5j-finetune-2026/best-epochepoch=12.ckpt` |

**Result: 0.885 LB — regressed −0.021 vs v5j baseline (0.906). REJECTED.**

Offline val/auroc 0.850 was essentially identical to v5j-control's 0.851
(Δ −0.001), but LB regressed clearly. Offline-LB anti-correlation
continues — this is the 22nd training-side change where offline metrics
gave no LB signal.

The result is consistent with the 2026-04-25 per-frame finding: v5j
sits at a soundscape-generalization equilibrium, and any training-side
perturbation (different init, different LR) tends to hurt LB even
when offline metrics look identical. 2025 pretraining is itself a
training perturbation — different starting point, different LR, mixed
warm/random head — and the result is what the structural argument
predicted.

Trajectory: 0.566 (ep 0, warm start) → 0.829 (ep 5) → 0.850 (ep 12, best)
→ plateau through ep 19. Earlier peak than v5j-control (~ep 14) consistent
with warm-started overlap species calibrating sooner.

**Apples-to-apples offline comparison** (same dual_validation schema):

| Slug | Init | Best val/auroc | Best Epoch | Δ |
|---|---|---:|---:|---:|
| v5j-control (exp 027) | ImageNet | 0.851 | ~14 | — |
| **exp 029 stage 2** | **2025-pretrained** | **0.850** | **12** | **−0.001** |

Offline result is essentially identical to v5j-control — within seed noise
(Mahog reports ±0.005). The 41-species warm start gave an epoch-0 head
start (0.566 vs ~0.5 random) but this washed out as the 193 random-init
classifier rows trained up. Offline being neutral does not predict LB
direction (offline-LB has been anti-correlated throughout this project).

Comparison baselines (LB):

- v5j single fold (exp 019): **0.906 LB** (ImageNet init, no dual_val)
- v5j-pp2 (exp 021): 0.909 LB (priors)
- v5j-4fold-pp2 (exp 021): 0.917 LB (4-fold + priors)
- Reference: Ali Ozan single model (filtered XC pretrain, much larger): 0.941 LB

**Decision: REJECTED.** Pretrain at our scale (28K 2025 rows, 41-species
overlap, reduced LR for warm start) does not transfer usefully to 2026
LB. Cheap retry candidate: `head_lr: 5e-4` with `lr: 2e-4` (split LR)
to test whether random-init head was undertrained. Otherwise, pivot to
exp 030 silent-class or Salman recipe replication.

## Experiment 030: Sydorskyi-faithful MixUp on v5j

A/B comparison testing whether replacing v5j's Zhang-style interpolated
spectrogram MixUp with Sydorskyi 2025 2nd-place's faithful waveform recipe
(`wave = (w1+w2)/2` + peak-normalize, `target = max(y1, y2)`, p=0.5,
deterministic 0.5 lambda, no theta cap, no warmup) moves LB.

Phase A: v5j HGNetV2-B0 ablation (single-variable mixup change).
Phase B (pending): EfficientNetV2-B0 backbone + 2x2 ablation.

Motivated by 2026-04-26 mixup audit (PR #42) — v5j and prior wave-mixup
attempts (v4j/v6b/v10b) all use interpolated multi-label targets, not
Sydorskyi's `max(y1, y2)`. Plus target-distribution simulation showing 80%
of v5j's training target values are sub-1, 59% have NO full positive.

### Phase A + Phase B: Training — DONE (LB PENDING)

All 4 cells use LSEHead + BCE + OneCycleLR (lr=5e-4) + 20 epochs + seed=42 +
dual_validation. Only backbone (Phase A vs B) and mixup recipe (control vs
sydorskyi) differ.

| Slug | Backbone | Mixup recipe | val/auroc | val/cmap | val/f1_macro | Best Ep | LB | Walltime | Checkpoint |
|------|----------|-------------|-----------|----------|--------------|---------|-----|----------|------------|
| `hgnet-b0-sed-v5j-control` | HGNetV2-B0 | spec, interp, theta=0.8 | **0.8490** | 0.5959 | 0.5039 | 14 | pending | 37m 44s | `030-.../2026-exp030-v5j-control/best-epochepoch=14.ckpt` |
| `hgnet-b0-sed-v5j-sydorskyi` | HGNetV2-B0 | wave, `(w1+w2)/2` + peak-norm, max(y1,y2), p=0.5 | 0.8472 | **0.6065** | **0.5467** | 14 | pending | 41m 45s | `030-.../2026-exp030-v5j-sydorskyi/best-epochepoch=14.ckpt` |
| `effnetv2b0-sed-control` | EffNetV2-B0 (192-ch features) | spec, interp, theta=0.8 | 0.8510 | 0.6124 | 0.5320 | 19 | pending | 41m 17s | `030-.../2026-exp030-effnetv2b0-control/best-epochepoch=19.ckpt` |
| **`effnetv2b0-sed-sydorskyi`** | **EffNetV2-B0** | **wave, sydorskyi** | **0.8518** | **0.6303** | **0.5751** | 19 | pending | 45m 37s | `030-.../2026-exp030-effnetv2b0-sydorskyi/best-epochepoch=19.ckpt` |

### 2x2 offline summary (val/auroc / val/cmap / val/f1_macro)

| | control (spec, interp) | sydorskyi (wave, max) |
|---|---|---|
| **HGNetV2-B0** (v5j arch) | 0.8490 / 0.5959 / 0.5039 | 0.8472 / 0.6065 / 0.5467 |
| **EfficientNetV2-B0** | 0.8510 / 0.6124 / 0.5320 | **0.8518 / 0.6303 / 0.5751** |

Main effects (averaged across the other axis):
- **Backbone** (v2-B0 − HGNet): +0.003 auroc, **+0.020 cmap**, **+0.028 f1_macro**
- **MixUp** (sydorskyi − control): −0.001 auroc, **+0.014 cmap**, **+0.043 f1_macro**
- Interaction: small positive on auroc/cmap (sydorskyi helps slightly more on v2-B0)

`effnetv2b0_sydorskyi` is the best cell on every offline metric. cmap +0.034
and f1_macro +0.071 vs v5j (HGNet + spec interp). LB will tell whether this
translates or hits the same offline-LB anti-correlation as our 22+ prior
recipe attempts.

### Reproducibility check (v5j_control)

| | val/auroc | Notes |
|---|---:|---|
| v5j-control (exp 027) | 0.851 | original baseline run |
| **v5j_control (exp 030)** | **0.849** | Δ −0.002, within seed noise |

**Pipeline reproducibility confirmed** — no codebase drift since exp 027.

### Phase B caveat (EffNetV2-B0 feature dim)

Encoder uses `tf_efficientnetv2_b0.in1k` stage-5 spatial features (192
channels), NOT the post-pool 1280-d projection set in the config. Both
v2-B0 cells share this same effective architecture, so the within-backbone
mixup comparison is internally clean. The v2-B0-vs-HGNet comparison is at
very different feature dims (192 vs 1024); take backbone main effect with
that caveat. A "faithful" v2-B0 setup might use a different feature
extraction layer with more channels.

See `experiments/030-sydorskyi-mixup/results.md` for full details.

## Checkpoint Base Path

All checkpoints under: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/`
