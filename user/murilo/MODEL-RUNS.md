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

## BirdCLEF 2026 — Perch Probes (exp 018)

Frozen Perch v2 as feature extractor + LogReg probe classifiers. No GPU training —
probes train in ~2s on CPU. All inference on Kaggle CPU kernel (~85 min for ~600 test files).

| Slug | Pipeline | Probe Features | OOF | LB | Kaggle Kernel |
|------|----------|---------------|-----|-----|---------------|
| `perch-raw` | Perch logits only | — | No | 0.825 | `submission_perch_raw.py` |
| `perch-cache-v1` | + genus proxy mapping | — | No | 0.836 | `submission_perch_cache.py` |
| `perch-probes-v1` | + LogReg probes, logit-space | 39-dim (PCA32 + 7) | No | 0.893 | `submission_perch_probes.py` v1 |
| `perch-probes-v2` | + OOF fold-safe priors | 39-dim (PCA32 + 7) | 5-fold | 0.898 | `submission_perch_probes.py` v2 (buggy) |
| **`perch-probes-v2-fix`** | **+ 6-cat fuse, sc_clean priors** | **39-dim (PCA32 + 7)** | **5-fold** | **0.911** | **`submission_perch_probes.py` v2-fix** |
| `perch-protossm-v4` | + BiSSM×2 + cross-attn + TTA + ResidualSSM + 6-stage PP | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.918** | `submission_perch_protossm.py` v1 |
| `perch-protossm-v4.1` | v4 + OOF prior by filename, per-stage clipping, all-data train | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.918** | `submission_perch_protossm.py` v4 (no improvement) |
| `perch-protossm-v4.2` | v4.1 + taxonomy 5-way, fixed ensemble 0.5, MLP seed, 1-based sites | 77-dim (PCA64 + 13) | 5-fold + 3-fold | **0.899** | `submission_perch_protossm.py` v5 (**REGRESSED −0.019**) |

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

## Checkpoint Base Path

All checkpoints under: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/`
