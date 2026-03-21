---
experiment: "013-mixit-pseudo-labeling"
status: proposed
date: 2026-03-21
author: Murilo Gustineli
---

# Experiment 013: MixIT Source Separation + Pseudo-Labeling

## Hypothesis

Source separation via Bird MixIT, applied as offline preprocessing to 10,592
unlabeled soundscapes, will produce cleaner per-channel audio that yields
higher-confidence pseudo-labels from the exp 012 teacher ensemble. Training
on these pseudo-labels will improve LB score by 0.02–0.05 over exp 012 alone.

## Motivation

BirdCLEF 2026 provides 10,592 unlabeled soundscapes — 160× more soundscape data
than the 66 labeled files used in exp 012. Direct pseudo-labeling on raw soundscapes
produces noisy multi-label predictions because:

1. **Dense overlapping vocalizations** — avg 4.2 species per 5s segment
2. **Low-SNR species are missed** — classifiers trained on focal data ignore quiet background species
3. **Co-occurring species confuse the classifier** — nocturnal frog choruses have 300+ co-occurrences

Bird MixIT (ICASSP 2022) was designed exactly for this scenario. It separates bird
mixtures into M=4 channels without requiring clean reference sources. The paper showed
+6.3% CMAP on dense dawn chorus data by classifying separated channels independently
and taking max predictions per species.

## Background

### Bird MixIT

- Pretrained on Xeno-Canto + Macaulay Library (704 species) at 22.05 kHz
- Architecture: TDCN++ with learned filterbank, mask-based separation, mixture consistency
- Domain-matched training gives +5 dB SI-SNR over generic AudioSet model
- Critical finding: max over (M separated channels + original) beats separated-only AND original-only

See `docs/vendor/bird-mixit.md` and `docs/references/bird-mixit-paper.md` for details.

### Teacher Models (from exp 012)

- EAT, ConvNeXt, ProtoCLR — 234-class models fine-tuned on labeled soundscapes
- Architecturally diverse (transformer vs CNN vs conv-transformer)
- Different pretraining domains (AudioSet vs ImageNet vs Xeno-Canto)
- Complementary error profiles make them ideal for ensemble pseudo-labeling

## Method

### Phase 1: TF1 → ONNX Conversion

Convert Google's pretrained Bird MixIT TF1 SavedModel to ONNX for portable inference:
- Use `tf2onnx` to convert the frozen TF1 graph
- Input: mono waveform at 22.05 kHz → Output: 4 separated channels
- Validate ONNX output matches TF1 output on test audio (atol < 1e-4)
- Benchmark CPU inference speed (target: <1s per 1-minute file)

### Phase 2: Offline Separation

Run MixIT on all 10,592 unlabeled soundscapes:
- Resample from 32 kHz → 22.05 kHz for MixIT input
- Save 4 separated channels per file as compressed numpy arrays
- Estimated storage: ~10,592 × 4 channels × ~160 KB ≈ 6.6 GB
- SLURM job: CPU-only, ~2–4 hours with batched processing

### Phase 3: Teacher Ensemble Pseudo-Labeling

Run exp 012 ONNX models on each separated channel + original:
- Per file: 1 original + 4 separated = 5 audio streams
- Per stream: 12 segments × 3 models = 36 predictions per file
- Aggregation: max-pool across channels per model, then average across 3 models
- Apply confidence threshold (e.g., 0.3–0.7) to filter low-confidence labels
- Save as CSV: filename, start, end, soft species probabilities

Validate pseudo-label quality on 66 labeled soundscapes (ground truth available).

### Phase 4: Student Training

Fine-tune from exp 012 checkpoint on pseudo-labeled data:
- Soft BCE loss (teacher probabilities as targets, not hard 0/1)
- Mix with focal + labeled soundscape data to prevent drift
- Compare: hard labels vs soft labels, threshold sweep

### Variables

- **Independent**: Separation (yes/no), aggregation strategy, confidence threshold
- **Dependent**: LB score, pseudo-label precision/recall on labeled soundscapes
- **Controlled**: Same models, same training pipeline as exp 012

## Evaluation

### Metrics
- **Pseudo-label quality**: Precision/recall vs expert labels on 66 labeled soundscapes
- **LB score improvement** over exp 012 baseline
- **Per-class improvement**, especially 28 soundscape-only species

### Baseline
- Exp 012 LB score (pending)

### Success Criteria
- **Confirm if**: LB improves by ≥0.02 over exp 012
- **Reject if**: No LB improvement or degradation on focal species

## Design Decisions

1. **Offline separation only** — inference-time MixIT is infeasible within 90-min Kaggle limit
2. **Max-pool channels, average models** — if any channel captures a species, max-pool retains it; model averaging reduces noise
3. **Soft pseudo-labels** — preserve teacher confidence calibration (vs hard 0/1)
4. **Validate on labeled soundscapes first** — measure pseudo-label quality before training
5. **Pretrained ONNX model first** — use Google's Bird MixIT before considering PyTorch reimplementation
6. **Domain-matched training as future work** — if ONNX results are promising, train a BirdCLEF-specific separator via Asteroid

## Limitations

- MixIT separation is imperfect; channels may contain artifacts or over-separated signals
- Sample rate mismatch: MixIT at 22.05 kHz, BirdCLEF at 32 kHz (loses info above 11 kHz)
- Pseudo-label noise compounds with separation artifacts
- Computational cost: separation + multi-model inference on 10,592 files
- TF1 dependency for Phase 1 (conversion only)

## Dependencies

- Exp 012 completed (234-class teacher models)
- Bird MixIT TF1 checkpoint downloaded from GCS
- `tf2onnx` for model conversion, `onnxruntime` for inference

## Leads to

- Potential PyTorch MixIT reimplementation (Asteroid-based) if domain-matched training is needed
- Student distillation for efficient Kaggle inference (future exp)
