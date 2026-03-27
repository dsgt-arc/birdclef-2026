---
experiment: "018-perch-ensemble"
status: proposed
date: 2026-03-27
author: Murilo Gustineli
---

# Experiment 018: Perch Ensemble

## Hypothesis

Ensembling our EfficientNet-B0 SED model (v4b, 0.859 LB) with Google Perch v2
(frozen foundation model, ~0.910 LB standalone) will significantly outperform either
model alone due to fundamental feature diversity: our SED is trained on BirdCLEF focal
recordings with high-res mel spectrograms, while Perch v2 is a global bioacoustic
foundation model trained on 10K+ species with PCEN frontend.

## Motivation

### The orthogonality argument

Our SED model and Perch v2 are fundamentally different:

| Aspect | Our SED (v4b) | Perch v2 |
|--------|--------------|----------|
| Architecture | EfficientNet-B0 (3.6M params) | EfficientNet-B3 (~120M params) |
| Training data | BirdCLEF 2026 focal + soundscapes | Global bird audio (10K+ species) |
| Frontend | n_fft=4096, 224 mels, AmplitudeToDB | PCEN mel-spectrogram |
| Training | Supervised, dual clip+frame loss | Self-supervised + supervised |
| Strengths | Focal recording species, high-freq detail | Soundscape robustness, broad coverage |
| Weaknesses | Domain shift to soundscapes | 25 insect sonotypes unmapped |
| LB score | 0.859 | ~0.910 (with probes + priors) |

### Competitor validation

Every top-scoring approach uses Perch:
- chiranjithdharma: Perch + LogReg probes = **0.910 LB** (simplest)
- zirach: Perch + ProtoSSM = **0.913 LB**
- pantanal-distill: Perch + ProtoSSM v3 + ResidualSSM = **0.917 LB**
- Tom at 0.929: likely SED + Perch combo

No public notebook combines a trained SED model with Perch — this is our unique angle.

### Species coverage

- **Perch v2**: 14,795 species, covers 203/234 BirdCLEF species directly + 6 via genus proxy
- **25 unmapped**: All insect sonotypes (47158son01-25) — only our SED model covers these
- **Ensemble advantage**: SED handles insects, Perch handles the rest with stronger features

## Approach

### v1: Simple ensemble (chiranjithdharma-style Perch + SED blend)

1. Run Perch v2 frozen on test soundscapes → 1536-dim embeddings + 14,795 logits per 5s window
2. Map Perch logits to 234 BirdCLEF species (203 direct + 6 genus proxy)
3. Apply per-class LogReg probes on PCA-compressed embeddings (trained on labeled soundscapes)
4. Apply site/hour metadata priors with taxa-aware lambda weights
5. Blend with SED: `final = 0.3 * sed_probs + 0.7 * perch_probs`

### v2 (future): Add ProtoSSM temporal modeling

If v1 works, add bidirectional SSM across 12 windows per file (zirach's 0.913 approach).

## Perch v2 Model

- **Kaggle source**: `google/bird-vocalization-classifier/TensorFlow2/perch_v2/1`
- **Download**: `kaggle models instances versions download google/bird-vocalization-classifier/TensorFlow2/perch_v2/1`
- **Local path**: `/storage/project/ps-clef2026_birdclef-0/shared/models/perch-v2/`
- **Format**: TensorFlow SavedModel (362MB)
- **Labels**: `assets/labels.csv` (14,795 scientific names + 1 non-bird class)
- **eBird codes**: `assets/perch_v2_ebird_classes.csv`

## Species Mapping

| Category | Count | Method |
|----------|-------|--------|
| Direct match | 203 | Scientific name match in Perch labels |
| Genus proxy | 6 | max(sigmoid(logits[same_genus_species])) |
| Unmapped | 25 | All insect sonotypes — SED-only predictions |

## Baseline

- SED v4b: 0.859 LB (single model)
- Perch + probes (chiranjithdharma): 0.910 LB

## Success Criteria

- **Confirm if**: Ensemble LB > max(SED alone, Perch alone)
- **Strong confirm if**: LB > 0.920
- **Target**: 0.92+ LB

## Runtime Budget

| Component | Est. Time | Notes |
|-----------|-----------|-------|
| SED inference (v4b ONNX) | ~30 min | Full-signal mel + batched ONNX |
| Perch v2 inference (TF) | ~15 min | 12 windows/file, CPU |
| Probe application | <1 min | sklearn predict |
| Blending + CSV | <1 min | numpy ops |
| **Total** | **~45 min** | **Fits in 90-min Kaggle budget** |

## Dependencies

- Exp 010 submission pipeline (ONNX export, upload infrastructure)
- Exp 017 v4b model weights (best SED model)
- Perch v2 SavedModel (downloaded to shared/models/)
- 66 labeled soundscapes (for training probes)

## References

- [Perch Probes (0.910 LB)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md)
- [Perch ProtoSSM (0.913 LB)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Pantanal Distill (0.917 LB)](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md)
- [Gap analysis](../../user/murilo/notes/gap-analysis-2026-03-26.md)
- [Exp 017 README](../017-mel-overhaul/README.md)

## Leads to

- If positive: Exp 020 (K-fold SED + OpenVINO for multi-model budget)
- If very positive: Exp 021 (pseudo-labeling with ensemble as teacher)
- v2: Add ProtoSSM temporal modeling on top of v1
