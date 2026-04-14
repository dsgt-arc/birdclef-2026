# Gemini Deep Research: BirdCLEF 2026 Strategy (0.906 → 0.93+)

- **Source**: Gemini Deep Research (unbiased — no context of our experiments)
- **Prompt**: Strategy to go from 0.906 to 0.93+ single model in BirdCLEF 2026
- **Date**: 2026-04-13
- **Context given**: Competition link, discussion thread, current score (0.906)
- **Context NOT given**: Our specific experiments, failures, architecture details

---

# Strategies for Attaining 0.93+ Macro-Averaged ROC-AUC in BirdCLEF 2026

## Executive Summary

Advancing a passive acoustic monitoring (PAM) classification system from a highly
competent 0.906 baseline to a robust 0.930+ score on the Kaggle BirdCLEF 2026 public
leaderboard necessitates a fundamental departure from standard supervised learning
paradigms. While baseline architectures relying on EfficientNet backbones paired with
standard Sound Event Detection (SED) heads provide a strong initialization, they
inherently plateau when subjected to the severe domain shift characteristic of the
Pantanal wetland test environment.

The optimal roadmap to transcending the 0.930 threshold with a single model relies on:

1. **Multi-Iterative Noisy Student (MINS) training** — utilizing unlabeled soundscapes
   to generate power-transformed pseudo-labels
2. **Contrastive Adversarial Domain (CAD) bottlenecks** — aligning focal and soundscape
   feature spaces
3. **Architectural upgrade** — higher-capacity convolutional backbones (ConvNeXt) from
   the timm ecosystem
4. **CPU inference optimization** — OpenVINO compilation for the 2-hour time constraint

---

## Ecological Context and the Pantanal Acoustic Challenge

The BirdCLEF 2026 challenge focuses on automatic identification of wildlife species
from vocalizations in audio recordings collected across the Pantanal wetlands. The
Pantanal is a habitat for over 650 bird species alongside countless amphibians, mammals,
and insects.

The ML challenge arises from the stark environmental differences between how training
data is collected and how models are deployed. Field microphones capture everything
simultaneously — multi-species soundscapes with heavy distortion from environmental
noise (rain, wind, water, anthropogenic sounds).

---

## The Mathematical Reality of the Evaluation Metric

The competition uses a **row-wise macro-averaged ROC-AUC** where classes with zero true
positive labels in a specific evaluation slice are skipped and ignored.

Key implications:

- **Egalitarian class weighting**: Every species contributes equally regardless of
  frequency. A ubiquitous bird carries the same weight as a rare insect sonotype.
- **Rare class failure = massive penalty**: When a model ignores underrepresented
  classes, it incurs massive penalties that drag down the macro-average.
- **A model at 0.906 has likely mastered the "head" but is failing on the "tail"**
  (rare birds, amphibians, insect sonotypes). Surpassing 0.930 requires explicitly
  targeting tail classes.

---

## Anatomy of the Dataset and Domain Shift

| Data Modality | Volume | Taxonomic Dominance | Signal | Environment |
|---|---|---|---|---|
| Focal Audio (train_audio) | ~35,000 clips | Aves (97.9%) | High SNR, clean, targeted | Global, single-species, directional mics |
| Labeled Soundscapes | 66 segments | Amphibia & Insecta | Low SNR, faint, overlapping | Pantanal-specific, omnidirectional |
| Unlabeled Soundscapes | ~600-10,500+ files | Unknown | Low SNR, heavy ambient noise | True test environment representation |

### The Inversion of Taxonomic Dominance

The focal data is 97.9% birds. But in labeled soundscapes, acoustic presence is
concentrated in **Amphibia** (4,174 mentions) and **Insecta** (1,136 mentions), with
Aves trailing (824 mentions). A model trained naively on focal data develops strong
bird bias and significant blind spots for dominant soundscape actors.

### Zero-Shot Realities: The 159 Missing Soundscape Species

159 of 234 species have zero soundscape training data. When the model learns a bird
call from clean focal recordings, it often uses the silence or background as a
discriminatory feature. When that call occurs in a test soundscape buried under rainfall
and insect stridulation, the feature extractors fail.

**Bridging this gap requires aggressive augmentation that synthetically places clean
focal recordings into noisy environments.**

### The "Invisible 28": Total Zero-Shot Constraints

28 target classes have zero focal audio — almost entirely 25 insect sonotypes and
3 amphibian species. These appear only as secondary labels in the 66 labeled soundscapes.
A standard training loop iterating over train_audio entirely ignores these classes.

**Any model at 0.906 is almost certainly achieving 0.0 on these 28 columns.**

---

## Validation Framework Engineering

### Constructing Validation Splits

Standard randomized K-Fold is disastrous for continuous audio — successive 5-second
segments from the same soundscape are highly correlated. Required approach:

- **Strict Grouping**: Group by `site_id + date + time`. All segments from the same
  recording session must be in the same fold.
- **Taxonomic Stratification**: Ensure rare species and labeled soundscapes are
  distributed evenly across folds.
- **Fast Replay Protocol**: Use 20% focal + 100% labeled soundscapes for rapid
  hypothesis testing before full 5-fold runs.

### Granular Evaluation Metrics

Track beyond global loss:
- **Per-class PR-AUC** — sensitive to rare class performance
- **Subgroup robustness** — segment by time of day, geographic site
- **Exact competition metric** — row-wise macro-averaged ROC-AUC with zero-positive
  class skipping

---

## Single-Model Architectural Evolution

### Feature Extraction: Beyond EfficientNet

**ConvNeXt** architectures (convnext_tiny or convnext_small from timm) have specific
advantages for audio:
- Larger kernel sizes (7x7 convolutions) capture wide-band harmonic structures
- Inverted bottlenecks and spatial depthwise convolutions
- Better at low-frequency amphibian calls than EfficientNet's 3x3 convolutions
- Single ConvNeXt models achieve **0.922-0.925 LB** without ensembling

### Integration of Global Audio Embeddings

Perch v2 (EfficientNet-B3, 1,536-dim, ~14,795 species) achieves 0.87-0.88 with just
KNN. But pure Perch caps around 0.899 — embeddings are too rigid for Pantanal-specific
tuning.

**Optimal approach**: Extract Perch embeddings as auxiliary inputs to the ConvNeXt
classification head. Injects global acoustic context while allowing ConvNeXt to
specialize on local taxonomy.

### Engineering the SED Head

Standard GAP destroys temporal localization of brief bird calls within 5s clips.
Required: attention-based SED head that preserves temporal dimension.

- Framewise linear layer → raw classification probabilities per time step
- Attention mechanism → importance weights per time step
- Clip-level prediction = attention-weighted sum

During inference: overlapping 5s windows (2.5s stride) → smoothed framewise predictions
→ aggregated, reducing false positives from transient noise.

### Domain Alignment via Contrastive Adversarial Domain (CAD) Bottlenecks

Force the feature extractor to produce embeddings indistinguishable between focal and
soundscape domains:

1. **Domain Discriminator**: MLP branched off bottleneck, binary task (focal vs soundscape)
2. **Gradient Reversal Layer (GRL)**: Identity on forward pass, multiply gradient by
   -lambda on backward pass
3. **Adversarial minimax game**: Discriminator minimizes domain classification error;
   feature extractor (via reversed gradients) maximizes it — scrubbing domain-specific
   acoustic signatures
4. **Supervised Contrastive Loss (SupCon)**: Ensures same-class embeddings cluster
   tightly regardless of domain

Uses ~600 unlabeled soundscapes as target domain examples.

---

## The Multi-Iterative Noisy Student (MINS) Paradigm

### Generation of Pseudo-Labels

1. Train robust Teacher on labeled data (focal + 66 labeled soundscapes)
2. Deploy Teacher on all unlabeled soundscapes → framewise soft predictions
3. Save predictions with high temporal resolution (4 frames per 5s segment)

### Power Transformation and Confidence Calibration

Raw pseudo-labels contain noise. Apply power transformation:

```
y_transformed = y^lambda    (lambda = 2 or 3)
```

- High-confidence predictions preserved: 0.9^2 = 0.81
- Low-confidence noise suppressed: 0.2^2 = 0.04

Dynamic sampling weight per soundscape = sum of max pseudo-label probabilities
(biases toward segments with rich bioacoustic activity).

### Iterative Distillation with Aggressive MixUp

Student model trains on combined: focal + labeled soundscapes + power-transformed PLs.

**Cross-domain MixUp**: Blend clean focal data with pseudo-labeled soundscapes at
waveform level. Dual purpose:
1. Regularization (Noisy Student methodology)
2. Synthetic simulation of test conditions (bird signals + Pantanal background noise)

Iterate 2-3 full cycles: Student → new Teacher → re-predict → retrain.

---

## Taxonomy-Guided Hierarchical Learning for Zero-Shot Taxa

### Exploiting the 66 Labeled Soundscapes

- **Aggressively oversample** during training (not just validation)
- Extract insect sonotype segments as **dynamic background noise** — layer beneath
  focal samples to force network to recognize insect presence as distinct class

### Structural Classification via Taxonomy

Instead of flat 234-class output, structure head hierarchically:
1. Predict broader taxonomic Class (Aves, Amphibia, Insecta, Mammalia)
2. Conditionally predict specific species/sonotype

If model detects high-frequency insect pulsing → propagate constructive gradients to
specific sonotype nodes even with low raw confidence.

Optional: parallel lightweight classifier focused only on Amphibia + Insecta as
auxiliary loss signal.

---

## Advanced Loss Functions and Audio Augmentations

### Loss Function

Weighted hybrid of BCE and Focal Loss:

```
L = gamma * L_BCE + (1 - gamma) * L_Focal
```

Focal Loss down-weights easy negatives (silence, common birds), focusing gradient on
hard/faint/overlapping vocalizations.

### Secondary Label Masking

Focal recordings have unannotated background species. Options:
- **Dynamic masking**: Set gradient to zero for unlisted secondary classes (don't
  penalize correct detections)
- **Label smoothing**: Assign secondary labels target of 0.3-0.5 instead of 1.0

### Multi-Domain Augmentations

**Waveform domain:**
- Gaussian noise injection
- Pink noise (wind simulation)
- Dynamic range compression
- **Background noise from unlabeled Pantanal soundscapes** mixed at varying SNRs

**Spectrogram domain:**
- SpecAugment (aggressive time + frequency masking)
- Forces reliance on full-spectrum harmonic structures rather than single dominant
  fundamental frequency

---

## CPU Inference Optimization

### OpenVINO and ONNX Runtime Conversion

OpenVINO specifically optimized for Intel CPUs — graph fusion, layout optimization,
AVX-512 vectorization. **Up to 3x speedup** over PyTorch. ConvNeXt + OpenVINO:
entire test pipeline in ~18 minutes of 120-minute allowance.

Dynamic INT8 quantization: compress memory + accelerate with <0.005 AUC impact.

### Multithreaded Dataloading

Use `concurrent.futures.ThreadPoolExecutor` for async audio decoding while CPU runs
inference on previous batch.

### Vectorized Processing

- Batch STFT/mel-filterbank operations (batch size 16-32)
- Sliding 5s window with 2.5s stride → overlapping predictions averaged
- Ensemble-smoothes output through temporal dimension

---

## Strategic Implementation Sequence

### Phase 1: Validation Security and Metric Fidelity
Establish hybrid Grouped + Stratified 5-fold CV using soundscape site_id and temporal
metadata. Ensure local metric exactly mirrors competition's row-wise macro-averaged
ROC-AUC.

### Phase 2: Architectural Scaling
Replace EfficientNet with convnext_small from timm. Implement attention-based SED head
for framewise predictions.

### Phase 3: Domain Alignment
Implement CAD bottleneck using ~600 unlabeled soundscapes for domain invariance.

### Phase 4: Noisy Student Distillation
Teacher inference on unlabeled soundscapes → power transform (lambda~2.0) → suppress
noise → retrain Student with heavy cross-domain MixUp.

### Phase 5: Zero-Shot Taxonomy Resolution
Isolate and oversample 66 labeled soundscapes with "Invisible 28" insect sonotypes.
Integrate hierarchical loss using taxonomy.csv.

### Phase 6: Inference Hardening
Convert to OpenVINO. Implement async audio loading + vectorized STFT. Verify <90min
local execution.
