---
title: "Improving Bird Classification with Unsupervised Sound Separation"
authors: "Tom Denton, Scott Wisdom, John R. Hershey (Google Research)"
venue: "ICASSP 2022"
arxiv: "https://arxiv.org/abs/2110.03209"
date_captured: "2026-03-13"
foundational_paper: "Unsupervised Sound Separation Using Mixture Invariant Training (Wisdom et al., NeurIPS 2020)"
related_vendor: "docs/vendor/bird-mixit.md"
---

# Improving Bird Classification with Unsupervised Sound Separation

**Paper**: [arXiv:2110.03209](https://arxiv.org/abs/2110.03209) (Denton, Wisdom, Hershey — Google Research, ICASSP 2022)

**Foundational method**: [Unsupervised Sound Separation Using MixIT](https://arxiv.org/abs/2006.12701) (Wisdom et al. — NeurIPS 2020)

**Blog post**: [Separating Birdsong in the Wild for Classification](https://research.google/blog/separating-birdsong-in-the-wild-for-classification/) (Google Research, Jan 2022)

**Plain-language explainer**: See [bird-mixit-explained.md](bird-mixit-explained.md) for a step-by-step intuitive walkthrough.

**Model & code**: See [docs/vendor/bird-mixit.md](../vendor/bird-mixit.md) for checkpoints, API, and PyTorch porting considerations.

## TL;DR

Training a MixIT source separation model specifically on bird audio (Xeno-Canto) produces +5 dB SI-SNR improvement over a general AudioSet model. Combining this separator with a downstream EfficientNet classifier — by taking the max probability over separated channels and the original audio — improves classification across three independent soundscape datasets. The paper also demonstrates that taxonomic multi-task training, low-pass augmentation, and PCEN channel normalization each independently improve classifier quality.

## 1. The Core Problem

Field recordings of birds have two intertwined problems:

1. **Overlapping vocalizations** — especially during dawn chorus, many species vocalize simultaneously.
2. **Weak labels with missing background species** — training data carries a single primary species label even when multiple species are present. Background species are unlabeled. This creates a domain shift: classifiers learn to ignore low-SNR vocalizations because they appear "unlabeled" in training. Common species like red-winged blackbird frequently appear unlabeled in backgrounds, teaching the classifier that quiet blackbird calls = no bird.

## 2. MixIT: Mixture Invariant Training

MixIT (Wisdom et al., NeurIPS 2020) solves the chicken-and-egg problem of needing clean sources to train a separator when clean sources don't exist.

### 2.1 Mixtures of Mixtures (MoM)

Two real-world recordings x₁, x₂ ∈ ℝᵀ are summed to create a "mixture of mixtures": x̄ = x₁ + x₂. Since each recording already contains multiple overlapping sounds, x̄ is even more complex — but crucially, we know which sounds came from which original recording.

### 2.2 Separation and Assignment

The model fθ predicts M source estimates ŝ = fθ(x̄) ∈ ℝᴹˣᵀ from the MoM. Each estimated source must be assigned to one of the two reference mixtures. The MixIT loss finds the optimal binary assignment matrix A ∈ 𝔹²ˣᴹ (each column sums to 1, meaning each source is assigned to exactly one reference):

```
L_MixIT({xₙ}, ŝ) = min_{A ∈ 𝔹^{2×M}} Σₙ L(xₙ, [Aŝ]ₙ)
```

This means: assign each of the M separated sources to one of the two reference mixtures, then compute the reconstruction loss between each reference and the sum of its assigned sources. Choose the assignment that minimizes total reconstruction loss.

### 2.3 Signal-Level Loss

Negative thresholded SNR:

```
L(y, ŷ) = -10 log₁₀ (‖y‖² / (‖y - ŷ‖² + τ‖y‖²))
```

where τ = 10^(-SNR_max/10) acts as a soft ceiling (SNR_max = 30 dB). This prevents already well-separated examples from dominating gradients.

### 2.4 Key Properties

- M is an **upper bound** on sources, not a fixed count. The model can produce fewer than M active sources by outputting near-zero signals for unused channels.
- The assignment search is over 2^M binary matrices (M=4 → only 16 possibilities). Tractable via brute force.
- No clean reference sources are needed — only pairs of real-world mixtures.

## 3. Separation Model Architecture (TDCN++)

The separator is a **masking-based architecture** operating in a learned time-domain basis (not STFT):

```
Input waveform → Learnable analysis basis (Conv1D)
                      ↓
              TDCN++ (Improved Time-Domain Conv Net)
                      ↓
              M sigmoid masks
                      ↓
              Mask × analysis coefficients
                      ↓
              Learnable synthesis basis (overlap-add)
                      ↓
              Mixture consistency projection
                      ↓
              M source waveforms (sum = input)
```

**Learnable basis**: Based on Conv-TasNet (Luo & Mesgarani, 2019). The input audio is projected into a learned filterbank domain via a 1D convolution (analysis), masks are applied, and waveforms are reconstructed via a transposed convolution (synthesis). This is more flexible than fixed STFT representations.

**TDCN++**: Improved time-domain convolutional network (Kavalerov et al., WASPAA 2019). Predicts M masks through sigmoid activation. Consists of stacked dilated convolutional blocks with skip connections.

**Mixture consistency projection** (Wisdom et al., ICASSP 2019): Constrains the M separated sources to sum exactly to the input mixture. This is a hard constraint applied after masking — not just a loss term. Ensures no energy is created or lost.

## 4. Bird-Specific Model Configuration

| Parameter | Value |
|-----------|-------|
| Training data | Xeno-Canto + Macaulay Library, 704 species |
| Sample rate | 22.05 kHz |
| Clip duration | 6 seconds |
| Output sources (M) | 4 (paper) or 8 (released checkpoint) |
| Learnable basis window | 1 ms (~22 samples at 22.05 kHz) |
| Learnable basis hop | 0.5 ms |
| Learnable basis coefficients | 256 |
| Loss | Negative thresholded SNR, SNR_max = 30 dB |

## 5. Training Data Activity Detection

All training data is weakly labeled (single primary species per file). Before training, an activity detector selects likely-active windows:

1. Compute log mel-spectrogram and per-frame energy
2. Apply wavelet peak detector (`scipy.signal.find_peaks_cwt`) to find high-energy frames
3. Extract 6-second windows around each peak, select up to 5 windows per recording in decreasing energy order

This reliably finds the labeled species but occasionally picks background vocalizations or human speech.

## 6. Separation Results

Evaluated on 3,856 MoMs from held-out Xeno-Canto data. Metric: MoMi (mean SI-SNR improvement over estimated mixtures).

| Train Data | Input SR | Eval SR | Window | MoMi |
|-----------|----------|---------|--------|------|
| AudioSet | 16 kHz | 16 kHz | 2.5 ms | 4.4 dB |
| Xeno-Canto | 22.05 kHz | 16 kHz | 2.5 ms | 9.6 dB |
| Xeno-Canto | 22.05 kHz | 16 kHz | 1.0 ms | 10.4 dB |
| **Xeno-Canto** | **22.05 kHz** | **22.05 kHz** | **1.0 ms** | **10.5 dB** |

**Key takeaways**:
- Domain-matched training (Xeno-Canto vs AudioSet) provides **+5 dB** — the single largest factor.
- Smaller analysis window (1 ms vs 2.5 ms) provides an additional +1 dB — finer temporal resolution helps for bird calls with rapid frequency modulation.

## 7. Downstream Classification Pipeline

### 7.1 Separate+Classify Strategy

```
Input (5s audio)
    ├──→ Separation model → M source channels
    │         │
    │         ↓
    │    Classify each channel independently
    │         │
    ↓         ↓
Classify original audio
    │
    ↓
Max probability per species across (M separated + 1 original) channels
```

**Critical finding**: Using **max over separated channels AND original audio** gives best results. Using separated channels alone actually degrades performance on some datasets (SSW: -12% CMAP). Including the original mixture as a channel is essential.

### 7.2 Classifier Architecture

- **Frontend**: mel-spectrogram → PCEN → channel-wise normalization (outlier-robust: remove values >1σ above mean, recompute stats, normalize)
- **Backbone**: EfficientNet-B0
- **Pooling**: AutoPool (learnable attention pooling over time)
- **Hidden**: 1280-dimensional
- **Heads**: separate heads for species, genus, family, order (taxonomic multi-task)
- **Training**: binary cross-entropy with label smoothing 0.1; 200k steps; Adam lr=0.01; batch 64; 32 v2 TPUs; ~3.5h per model
- **Ensemble**: 5 models trained from scratch, logit-domain probability averaging
- **Mel config**: 100 Hz frame rate, 80ms frame length, 60–10,000 Hz, 160 channels
- **Background species handling**: For Xeno-Canto examples with background labels, loss is set to zero for those species (avoids punishing correct detections)

### 7.3 Taxonomic Multi-Task Training

Classification heads at every taxonomy level (species, genus, family, order) plus a binary bird/no-bird head. Higher-level losses weighted at 0.1×. This provides a natural curriculum — the model learns "this is a warbler" before sorting out "yellow-rumped vs. magnolia warbler."

Context: this is a global classifier approach to hierarchical classification (as opposed to nested per-node classifiers). The taxonomic outputs are used only during training, not at inference.

### 7.4 Data Augmentation

| Augmentation | Description |
|-------------|-------------|
| Random time shift | Random 5s crop from 6s example |
| Random gain | Peak normalize to uniform [0.05, 0.75] |
| Example mixing | 50% prob: mix in second labeled example, union labels |
| Noise mixing | 75% prob: mix noise at SNR ∈ [0, 40] dB. Sources: DCASE 2018 negatives, BBC Nature SFX, Common Voice. 10% noise-only |
| Random low-pass | Scale frequency bands in mel-spectrogram to simulate distance |

## 8. Classification Results

### 8.1 Evaluation Datasets

| Dataset | Location | Segments | Species | Source |
|---------|----------|----------|---------|--------|
| Sapsucker Woods (SSW) | Ithaca, NY | 40,000 × 5s | 70 | BirdCLEF 2019 test set, ARU-recorded, expert-annotated |
| High Sierras (HSN) | Sierra Nevada, CA | 4,928 × 5s | 18 (16 with ≥5 obs) | BirdCLEF 2021, dawn chorus (4–8 AM) |
| Caples (CAP) | Lake Tahoe, CA | 2,944 × 5s | 42 (36 with ≥5 obs) | 80 ARU points, volunteer-annotated |

### 8.2 Separate+Classify Comparison (Table 2)

| Method | SSW CMAP | SSW lwlrap | SSW d' | SSW Top-1 | CAP CMAP | CAP lwlrap | CAP d' | CAP Top-1 | HSN CMAP | HSN lwlrap | HSN d' | HSN Top-1 |
|--------|----------|------------|--------|-----------|----------|------------|--------|-----------|----------|------------|--------|-----------|
| Mix Only | 0.304 | 0.431 | 1.117 | **0.398** | 0.334 | 0.569 | 1.144 | 0.496 | 0.527 | 0.531 | 1.149 | 0.432 |
| Separation Only | 0.268 | 0.413 | 1.116 | 0.360 | 0.327 | 0.581 | 1.154 | 0.506 | 0.548 | 0.548 | 1.149 | 0.448 |
| **Mix+Separation** | **0.306** | **0.441** | **1.123** | 0.397 | **0.341** | **0.590** | **1.155** | **0.517** | **0.560** | **0.560** | **1.153** | **0.451** |

**Observations**:
- Nearly uniform improvement across all datasets from Mix+Separation.
- HSN benefits most (+6.3% CMAP) — this is the dawn chorus dataset with the most overlapping vocalizations.
- SSW shows minimal CMAP improvement (+0.7%) but noticeable lwlrap improvement (+2.3%).
- Separation Only performs worse than Mix Only on SSW (CMAP -12%), confirming that the original mixture context is important.

### 8.3 Classifier Ablation (Table 3)

| Component | CAP CMAP | CAP lwlrap | CAP Top-1 | HSN CMAP | HSN lwlrap | HSN Top-1 |
|-----------|----------|------------|-----------|----------|------------|-----------|
| B0 Ensemble (5×) | 0.334 | 0.569 | 0.496 | 0.527 | 0.531 | 0.432 |
| B0 Single (mean) | 0.284 | 0.517 | 0.458 | 0.479 | 0.483 | 0.392 |
| − Taxonomic loss | 0.272 | 0.491 | 0.434 | 0.465 | 0.469 | 0.376 |
| − Low-pass aug | 0.283 | 0.467 | 0.423 | 0.401 | 0.398 | 0.319 |
| − Channel norm | 0.270 | 0.484 | 0.437 | 0.438 | 0.447 | 0.362 |
| B1 | 0.275 | 0.503 | 0.448 | 0.477 | 0.479 | 0.396 |
| B2 | 0.268 | 0.490 | 0.437 | 0.486 | 0.477 | 0.390 |

**Ablation takeaways**:
- **Low-pass augmentation** is the single most impactful improvement on HSN: removing it drops CMAP by -16% (0.479 → 0.401). HSN is recorded at high elevation where vocalizations travel unobstructed — simulating distant calls via low-pass filtering is especially valuable.
- **Channel normalization** provides +4–9% CMAP across both datasets.
- **Taxonomic loss** provides +3–5% CMAP.
- **Ensembling** (5×) provides +18% CMAP on CAP, +10% on HSN.
- Larger EfficientNets (B1, B2) do not help — overfitting on relatively small training sets.

## 9. Qualitative Findings

### Benefits of Separation
1. **Low-SNR calls recovered**: Isolated (non-overlapping) vocalizations with very low SNR get dramatically better probabilities after noise removal. PCEN processing brings denoised calls clearly into the foreground. Weakly labeled training data teaches models to ignore these low-SNR calls; separation removes the masking noise.
2. **Common species suppression fixed**: Common species appearing frequently in backgrounds of other species' recordings get suppressed probabilities. After separation, they're isolated and score correctly.
3. **Overlapping vocalizations separated**: Even when species' frequency ranges overlap, the model successfully separates them into distinct channels.

### Limitations of Separation
1. **Prominent species scores decrease** after separation — loss of audio context. Mitigated by including original mixture as a channel.
2. **Over-separation**: Single songs sometimes split across channels. Isolated notes may resemble another species, causing misclassification.
3. **Domain shift**: Separated audio looks different from training data when classifier was trained on unseparated audio.

## 10. Relevance to Our BirdCLEF Work

### Direct parallels to our problems
- **Focal → soundscape domain gap** (our key challenge): MixIT separation directly addresses the overlapping vocalizations and background noise that make soundscapes harder than focal recordings.
- **Weak labels / missing backgrounds**: Our BirdCLEF 2025 training data has the same single-primary-label problem.
- **Dawn chorus complexity**: BirdCLEF soundscape evaluation includes dawn chorus periods.

### Ideas for our pipeline
1. **Inference-time preprocessing**: Run bird MixIT on soundscapes before our teacher ensemble (EAT, ConvNeXt, ProtoCLR), take max prediction per species across channels + original.
2. **Cleaner pseudo-labels**: Separate soundscapes before generating pseudo-labels on the 9,726 unlabeled files.
3. **Augmentation ideas from the paper**: Low-pass augmentation (+19% on their HSN — we haven't tried this), taxonomic multi-task training, example mixing with union labels.
4. **Train domain-matched separator**: The paper showed +5 dB from domain-matched training. Training a MixIT model on BirdCLEF 2025 data could provide even better separation.

### What we'd borrow vs. what's different
- Their classifier is EfficientNet-B0 with PCEN; ours are LoRA-tuned EAT/ConvNeXt/ProtoCLR — stronger classifiers that may benefit more from clean inputs.
- Their augmentation suite (example mixing, noise mixing, low-pass) is more extensive than anything we've tested — this is orthogonal to separation and worth exploring independently.
- They ensemble 5 classifiers; we have 3 architecturally diverse models — a natural ensemble.

## References

1. Denton, Wisdom, Hershey. "Improving Bird Classification with Unsupervised Sound Separation." ICASSP 2022. [arXiv:2110.03209](https://arxiv.org/abs/2110.03209)
2. Wisdom et al. "Unsupervised Sound Separation Using Mixture Invariant Training." NeurIPS 2020. [arXiv:2006.12701](https://arxiv.org/abs/2006.12701)
3. Kavalerov et al. "Universal Sound Separation." WASPAA 2019. (TDCN++ architecture)
4. Luo & Mesgarani. "Conv-TasNet: Surpassing ideal time–frequency magnitude masking for speech separation." IEEE/ACM TASLP 2019. (Learnable basis / encoder-decoder)
5. Wisdom et al. "Differentiable Consistency Constraints for Improved Deep Speech Enhancement." ICASSP 2019. (Mixture consistency projection)
6. Wang et al. "Trainable frontend for robust and far-field keyword spotting." ICASSP 2017. (PCEN)
7. McFee et al. "Adaptive Pooling Operators for Weakly Labeled Sound Event Detection." IEEE/ACM TASLP 2018. (AutoPool)
