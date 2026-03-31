# Model Training Reference

Single source of truth for all training techniques explored, what worked, what didn't,
and why. Organized by component. Updated as we learn more.

See `MODEL-RUNS.md` for individual model results and `docs/references/deep-research/`
for external research.

---

## Table of Contents

1. [Backbones](#backbones)
2. [SED Heads](#sed-heads)
3. [Mel Spectrogram Preprocessing](#mel-spectrogram-preprocessing)
4. [Loss Functions](#loss-functions)
5. [Data Augmentation (MixUp)](#data-augmentation-mixup)
6. [Optimizers and Schedulers](#optimizers-and-schedulers)
7. [Training Data Strategy](#training-data-strategy)
8. [Context Windows](#context-windows)
9. [Cross-Validation and Ensembles](#cross-validation-and-ensembles)
10. [Post-Processing](#post-processing)
11. [Foundation Models (Perch)](#foundation-models-perch)
12. [Pseudo-Labeling](#pseudo-labeling)
13. [Key Lessons Learned](#key-lessons-learned)

---

## Backbones

### EfficientNet-B0 (`efficientnet_b0`)
- **Our default SED backbone** (exps 016-019)
- ImageNet pretrained, 5.3M params, embed_dim=320 (features_only) or 1280 (global_pool)
- Best result: **0.860 LB** (v4b, with AmplitudeToDB + AttHead + soft_ce)
- 1-channel input via weight averaging of 3-channel pretrained stem

### EfficientNet-B0 NS-JFT (`tf_efficientnet_b0.ns_jft_in1k`)
- Noisy Student + JFT pretrained (stronger init than standard ImageNet)
- Competitor's 0.872 LB model uses this with **3-channel input** (mel repeated 3x)
- Our v2 (exp 016) used this but regressed: 0.837 val, 0.755 LB
- Difference: competitor keeps 3 channels, we adapted to 1 channel

### EfficientNetV2-B0 (`tf_efficientnetv2_b0.in1k`)
- Tested as v3d (exp 016): **0.841 val, early-stopped at epoch 13**
- Regressed -0.019 vs v1. Only tested with v4b recipe — never tried with ttahara recipe.

### HGNetV2-B0 (`hgnetv2_b0.ssld_stage2_ft_in1k`)
- SSLD (Semi-supervised Label Distillation) pretrained, ~6M params, embed_dim=1024
- ttahara's 0.884 LB model (4-fold ensemble)
- **Our v5g**: 0.872 val (single fold, 21 min training)
- v5e (HGNet + v4b recipe): 0.861 val — recipe matters more than backbone
- Re-parameterized architecture optimized for deployment (inference speed)

### Input Channels: 1-Channel vs 3-Channel

- **1-channel** (our default): Mel spectrogram as single channel `(B, 1, H, W)`.
  Pretrained 3-ch stem weights averaged to 1 channel — loses information.
- **3-channel** (competitor's approach): Repeat mono mel to 3 channels `(B, 3, H, W)`.
  Preserves full RGB pretrained weights from ImageNet/SSLD pretraining.
- Source: `ravi20076/birdclef2026-public-blend-v6` uses `mel.repeat(1, 3, 1, 1)` with
  `tf_efficientnet_b0.ns_jft_in1k` (0.872 LB model).
- **v5k** tests 3-channel on HGNet with ttahara recipe — pending results.

### Key Finding
**Backbone alone doesn't determine performance.** HGNet with v4b recipe (0.861) < EfficientNet
with v4b recipe (0.865). But HGNet with ttahara recipe (0.872) > EfficientNet with ttahara
recipe (0.866). The recipe must match the backbone.

---

## SED Heads

### AttHead (our default)
- GeM frequency pool → Linear+ReLU+Dropout → Conv1d (frame logits) + Conv1d (attention weights)
- Attention: `softmax(attention_conv)` over time → weighted sum
- Returns (clipwise, framewise) for dual loss supervision
- Used in v1 through v4b (our best: 0.860 LB)

### AttBlockV2 (PANNs-style)
- Same structure but: `tanh(value) × softmax(weight)` attention
- Framewise: sigmoid-bounded [0,1] probabilities
- Tested as v4c (exp 017): **0.815 val** — significant regression
- Also v4g (BCE + AttBlockV2): 0.868 val, 0.819 LB — still worse than AttHead

### LSEHead (ttahara's approach)
- **No attention weights** — just Log-Sum-Exp temporal pooling
- Two variants of frequency pooling:
  - `freq_pool="gem"` (our default): Learnable GeM power-mean
  - `freq_pool="mean"` (ttahara): Simple `torch.mean(dim=2)`
- v5d (GeM + LSE, EfficientNet): 0.861 val — slight regression from AttHead
- v5g (mean + LSE, HGNet, ttahara recipe): **0.872 val** — best single-fold SED
- Simpler, fewer parameters, but works better with the right recipe

### Competitor's Head (0.872 LB notebook)
- `tanh+softmax` attention (similar to AttBlockV2)
- `hidden_dim = backbone.num_features` (1280 for NS-JFT EfficientNet, not 512)
- `dropout=0.1` (vs our 0.5)

### Key Finding
**LSEHead + mean freq pooling works best with ttahara's recipe** (BCE, OneCycleLR, 5s).
AttHead works best with our v4b recipe (soft_ce, CosineAnnealing, 10s). The head and
training recipe are coupled — can't swap one without the other.

---

## Mel Spectrogram Preprocessing

### Parameters Explored

| Config | n_fft | n_mels | hop | norm | Resize | Best LB |
|--------|-------|--------|-----|------|--------|---------|
| v1 (exp 016) | 1024 | 128 | 320 | audioset | No | 0.827 |
| v4a (exp 017) | 4096 | 224 | 320 | audioset | No | 0.830 |
| **v4b (exp 017)** | **4096** | **224** | **320** | **amplitude_to_db** | **No** | **0.860** |
| ttahara (v5f/g) | 2048 | 256 | 313 | min-max | 256×256 | 0.855/pending |
| Competitor 0.872 | 2048 | 224 | 512 | min-max | No | 0.872 |

### Normalization Methods

1. **AudioSet** (`log(mel+1e-6) → (x - mean) / (std * 2)`): Global stats, used in v1-v4a
2. **AmplitudeToDB** (`10*log10(mel) → standardize → min-max [0,1]`): Per-sample, used in v4b+
3. **Min-max only** (`AmplitudeToDB → min-max [0,1]`, no standardization): ttahara's approach
4. **Min-max + 3-channel repeat**: Competitor's approach (matches ImageNet 3-ch pretrained weights)

### Key Finding
- **n_fft=4096 was the single biggest lever**: v4a raw (0.825) nearly matched v1+pp2 (0.827)
- **AmplitudeToDB norm** added +0.030 LB over audioset (v4a 0.830 → v4b 0.860)
- **Resize to square (256×256)** may help match ImageNet pretraining spatial assumptions
- ttahara's simpler mel (n_fft=2048) works because the recipe compensates

---

## Loss Functions

### Soft Cross-Entropy (our default)
- Normalize multi-hot labels to sum-to-1, then `F.cross_entropy`
- Treats multi-label as soft single-label distribution
- Used in v4b (0.860 LB) — our best with AttHead

### BCEWithLogitsLoss
- Native multi-label: independent sigmoid per class
- v5a (EfficientNet): 0.875 val but **0.851 LB** (regressed -0.009)
- v5f (EfficientNet + ttahara): 0.866 val, 0.855 LB
- v5g (HGNet + ttahara): **0.872 val** — works with the right backbone+recipe
- ttahara's 0.884 uses BCE exclusively

### Standard Cross-Entropy
- Single-label (primary_label only), ignores secondary species
- v4i: 0.835 val — significant regression. CE is wrong for multi-label.

### Asymmetric Loss (ASL)
- Separate focusing for positives (gamma_pos=0) and negatives (gamma_neg=4)
- v4d: 0.858 val, **0.808 LB** — bad regression
- Recommended by Gemini framework but didn't work in our setup
- May need different hyperparams or larger training data to shine

### Focal BCE (used in Perch ProtoSSM)
- `(1 - p_t)^gamma * BCE`, gamma=2.0-2.5
- Used in pantanal-distill V18 (0.930 LB) with gamma=2.5
- Not tested in our SED pipeline yet

### Key Finding
**Loss function interacts strongly with head type.** Soft_ce works with AttHead (dual loss
on clipwise + framewise). BCE works with LSEHead (single clipwise loss). Can't mix freely.
The **dual loss** (clip + frame weights) is specific to AttHead — LSEHead uses clipwise only.

---

## Data Augmentation (MixUp)

### Spectrogram MixUp (our default)
- Mix mel spectrograms: `mel = λ*mel_a + (1-λ)*mel_b`
- `Beta(0.5, 0.5)` — bimodal, most mixes barely mixed
- Applied with 50% probability per batch
- Single lambda for entire batch

### Stronger MixUp (ttahara's approach)
- `Beta(1.0, 1.0)` = Uniform[0,1] — much more aggressive mixing
- **Per-sample lambda** (each sample gets its own mixing ratio)
- **Label clipping θ=0.8**: if mixed label ≥ 0.8, snap to 1.0
- **5-epoch warmup**: no MixUp during first 5 epochs
- Applied 100% of batches (after warmup)
- v5b (stronger MixUp alone): 0.869 val, **0.840 LB** — regressed as isolated change
- v5g (full ttahara recipe): **0.872 val** — works in combination

### Raw Waveform MixUp
- Mix before mel: `wave = λ*wave_a + (1-λ)*wave_b`
- Gemini framework recommends this (simulates physical sound superposition)
- v3b (exp 016): 0.862 val, 0.802 LB
- v4j (exp 017): 0.868 val, 0.853 LB — slight val gain, LB regression

### Key Finding
**Individual MixUp changes don't help on LB** despite improving val/auroc. The val/auroc ↔ LB
disconnect is a persistent problem for SED models. MixUp must be part of a complete recipe.

---

## Optimizers and Schedulers

### CosineAnnealingLR + LinearLR Warmup (our default)
- Two param groups: encoder (1e-4) + head (1e-3)
- Per-epoch stepping
- 50 epochs, early stopping (patience 10)
- Effective batch size 256 (64 × 4 grad accum)

### OneCycleLR (ttahara's approach)
- **Single param group** (uniform lr=5e-4 for all params)
- Per-step stepping (smoother LR curve)
- 20 epochs, no early stopping
- `pct_start=0.25` (5/20 warmup), `div_factor=25`, `final_div_factor=4`
- Effective batch size 64 (no grad accumulation)
- **8-10x faster training** (21 min vs 3-4h)

### CosineAnnealing with Warm Restarts (pantanal-distill V18)
- `CosineAnnealingWarmRestarts(T_0=20, T_mult=1)`
- Periodic LR resets for escaping local optima
- Used with 80 epoch training on Perch ProtoSSM

### Weight Decay
- Our default: 0.01 (strong regularization)
- ttahara: **1e-4** (100x less)
- v5c (1e-4 alone): 0.867 val — marginal improvement as isolated change

### Key Finding
**OneCycleLR + uniform LR + no grad accumulation is the better recipe for 20-epoch training.**
CosineAnnealing + two param groups + grad accum works for 50-epoch training. Different
philosophies: "train hard and fast" vs "train slow and carefully."

---

## Training Data Strategy

### Data Composition (BirdCLEF 2026)
- **Focal recordings**: 35,549 files, 206 species (Aves dominated, 97.9%)
- **Labeled soundscapes**: 66 files, 739 segments, 75 species (234 total including 28 soundscape-only)
- **Unlabeled soundscapes**: 10,592 files — same domain as test set, completely unannotated
- **Test set**: ~600 soundscape files (hidden)

### The 28-Species Problem
28 species (insects, amphibians) have ZERO focal recordings. They exist only in the 66
labeled soundscapes (~200-300 positive windows). This creates a hard ceiling for any model
trained only on competition data — the model can't learn these species well enough to
achieve high macro-averaged AUC.

**Estimated ceiling**: If model scores 0.92 on 206 focal species but 0.5 on 28 soundscape-only
species: `(206*0.92 + 28*0.5) / 234 ≈ 0.87`. This matches our observed ~0.860 ceiling.

### Why Perch Breaks Through
Perch v2 was pre-trained on massive global bioacoustic data (tens of thousands of taxa).
It already has representations for insects, amphibians, and rare species. Our SED models
(trained from ImageNet init) must learn these from scratch with minimal data.

### Split Strategies
- **90/10 random stratified** (our default): Simple, used for all SED experiments
- **4-fold MultiLabelStratifiedGroupKFold** (ttahara): Grouped by audio_id, multi-hot stratified
- **5-fold GroupKFold** (pantanal-distill): 5-fold for Perch ProtoSSM OOF

---

## Context Windows

### 5-Second Windows
- Competition standard: predictions required every 5s
- ttahara's recipe: train on 5s, infer on 5s
- Faster training, more samples per epoch
- May miss long-duration insect/amphibian patterns

### 10-Second Windows (our default for SED)
- v4b: train on 10s (random crops from focal, 10s context around 5s soundscape segments)
- Infer with 10s centered on each 5s target (+2.5s context each side)
- +0.003 LB from 10s alignment (exp 016: 0.823 → 0.826)

### 15-20 Second Windows (Gemini framework recommendation)
- Not tested. Framework argues long windows capture insect/anuran rhythmic patterns.
- Would require hop_length scaling to manage tensor size.
- May help with the 28 texture-species (continuous stridulators).

---

## Cross-Validation and Ensembles

### Single 90/10 Split (our default)
- Fast iteration, one model
- No ensemble benefit
- Risk of split-dependent results

### 4-Fold Ensemble (ttahara, 0.884 LB)
- MultiLabelStratifiedGroupKFold, grouped by audio_id
- 4 models × sigmoid average at inference
- Estimated +0.010-0.015 LB over single fold
- **Our v5g-4fold**: trained but not yet submitted to LB
- **Fold imbalance issue**: 55/66 soundscape files cluster in fold 0 (only 66 groups)

### Multi-Model Blending (competitor's notebook)
- Two different checkpoints (different training recipes), weighted probability average
- 75% finetuned + 25% baseline
- More diverse than same-recipe 4-fold

### Perch OOF (exp 018)
- 5-fold GroupKFold for MLP probes + 3-fold for ProtoSSM
- OOF predictions used for ensemble weight optimization
- pantanal-distill V18: found MLP-only beat ProtoSSM+MLP blend

---

## Post-Processing

### Temperature Scaling
- Divide logits by T before sigmoid: `probs = sigmoid(logits / T)`
- T > 1: softens (spreads distribution, helps rare species)
- T < 1: sharpens (concentrates on confident predictions)
- Per-taxon: Aves T=1.10, texture T=0.95 (pantanal-distill)
- Our pp2: T=1.15 (global), added +0.001 LB

### Temporal Smoothing
- **Gaussian**: `[0.1, 0.2, 0.4, 0.2, 0.1]` kernel (our exp 018)
- **Confidence-sharpened**: `p^1.5 → smooth → p^(1/1.5)` (competitor's notebook)
  Preserves peaks better than raw smoothing
- **Adaptive delta shift**: `alpha = 0.20 * (1 - confidence)` (pantanal-distill V18)
  High-confidence windows stay, uncertain ones smooth
- **Bifurcated by taxonomy**: Different sigma for Aves (event) vs Insecta (texture)

### File-Level Operations
- **TopN multiplication**: `probs *= max(probs_across_file)` — boosts species across file
- **File-max additive**: `probs += 0.05 * file_max` (competitor's notebook)
- **Rank-aware scaling**: `probs *= file_max^0.4` (pantanal-distill V18)

### Per-Class Threshold Sharpening
- OOF-optimized threshold per species (pantanal-distill: 10-value grid)
- Maps probabilities to sharper [0,1] range around species-specific threshold
- Not implemented in our pipeline yet

### TTA (Test-Time Augmentation)
- Time shifts: [0, ±2.5s] for 3-shift TTA (our default)
- Wider: [0, ±1, ±2 windows] for 5-shift TTA (pantanal-distill)
- Weighted: center=0.50, ±1=0.25, ±2=0.125

### Metadata Priors
- Site+hour conditional priors from labeled soundscapes
- Fold-safe: exclude val sites when building prior tables
- 6-category fusion: mapped_event, mapped_texture, proxy, prior-only, unmapped

---

## Foundation Models (Perch)

### Perch v2 (Google Bird Vocalization Classifier)
- EfficientNet-B3 backbone, pre-trained on massive global bioacoustic data
- Frozen TensorFlow model (CPU inference)
- Outputs: 1536-dim embeddings + 234-class logits per 5s window
- **Our best**: 0.918 LB (ProtoSSM v4 ensemble)

### ProtoSSM Pipeline (0.918 → 0.930)
1. Perch embeddings → PCA reduction (64-128 dim)
2. OOF fold-safe prior fusion (site/hour priors)
3. MLP probes on PCA-reduced embeddings
4. ProtoSSM (BiSSM + cross-attention + prototypical head)
5. Ensemble: probe + ProtoSSM weighted average
6. ResidualSSM (lightweight correction pass)
7. TTA + post-processing pipeline

### Scaling Recipe (pantanal-distill V18, 0.930 LB)

| Component | Our v4 (0.918) | V18 (0.930) |
|-----------|---------------|-------------|
| d_model | 128 | 320 |
| SSM layers | 2 | 4 |
| Cross-attn heads | 4 | 8 |
| MLP probes | (128,), PCA-64 | (256,128), PCA-128 |
| ResidualSSM | d=64, 1 layer | d=128, 2 layers |
| OOF folds | 3 | 5 |
| TTA shifts | 3 | 5 |
| Training epochs | 60 | 80 |
| Ensemble weight | fixed 0.5 | OOF-optimized (MLP-only won) |

---

## Pseudo-Labeling

### Exp 014: Teacher Predictions (ProtoCLR + ConvNeXt ensemble)
- Teacher: exp 012 models (0.767 LB best), macro AUROC 0.885 on labeled soundscapes
- Generated soft labels for 10,592 unlabeled soundscapes (127K windows)

### Exp 015: Student Training (FAILED)
- Trained on focal + labeled soundscape + 127K pseudo-labeled windows
- Per-sample weighted BCE: w=1.0 ground truth, w=0.5 pseudo-labels
- **Result: all models regressed** (ProtoCLR -0.005 LB, EAT -0.024 LB)
- Root cause: teacher too weak (0.767 LB), pseudo-labels introduced noise

### Untested: Perch Teacher (0.918 LB)
- Much stronger teacher → higher quality pseudo-labels
- Power transform (`p^1.5`) to suppress uncertain pseudo-labels
- WeightedRandomSampler by prediction density
- Multi-iterative refinement (student becomes new teacher)
- **This is the most promising untested approach for >0.900 SED.**

---

## Key Lessons Learned

### 1. Val/auroc is a poor LB predictor for SED
Every v5 ablation that improved offline val regressed on LB:
- v5a: +0.010 val → **-0.009 LB**
- v5b: +0.004 val → **-0.020 LB**
- v4e: +0.003 val → **-0.020 LB**
- v4j: +0.003 val → **-0.007 LB**

### 2. Individual changes don't help — recipes must be adopted wholesale
Swapping one component (loss, MixUp, head) while keeping everything else from v4b
consistently fails. ttahara's 0.884 requires ALL changes together: LSEHead + BCE +
per-sample MixUp + OneCycleLR + 5s windows + min-max mel.

### 3. The 0.860 SED ceiling is a data problem, not an architecture problem
After exhaustive ablations (v4a-v4k, v5a-v5g), no single-model SED exceeds 0.860 LB.
The bottleneck is the 28 soundscape-only species with minimal training data. Breaking
through requires either a foundation model (Perch) or pseudo-labels from a strong teacher.

### 4. Training speed matters for iteration
ttahara's recipe trains in 21 min (vs 3-4h for v4b recipe). This enables faster
experimentation. OneCycleLR + 20 epochs + 5s + no grad accum = 8-10x speedup.

### 5. Perch-based approaches have a fundamentally higher ceiling
Our SED best: 0.860. Our Perch best: 0.918. The 0.058 gap comes from Perch's
pre-trained bioacoustic knowledge, especially for rare/novel taxa. Scaling ProtoSSM
(pantanal-distill V18) reaches 0.930.
