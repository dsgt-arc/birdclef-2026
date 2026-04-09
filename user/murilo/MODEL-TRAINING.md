# Model Training Reference

Single source of truth for all training techniques explored, what worked, what didn't,
and why. Organized by component. Updated as we learn more.

Last updated: 2026-04-08

See `MODEL-RUNS.md` for individual model results, `FINDINGS.md` for cross-cutting
patterns, and `docs/references/deep-research/` for external research.

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

| Config | n_fft | n_mels | hop | norm | Resize | Mel Library | Best LB |
|--------|-------|--------|-----|------|--------|-------------|---------|
| v1 (exp 016) | 1024 | 128 | 320 | audioset | No | librosa | 0.827 |
| v4a (exp 017) | 4096 | 224 | 320 | audioset | No | librosa | 0.830 |
| v4b (exp 017) | 4096 | 224 | 320 | amplitude_to_db | No | librosa | 0.859 |
| v5g (exp 019) | 2048 | 256 | 313 | min-max | 256×256 | torchaudio | 0.892 |
| **v5j (exp 019)** | **2048** | **256** | **313** | **amplitude_to_db** | **No (256×511)** | **NNAudio** | **0.906** |
| v10a (exp 022) | 2048 | 256 | 313 | PCEN | No | NNAudio | 0.880 |
| Competitor 0.872 | 2048 | 224 | 512 | min-max | No | — | 0.872 |

### Normalization Methods

1. **AudioSet** (`log(mel+1e-6) → (x - mean) / (std * 2)`): Global stats, used in v1-v4a
2. **AmplitudeToDB** (`10*log10(mel) → standardize → min-max [0,1]`): Per-sample, used in v4b+
3. **Min-max only** (`AmplitudeToDB → min-max [0,1]`, no standardization): ttahara's approach
4. **Min-max + 3-channel repeat**: Competitor's approach (matches ImageNet 3-ch pretrained weights)

### NNAudio (GPU mel — used in v5j)

NNAudio (`nnAudio.features.MelSpectrogram`) computes mel spectrograms on GPU rather
than CPU. Critical for v5j because:
- Mel computed during forward pass (not in DataLoader workers)
- Enables learnable mel parameters (though v5j uses fixed params)
- Avoids CPU bottleneck for n_fft=2048 at 32kHz
- v5j uses NNAudio + AmplitudeToDB norm + no resize → output shape (256, 511)

The switch from torchaudio (v5g, resize to 256×256) to NNAudio (v5j, no resize 256×511)
is confounded with the AmplitudeToDB normalization change. MODEL-RUNS footnote notes
v5j "also changes win_length (626→2048), removes resize, uses NNAudio." The +0.014
gain (v5g 0.892 → v5j 0.906) is from all three changes combined, not AmplitudeToDB alone.

### PCEN Normalization (exp 022 — REJECTED)

Per-Channel Energy Normalization: IIR smoothing + power-law compression.
Theoretically motivated for domain invariance (adapts to local signal energy).
v10a (learnable PCEN): **0.880 LB (−0.026 vs v5j)**. Rejected.

PCEN *transforms* the representation; AmplitudeToDB *adds* dynamic range information.
The model had already learned to exploit AmplitudeToDB's characteristics.

### Key Finding
- **AmplitudeToDB norm** is the biggest mel lever: +0.030 on EffNet (v4a→v4b), +0.014 on HGNet (v5g→v5j, confounded with NNAudio)
- **n_fft=4096** added +0.030 raw LB (v1→v4a) but v5j uses n_fft=2048 — the recipe compensates
- **NNAudio + no resize (256×511)** contributed to v5j's 0.906, confounded with AmplitudeToDB
- **PCEN failed** (−0.026): transforms representation rather than adding information

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

## v5j Winning SED Recipe (0.906 LB single, 0.917 4-fold+pp2)

The best SED recipe, discovered through exp 019 ablations. Every single-variable
deviation from this recipe regressed on LB (14 attempts, 0 successes in exp 021).

| Component | Value | Notes |
|-----------|-------|-------|
| **Backbone** | HGNetV2-B0 (`hgnetv2_b0.ssld_stage2_ft_in1k`) | 6M params, embed_dim=1024, SSLD pretrained |
| **Head** | LSEHead (mean freq pool + LSE temporal pool) | No attention — simpler than AttHead |
| **Loss** | BCEWithLogitsLoss (clipwise only) | Multi-label native. No dual loss. |
| **MixUp** | Per-sample spectrogram MixUp (α=1.0, θ=0.8, 5-ep warmup) | ttahara's stronger MixUp recipe |
| **Mel** | NNAudio: n_fft=2048, n_mels=256, hop=313, fmax=None | GPU-computed, no resize (256×511 output) |
| **Mel norm** | AmplitudeToDB + per-sample standardize + min-max [0,1] | Critical: +0.014 over min-max alone |
| **Optimizer** | AdamW, OneCycleLR (lr=5e-4, wd=1e-4, pct_start=0.25) | Single param group, uniform LR |
| **Training** | 20 epochs, batch_size=64, no grad accumulation | ~21 min per fold on RTX 6000 |
| **Windows** | 5s chunks, 32kHz | Competition segment size |
| **Data** | Focal + labeled soundscapes (234 classes) | 90/10 train/val split |
| **Inference** | TTA ±2.5s (50/25/25), temporal smoothing, TopN | All baked into LB scores from v5g onward |

### Why This Recipe Works

The recipe was adopted wholesale from ttahara's 0.884 LB pipeline, then improved
by adding AmplitudeToDB normalization and NNAudio mel. Individual components cannot
be swapped freely — the head, loss, scheduler, MixUp, and mel are all coupled.
See exp 021 (7 phases, all regressed) for evidence that perturbations fail.

### What This Recipe Cannot Do

The recipe plateaus around 0.906 single / 0.917 4-fold. The gap to 0.920+ single-model
competitors (Boredom 0.922, Salman 0.921) is NOT in the training recipe — it's in
something else (possibly data augmentation details, inference tricks, or architecture
differences we haven't identified). See `FINDINGS.md` insight #3.

---

## Perch + SED Ensemble (exp 023, 0.929 LB)

Score-level fusion of Perch ONNX (MLP probes) + SED v5j-4fold-pp2.

| Component | Value | Notes |
|-----------|-------|-------|
| **Perch backend** | ONNX (0.920 standalone) | TF (0.925) timed out for ensemble |
| **Perch probes** | MLP-only (proto_weight=0.0) | ProtoSSM hurts under ONNX mismatch |
| **SED model** | v5j-4fold-pp2 (0.917 standalone) | 4 folds averaged |
| **Ensemble weight** | w=0.5 (flat curve: 0.3/0.5/0.7 all ≈ 0.927) | MLP-only shifts best to 0.929 |
| **PP** | Each pipeline's own PP, then average | Disabling PP costs −0.005 |
| **Runtime** | ~64 min for 600 files | Within 90 min Kaggle limit |

### Key Ensemble Findings

- **MLP-only > ProtoSSM+MLP** (+0.002): simpler probes are more robust to ONNX embedding shift
- **PP matters** (+0.005): both pipelines' individual PP contribute positively even after averaging
- **Weight doesn't matter**: 0.926-0.927 across w=0.3/0.5/0.7 — models are equally complementary
- **TF Perch not viable**: ~10s/file exceeds 90 min Kaggle limit for 600 files

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

### 3. The SED ceiling is a domain shift problem, not architecture
After exhaustive ablations (v4a-v4k, v5a-v5n, v6a-v6l, v8a-v8b, v10a-v10c),
the best single SED is 0.906 LB (v5j). 14 training recipe changes in exp 021
all regressed. The gap to 0.920+ competitors is not in the recipe — it's in
the domain shift between focal recordings and soundscapes.

### 4. Training speed matters for iteration
ttahara's recipe trains in 21 min (vs 3-4h for v4b recipe). This enables faster
experimentation. OneCycleLR + 20 epochs + 5s + no grad accum = 8-10x speedup.

### 5. Perch-based approaches have a fundamentally higher ceiling
Our SED best: 0.906 (v5j). Our Perch best: 0.925 (V18+TF). The 0.019 gap comes
from Perch's pre-trained bioacoustic knowledge, especially for rare/novel taxa.
Ensemble of both: 0.929 (MLP-only).

### 6. Ensembling orthogonal models works — simpler is better
Perch + SED ensemble (0.929) beats both individuals. MLP-only probes outperform
ProtoSSM+MLP by +0.002 in the ensemble context. The ONNX embedding mismatch
hurts complex models (ProtoSSM) more than simple ones (MLP probes).
