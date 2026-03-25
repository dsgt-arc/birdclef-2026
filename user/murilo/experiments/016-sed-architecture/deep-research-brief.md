# BirdCLEF 2026 — Deep Research Brief

Self-contained context document for LLM deep research sessions.
Last updated: 2026-03-24.

---

## 1. Competition Overview

**BirdCLEF+ 2026** is a Kaggle competition for automated bird/amphibian/insect identification from soundscape audio. The evaluation metric is **macro-AUROC** across 234 species on hidden test soundscapes (~600 files, 60s each, predictions every 5s = 12 segments per file).

**Key constraints:**
- CPU-only inference, 90-minute Kaggle time limit
- 234 species: 162 birds, 35 amphibians, 28 insects, 8 mammals, 1 reptile
- **Amphibian-dominant**: Pantanal soundscapes are 76.6% amphibian by labeled window count
- Class imbalance: 499× between most/least common species

**Dataset:**
- 35,549 focal recordings (344.5h) — clean, single-species XenoCanto clips
- 66 labeled soundscape files (0.6%) — multi-species, noisy
- 10,592 unlabeled soundscape files (177.6h) — the pseudo-labeling goldmine
- Only 18% species overlap with BirdCLEF 2025

---

## 2. Our Experiment History (Exps 001–016)

### Quick Reference

| Exp | Name | Key Finding | Best Metric |
|-----|------|-------------|-------------|
| 001 | BirdSet Benchmark | Audio pretraining >> ImageNet (2.5×) | 0.480 cmAP (HSN) |
| 002 | ViT Baseline | ImageNet ViT reaches only 39% of audio baseline | 0.186 cmAP |
| 003 | Architecture Comparison | Pretraining gives 2× lift; audio > ImageNet | 0.191 cmAP |
| 004 | Frame Tokenization | Vertical 128×4 patches +37% from scratch | 0.118 cmAP |
| 005 | EAT Frozen Probe | AudioSet features contain species signal | 0.890 AUROC |
| 006 | EAT LoRA | MLP head matters more than LoRA scope | **0.945 AUROC** |
| 007 | ConvNeXt LoRA | ImageNet nearly matches AudioSet with LoRA | 0.941 AUROC |
| 008 | Frame Tokens + EAT | Frame tokens HURT pretrained models (−6.4%) | Rejected |
| 009 | ProtoCLR LoRA | 5× smaller model matches EAT/ConvNeXt | 0.942 AUROC |
| 010 | Submission Pipeline | ONNX + Kaggle kernel works end-to-end | 0.531 LB (2025 data) |
| 011 | LoRA Baseline 2026 | Pipeline transfers to 2026 with zero changes | 0.736 LB |
| 012 | Soundscape Fine-Tuning | Soundscape domain adaptation universally helps | **0.767 LB** |
| 013 | Weak Supervision | FlyingSquid soft labels generated | Labels ready |
| 014 | Soft Label Training | Soft labels HURT (0.238 vs 0.336 baseline) | Rejected |
| 015 | SED + Soft Labels | SED head adds nothing with soft labels | Rejected |
| 016 | SED Architecture | SED + post-processing = **0.823 LB** | **0.823 LB** |

### Three Encoder Family

We have three architecturally diverse encoders at AUROC parity (~0.94 on BirdCLEF 2025):

| Encoder | Params | Pretraining | LoRA Targets | BirdCLEF 2025 AUROC | BirdCLEF 2026 LB |
|---------|--------|-------------|-------------|---------------------|-----------------|
| EAT (ViT) | 90M | AudioSet SSL | Attention | 0.945 | 0.720 |
| ConvNeXt-Base | 89M | ImageNet-22K | MLP layers | 0.941 | 0.755 |
| ProtoCLR (CvT-13) | 19M | XenoCanto contrastive | Attention | 0.942 | 0.767 |

Key insight: LoRA equalizes pretraining domains — 5× smaller ProtoCLR matches 90M models.

### Cross-Experiment Insights

1. **LoRA > full fine-tuning**: <1% params adapted gives best results; full FT causes catastrophic forgetting
2. **Pretrained patch embeddings are sacred**: Changing tokenization destroys pretrained value (exp 008)
3. **Soundscape domain adaptation is universally beneficial**: +0.022 to +0.060 LB from just 739 labeled segments
4. **Offline metrics don't predict LB ranking**: Must submit to Kaggle for reliable comparison
5. **Overfitting is pervasive**: No data augmentation used in exps 001–012; largest unexplored axis
6. **Fulldata training is risky**: Without early stopping, fast-converging models overshoot

---

## 3. Experiment 016: SED Architecture (Current Focus)

### What Was Built

Replaced clip-level classification (global avg pool → linear head) with Sound Event Detection:
- **Architecture**: EfficientNet-B0 backbone + GeM frequency pooling → attention temporal aggregation → dual clipwise/framewise output
- **Loss**: Soft CE (normalized multi-hot → F.cross_entropy) on both clipwise and framewise-max (0.5/0.5)
- **Training**: 10s windows, spectrogram mixup, 32kHz, ImageNet pretrained, all params trainable (3.6M)
- **Inference**: Blended sigmoid: 0.75×clipwise + 0.25×framewise_max
- **Post-processing**: TTA 2.5s shifts + Gaussian temporal smoothing + TopN file-level max

### Results: Every Change from v1 Made Things Worse

| Version | Change from v1 | Test AUROC | Test cmAP | LB (+pp) | Verdict |
|---------|---------------|-----------|----------|----------|---------|
| **v1** | Baseline (ImageNet RA + soft CE + spec mixup) | **0.860** | **0.593** | **0.823** | Best |
| v2 | NS-JFT weights + CE loss (2 changes) | 0.837 | 0.539 | 0.755 | Regression |
| v3a | TF-ported weights only | 0.857 | 0.599 | 0.813 | Regression |
| v3b | Raw waveform mixup only | 0.862 | 0.607 | 0.802 | Regression |
| v3c | Standard CE loss only | 0.850 | 0.552 | pending | Regression |
| v3d | EfficientNetV2-B0 | 0.841 | 0.523 | pending | Regression |
| v1-10s | 10s overlapping inference | — | — | pending | Awaiting LB |

**v1's recipe is a robust local optimum**. The gap to 0.889–0.908 is NOT in model variant, weights, mixup domain, or loss function.

### Gap Analysis: 0.823 → 0.908

The remaining ~0.085 gap likely comes from multiple small cumulative gains:

1. **Mel spectrogram parameters** (NOT investigated): Our n_fft=1024, 128 mels vs 2025 1st place n_fft=4096, 224 mels. 4× less frequency resolution.
2. **Train/inference window alignment** (just submitted): Trained on 10s, inferred on 5s. v1-10s fixes this. Expected +0.015.
3. **Pseudo-labeling** (not started for SED): 2025 1st place gained +0.032 from 4-round PowerTransform pseudo-labels.
4. **Multi-model ensemble**: All top solutions use 3-7 models.
5. **Model soup**: Average last 3-5 checkpoints. Free, not done.
6. **ASL loss**: +0.023 over BCE in competitor ablation. Not tested (v3c tested standard CE, not ASL).
7. **Perch knowledge distillation**: +0.030 val in competitor experiments.
8. **Prior-year pretraining**: +0.014 LB from BirdCLEF 2021-2024 data (819K recordings).

---

## 4. Competition Landscape (2026 Leaderboard)

### Current Standings

| Approach | Best LB | Strategy |
|----------|---------|----------|
| Pantanal Distill (Perch+ProtoSSM v3) | **0.917** | Frozen Perch + bidirectional SSM + ResidualSSM boosting |
| Perch+ProtoSSM | **0.913** | Frozen Perch + SSM temporal + MLP probes + metadata priors |
| Perch+LogReg | **0.910** | Frozen Perch + sklearn LogisticRegression (simplest!) |
| Perch baseline | **0.907** | Foundation model + simple probes |
| Competitor SED (commenter) | **0.908** | "SED EfficientNet B0" single model |
| Competitor SED (post author) | **0.889** | EfficientNetV2-B0 SED single model |
| **Our SED v1+pp** | **0.823** | EfficientNet-B0 SED + TTA + smoothing + TopN |
| Our clip-level (exp 012) | 0.767 | ProtoCLR LoRA + soundscape fine-tuning |

### Two Paradigms

**Paradigm 1: Perch Foundation Model (0.907–0.917)**
- Frozen Perch v2 as 1536-dim feature extractor (trained on 10K+ bird species by Google)
- Lightweight heads: ProtoSSM (bidirectional SSM), MLP probes, LogReg
- Metadata priors: site-specific species pools, hour-of-day patterns
- Taxa-aware processing: birds (events, light smoothing) vs frogs/insects (textures, heavy smoothing)
- No focal training needed — soundscapes only
- Key: Perch alone = 0.825 LB out-of-the-box (better than our best trained model at 0.823)

**Paradigm 2: Trained SED Models (0.823–0.908)**
- Train from ImageNet-pretrained backbone (EfficientNet-B0/B3/B4)
- SED head: GeM + attention pooling + framewise max
- Iterative pseudo-labeling (2025 1st place: 4 rounds with PowerTransform)
- CE or ASL loss (NOT BCE)
- Data augmentation: audio-domain MixUp (+0.036 confirmed)

---

## 5. Competitor Technique Rankings (from 28-experiment ablation study)

### Tier 1: Architecture & Loss (+0.02–0.05 each)

| Technique | Impact | Evidence |
|-----------|--------|----------|
| CE loss (over BCE) | +0.044 LB | 2024 1st place ablation |
| ASL loss (over BCE) | +0.023 val | Competitor ablation |
| Audio-domain MixUp | +0.036 Private | 2025 2nd place confirmed |
| Soft KD from Perch | +0.030 val | Competitor progression |
| Multi-round pseudo-labels | +0.025 val | Competitor progression |

### Tier 2: Inference & Post-Processing (+0.01–0.02 each)

| Technique | Impact | Evidence |
|-----------|--------|----------|
| 10s context windows | +0.015 LB | 2024 1st place |
| TopN postprocessing | +0.011 LB | 2025 2nd place |
| TTA: 2.5s shifts | +0.012 LB | Competition analysis |
| Temporal smoothing | +0.010 LB | Gaussian kernel |
| Model soup | Free | Average 3-5 checkpoints |

### Tier 3: Pretraining & Data (+0.01–0.02 each)

| Technique | Impact |
|-----------|--------|
| NS-JFT pretrained B0 | Consistent improvement (but hurt in our v2, confounded) |
| 819K prior-year pretrain | +0.014 LB |
| Separate non-bird pipeline | +0.003 LB |

### What DOESN'T Work

| Technique | Impact | Reason |
|-----------|--------|--------|
| Background noise aug | −0.041 | Distribution mismatch with pseudo-labels |
| Soft secondary labels | −0.043 | Conflicting gradients from co-occurrence |
| Rating filter ≥3 | −0.128 | Reduces diversity too aggressively |
| EfficientNetV2-S | −0.043 vs B0 | ImageNet-21k transfers poorly to spectrograms |
| Soft labels as training targets | −0.098 cmAP | Our exp 014: replacing hard labels hurts |
| SED on frozen backbone | No gain | Our exp 015/Anthony's exp: backbone must be trainable |

---

## 6. 2025 Winning Solutions Breakdown

### 1st Place (0.937 Private LB)

```
Baseline (B0 + RegNetY + CE):           0.872
+ Pseudo round 1 + MixUp:              0.898  (+0.026)
+ PowerTransform × 4 rounds:           0.930  (+0.032)
+ Separate insect/amphibian pipeline:   0.933  (+0.003)
```

- **PowerTransform pseudo-labels**: `p^γ` (γ=2.0) to sharpen predictions before next round
- **SoftAUCLoss** in later rounds: directly optimizes macro AUROC
- 7 models: EfficientNet-L0, B4, B3, RegNetY-016/008, B0 (insects)
- Extra data: 5,489 XC + 17,197 insect/amphibian recordings
- Augmentation: MixUp + StochasticDepth (drop_path_rate=0.2)
- Inference: OpenVINO CPU

### 2nd Place (0.928 Private LB)

```
NFNet-L0 Base:                          0.868
+ Label smoothing, class balancing:     0.875  (+0.007)
+ EfficientNetV2-S (in21k):            0.889  (+0.014)
+ OOF Pseudo × 3 rounds:               0.913  (+0.024)
+ TopN + Optuna ensemble:              0.928  (+0.015)
```

- **Audio MixUp**: +0.036 Private LB confirmed
- **819K prior-year pretraining**: +0.014 LB
- 15 models (3 experiments × 5 folds), Optuna-optimized weights
- GeM frequency pooling

### Consistent Winning Patterns (3/4+ solutions)

1. **SED Architecture** (3/4): GeM freq pooling + attention temporal + framewise max
2. **Pseudo-labeling** (2/4): Iterative self-training, 3-4 rounds, temperature scaling
3. **Overlapping context** (3/4): 10-20s windows for 5s predictions
4. **CE/focal loss** (3/4): NOT BCE
5. **Ensemble** (4/4): 3-7 diverse models

---

## 7. Perch-Based Approaches (Detailed)

### Architecture: Perch + ProtoSSM (0.913 LB)

```
Soundscape (60s, 32kHz)
  → 12 × 5s windows
  → Perch v2 (frozen TF model) → 1536-dim embeddings + logits
  → Three prediction streams:
      1. ProtoSSM: bidirectional SSM across 12 windows
      2. MLP probes: per-class classifiers on PCA embeddings
      3. Metadata priors: site/hour probability tables
  → Ensemble (50/50 ProtoSSM + MLP) → temperature scaling → submit
```

**ProtoSSM**: d_model=128, d_state=16, 2 SSM layers, prototypical head with cosine similarity, multi-task training (species BCE + Perch KD + taxonomy aux).

**Per-class MLP probes**: PCA-compressed Perch embeddings (1536→64) + temporal context (prev/next window, file mean/max) + onset/offset detection. sklearn MLPClassifier.

**Metadata priors**: Site-specific species tables, hour-of-day patterns, Bayesian smoothing. Taxa-aware: events (λ=0.4), textures (λ=1.0).

### Pantanal Distill (0.917 LB)

Adds over the 0.913 baseline:
- ProtoSSM v3 with site/hour embeddings injected into SSM hidden state
- ResidualSSM second-pass: learns to correct first-pass errors (like gradient boosting)
- Conservative distillation weight (0.1 vs 0.3)
- 60% ProtoSSM + 40% MLP (optimized on OOF)

### Perch + LogReg (0.910 LB)

The simplest high-scoring approach:
- Frozen Perch v2 + PCA (1536→64) + per-class sklearn LogisticRegression
- Bayesian site/hour priors with taxa-aware fusion
- Dynamic alpha blending based on training data quantity per class
- No PyTorch, no GPU training, no SSM/attention — just sklearn

---

## 8. Available Infrastructure

### Ready to Use
- **Three LoRA-adapted encoders** (ProtoCLR, ConvNeXt, EAT) with BirdCLEF 2026 soundscape checkpoints
- **SED pipeline**: EfficientNet-B0 + AttHead, training + ONNX export + Kaggle submission
- **Teacher labels**: BirdNET + Perch + BamBird predictions on full dataset (88 GB)
- **Weak supervision labels**: FlyingSquid soft labels (but proven to hurt as training targets)
- **Post-processing**: TTA + temporal smoothing + TopN (confirmed +0.028 LB)
- **10,592 unlabeled soundscapes** ready for pseudo-labeling

### Compute
- PACE cluster: RTX 6000 (24GB VRAM), typical allocation 300-500 GPU hours
- Training time: ~2-4h per EfficientNet-B0 SED run, ~3-9h for larger encoders

### What's NOT Available
- No Perch v2 TF model on our cluster (would need to set up)
- No source separation model trained yet (concept ready, not implemented)
- No ensemble inference within Kaggle budget (single model only currently, ~120 min for 2 models)

---

## 9. Open Research Questions

### Near-Term (Next Experiments)

1. **Mel spectrogram overhaul**: Our n_fft=1024, 128 mels vs 2025 1st place n_fft=4096, 224 mels. Does 4× frequency resolution close the gap? Requires retraining.

2. **ASL loss**: Handles multi-label natively (unlike soft CE which normalizes multi-hot). +0.023 in competitor ablation. We tested standard CE (v3c, regressed) but NOT ASL — they're fundamentally different.

3. **Model soup**: Average last 3-5 epoch checkpoints. Free stability, zero cost, not done yet.

4. **Perch as teacher for knowledge distillation**: Competitor showed +0.030 val. Our teachers (ProtoCLR/ConvNeXt) scored 0.885 AUROC on labeled val; Perch covers 203/234 species natively.

5. **Pseudo-labeling with SED teacher**: Use current 0.823 SED model to label 10,592 unlabeled soundscapes, retrain. 2025 1st: +0.032 from 4 rounds with PowerTransform.

### Strategic Questions

6. **Should we build on Perch instead of training from scratch?** Perch alone = 0.825 LB. Our best trained model = 0.823. The Perch-based approaches (0.907-0.917) sidestep all our architectural struggles.

7. **Is the SED architecture gap solvable or should we pivot to ensemble/Perch hybrid?** Our SED v1 recipe is a local optimum — no single change helped. The remaining gap may require multiple simultaneous changes or a fundamentally different approach.

8. **What's the fastest path to 0.85+?** Options: (A) continue SED improvements (mel params + pseudo-labels), (B) implement Perch + ProtoSSM pipeline, (C) ensemble SED + Perch predictions.

9. **How to handle the 28 soundscape-only species?** Only 739 labeled segments total. These species (mostly insect song-types) get minimal training signal. Perch covers most via genus proxies; our model trains on them directly but with tiny data.

10. **Metadata priors**: Site/hour info is available in soundscape filenames. The Perch notebooks get huge gains from this. We don't use it at all. How much would it help our SED model?

### Longer-Term

11. **Source separation**: Concept ready (ConvTasNet + MixIT at 32kHz), not implemented. No 2025 winner used it. Potential unexplored competitive advantage or too expensive for marginal gain?

12. **OpenVINO for multi-model ensemble**: Currently can't fit 2+ models in 90-min Kaggle budget. OpenVINO CPU optimization could enable 3-4 model ensemble.

13. **Track 2 (Prompted SED with Mamba)**: Novel research direction — reference audio conditions Mamba for few-shot/zero-shot detection. Exp 011 was inconclusive. High risk, high potential reward.

---

## 10. What We Know Doesn't Work (Don't Suggest These)

| Approach | Evidence | Why It Failed |
|----------|----------|---------------|
| Soft labels as training targets | Exp 014: −0.098 cmAP | Dilutes gradient signal vs hard labels |
| SED on frozen backbone | Exp 015/Anthony's exp | Backbone must co-adapt with attention head |
| NS-JFT pretrained weights (alone) | v2: −0.068 LB (confounded) | May need different LR; confounded with CE loss |
| Standard CE loss (argmax of multi-hot) | v3c: −0.041 cmAP | Picks arbitrary species for multi-label; soft CE is better |
| Raw waveform mixup (alone) | v3b: −0.021 LB | Better offline but worse LB — calibration issue? |
| TF-ported V1 weights (alone) | v3a: −0.010 LB | Marginal difference, not worth the risk |
| EfficientNetV2-B0 | v3d: −0.070 cmAP | Worse with our recipe; may need different hyperparameters |
| Frame tokenization on pretrained models | Exp 008: −6.4% AUROC | Incompatible with pretrained position embeddings |
| Full fine-tuning (all params) | Exp 009: −1.7% AUROC | Catastrophic forgetting of pretrained features |
| Rating-based quality filtering | Competitor: −0.128 | Reduces diversity too aggressively |
| Background noise augmentation | Competitor: −0.041 | Distribution mismatch with pseudo-labels |
| Fulldata training without early stopping | Exp 011: −0.025 LB | Fast-converging models overshoot |
