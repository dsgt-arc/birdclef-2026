---
experiment: "021-sed-training"
status: proposed
date: 2026-04-03
author: Murilo Gustineli
---

# Experiment 021: SED Training — Closing the Gap to 0.920+

## Hypothesis

Applying insights from top competitors (site/hour priors at inference, CE+sigmoid
training/inference split, waveform augmentations, and backbone exploration beyond
small EfficientNet-B0) will push our single SED model from 0.906 LB to 0.920+ LB,
matching or exceeding our Perch ProtoSSM pipeline (0.925) with a simpler, faster,
more generalizable model.

## Background

### Current state

Our best SED models (exp 019):

| Model | Backbone | Params | LB | Key recipe |
|-------|----------|--------|-----|-----------|
| v5j (single) | HGNetV2-B0 (1024 embed) | 6M | **0.906** | NNAudio AmplitudeToDB, LSEHead, BCE, 20ep |
| v5j-4fold | HGNetV2-B0 ×4 | 6M×4 | **0.913** | Same, 4-fold ensemble |
| v5m (single) | EfficientNet-B0 + proj 320→1024 | 5.3M | **0.898** | v5j recipe with 1x1 conv projection |
| v5n (single) | EfficientNet-B0 (native 320) | 5.3M | 0.875 | v5j recipe, no projection (control) |

### What top competitors achieve (from Kaggle discussion #683791, 2026-04-03)

| Person | Rank | LB | Key details |
|--------|------|-----|------------|
| **Boredom** | 8th | **0.922** (single/single-fold) | EfficientNet, "easily 925+", train_audio + labeled soundscapes |
| **Salman Ahmed** | 396th | **0.921** solo, **0.923** w/ priors | CE loss + sigmoid infer, waveform MixUp, 25ep, priors +0.006 |
| **Ali Ozan** | 2nd | **0.925** single model | No details, no unlabeled data |
| yukiZ | 257th | 0.904 (fold 0 only) | Raw waveform augmentation (pink noise, time stretch) |

### Gap analysis: our 0.906 vs 0.921-0.925

**1. No site/hour priors on SED output (estimated +0.006)**

Salman reports priors give +0.006 (0.917→0.923). We have `fuse_scores_with_tables`
implemented in exp 020 Perch pipeline but never applied to SED. The code builds
prior probability tables from site + hour metadata in labeled soundscapes and
blends them with model logits. This is pure post-processing — no retraining needed.

**2. Loss function: CE + sigmoid inference (unknown impact)**

Salman uses "CrossEntropyLoss on clipwise and framewise max" during training but
"Sigmoid at Infer." This is a single-label training assumption (CE picks the
primary label) with multi-label inference (sigmoid allows multiple species).

Our attempts with CE (exp 017 v4i, v4k) regressed badly (0.835 val), but those
used a different setup. Salman's approach may work because:
- CE on primary_label during training provides sharper gradients (one target)
- Multi-label handling at inference via sigmoid recovers co-occurrence
- Avoids the gradient dilution of soft_ce across rare co-occurring species

We never tested CE(primary_label) + sigmoid inference on the v5j/HGNet recipe.

**3. Waveform augmentation (unknown impact)**

Salman emphasizes "mixup on waves" (not spectrograms). yukiZ reports waveform
augmentations (Pink Noise, Time Stretch) are effective and used in past BirdCLEF
winning solutions.

Our test (exp 017 v4j): waveform MixUp regressed −0.006 LB on EfficientNet-B0.
But we never tested it on HGNetV2-B0 with the v5j recipe. The interaction may
differ on a stronger backbone.

**4. Backbone: MBConv (EfficientNet) struggles, other architectures work**

Salman explicitly says "Efficientnet NS (with MBConv Architecture) were not
converging fast and scored much lower on LB, compared to other architectures"
and "For now, Efficientnet B0 didn't help." His 0.921 uses a non-EfficientNet
backbone. Meanwhile Boredom (8th) says EfficientNet can reach 925+ — possibly
with a very different recipe.

Our own data confirms this: HGNetV2-B0 (non-MBConv) outperforms EfficientNet-B0
by +0.031 LB (v5j 0.906 vs v5f 0.855) on the same recipe. The MBConv
architecture may be suboptimal for audio mel spectrograms.

**5. Larger backbones with ONNX inference**

Our models are small (5-6M params). Top competitors likely use larger backbones.
Our exp 010 ONNX pipeline handles 20-30M param models (EAT 86M, ConvNeXt 88M)
within the 90-min Kaggle budget. Candidates for exploration:

| Model | Params | Embed dim | Architecture | timm slug |
|-------|--------|-----------|-------------|-----------|
| HGNetV2-B0 | 6M | 1024 | Non-MBConv (our current best) | `hgnetv2_b0.ssld_stage2_ft_in1k` |
| HGNetV2-B1 | 8M | 1024 | Non-MBConv | `hgnetv2_b1.ssld_stage2_ft_in1k` |
| HGNetV2-B2 | 12M | 1536 | Non-MBConv | `hgnetv2_b2.ssld_stage2_ft_in1k` |
| ConvNeXt-V2 Nano | 15M | 640 | Modern ConvNet, no MBConv | `convnextv2_nano.fcmae_ft_in22k_in1k` |
| ConvNeXt-V2 Tiny | 28M | 768 | Modern ConvNet, no MBConv | `convnextv2_tiny.fcmae_ft_in22k_in1k` |
| EfficientNet-B2 NS | 9M | 352 | MBConv (may struggle) | `tf_efficientnet_b2.ns_jft_in1k` |
| EfficientNet-B3 NS | 12M | 384 | MBConv (may struggle) | `tf_efficientnet_b3.ns_jft_in1k` |
| EfficientNet-B4 NS | 19M | 448 | MBConv (may struggle) | `tf_efficientnet_b4.ns_jft_in1k` |

Priority: larger HGNet and ConvNeXt-V2 (non-MBConv, aligned with Salman's findings).
EfficientNet NS as a lower-priority test — Salman explicitly warns against MBConv.

### Why not keep tuning Perch?

Perch ProtoSSM (0.925 LB) relies on a frozen feature extractor with
hand-tuned post-processing (6-stage PP, per-class thresholds, metadata priors).
Each increment requires LB-validated submission cycles with diminishing returns
(−0.005 gap to pantanal 0.930). A well-trained SED model is:
- More generalizable (learns directly from audio, not fixed embeddings)
- Faster to iterate (train on PACE, validate offline, fewer LB subs needed)
- Better foundation for pseudo-labeling and ensemble work

Perch remains valuable as a pseudo-labeling teacher for future experiments.

## Method

### Approach

Three phases, each building on findings:

**Phase A: Quick wins (no retraining)** — Apply site/hour priors to existing
v5j model output. Pure post-processing, just needs kernel changes.

**Phase B: Training recipe improvements** — Test CE+sigmoid, waveform augmentations
on v5j recipe with HGNetV2-B0. Single-variable ablations on PACE.

**Phase C: Backbone scaling** — Test larger non-MBConv backbones (HGNetV2-B1/B2,
ConvNeXt-V2) and larger EfficientNet-NS variants for comparison. ONNX export
for Kaggle inference.

### Setup

| Component | Details |
|-----------|---------|
| Data | 35K focal + 708 labeled soundscape windows (same as exp 019) |
| Compute | PACE RTX 6000 GPU, ~2-4h per training run |
| Dependencies | timm, torch, torchaudio, nnAudio, pytorch-lightning |
| Code | `user/murilo/experiments/021-sed-training/` (extends exp 019 train.py) |
| Branch | `mg/021-sed-training` |
| Base model | v5j recipe: HGNetV2-B0, LSEHead, BCE, NNAudio AmplitudeToDB, 5s, 20ep |

### Procedure

#### Phase A: Site/Hour Priors on SED (1 day, 1-2 LB subs)

Apply the prior fusion from exp 020 Perch pipeline to SED inference.
The prior tables are built from the 739 labeled soundscape annotations
(site × hour × species co-occurrence statistics).

- [ ] A.1 Port `fuse_scores_with_tables` + `fit_prior_tables` to SED submission kernel
- [ ] A.2 Submit v5j + priors → target ≥0.912 (v5j 0.906 + expected 0.006)
- [ ] A.3 Submit v5j-4fold + priors → target ≥0.919

#### Phase B: Training Recipe Ablations (3-5 days, 4-6 LB subs)

All ablations from v5j baseline (HGNetV2-B0, 0.906 LB). Single-variable changes.

| Run | Change | Hypothesis |
|-----|--------|-----------|
| v6a | CE(primary_label) + sigmoid inference | Sharper gradients, Salman's approach |
| v6b | Waveform MixUp (instead of spectrogram) | Salman + yukiZ both use this |
| v6c | Waveform MixUp + Pink Noise + Time Stretch | yukiZ's full augmentation stack |
| v6d | 25 epochs (vs 20) | Salman's epoch count, slightly longer training |
| v6e | Combined winners from v6a-d | Best recipe |

- [ ] B.1 Implement CE(primary_label) training mode in train.py
- [ ] B.2 Implement waveform augmentations (MixUp, pink noise, time stretch)
- [ ] B.3 Train v6a-v6d on PACE (parallel where possible)
- [ ] B.4 Submit winners to LB
- [ ] B.5 Train combined recipe v6e, submit to LB

#### Phase C: Backbone Scaling (2-3 days, 2-3 LB subs)

Test larger backbones with the best recipe from Phase B. Priority: non-MBConv
architectures (HGNet, ConvNeXt-V2), with EfficientNet-NS as a comparison.
All models exported to ONNX for Kaggle inference validation.

| Run | Backbone | Params | Embed dim | Architecture |
|-----|----------|--------|-----------|-------------|
| v6f | HGNetV2-B2 | 12M | 1536 | Non-MBConv (larger HGNet) |
| v6g | ConvNeXt-V2 Nano | 15M | 640 | Modern ConvNet |
| v6h | EfficientNet-B3 NS | 12M | 384 | MBConv (comparison) |

- [ ] C.1 Verify backbones work with v5j recipe + ONNX export
- [ ] C.2 Train v6f-v6h on PACE
- [ ] C.3 ONNX export + Kaggle inference timing validation
- [ ] C.4 Submit best to LB — target ≥0.920
- [ ] C.5 4-fold ensemble of best backbone — target ≥0.925+

### Variables

- **Independent**: Loss function, augmentation strategy, backbone architecture/scale
- **Dependent**: LB score, offline val/auroc, inference time
- **Controlled**: Data (35K focal + 708 soundscape), mel config (NNAudio AmplitudeToDB),
  head (LSEHead), optimizer (AdamW), scheduler (OneCycleLR), batch size (64)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| LB score | Competition metric (row-wise AUROC) — ground truth |
| val/auroc | Offline validation — UNRELIABLE for ranking (see exp 019 findings) |
| Inference time | Must be < 90 min on Kaggle CPU (or via ONNX) |

### Baselines

| Model | LB | Notes |
|-------|-----|-------|
| v5j (our best single) | 0.906 | HGNetV2-B0, v5j recipe |
| v5j-4fold | 0.913 | 4-fold ensemble |
| Perch ProtoSSM V18 | 0.925 | Two-kernel TF, exp 020 |
| Salman single | 0.921 | Non-EfficientNet, CE + waveform MixUp + priors |
| Boredom single | 0.922 | EfficientNet, recipe unknown |
| Competition #1 | 0.942 | Unknown |

### Success Criteria

- **Phase A confirm**: v5j + priors ≥ 0.910
- **Phase B confirm**: any v6 variant ≥ 0.915 (single-fold)
- **Phase B stretch**: ≥ 0.920 (matching Salman/Boredom)
- **Phase C confirm**: larger backbone ≥ v5j baseline (0.906)
- **Phase C stretch**: 4-fold ensemble ≥ 0.925 (matching Perch V18)

## Limitations

- Offline val/auroc is unreliable for ranking — each promising variant needs LB submission
- 5 LB subs/day constrains ablation speed
- CE loss on primary_label assumes single-label per clip — may fail on soundscapes
  with dense co-occurrence
- Larger backbones increase training time (~2x per size level) and may overfit on 708 windows
- Salman's warning about EfficientNet-NS MBConv may limit backbone options

## Leads To

- Stronger SED models for Perch + SED ensemble (exp 020 Phase D)
- Better teacher models for pseudo-labeling (future exp)
- Multi-model ensemble (SED + Perch + diverse backbones)

## References

- [Exp 019 SED Ablations](../019-sed-ablations/tasks.md) — v5a-v5n results
- [Exp 017 Mel Overhaul](../017-mel-overhaul/tasks.md) — v4a-v4k results
- [Exp 020 Perch ONNX](../020-perch-onnx/tasks.md) — Perch V18 baseline, prior tables code
- [Kaggle Discussion #683791](https://www.kaggle.com/competitions/birdclef-2026/discussion/683791) — competitor single-model scores
- [ttahara annotated](../../notes/ttahara-hgnetv2-b0-annotated.md) — base recipe
- [EfficientNet NS models](https://huggingface.co/timm/tf_efficientnet_b0.ns_jft_in1k) — Noisy Student pretrained
