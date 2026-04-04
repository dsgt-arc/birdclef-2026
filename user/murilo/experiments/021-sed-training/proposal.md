---
experiment: "021-sed-training"
status: in-progress
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

Salman reports priors give +0.006 (0.917→0.923). We have prior infrastructure
in exp 010's submission pipeline (`_apply_metadata_priors`) but never generated
the priors JSON for our SED models. Pure post-processing — no retraining needed.

**How site/hour priors work:**

1. Parse test soundscape filename → extract **site** (e.g., S08) and **hour** (e.g., 03 UTC)
2. Look up each species' prior probability using a fallback hierarchy:
   site+hour → site only → hour only → global frequency
3. Apply multiplicatively, scaled by taxa class:
   `adjusted_prob = model_prob × prior^lambda`
   - Aves: λ=0.3 (birds are mobile, weak prior)
   - Amphibia/Insecta: λ=0.5 (strongly tied to site/time)
   - Mammalia/Reptilia: λ=0.2 (most mobile, weakest prior)

**Prior source:** 66 labeled soundscapes (1478 annotated segments), 9 sites,
13 hours. Laplace-smoothed (α=0.01) to avoid zeroing out any species.

**Effect:** Suppresses unlikely species for a given location/time (e.g., a frog
at a site where it's never been observed gets its score reduced), while leaving
common species mostly unchanged. Fallback to global priors for unseen sites/hours
in the test set.

**Risk:** Only 9 sites in training data. If the test set has different sites,
priors fall back to global rates (minimal effect). Laplace smoothing prevents
any species from being fully zeroed out.

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

**Phase A: Quick wins (no retraining)** — COMPLETE. Site/hour priors on SED output.
v5j-pp2 0.909 (+0.003), v5j-4fold-pp2 0.917 (+0.004).

**Phase B: Training recipe ablations** — COMPLETE (no gains). CE loss, waveform
MixUp, noise/stretch, 25 epochs — all regressed or flat on LSEHead.

**Phase C: AttHead + backbone scaling** — IN PROGRESS. Winning solution analysis
revealed the gap is architectural: every BirdCLEF winner (2023/2024/2025) uses
**AttHead with GeM freq pooling + dual supervision + 75/25 clip/frame blending**.
Phase B's CE loss regression was because CE needs AttHead's framewise max channel.

### Winning Solution Evidence for AttHead

| Year | Winner | Head | Loss | Blending |
|------|--------|------|------|----------|
| 2023 1st | Volodymyr | AttHead (GeM freq + attention) | Focal | 75% clip + 25% frame max |
| 2024 1st | Team Kefir | Pure backbone (no SED) | CE | sigmoid inference |
| 2025 1st | Nikita Babich | AttHead (GeM freq + attention) | CE | 75% clip + 25% frame max |
| 2025 9th | i2nfinit3y | AttBlockV2 | BCE | framewise max |
| Salman (0.921) | — | AttHead | CE | framewise max > LSEHead |

### Setup

| Component | Details |
|-----------|---------|
| Data | 35K focal + 708 labeled soundscape windows (same as exp 019) |
| Compute | PACE GPU (~20-30 min per training run) |
| Dependencies | timm, torch, torchaudio, nnAudio, pytorch-lightning |
| Code | `user/murilo/experiments/021-sed-training/` |
| Branch | `mg/021-sed-training` |
| Base mel | v5j recipe: NNAudio AmplitudeToDB, n_fft=2048, n_mels=256, 5s, 20ep, OneCycleLR |
| Head | **AttHead** (GeM freq pooling + attention) with dual supervision |
| Dual supervision | clipwise_weight=0.5, framewise_weight=0.5, blend 75% clip + 25% frame max |

### Phase C Config Table

All configs use AttHead with dual supervision. Base mel pipeline from v5j.

| Config | Backbone | Params | Head | Loss | What it tests | Priority |
|--------|----------|--------|------|------|--------------|----------|
| **v6e** | HGNetV2-B0 | 6M | **AttHead** | BCE | Isolate head: LSEHead → AttHead (same backbone as v5j) | **Highest** |
| **v6f** | HGNetV2-B0 | 6M | **AttHead** | **CE** | Winning loss combo: AttHead + CE (2024+2025 1st place recipe) | **High** |
| **v6g** | **EffNet-B3 NS** | 12M | **AttHead** | **CE** | Full winning recipe on larger EfficientNet (NS-JFT weights) | **High** |
| **v6h** | **HGNetV2-B2** | 12M | **AttHead** | BCE | Backbone capacity: 2x params, 1536 embed (vs B0's 1024) | Medium |
| **v6i** | **ConvNeXt-V2 Nano** | 15M | **AttHead** | BCE | Architecture diversity: non-MBConv, non-HGNet backbone | Medium |
| **v6j** | **EffNet-B0 NS** | 5.3M | **AttHead** | BCE | Boredom says EffNet reaches 925+; NS-JFT weights + proj 1024 | Medium |

Detailed config parameters:

| Config | timm model | Embed dim | proj_dim | hidden_dim |
|--------|-----------|-----------|----------|------------|
| v6e | `hgnetv2_b0.ssld_stage2_ft_in1k` | 1024 | — | 1024 |
| v6f | `hgnetv2_b0.ssld_stage2_ft_in1k` | 1024 | — | 1024 |
| v6g | `tf_efficientnet_b3.ns_jft_in1k` | 384 | 1024 | 1024 |
| v6h | `hgnetv2_b2.ssld_stage2_ft_in1k` | 1536 | — | 1024 |
| v6i | `convnextv2_nano.fcmae_ft_in22k_in1k` | 640 | — | 640 |
| v6j | `tf_efficientnet_b0.ns_jft_in1k` | 320 | 1024 | 1024 |

### Variables

- **Independent**: SED head type (AttHead vs LSEHead), loss function, backbone architecture/scale
- **Dependent**: LB score, offline val/auroc, inference time
- **Controlled**: Data (35K focal + 708 soundscape), mel config (NNAudio AmplitudeToDB),
  optimizer (AdamW), scheduler (OneCycleLR), batch size (64), seed (42)

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
| v5j (our best single) | 0.906 | HGNetV2-B0, LSEHead, BCE |
| v5j-pp2 | 0.909 | v5j + site/hour priors |
| v5j-4fold-pp2 | **0.917** | Current best SED |
| Perch ProtoSSM V18 | **0.925** | Two-kernel TF, exp 020 |
| Salman single | 0.921 | AttHead, CE, non-EfficientNet + priors |
| Boredom single | 0.922 | EfficientNet, "easily 925+" |
| Competition #1 | 0.942 | Unknown |

### Success Criteria

- **Phase A**: ~~v5j + priors ≥ 0.910~~ → **DONE: 0.909 single, 0.917 4-fold**
- **Phase B**: ~~recipe ablation ≥ 0.915~~ → **DONE: no gains on LSEHead**
- **Phase C confirm**: any AttHead variant ≥ 0.910 single (beats v5j-pp2)
- **Phase C stretch**: ≥ 0.920 single (matching Salman/Boredom)
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
