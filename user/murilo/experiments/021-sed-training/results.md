# Experiment 021: SED Training — Results

## Phase A: Site/Hour Priors

### v5j-pp2: 0.909 LB (+0.003)

Priors added +0.003 over the v5j baseline (0.906 → 0.909). Below the +0.006
expected from Salman's report.

| Submission | LB | Runtime | Notes |
|-----------|-----|---------|-------|
| v5j (baseline, exp 019) | **0.906** | ~20 min | No priors |
| **v5j-pp2** | **0.909** | 11 min | + site/hour priors |
| v5j-4fold (baseline, exp 019) | **0.913** | ~50 min | No priors |
| **v5j-4fold-pp2** | **0.917** | ~50 min | **+ site/hour priors (+0.004)** |

### Why only +0.003 (not +0.006)?

Possible explanations for the smaller-than-expected prior boost:

1. **Different prior implementation**: Our exp 010 applies priors multiplicatively
   in probability space (`prob *= prior^lambda`). Salman may use logit-space
   blending (like exp 020's `fuse_scores_with_tables`) which is more nuanced
   with 6-category taxa handling and different lambda per category.

2. **Lambda tuning**: Our lambdas are fixed (Aves=0.3, Amphibia/Insecta=0.5,
   Mammalia/Reptilia=0.2). Salman may have tuned these on a validation set.

3. **Model strength matters**: v5j (0.906) may already be capturing some
   site/time patterns that priors are redundant with. Salman's base model
   (0.917 → 0.923) may have different failure modes that priors correct better.

4. **Test set sites**: If the hidden test set includes sites not in our 9
   training sites, priors fall back to global rates (minimal effect).

### Potential follow-up

- Try exp 020's logit-space `fuse_scores_with_tables` approach on SED output
- Tune lambda values via cross-validation
- But +0.003 suggests priors are a minor factor — training improvements (Phase B)
  are likely more impactful

## Phase B: Training Recipe Ablations

All ablations from v5j baseline (HGNetV2-B0, val/auroc 0.874, LB 0.906).
Single-variable changes, trained with seed=42, deterministic=True.

### Offline Results

| Model | Change | val/auroc | val/cmap | Best Epoch | Delta vs v5j |
|-------|--------|-----------|----------|------------|-------------|
| v5j (baseline) | — | **0.874** | — | 19 | — |
| v6a | CE loss | 0.854 | 0.580 | 18 | **−0.020** |
| v6b | Waveform MixUp | 0.871 | 0.625 | 16 | −0.003 |
| v6c | Waveform MixUp + noise + stretch | 0.861 | 0.608 | 16 | −0.013 |
| v6d | 25 epochs | **0.873** | **0.651** | 22 | −0.001 |

### Analysis

**v6a (CE loss): −0.020 val/auroc — clear regression.**
CE on primary_label significantly hurts. Despite Salman's success with CE, our
HGNetV2 + LSEHead + v5j recipe doesn't benefit. The multi-label nature of
soundscape data means BCE (per-species binary) is more appropriate than CE
(single-label assumption). Salman may use a different head architecture that
pairs better with CE.

**v6b (waveform MixUp): −0.003 val/auroc — slight regression.**
Waveform MixUp didn't help on HGNetV2, consistent with our earlier finding on
EfficientNet-B0 (exp 017 v4j, −0.006 LB). Spectrogram MixUp remains better
for our recipe. However, offline val/auroc is unreliable — this could still
improve on LB. Worth a submission if budget allows.

**v6c (full waveform augs): −0.013 val/auroc — clear regression.**
Adding pink noise + time stretch on top of waveform MixUp made things worse.
The augmentations may be too aggressive, or the SNR range (10-30dB) and stretch
range (0.8-1.25x) need tuning. Also possible that NNAudio's mel normalization
(AmplitudeToDB) already handles noise robustness well enough.

**v6d (25 epochs): −0.001 val/auroc — essentially flat, but best cmap.**
Nearly identical to v5j on AUROC. The extra 5 epochs improved cmap from ~0.625
(v5j approximate) to 0.651, suggesting the model learned better class-specific
patterns with slightly longer training. Best epoch was 22 (used extra headroom).
Worth submitting to LB as it's essentially free.

## Phase C: AttHead + Backbone Scaling

All models use **AttHead with dual supervision** (clipwise_weight=0.5,
framewise_weight=0.5, blend 75% clip + 25% frame max). GeM freq pooling + attention
head, replacing LSEHead from v5j baseline. Motivated by BirdCLEF 2023/2024/2025
winning solutions and Salman's explicit recommendation ("Attn head's framewise max
is better than LSE head").

### Offline Results

| Model | Backbone | Params | Head | Loss | val/auroc | val/cmap | Best Epoch | Delta vs v5j |
|-------|----------|--------|------|------|-----------|----------|------------|-------------|
| v5j (baseline) | HGNetV2-B0 | 6M | LSEHead | BCE | **0.874** | — | 19 | — |
| **v6h** | **HGNetV2-B2** | **12M** | AttHead | BCE | **0.876** | **0.658** | 18 | **+0.002** |
| v6i | ConvNeXt-V2 Nano | 15M | AttHead | BCE | 0.874 | **0.663** | 16 | ±0 |
| v6j | EffNet-B0 NS | 5.3M | AttHead | BCE | 0.874 | 0.626 | 15 | ±0 |
| v6e | HGNetV2-B0 | 6M | AttHead | BCE | 0.870 | 0.611 | 18 | −0.004 |
| v6g | EffNet-B3 NS | 12M | AttHead | CE | 0.859 | 0.611 | 19 | −0.015 |
| v6f | HGNetV2-B0 | 6M | AttHead | CE | 0.850 | 0.577 | 18 | −0.024 |

### Analysis

**v6h (HGNetV2-B2, AttHead, BCE): +0.002 val/auroc — best offline performer.**
Scaling HGNet from B0 (6M) to B2 (12M) with AttHead achieves the best offline AUROC
(0.876) and strong cmap (0.658). The larger backbone provides modest but consistent
improvement. LB validation required — offline gains have been unreliable predictors.

**v6i (ConvNeXt-V2 Nano, AttHead, BCE): ±0 val/auroc — matches baseline.**
Architecture diversity test. ConvNeXt matches v5j AUROC exactly (0.874) but achieves
the highest cmap of all Phase C models (0.663), suggesting better per-class calibration.
Worth a LB submission for ensemble diversity even if AUROC is flat.

**v6j (EffNet-B0 NS, AttHead, BCE): ±0 val/auroc — matches baseline.**
Tests Boredom's claim that EfficientNet can "easily" reach 925+ LB. Offline AUROC
matches v5j (0.874) but doesn't show the improvement Boredom implies. The gap may
be in recipe details we haven't reproduced (training schedule, augmentations, etc.).

**v6e (HGNetV2-B0, AttHead, BCE): −0.004 val/auroc — AttHead regressed on B0.**
Isolating the head swap (LSEHead → AttHead) on HGNetV2-B0 actually hurt by −0.004.
This is surprising given winning solutions' preference for AttHead. Possible
explanations: the dual supervision (50/50 clip/frame) may need different weighting
for smaller backbones, or the 75/25 blend ratio needs tuning for HGNet's embedding
structure.

**v6f (HGNetV2-B0, AttHead, CE): −0.024 val/auroc — CE with AttHead still fails.**
CE loss regressed even worse with AttHead than with LSEHead (v6a: −0.020). This
definitively rules out the hypothesis that CE needs AttHead to work. CE loss with
single-label primary_label is fundamentally incompatible with our multi-label
soundscape data setup, regardless of head architecture.

**v6g (EffNet-B3 NS, AttHead, CE): −0.015 val/auroc — larger EffNet + CE fails.**
EfficientNet-B3 (12M params) with CE loss underperforms. Confirms both findings:
CE loss hurts (−0.015 vs EfficientNet with BCE), and MBConv architecture struggles
on mels compared to HGNet. Salman also noted "MBConvs from EfficientNets are not
learning that well."

### LB Results

| Model | Backbone | Head | val/auroc | LB | Delta vs v5j (0.906 LB) |
|-------|----------|------|-----------|-----|-------------------------|
| v5j (baseline) | HGNetV2-B0 | LSEHead | 0.874 | **0.906** | — |
| **v6j** | **EffNet-B0 NS** | AttHead | 0.874 | **0.893** | **−0.013** |
| v6h | HGNetV2-B2 | AttHead | **0.876** | 0.889 | −0.017 |

**Both AttHead models regressed on LB** despite matching or beating v5j offline.
v6h was the best offline (+0.002) but worst on LB (−0.017). v6j (EffNet) beat v6h
on LB despite identical offline AUROC — further proof that offline metrics don't
predict LB performance.

### Key Takeaways

1. **AttHead with 75/25 blend regresses on LB** — v5j (LSEHead, clipwise-only) at 0.906 vs v6h/v6j (AttHead, 75/25 blend) at 0.889/0.893. The dual supervision + framewise blending hurts LB despite being competition-standard. Our AttHead implementation differs from the reference (see `atthead-comparison.md`).
2. **CE loss is dead** — consistently −0.015 to −0.024 across LSEHead (Phase B) and AttHead (Phase C), on both HGNet and EffNet.
3. **Offline metrics are anti-correlated with LB** — v6h had best offline (+0.002) but worst LB (−0.017). v6j had same offline as v5j but lost −0.013 on LB. Head type and backbone scaling (Phase B/C) are not the lever.

## Phase D: Multi-Context Training (AttHeadMultiContext)

Multi-context training using `AttHeadMultiContext` (competition-standard attention head
with train_period/infer_period cropping). Trains on 15-20s windows, predicts center 5s.

### Offline Results

| Model | Backbone | Head | Context | val/auroc | val/cmap | Best Epoch | Delta vs v5j |
|-------|----------|------|---------|-----------|----------|------------|-------------|
| v5j (baseline) | HGNetV2-B0 | LSEHead | 5s | **0.874** | — | 19 | — |
| v7c | EffNet-B0 NS | AttHeadMC | 3×5s (15s) | 0.851 | 0.605 | 19 | −0.023 |
| v7a | HGNetV2-B0 | AttHeadMC | 3×5s (15s) | 0.820 | 0.579 | 16 | −0.054 |
| v7b | HGNetV2-B0 | AttHeadMC | 4×5s (20s) | 0.799 | 0.542 | 16 | −0.075 |

**All v7 models massively regressed.** Longer context = worse (v7b 20s < v7a 15s).
HGNet regressed more than EffNet (opposite of all prior experiments).

### Analysis: Why multi-context failed

The v7 runs changed **too many variables at once** from v5j:
1. Head: LSEHead → AttHeadMultiContext (different architecture, loss gradient path)
2. Window: 5s → 15-20s (3-4× more context)
3. Batch size: 64 → 32/24 (halved due to memory)
4. Clipwise aggregation: LSE pooling → sigmoid probability-space attention

**Confounds make it impossible to isolate the multi-context effect.** The head change
alone already hurts (Phase C showed AttHead regresses −0.004 to −0.017 vs LSEHead).
Adding multi-context on top of an already-worse head compounds the regression.

**Possible root causes:**

1. **AttHeadMultiContext itself** — the `logit(Σ(att × sigmoid(logits)))` loss path
   differs from Tattaka's `BCEWithLogitsLoss(Σ(att × logits))`. This may produce
   weaker gradients or training instability (noted in `atthead-comparison.md`).

2. **Zero-padding of short recordings** — many focal recordings are shorter than 15s.
   With 15s window_duration, these get zero-padded. The model sees lots of silence,
   diluting the training signal. v5j's 5s windows had much less padding.

3. **Label mismatch with context** — during training, the label applies to the center
   5s but the model aggregates over all 15s. Flanking soundscape segments may contain
   different species, confusing the attention. For focal recordings, the label covers
   the whole file but padded silence contributes noise.

4. **Batch size halved** — 32 (v7a/c) or 24 (v7b) vs 64 (v5j). Changes MixUp
   pairing diversity and gradient noise characteristics.

5. **HGNet collapses with long context** — v7a/v7b (HGNet, 0.820/0.799) regressed
   far more than v7c (EffNet, 0.851). HGNet's embedding structure may not handle
   3-4× longer temporal features well with the attention mechanism.

### Lesson: isolate variables, keep LSEHead

## Phase E: LSEHead + Temporal Split (correct multi-context)

Correct implementation based on hengck23's image: keep LSEHead exactly as v5j,
add `n_chunks` to split temporal features before LSE pooling. Only TWO changes
from v5j: `window_duration` and `n_chunks`. See `multi-context-data-flow.md`.

### Offline Results

| Model | Backbone | Head | Context | n_chunks | val/auroc | val/cmap | Best Epoch | Delta vs v5j |
|-------|----------|------|---------|----------|-----------|----------|------------|-------------|
| v5j (baseline) | HGNetV2-B0 | LSEHead | 5s | 1 | 0.874 | ~0.625 | 19 | — |
| **v8a** | HGNetV2-B0 | LSEHead | **3×5s (15s)** | **3** | **0.882** | **0.719** | 13 | **+0.008** |
| **v8b** | HGNetV2-B0 | LSEHead | **4×5s (20s)** | **4** | **0.881** | **0.727** | 9 | **+0.007** |

**v8a shows the largest offline improvement we've ever seen** (+0.008 AUROC, +0.094 cmap
vs v5j). The cmap of 0.719 is by far the best across all experiments — the multi-context
backbone features are helping the model detect per-class patterns much better.

Critically, v8a uses the exact same LSEHead as v5j — the improvement comes purely from
the backbone seeing 15s of context and splitting into 3 chunks before pooling. This
validates hengck23's approach: context from the backbone, precision from the head.

Both v8a and v8b show best-ever offline metrics. v8b (4×5s=20s) matches v8a on AUROC
(0.881 vs 0.882) and has even better cmap (0.727 vs 0.719).

### LB Results

| Model | Context | n_chunks | Inference | val/auroc | LB | Delta vs v5j |
|-------|---------|----------|-----------|-----------|-----|-------------|
| v5j (baseline) | 5s | 1 | — | 0.874 | **0.906** | — |
| v8b | 4×5s (20s) | 4 | avg all 4 | 0.881 | **0.896** | −0.010 |
| v8a | 3×5s (15s) | 3 | avg all 3 | 0.882 | 0.882 | −0.024 |
| v8a (center) | 3×5s (15s) | 3 | **center chunk** | 0.882 | **0.883** | −0.023 |

**Multi-context improves offline but regresses on LB.** Center-chunk inference
gave only +0.001 over averaging (0.883 vs 0.882) — negligible. The hypothesis
that averaging dilutes predictions was wrong, or at least not the main issue.

v8b (0.896) outperformed v8a (0.883) significantly despite more chunks (4 vs 3).
Longer backbone context (20s) helps more than inference strategy. Since center-chunk
didn't help v8a, re-exporting v8b with center chunks is unlikely to improve on 0.896.

### Analysis: why multi-context regresses on LB despite best-ever offline

The +0.008 offline AUROC gain doesn't translate to LB. Possible explanations:

1. **Domain shift amplification**: longer windows from focal recordings (clean,
   single-species) may teach backbone features that don't transfer to soundscapes
   (noisy, multi-species). The 15-20s focal context is acoustically different from
   15-20s of Pantanal soundscape.

2. **Zero-padding artifacts**: ~30% of focal recordings are <15s. Zero-padded regions
   produce low-information backbone features that may confuse the head at test time
   when all 15-20s contain real audio.

3. **Training signal mismatch**: all 3-4 chunks get the same label during training,
   but test soundscapes have different species in adjacent 5s segments. The model
   may learn to spread predictions across chunks rather than localizing them.

4. **hengck23's missing ingredient**: his 0.896 used 4-fold ensemble (we submitted
   single fold) and "wave mix augmentation specially designed for 3×5sec." The
   multi-context approach may need both 4-fold and specialized augmentation to work.

## Phase F: Gap Analysis Fixes

Motivated by gap analysis (`notes/2026-04-05-sed-gap-analysis.md`) comparing our
v5j (0.906) against Nikita 1st place 2025 and Salman (0.921).

### F.1: Framewise-max inference — no gain

Re-exported v5j with `blend_clip=0.0, blend_frame=1.0` (hard framewise max).
LB: **0.906** — identical to v5j's 0.75/0.25 blend. LSEHead with T=1.0 already
approximates hard max, so the pooling method is not the gap.

### F.4: Competition-standard AttHead (v6k)

Fixed AttHead to match Nikita/Tattaka exactly: pre-dropout(p/2), softmax(tanh(x))
attention, probability-space clipwise aggregation. Trained with dual supervision
(50% clipwise + 50% framewise max). Everything else identical to v5j (HGNetV2-B0,
BCE, OneCycleLR, hidden_dim=1024).

| Model | Head | Supervision | val/auroc | val/cmap | Best Epoch | Delta vs v5j |
|-------|------|-------------|-----------|----------|------------|-------------|
| v5j (baseline) | LSEHead | clip only | **0.874** | ~0.625 | 19 | — |
| v6e (old AttHead) | AttHead (buggy) | clip 0.5 + frame 0.5 | 0.870 | 0.611 | 18 | −0.004 |
| **v6k (fixed AttHead)** | **AttHead (Nikita)** | **clip 0.5 + frame 0.5** | **0.871** | **0.596** | **19** | **−0.003** |

**The competition-standard fixes barely changed anything** (+0.001 vs old AttHead v6e).
The three fixes (pre-dropout, tanh, probability-space) are not the differentiator.
AttHead consistently underperforms LSEHead on HGNetV2-B0 by −0.003 to −0.004 on
AUROC and significantly on cmap (0.596 vs ~0.625).

**Interpretation**: The gap between us (0.906) and Salman (0.921) is NOT the AttHead
implementation. LSEHead may actually be the better head for our backbone/recipe.
The gap must come from something else entirely: backbone choice, CE loss with
proper pairing, waveform augmentation, or a combination thereof.
