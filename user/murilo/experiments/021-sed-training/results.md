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

### Key Takeaways

1. **CE loss is dead** — consistently −0.015 to −0.024 across LSEHead (Phase B) and AttHead (Phase C), on both HGNet and EffNet. Despite winning solutions using CE, our setup can't make it work.
2. **AttHead is not a universal upgrade** — on HGNetV2-B0, it regressed −0.004 vs LSEHead. The improvement comes from pairing AttHead with a larger backbone (B2).
3. **Backbone scaling matters more than head type** — v6h (B2+AttHead, 0.876) > v5j (B0+LSEHead, 0.874) > v6e (B0+AttHead, 0.870).
4. **ConvNeXt-V2 has ensemble potential** — matches AUROC with best cmap (0.663), different architecture from HGNet.
5. **Offline metrics are modest** — best delta is +0.002. LB submission is essential to validate whether these translate.

### LB Submission Priority

1. **v6h** (HGNetV2-B2, AttHead, BCE) — best offline AUROC, highest priority
2. **v6i** (ConvNeXt-V2 Nano) — ensemble diversity, highest cmap
3. **v6j** (EffNet-B0 NS) — tests Boredom's EffNet claim on LB
