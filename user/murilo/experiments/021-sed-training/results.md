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
