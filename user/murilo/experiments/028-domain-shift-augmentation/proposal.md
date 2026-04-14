# Proposal: Domain Shift Augmentation & Long-Tail Coverage

**Status**: proposed
**Created**: 2026-04-13
**Author**: Murilo

## Hypothesis

The 0.906→0.93+ gap versus competitors (Boredom 0.938, Salman 0.922) using the
same labeled data is caused by undertreated domain shift in our augmentation stack
and inadequate long-tail/non-bird taxa coverage. Specifically:

1. Our model never trains with environmental background noise mixed into focal
   recordings, so it can't extract bird signals from noisy soundscapes
2. We ignore secondary species labels in focal recordings, so the model never
   sees multi-species supervision (which is the norm in soundscapes)
3. We don't oversample the 66 labeled soundscapes, undertreating the only
   in-domain training signal we have
4. We don't use SpecAugment, missing a proven robustness augmentation
5. We don't balance sampling for long-tail species with few recordings
6. We don't apply any post-processing to exploit temporal coherence in soundscapes

**A combined training recipe addressing all five training gaps, plus TopN
post-processing at inference, will improve LB over the v5j baseline (0.906).**

This hypothesis is falsifiable: if the combined recipe + post-processing still
regresses on LB, then the gap is caused by something else entirely (e.g.,
pre-training data, backbone choice, or a systematic pipeline issue we haven't
identified).

## Background

After 21+ changes across 7 experiments (19 training recipe changes + 2 pseudo-label
approaches), we have 0 LB improvements over the v5j baseline (0.906). Every change
modified either HOW the model trains or WHAT extra data it sees.

Two independent deep research analyses (Claude and Gemini, run without context of
our experiments) converged on the same pattern: the gap at the 0.90 level is about
**domain adaptation**, not architecture. Both identified:

- **Environmental noise augmentation** as critical (different from bird+bird MixUp)
- **Secondary labels** as underutilized
- **Long-tail/non-bird taxa** as systematically undertreated
- **Post-processing** (TopN temporal consistency) as a free +0.01-0.015
- **SpecAugment** as a standard robustness technique we're missing

Key competitor reference points (labeled data only, no KD):
- Boredom: 0.938 (non-Perch, non-KD, no unlabeled data)
- Salman: 0.922 (EfficientNet-B0, uses BG noise from prior competitions)
- Us: 0.906 (HGNetV2-B0, LSEHead, BCE)

References:
- `notes/2026-04-13-claude-deep-research-birdclef-strategy.md`
- `notes/2026-04-13-gemini-deep-research-birdclef-strategy.md`
- `notes/kaggle-discussion-best-single-model-lb.md`

## Method

### Approach

Three-phase experiment: (A) diagnostic analysis to understand where we fail,
(B) inference-only post-processing test, (C) combined training recipe addressing
all identified gaps. We test the combined recipe as a single change because
individual recipe changes have consistently failed — the hypothesis is that these
improvements are complementary and need to work together.

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 focal (35K) + labeled soundscapes (66 files), 90/10 split |
| Compute | 1x RTX 6000 (PACE cluster), ~1-2h per training run |
| Dependencies | Existing v5j pipeline + new augmentation modules |
| Code | Branch `mg/028-domain-shift-augmentation`, based on exp 025 codebase |
| Baseline | v5j single fold = 0.906 LB (exp 019) |

### Procedure

#### Phase A: Diagnostic — Where are we failing?

v5j inference already exists at `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/`.

1. Load existing v5j logits on labeled soundscapes (no new inference needed)
2. Compute per-species ROC-AUC on the 75 species with soundscape labels
3. Compute per-species ROC-AUC grouped by taxonomy (Aves vs Amphibia vs Insecta)
4. Identify worst-performing species and check the "Invisible 28" (soundscape-only)
5. Analyze confidence distributions: focal-trained species vs soundscape-only species
6. Document specific failure modes to inform Phase C

#### Phase B: TopN=1 Post-Processing (inference-only)

7. Implement TopN=1 post-processing in submission kernel:
   `prob[file, segment, class] *= max(prob[file, :, class])`
8. Apply to existing v5j ONNX model predictions
9. Submit to LB — expected gain: +0.01-0.015 (zero training cost)

#### Phase C: Combined Training Recipe

10. Implement augmentation changes in `src/birdclef/data/sed.py`:

    **C.1: Environmental noise mixing**
    - Extract noise segments from labeled soundscapes (segments with low bird activity)
    - Also: random Gaussian noise, pink noise at varying SNR (-6 to +12 dB)
    - Apply to 50% of focal samples (additive, no label change)
    - This is NOT MixUp — we're adding noise, not mixing two bird recordings

    **C.2: Secondary label supervision**
    - Parse `secondary_labels` column from train.csv (4,372 recordings have them)
    - Add secondary species to the multi-hot label vector with weight 1.0
    - The model now sees multi-species labels even from focal data

    **C.3: Soundscape oversampling**
    - Oversample the 66 labeled soundscape files (e.g., 5-10x weight)
    - These are the only in-domain training data with ground truth
    - Currently treated equally with 35K focal recordings

    **C.4: SpecAugment**
    - Time masking: 1-2 masks, max width 40 frames
    - Frequency masking: 1-2 masks, max width 20 mel bins
    - Applied after mel computation, before MixUp

    **C.5: Sqrt-balanced sampling**
    - Sample weight per species: w_i = (count_i / total)^(-0.5)
    - Addresses long-tail species with <10 recordings
    - Ensures rare species appear more frequently during training

11. Create config `configs/v12_domain_shift.yaml` with all changes
12. Fast-dev-run sanity check
13. Full training (single fold, 90/10 split, 40 epochs)
14. ONNX export + LB submission (with and without TopN post-processing)

### Variables

- **Independent**: Augmentation stack (noise mixing, secondary labels, soundscape
  oversampling, SpecAugment, balanced sampling) + TopN post-processing
- **Dependent**: Public LB score (row-wise macro-averaged ROC-AUC)
- **Controlled**: Model architecture (HGNetV2-B0 + LSEHead), loss (BCE), optimizer
  (AdamW + OneCycleLR), window duration (5s), MixUp (spectrogram, alpha=1.0)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Public LB score | Primary — row-wise macro-averaged ROC-AUC on test soundscapes |
| val/auroc | Focal validation AUROC (informational only — anti-correlated with LB) |
| val_sc/auroc | Soundscape validation AUROC (informational) |
| Per-species AUC (Phase A) | Diagnostic — identifies specific failure modes |
| Per-taxonomy AUC (Phase A) | Diagnostic — Aves vs Amphibia vs Insecta performance |

### Baseline

- v5j single fold: **0.906 LB** (exp 019)
- v5j + TopN post-processing: **TBD** (Phase B result)

### Success Criteria

- **Confirm if**: Phase C LB > 0.906 (any improvement over v5j baseline)
- **Strong confirm if**: Phase C LB > 0.910 (meaningful improvement)
- **Reject if**: Phase C LB ≤ 0.906 AND Phase B (TopN) LB ≤ 0.906
- **Pivot if**: Phase A diagnostics reveal an unexpected failure mode that
  changes the hypothesis

## Limitations

- We're combining 5 training changes at once, so if it works we won't know
  which individual change contributed most. This is intentional — 21 individual
  changes have all failed, suggesting they need to work together.
- If the combined recipe also fails, we'll need to investigate whether there's
  a systematic pipeline issue (e.g., ONNX export, inference kernel, mel config)
  rather than a training gap.
- Offline metrics (val/auroc) are anti-correlated with LB. LB is the only
  meaningful signal.
- The "Invisible 28" species may or may not appear in the test soundscapes.
  If they don't, improvements on those species won't affect LB.
- TopN post-processing (+0.01-0.015 cited) was documented for BirdCLEF 2025.
  The gain may be different for 2026.
