# Proposal: Background-Noise Mixing on v5j

**Status**: proposed
**Created**: 2026-04-13 (as `028-domain-shift-augmentation`, bundled-everything plan)
**Rewritten**: 2026-04-25 — renamed and rescoped to a single-variable experiment after audio validation
**Author**: Murilo

## Hypothesis

v5j fails on focal crops where the bird call is masked by foreground ambient
noise (low SNR within the 5s window), because the training pipeline never
exposes the model to focal recordings mixed with environmental or anthropogenic
background noise.

**Adding additive background-noise mixing to the focal training pipeline,
on the otherwise-unchanged v5j recipe, will improve LB over the v5j baseline
(0.906).**

Falsifiable: if BG noise mixing alone, with reasonable parameters, produces
flat or negative LB movement, the failure mode may be real but this
intervention is the wrong lever.

## Background

### The failure mode (validated 2026-04-25)

Murilo listened to all 15 `mixed_uncertainty_roahaw` clips and 10
`correct_confident_roahaw` baseline clips, plus samples from `coffal1`,
`socfly1`, `epaori4`. Pattern across all listened-to mixed_uncertainty clips:

- Foreground is dominated by ambient hiss, white noise, or anthropogenic
  noise (e.g. chainsaw)
- Bird calls are weak, distant, partial, or absent
- Sometimes the call is genuinely missing from the 5s window (silent crop)

Baseline correct clips have the call as the foreground signal. **The
discriminator is foreground/background ratio, not call presence.** The
model output ("I don't hear roahaw confidently") is the honest answer
for these crops; BCE penalizes it because the recording-level label is
roahaw.

Full per-rank observations, cross-species generalization, and confidence
audit: [`analysis/notes/2026-04-25-audio-validation-low-snr-crops.md`](../../analysis/notes/2026-04-25-audio-validation-low-snr-crops.md).

### Why BG noise injection should help

Two arguments converge on this lever:

1. **Direct response to the failure mode.** If the model fails on low-SNR
   crops at inference time, the cleanest fix is to expose it to low-SNR
   conditions during training. Random cropping creates some natural
   low-SNR exposure, but limited and not parameterized. Explicit BG noise
   mixing controls SNR distribution directly.

2. **Competitor pattern.** Salman (0.937 single-model LB) explicitly uses
   external BG noise from prior competitions (BirdCLEF 2020/21 nocall +
   freefield1010 + warblrb + birdvox) "to make the model generalize better."
   Boredom (0.938-0.939 labeled-only) uses augmentation he hasn't fully
   disclosed but trains at 0.93+ levels. Both train at performance levels
   we haven't reached.

### Why this is being run as a single variable

Across exps 016–027, 21+ training-recipe changes have produced 0 LB wins
over v5j. Every positive LB delta has been input representation,
post-processing, or ensembling — never recipe.

The original 04-13 version of this experiment (then named
`028-domain-shift-augmentation`) bundled five training changes (BG noise
+ secondary labels + soundscape oversampling + SpecAugment + sqrt-balanced
sampling), arguing that individual recipe changes had failed and the
improvements might be complementary. That argument changed when the audio
listening produced direct evidence for *one* specific lever. Bundling it
with four un-evidenced changes would surrender attribution. This proposal
is BG noise mixing in isolation, on the otherwise-unchanged v5j recipe.

### Reference notes

- [`analysis/notes/2026-04-25-audio-validation-low-snr-crops.md`](../../analysis/notes/2026-04-25-audio-validation-low-snr-crops.md)
- [`analysis/notes/2026-04-24-three-regimes-of-mixed-uncertainty.md`](../../analysis/notes/2026-04-24-three-regimes-of-mixed-uncertainty.md)
- [`analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md`](../../analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md)
- [`analysis/notes/2026-04-24-triage-v5j-failure-modes.md`](../../analysis/notes/2026-04-24-triage-v5j-failure-modes.md)
- [`notes/kaggle-discussion-best-single-model-lb.md`](../../notes/kaggle-discussion-best-single-model-lb.md)

## Method

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026 focal (~35K) + labeled soundscapes (66 files), 90/10 split (same as v5j) |
| BG noise source | TBD — see scoping questions below |
| Compute | 1× RTX 6000 (PACE cluster), ~1–2h per training run |
| Code | Branch `mg/028-background-noise-mixing`. Train script copied from a recent v5j-compatible experiment (027 or 025). Config copied from `019-sed-ablations/config/hgnet_b0_sed_v5j.yaml` and modified only to add BG noise parameters |
| Baseline | v5j single fold = **0.906 LB** (exp 019) |

### Procedure

1. **Resolve scoping decisions** (see open questions below) — all four
   decisions made before any code is written.
2. **Implement BG noise dataset**: load BG noise segments from chosen
   source(s); return random segment, length-matched to the focal sample.
3. **Implement noise mixing augmentation** in the focal dataloader:
   additive `x_focal + α · x_bg` where `α` is computed from a sampled
   target SNR. **No label change** — BG noise contains no target species
   (or in the soundscape-derived case, low-activity segments specifically),
   so the focal recording's label vector is unaffected.
4. **Sanity check**: fast-dev-run + listen to a handful of mixed samples
   to confirm SNR sounds right and the focal call is still audible at
   the upper end of the SNR range.
5. **Full training**: single fold, 90/10 split, identical to v5j (20
   epochs, AdamW + OneCycleLR, BCE, spectrogram MixUp, etc.). Save best
   by val/auroc.
6. **ONNX export** via the standard pipeline.
7. **LB submission** — single submission against the v5j baseline.
8. **Re-run triage** on the new model: does mixed_uncertainty count
   drop, especially for the bimodal-regime species? This is a sanity
   check on whether the intervention actually shifted the failure mode,
   regardless of LB direction.

### Variables

- **Independent**: BG noise mixing — source(s), SNR range, application
  probability, soundscape-sample inclusion
- **Dependent**: Public LB score (row-wise macro-averaged ROC-AUC)
- **Controlled** (held at v5j values):
  - Architecture: HGNetV2-B0 + LSEHead
  - Loss: BCE
  - Optimizer: AdamW + OneCycleLR
  - Window: 5s
  - Spectrogram MixUp: α=1.0
  - Mel pipeline: NNAudio + AmplitudeToDB, n_fft=2048, n_mels=128
  - Epochs: 20
  - Data split: 90/10

### Open scoping questions (resolve before implementation)

1. **BG noise source.**
   - Internal: low-energy segments extracted from labeled soundscapes
     (in-domain, limited size, possibly redundant with what the model
     already sees from the 66 labeled files)
   - External: prior-comp BG datasets (BirdCLEF 2020/21 nocall +
     freefield1010 + warblrb + birdvox) — what Salman uses, larger and
     more varied
   - Both: external as primary, internal as supplementary
2. **SNR range.** Original bundled proposal said `[-6, +12]` dB.
   Audio listening (chainsaw masking case at rank 04) suggests widening
   the lower bound to `[-10, +12]` dB.
3. **Application probability.** Original said 50%. Worth considering
   75% given that the failure pattern shows up across most low-confidence
   focal crops.
4. **Apply to labeled soundscape samples too?** The 66 labeled
   soundscapes already contain natural BG noise. Default: skip noise
   injection on them.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **Public LB** | **Primary** — row-wise macro-averaged ROC-AUC on test soundscapes |
| val/auroc | Focal validation AUROC — informational only (anti-correlated with LB) |
| val_sc/auroc | Soundscape validation AUROC — informational |
| Triage on new model | Sanity check: does `mixed_uncertainty` drop for bimodal-regime species? |

### Baseline

- v5j single fold: **0.906 LB** (exp 019)

### Success Criteria

- **Confirm**: LB > 0.906
- **Strong confirm**: LB ≥ 0.910
- **Reject**: LB ≤ 0.906 across at least one full training run with
  reasonable BG noise parameters AND triage shows no shift in the
  bimodal-regime failure rate
- **Retry**: LB ≤ 0.906 BUT triage shows mixed_uncertainty did drop —
  the intervention worked on the diagnostic but didn't translate to LB;
  retry with different SNR/source/probability before giving up
- **Pivot**: independent diagnostic shows the bimodal-regime species
  aren't the main contributors to test-set loss, regardless of training
  outcome — would change the relevance of the audio finding

## Limitations

- **Bimodal-regime only.** The audio finding is high-confidence for
  species with median gt_prob ≥ 0.8 (the bimodal regime, ~50% of
  mixed_uncertainty). It says nothing about the other ~50% (uniformly
  uncertain or model-blind regimes). BG noise mixing may help only the
  bimodal portion of failures.
- **Test-set relevance unverified.** We have not confirmed that the 87
  bimodal species, or their failure modes, actually appear in the public
  test soundscapes. If they don't, fixing them won't move LB.
- **Offline metrics remain anti-correlated with LB.** LB is the only
  meaningful signal. Triage on the new model is a structural sanity
  check, not a metric.
- **Seed sensitivity.** Mahog reports ±0.005 from seed alone. One run
  can detect a clear win or loss; close results may need a seed sweep.
- **Parameter sensitivity unknown.** A first run with reasonable defaults
  is the test of *whether the lever exists*. If it helps, parameter
  sweeps come next; if it regresses, parameters are not the first thing
  to blame.

## Related work and potential follow-ups

These are NOT part of this experiment. Listed only for reference and to
be honest about the broader option space.

- **Other domain-shift levers from the 04-13 deep-research analyses**:
  secondary label supervision, soundscape oversampling, SpecAugment,
  sqrt-balanced sampling. None has direct evidence from the audio
  validation. If exp 028 produces a positive result, any of these
  becomes a candidate for its own future single-variable experiment.
  Original specifications are preserved in git history under
  `028-domain-shift-augmentation/proposal.md` (pre-2026-04-25 commits).
- **TopN=1 post-processing**. Inference-only, no training. Independent
  bet, future experiment if pursued.
- **2025 pretraining (Ali Ozan path)**. Different bet — more data, not
  better augmentation. Future experiment.
- **Seed sweep on v5j**. Establishes the ±σ for LB results; useful for
  judging marginal wins. Future small experiment.

If exp 028 fails, none of the above is automatically the next bet.
The next call depends on the *kind* of failure (LB flat with triage
shift vs LB regression with no shift, etc.).
