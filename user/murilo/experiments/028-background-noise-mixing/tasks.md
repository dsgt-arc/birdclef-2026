# Experiment 028: Background-Noise Mixing on v5j — Tasks

## Goal

Test whether adding additive background-noise mixing to the v5j training
pipeline, on the otherwise-unchanged v5j recipe, improves LB over the
0.906 baseline. **Single-variable, evidence-led** — motivated by the
2026-04-25 audio validation that confirmed v5j fails on low-SNR focal
crops.

**Branch**: `mg/028-background-noise-mixing`
**Baseline**: v5j single fold = 0.906 LB (exp 019)
**Reference**: [audio-validation note](../../analysis/notes/2026-04-25-audio-validation-low-snr-crops.md)

---

## 1. Scoping decisions (resolve before any implementation)

- [ ] 1.1 BG noise source — internal (low-activity soundscape segments) /
      external (BirdCLEF 2020/21 nocall + freefield1010 + warblrb +
      birdvox) / both
- [ ] 1.2 SNR range — proposal: widen from `[-6, +12]` to `[-10, +12]` dB
      given chainsaw-masking case from audio listening
- [ ] 1.3 Application probability — proposal: 75% (up from 50%)
- [ ] 1.4 Apply BG noise to labeled soundscape samples too — proposal:
      no (soundscapes already contain natural BG noise)

---

## 2. Implementation

- [ ] 2.1 Copy training pipeline from a recent v5j-compatible exp (027 or 025)
- [ ] 2.2 Copy v5j config (`019-sed-ablations/config/hgnet_b0_sed_v5j.yaml`)
      and add BG noise parameters
- [ ] 2.3 Acquire and stage the chosen BG noise source(s)
- [ ] 2.4 Implement BG noise dataset (load segment, length-match to focal)
- [ ] 2.5 Implement noise mixing augmentation in focal dataloader
      (additive, SNR-controlled, no label change)
- [ ] 2.6 Listen to a handful of mixed samples — confirm SNR sounds right
      and focal call is still audible at upper end of the range
- [ ] 2.7 Fast-dev-run sanity check (1–2 epochs, no checkpoints)

---

## 3. Train

- [ ] 3.1 Full training: single fold, 90/10, 20 epochs, identical to v5j
      apart from the BG noise change
- [ ] 3.2 ONNX export

---

## 4. Evaluate

- [ ] 4.1 Submit to LB
- [ ] 4.2 Record result in `results.md` and `MODEL-RUNS.md`
- [ ] 4.3 Re-run triage on the new model: does mixed_uncertainty drop
      for bimodal-regime species?
- [ ] 4.4 Spot-check a handful of high-loss samples on the new model
      (compare to v5j on the same samples, are predictions saner?)

---

## 5. Decision gate

After step 4, classify the outcome:

| Outcome | LB delta | Triage shift on bimodal regime | Action |
|---------|---|---|---|
| Confirm | > 0 | yes | Record win, update FINDINGS, plan 4-fold + propose next experiment |
| Strong confirm | ≥ +0.004 | yes | Same, with stronger priority for 4-fold + ensemble |
| Retry | ≤ 0 | yes | Lever is real but parameters are off — retry with different SNR/source/rate |
| Reject | ≤ 0 | no | Intervention didn't shift the failure mode — pivot, don't retry parameters |
| Pivot | any | n/a | Independent diagnostic shows bimodal-regime species don't drive test-set loss — audio finding's relevance is in question |

---

## What is NOT in this experiment

For reference; each of these would be its own future experiment if pursued.

- Secondary label supervision
- Soundscape oversampling
- SpecAugment (time/frequency masking)
- Sqrt-balanced long-tail sampling
- TopN=1 post-processing (inference-only)
- 2025 pretraining (Ali Ozan path)
- Seed sweep on v5j to characterize ±σ

See `proposal.md` § "Related work and potential follow-ups."
