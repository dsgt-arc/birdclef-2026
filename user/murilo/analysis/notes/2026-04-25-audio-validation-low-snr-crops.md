# 2026-04-25 — audio validation: low-SNR crops drive mixed_uncertainty

Resolves the biggest gap in [the three-regimes note](2026-04-24-three-regimes-of-mixed-uncertainty.md):
no audio had been listened to. This session, Murilo listened to the 15 roahaw
`mixed_uncertainty` clips and the 10 `correct_confident` clips, then sampled
across `coffal1`, `socfly1`, `epaori4`. The bimodal-regime hypothesis survives,
and the framing tightens from "silent crop" to **"low SNR within the 5s crop."**

## What we did

- Listened to all 15 WAVs in `figures/v5j/mixed_uncertainty_roahaw/`.
- Listened to all 10 WAVs in `figures/v5j/correct_confident_roahaw/`.
- Sampled across `mixed_uncertainty_coffal1/`, `mixed_uncertainty_socfly1/`,
  `mixed_uncertainty_epaori4/` for cross-species generalization.

## Per-rank observations (roahaw mixed_uncertainty)

| Rank | What's audible | Failure type |
|---|---|---|
| 01 | Quiet bird call + foreground white noise; call appears twice, mostly noise | Low-SNR crop |
| 02 | No bird call at all in the 5s window — pure noise | True silent crop |
| 03 | One faint distant call; otherwise white noise | Low-SNR crop |
| 04 | One call + prominent chainsaw noise across the entire clip | Masked by anthropogenic noise |
| 05 | Loud clip with strong white noise; calls present but disrupted by foreground noise | Low-SNR crop |

Spot-checked the rest — all match one of these three patterns. The model output
("I don't hear roahaw confidently") is the **honest answer** for these crops; BCE
penalizes it because the recording-level label is roahaw.

## Cross-species generalization (informal)

Listened to a handful from each of `coffal1`, `socfly1`, `epaori4` mixed_uncertainty.
Same pattern: ambient/white noise dominates the foreground, bird calls are weak,
distant, partial, or absent. The pattern is not roahaw-specific.

## Baseline contrast (correct_confident_roahaw)

All 10 clips are clean: bird call is the foreground, ambient noise is background.
rank10 is the one outlier — call is quieter than the others — but it's still
clearly the dominant signal. **The discriminating factor is foreground/background
ratio, not absolute call presence.**

## Reframing: it's SNR, not silence

The earlier "silent crop label noise" framing in the
[roahaw note](2026-04-24-roahaw-failure-mode-analysis.md) is mostly correct but
slightly misleading. Of the 5 representative ranks above, only rank02 is literally
silent. The other four contain calls — they're just at a low signal-to-noise ratio.

This matters for choice of intervention:

- **Pure silent-crop framing** would suggest energy-based VAD filtering (drop windows
  with no signal). But that would also drop rank01/03/05 — windows where a real
  call exists, just quietly. We'd be throwing away weak-signal training data that
  the model needs to learn from.
- **Low-SNR framing** suggests the opposite: train the model to be robust to low
  SNR, not avoid it. **Background-noise injection during training** (mix focal
  recordings with environmental/anthropogenic noise) directly targets this. Salman
  uses external BG noise data from prior comps for exactly this purpose.

## Confidence audit (updated from 04-24)

| Claim | Confidence | Why |
|---|---|---|
| roahaw mixed_uncertainty is dominated by low-SNR crops | **High** | All 15 audios listened |
| Pattern generalizes to coffal1/socfly1/epaori4 mixed_uncertainty | **Medium-High** | Sampled, not exhaustive |
| Pattern generalizes to all 87 "bimodal" species | Medium | Not directly tested; structural argument |
| BG noise injection will help v5j on LB | **Hypothesis only** | Plausibly motivated; needs experiment |
| The 8 abundant-but-model-blind species are the cleanest target | Low (unchanged) | Test-set relevance still unverified |

The roahaw audio listening was the single biggest verification step we'd skipped.
With it done, the bimodal-regime claim moves from medium to high confidence. The
other two regimes (uniformly-uncertain, model-blind) remain at their prior
confidence levels — the audio listening did not address them.

## Salman recipe audit (separate but related)

This session also surfaced that earlier framings of "Salman's recipe" conflated
documented details with assumptions. To keep this honest for future work:

**What Salman has actually disclosed (in discussion #683791):**
- SED architecture, EfficientNet-B0 → EfficientNetV2-B0 → B0 NS (current matches v2B0)
- 20s training windows (was 10s; "10s/15s scores almost the same / sometimes better")
- CE loss on clipwise + framewise max, sigmoid at inference
- Raw waveform MixUp (not spectrogram)
- 25 epochs (more epochs hurt — model "too confident")
- Train_audio + labeled soundscapes at 0.917→0.923; train_audio ONLY at 0.937
  (says soundscape training is currently a "bug" on his end)
- No KD, no model soup
- PP/TTA at inference (not specified)
- AttHead's framewise max (preference statement; not confirmed used in 0.937)
- External BG noise from prior comps (BirdCLEF 2020/21 + freefield1010 + warblrb +
  birdvox); says LB is "same without it" but uses for generalization

**What he has NOT disclosed:**
- The specific attention/modeling change that drove 0.922 → 0.937 — **the
  +0.015 jump is opaque**. He says "investigating attention values on highest/lowest
  loss examples… now it's much better" with no detail.
- Mel parameters (n_fft, n_mels, hop, fmax, normalization)
- MixUp alpha/probability
- Optimizer / LR schedule / weight decay
- BG noise mixing parameters
- Train/val split strategy

**Implication**: "replicate Salman's full recipe" is not a clean option.
Replicating the disclosed parts means reproducing his 0.922 baseline at best,
not the 0.937 jump. The disclosed parts are useful as a *direction* (waveform
mixup, longer windows, framewise-max-aware loss, BG noise from external data)
but not as a copy-paste solution.

## Decision: pivot to BG noise mixing experiment

Given the audio validation, the next experiment is **single-variable
background-noise injection on the v5j recipe**:

- Keep v5j architecture, head, loss, schedule, mel pipeline.
- Add additive mixing of environmental/anthropogenic noise to focal recordings
  during training, at varying SNR.
- One change vs. v5j. Single LB submission. Compares directly against the 0.906
  baseline.

This is a cleaner version of exp 028 phase C.1, run in isolation rather than
bundled with four other changes. Rationale: we now have direct audio evidence
motivating this specific change, so we should run it on its own to attribute
any LB movement (positive or negative) to the right cause.

Open scoping questions for the experiment proposal:
- BG noise source: extract low-activity segments from labeled soundscapes only,
  or pull external prior-comp data (BirdCLEF 2020/21, freefield1010, warblrb,
  birdvox) like Salman?
- SNR range: −6 to +12 dB (per exp 028 proposal) or wider/narrower?
- Application rate: 50% of focal samples (per exp 028) or different?
- Apply to soundscape samples too, or focal-only?

## Future work

- **Option B (Ali Ozan path)**: pretrain on BirdCLEF 2025 focal data (already in
  project storage), fine-tune on 2026. Different bet, orthogonal to BG noise.
  Defer until we see the BG noise result.
- **Validation queue from 04-24** (gt_prob histograms, threshold sensitivity,
  random-sample figures, cross-model on v5g): defer. The audio listening already
  validated the part of the regime story that mattered most for picking a lever.
  Re-prioritize only if BG noise mixing fails and we need to pick a different
  regime to target.

## Files

- `analysis/figures/v5j/mixed_uncertainty_roahaw/*.wav` — 15 audios listened
- `analysis/figures/v5j/correct_confident_roahaw/*.wav` — 10 audios listened
- `analysis/figures/v5j/mixed_uncertainty_{coffal1,socfly1,epaori4}/*.wav` — sampled
