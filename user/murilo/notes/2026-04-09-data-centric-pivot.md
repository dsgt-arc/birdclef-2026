# Data-Centric Pivot: Analysis & Action Plan

**Date**: 2026-04-09 (revised)
**Context**: Brainstormed with Google Gemini Deep Research based on our 23-experiment history,
Karpathy's "Recipe for Training Neural Networks", and MIT's "Hacker's Guide to Deep Learning."
Revised after Kaggle discussion thread update (Gaurav Rawat, Anil Ozturk, Tucker Arrants CV data).
**Goal**: Build a robust, generalizable BirdCLEF solution via data-centric approaches.
Close the gap to competition leaders (0.929 → 0.948 #1) by improving the *data*, not the recipe.

---

## The Core Thesis

After 23 experiments and 40+ model variants, we've established an overwhelming pattern:
**only input representation and inference-time changes improve LB.** 15+ training recipe
changes (loss, augmentation, head, regularization) have 0 successes. This suggests the
remaining gap is NOT in the model architecture or training recipe — it's in the **data**.

Two foundational references support this pivot:

### Andrej Karpathy — "A Recipe for Training Neural Networks"
> *"The first step to training a neural net is to not touch any neural net code at all and
> instead begin by thoroughly inspecting your data. This step is critical. I like to spend
> copious amount of time (measured in units of hours) scanning through thousands of
> examples... adding more data is pretty much the only guaranteed way to monotonically
> improve the performance..."*
— [karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/)

### MIT OpenCourseWare — "Hacker's Guide to Deep Learning"
> The lecture emphasizes: **"Become Friends with Your Data"** and **"Beware of Spurious
> Correlations."** We must ensure the model is learning biological acoustic signals rather
> than exploiting background noise, data leaks, or formatting artifacts.
— [MIT OCW Lecture (YouTube)](https://youtu.be/DC2Hw9DiLCg?si=7LRkAZ4im13NDMcj)

### Why This Fits Our Evidence

Our findings align perfectly with this thesis:
- **Exp 012**: Adding 739 labeled soundscape segments (data change) → +0.022–0.060 LB for all 3 models
- **Exp 019**: HGNetV2-B0 backbone (input representation) → +0.037 LB
- **Exp 019**: AmplitudeToDB + NNAudio (input representation) → +0.014 LB
- **Exp 023**: Perch+SED ensemble (inference combination) → +0.004 LB
- **Exp 021**: 14 training recipe changes → 0-for-14 on LB

The pattern is clear: change the data or the inference, not the training recipe.

---

## Gemini's Proposed 5 Steps

### Step 1: Fix the Validation Disconnect (Site-Level Soundscape CV)

**Proposal**: Replace our 90/10 focal split (anti-correlated with LB) with a GroupKFold
CV on the 66 labeled soundscapes, grouped by recording site.

**Our Assessment: HIGH VALUE — Do this first, but with eyes open about limitations.**

| Aspect | Detail |
|--------|--------|
| Why it matters | Current val/auroc is anti-correlated with LB (FINDINGS.md). We cannot evaluate changes offline. Every model ranking requires a Kaggle submission. |
| Site distribution | **Severely imbalanced**: Site 22 has 40/66 files (61%), 8 other sites share remaining 26 files. GroupKFold by site gives very few balanced groups. |
| Species coverage | Only **75/234 species** appear in soundscape labels. 159 species will have no validation signal. |
| Species with zero focal data | **28** species (not 26 as Gemini states) have zero focal recordings — only soundscape segments |
| Existing infrastructure | `MultiLabelStratifiedGroupKFold` already implemented in `experiments/021-sed-training/generate_folds.py` |

**Key risks**:
1. With only 9 sites and Site 22 dominating (40 files), the CV will be noisy
2. 75/234 species means we're validating on only 32% of the label space
3. This is a *diagnostic tool*, not a direct LB improvement — it won't close the 0.019 gap itself

**Recommended approach**:
- Build soundscape-only CV as a *secondary* metric alongside our existing val/auroc
- Use leave-one-site-out (LOSO) for sites with enough data, or group small sites together
- Track macro-AUROC on soundscape CV and correlate with LB scores from past submissions
- If it correlates, trust it for future model selection without burning LB submissions

---

### Step 2: Clean Focal Data with VAD (Silero)

**Proposal**: Run Silero VAD (`snakers4/silero-vad`) on 35,549 focal recordings to detect
and filter/mask human speech contamination.

**Our Assessment: LOW-MEDIUM VALUE — Worth doing as hygiene, low expected LB gain.**

| Aspect | Detail |
|--------|--------|
| Reference implementation | `docs/references/kaggle/birdclef-2025/bc25-separation-voice-from-data.md` — full pipeline documented |
| BirdCLEF 2025 usage | Nikita's 1st place solution used this; also referenced by 20th place |
| Expected impact | Our SED model is at 0.906 single / 0.917 4-fold. Speech contamination in a small fraction of 35K focal files is unlikely to be the bottleneck. |
| Effort | Low — Silero VAD is off-the-shelf, runs on CPU in minutes |

**Why I'm skeptical this moves the needle much**:
- The domain shift problem (focal → soundscape) is the fundamental bottleneck, not noise in focal data
- Perch succeeds precisely because it does NOT fine-tune on focal data (FINDINGS.md insight #5)
- If speech contamination were a major issue, we'd expect higher LB from cleaner data — but our
  biggest gains came from representation changes, not data cleaning
- The BirdCLEF 2026 dataset may have different contamination patterns than BirdCLEF 2025

**Recommended approach**:
- Run VAD scan as a batch job, save timestamps to a pickle/CSV (like the reference notebook)
- Use the output for data exploration first: how many files are affected? Which species?
- Try training v5j with speech-masked data as a single controlled experiment
- Don't expect more than +0.001-0.002 LB

---

### Step 3: Hard Negative Mining from Unlabeled Soundscapes

**Proposal**: Run our 0.917 SED model on 10,592 unlabeled soundscapes, identify segments
with high-confidence but ecologically impossible predictions, and add them as explicit
negative examples.

**Our Assessment: MEDIUM VALUE — Novel idea, but needs careful design.**

| Aspect | Detail |
|--------|--------|
| Unlabeled data | 10,592 files (127K 5s segments) — same domain as test set, completely unannotated |
| Ecological constraints | Would need species range/site/season data for "impossible" detection |
| Available priors | We have site×hour priors from 66 labeled soundscapes (9 sites, 13 hours) |
| Risk | False negatives: some "impossible" predictions might be real rare occurrences |

**Key considerations**:
1. We don't have comprehensive ecological range data for all 234 species in the Pantanal biome
2. Our site/hour priors cover only 9 sites — the unlabeled data may cover different sites
3. A simpler version: find segments where model predicts species at very high confidence (>0.9)
   across many segments — if a species appears in 80% of unlabeled files, it's likely false positives
4. This is related to but different from pseudo-labeling — it's adding *confident negatives*

**Recommended approach**:
- Start with inference on unlabeled data (needed for pseudo-labeling too)
- Analyze prediction distributions before deciding on filtering criteria
- Use frequency-based heuristics (species predicted too often = likely FP) rather than
  ecological rules we might get wrong

---

### Step 4: Multi-Iterative Noisy Student Pseudo-Labeling

**Proposal**: Use our 0.929 ensemble as teacher to pseudo-label unlabeled soundscapes,
with power-transform filtering and density-weighted sampling, iterating 3-4 times.

**Our Assessment: HIGHEST VALUE — This is the most promising direction by far.**

| Aspect | Detail |
|--------|--------|
| Previous attempt | Exp 015 failed (ProtoCLR -0.005, EAT -0.024 LB) |
| Why it failed | Teacher was weak (0.767 LB), clip-level architecture couldn't use frame-level labels |
| What's different now | Teacher is 0.929 (5× stronger), SED produces frame-level predictions, Perch contributes domain-general knowledge |
| Nikita's approach | 3-4 iterations, temperature 0.55-0.6, power transform, density-weighted sampling |
| Data scale | 127K segments of same-domain soundscape data — exactly what the test set looks like |

**Why this should work now (the key differences from exp 015)**:

1. **Teacher quality**: 0.929 LB ensemble vs 0.767 LB clip models. FINDINGS.md explicitly notes:
   "Untested: Perch Teacher (0.918 LB) — Much stronger teacher → higher quality pseudo-labels."
   We now have an even stronger teacher (0.929).

2. **Architecture match**: SED produces frame-level predictions that match the 5s evaluation windows.
   Exp 015 used clip-level predictions that couldn't resolve temporal detail.

3. **Domain match**: The unlabeled soundscapes are the *same domain* as the test set (Pantanal
   soundscapes). This is NOT a training recipe change — it's a DATA change. Our evidence
   consistently shows that data changes (exp 012: +0.022–0.060) help while recipe changes don't.

4. **The 28-species problem**: 28 species have zero focal data. The only way to get training
   signal for these species is from soundscape data. Pseudo-labels from unlabeled soundscapes
   are the ONLY viable path to cover these species.

**Nikita's specific techniques (from 1st place writeup)**:

| Technique | Purpose | Reference |
|-----------|---------|-----------|
| Power transform (`P^1.5`) | Suppress uncertain pseudo-labels, preserve confident peaks | birdclef2025-1st-place-inference.md |
| Temperature scaling (T=0.55-0.6) | Sharpen pseudo-label distribution | birdclef2025-1st-place-inference.md |
| Density-weighted sampling | Sample "cleaner" (less confused) soundscapes more often | Gemini brainstorm |
| Multi-iteration (3-4 rounds) | Student becomes teacher → generates better pseudo-labels → iterate | birdclef2025-1st-place-inference.md |

**Critical nuance**: Nikita used 7 diverse models (5 SSL + 2 supervised), not just one.
The iteration works because each student model is different from the teacher ensemble,
adding genuine diversity. If we iterate with the same v5j recipe, we might just converge
to the same failure mode as exp 015. **Backbone diversity may be required for iteration
to work.** (See "What Gemini Missed" below.)

**Recommended approach**:
1. Run ensemble inference on all 10,592 unlabeled soundscapes (saves both pseudo-labels and predictions)
2. Apply power transform (P^1.5) and temperature scaling
3. Train v5j on focal + labeled soundscape + filtered pseudo-labels
4. Submit to LB — this is the key test
5. If gain > 0, iterate with student-as-teacher
6. Use backbone diversity (e.g., EfficientNet-B3 NS, RegNetY-008) in later iterations

---

### Step 5: Fix Waveform MixUp (Absmax Normalization + Left-Padding)

**Proposal**: Rewrite waveform MixUp to normalize by absolute maximum before mixing
(Nikita's approach), and left-pad shorter signals.

**Our Assessment: LOW VALUE — 3 failed attempts, recipe coupling problem. Skeptical.**

| Aspect | Detail |
|--------|--------|
| Previous results | v3b (exp 016): 0.802 LB (−0.025 vs v1), v4j (exp 017): 0.853 LB (−0.006 vs v4b), v6b (exp 021): 0.887 LB (−0.019 vs v5j) |
| Times tested | 3x, always regressed on LB |
| Implementation gap | Current code does NOT normalize waveforms by absmax before mixing (verified in `sed_finetune.py:283-307`) |
| Left-padding | **Not applicable**: v5j trains on fixed 5s chunks — all waveforms in a batch are the same length (160,000 samples at 32kHz). No variable lengths to pad. |

**Why I'm skeptical**:
1. FINDINGS.md establishes that **every training recipe change fails on LB**, even when it
   improves offline val. This is the strongest pattern in 23 experiments.
2. The v5j recipe is a local optimum — MODEL-TRAINING.md: "every perturbation falls off the
   generalization cliff." Waveform MixUp is a perturbation.
3. Nikita's recipe differs from ours in many dimensions simultaneously (CE loss, AttHead,
   EfficientNet backbone, different mel config). Absmax normalization is ONE of ~20 differences.
4. The argument "the failure was in implementation details" is speculative — Gemini is
   hypothesizing a specific cause without evidence.

**What IS valid about Gemini's point**:
- Our current MixUp does `lam * x_a + (1-lam) * x_b` on raw waveforms without any normalization.
  If recording A is much louder than recording B, the mixture is dominated by A regardless of lambda.
  Absmax normalization (`x / max(|x|)`) before mixing ensures equal energy contribution.
- This is a genuine implementation quality issue worth fixing *if* we decide to test waveform MixUp again.

**Recommended approach**:
- Fix the absmax normalization in our MixUp code as a code quality improvement
- Do NOT prioritize testing waveform MixUp as a standalone experiment — too many failed attempts
- IF pseudo-labeling (Step 4) changes the data distribution enough that the v5j recipe is no
  longer the optimum, THEN revisit waveform MixUp as part of a new recipe exploration

---

## What Gemini Missed

### A. Backbone Diversity for Ensemble (HIGH VALUE)

Nikita's 1st place used **7 models across 5 different backbones** (EfficientNet B0/B3/B4,
RegNetY 008/016, ECA-NFNetL0). We use only HGNetV2-B0 for SED. Training v5j on a different
backbone (e.g., EfficientNet-B3 NS, which scored 0.893 with AttHead in exp 021 Phase C)
would add diversity to both:
- The SED ensemble (currently 4 identical HGNet folds)
- The teacher for pseudo-labeling iterations

Expected gain: +0.003-0.007 from backbone-diverse ensemble.

### B. Overlap Reconstruction with Delta Blending (MEDIUM VALUE)

Nikita's inference uses 20s windows with 5s step (75% overlap), then reconstructs via
`overlap_average_max_delta`: average overlapping regions → max per 5s → blend center/±2
frames (50/25/25). This is an **inference-only change** — exactly the category that works.

Already documented: `user/murilo/notes/2026-04-05-sed-gap-analysis.md` (GAP 3).

### C. Multi-Taxonomy Training (MEDIUM VALUE)

Nikita trained one model on an **expanded 700-class taxonomy** (birds + insects + amphibians
+ mammals), then mapped back to competition classes. This directly addresses our 28-species
problem — the model learns to recognize non-bird taxa from external data.

### D. OpenVINO Acceleration (LOW-MEDIUM VALUE)

Nikita used OpenVINO INT8/FP16 for backbone inference. Could enable TF Perch in ensemble
(which timed out at ~10s/file). If we could cut Perch inference to ~3s/file, the TF+SED
ensemble becomes viable, recovering the −0.005 ONNX penalty.

---

## Revised Philosophy: Robustness Over Score-Chasing

**Update (2026-04-09 PM)**: After reviewing the Kaggle discussion thread, we have critical
new data on CV-LB correlation that changes our approach:

- **Gaurav Rawat** (GroupK on site): CV values of 0.82/0.94/0.97/0.87/0.942 map to LB
  0.908/0.913/0.926/0.916/0.915. Not monotonic — no reliable ranking.
- **Anil Ozturk** (soundscape-val-only GKF on sites): OOF 0.608 → LB 0.879. Essentially useless.
- **Tucker Arrants** explains why: train and test sites **overlap**. GKF forces generalization
  to *unseen* sites, but the test set includes *seen* sites. GKF is "overly harsh."

**The key realization**: No one has solved CV-LB correlation for this competition. And the
public LB itself is only 34% of the test data — the private LB (66%) determines final standings.
Optimizing for a metric that tracks 34% of a hidden test set is chasing noise on top of noise.

**Instead, we focus on robustness and generalizability:**
- Does the model learn biological acoustic signals, not artifacts? → **data quality**
- Does the model see enough diverse examples to generalize? → **data quantity/diversity**
- Is the training pipeline sound? → **techniques, augmentation, regularization**
- Does the ensemble cover failure modes? → **model diversity**

A model built on these principles will perform well on *any* reasonable test distribution —
whether that's the 34% public LB or the 66% private LB.

---

## Revised Priority Ordering

Reordered for robustness, not score-chasing. The goal is to build a system where every
component makes the model more generalizable, regardless of which test subset we're scored on.

| Priority | Step | Why (robustness framing) | Effort |
|----------|------|--------------------------|--------|
| **1** | **Data exploration & diagnostics** | Karpathy: "spend copious time scanning data." Understand species distribution, per-class failures, data quality issues before changing anything. Implement hengck23's diagnostics (logit std, per-group AUC). | Low-Medium |
| **2** | **VAD cleaning** (Step 2) | Remove spurious correlations (human speech → bird class). Makes model learn the right features. Data quality is the foundation. | Low |
| **3** | **Pseudo-labeling v2** (Step 4) | The single biggest untapped data source: 127K same-domain soundscape segments. Addresses the 28-species problem (zero focal data). Karpathy: "adding more data is the only guaranteed way to improve." | High |
| **4** | **Backbone diversity** (Gemini missed) | Diverse backbones = robust ensemble. Also required for pseudo-labeling iteration (prevents convergence to single-model failure mode). Nikita used 7 models × 5 backbones. | Medium |
| **5** | **Hard negative mining** (Step 3) | Uses unlabeled data to find confident false positives. Improves training signal quality — the model learns what ISN'T a bird call. | Medium |
| **6** | **Soundscape CV** (Step 1, demoted) | Lightweight diagnostic — NOT a gate. Use Tucker's joint validation (within-site + cross-site). Accept imperfection. Use to catch gross regressions only. | Low |
| **7** | **Overlap reconstruction** (Gemini missed) | Inference improvement (Nikita's delta blending). Worth testing but lower priority than data work. | Low |
| **8** | **Waveform MixUp fix** (Step 5) | Fix absmax normalization as code quality. Don't prioritize standalone testing (3 failed attempts). May become relevant if pseudo-labeling shifts the training distribution enough. | Low |

**The through-line**: Priorities 1-5 are all about the DATA. Priority 6 is a diagnostic tool.
Priorities 7-8 are recipe/inference tweaks that can wait.

---

## Key Unknowns & Risks

1. **Will pseudo-labeling work this time?** Our one attempt (exp 015) failed badly. The
   situation is fundamentally different now (5× stronger teacher, SED architecture, frame-level
   labels), but there's no guarantee. We submit to LB to test, but also monitor hengck23's
   robustness metrics (logit std, per-group AUC) to understand *why* it helps or hurts.

2. **Is the v5j recipe still optimal with more data?** If we add 127K pseudo-labeled segments,
   the training distribution changes fundamentally. The "generalization cliff" around v5j may
   shift, potentially opening up recipe changes that currently fail. Monitor this — if pseudo-
   labels work, revisit waveform MixUp and other recipe changes that failed on focal-only data.

3. **How much does data quality matter vs quantity?** VAD cleaning removes some focal files.
   Pseudo-labeling adds 127K noisy segments. The net effect depends on the quality of the
   pseudo-labels. Power transform and temperature scaling are the quality controls — but we
   should also try a "conservative" variant (only high-confidence pseudo-labels) vs "aggressive"
   (all pseudo-labels with soft weighting).

4. **How many pseudo-labeling iterations are needed?** Nikita used 3-4 with 7 diverse models.
   We may need backbone diversity (#4) before iteration can add value — iterating with the
   same v5j recipe may just amplify the same errors.

5. **Private LB surprise risk**: Our 0.929 is based on 34% of test data. The private LB could
   be significantly different. A robust, generalizable approach insures against this — a model
   that works on diverse acoustic conditions will be stable across both LB splits.

---

## Experiment Proposal: Exp 024 — Data-Centric Pipeline

### Phase A: Data Exploration & Diagnostics
- **Species distribution analysis**: per-class sample counts (focal vs soundscape), identify
  the 28 species with zero focal data, find species with < 10 training examples
- **Per-class performance breakdown**: run our v5j model on labeled soundscapes, compute
  per-species AUC. Which species do we fail on? Are they the 28 soundscape-only species?
  Are they rare Aves? Insects?
- **hengck23's diagnostics**: implement logit std monitoring, per-group AUC splits
  (common Aves, rare Aves, Insecta, Amphibia, Reptilia, Mammalia)
- **Labeled soundscape quality check**: listen to a sample of labeled soundscapes. Are
  labels accurate? Are there systematic issues?
- Output: a data quality report that informs all subsequent phases

### Phase B: VAD Cleaning
- Run Silero VAD on 35,549 focal recordings
- Output: speech timestamps per file, contamination statistics by species
- Analysis: how many files affected? Which species most contaminated?
- Train v5j with speech-masked data as controlled experiment
- Also run VAD on labeled soundscapes (66 files) — are there labeling issues?

### Phase C: Teacher Inference on Unlabeled Data
- Run 0.929 ensemble (Perch ONNX MLP-only + SED v5j-4fold-pp2) on all 10,592 unlabeled
  soundscapes
- Save: per-species probabilities for every 5s segment (127K segments total)
- Analyze: prediction distributions, species frequency, high-confidence segments
- Identify potential hard negatives (Step 3 preparation)

### Phase D: Pseudo-Labeled Training Round 1
- Power transform (P^1.5) on pseudo-labels
- Temperature scaling (T=0.6 based on Nikita's range)
- Conservative variant: only segments where max species prob > 0.7
- Aggressive variant: all segments with soft weighting by confidence
- Train v5j on: focal + labeled soundscape + pseudo-labels
- Monitor hengck23's robustness metrics during training
- Submit both variants to LB
- **Decision gate**: if LB improves, proceed to backbone diversity + iteration

### Phase E: Backbone Diversity
- Train v5j recipe on EfficientNet-B3 NS (scored 0.893 in exp 021 Phase C with AttHead)
- Also try RegNetY-008 (one of Nikita's backbones)
- Compare: with and without pseudo-labels
- Add to ensemble alongside HGNet v5j-4fold
- Submit to LB

### Phase F: Pseudo-Labeling Iteration 2-3 (if Phase D succeeds)
- Backbone-diverse ensemble becomes new teacher
- Generate new pseudo-labels with improved, diverse teacher
- Train new students, submit after each round
- Monitor: are iterations still adding value? When to stop?

### Phase G: Lightweight Soundscape CV (diagnostic, non-blocking)
- Build alongside other phases, not as a gate
- Tucker's joint validation: within-site metric + cross-site metric
- Use to catch gross regressions, not for fine-grained model selection
- Backtest against known LB scores to calibrate trust level

---

## References

| Document | Location |
|----------|----------|
| Karpathy: Recipe for Training NNs | [karpathy.github.io/2019/04/25/recipe/](https://karpathy.github.io/2019/04/25/recipe/) |
| MIT OCW: Hacker's Guide to DL | [YouTube](https://youtu.be/DC2Hw9DiLCg?si=7LRkAZ4im13NDMcj) |
| Nikita 1st Place BirdCLEF 2025 | [Kaggle writeup](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n) |
| Nikita 1st Place Inference Code | `docs/references/kaggle/birdclef-2025/birdclef-2025-nikita-1st-place-sed.py` |
| Nikita 1st Place Summary | `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` |
| Silero VAD Reference | `docs/references/kaggle/birdclef-2025/bc25-separation-voice-from-data.md` |
| Our SED Gap Analysis | `user/murilo/notes/2026-04-05-sed-gap-analysis.md` |
| Our Pattern Analysis | `user/murilo/notes/2026-04-06-experiment-pattern-analysis.md` |
| FINDINGS.md | `user/murilo/FINDINGS.md` |
| MODEL-TRAINING.md | `user/murilo/MODEL-TRAINING.md` |
