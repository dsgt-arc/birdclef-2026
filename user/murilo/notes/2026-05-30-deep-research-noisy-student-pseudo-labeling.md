# Deep Research — Multi-Iterative Noisy-Student Pseudo-Labeling (to break 0.929)

**Captured**: 2026-05-30 · **Source**: Claude deep research (run #1 of 2) · **Author**: Murilo Gustineli
**Companion**: [2026-05-30-deep-research-xeno-canto-pretraining.md](2026-05-30-deep-research-xeno-canto-pretraining.md)
**Discriminator review / our-evidence cross-check**: see session discussion 2026-05-30 (verdict: PL is the
under-explored lever — our 4 past PL failures used *sharpening* (T=0.6, P^1.5), the documented
overconfidence failure mode; the winning recipe *softens* (p^0.5). We have never run the winning direction.)

---

## Prompt context

This research run answered: *what single experiment most likely takes us from 0.929 → >0.94?*
Clarifications established up front: the competition train set already includes an XC-derived
`train_audio` folder; "use XC data" could mean expanding target-species clips, pulling non-target
neighbors for pretraining, or both; and whether to compare vs / treat as additive to pretraining.

---

## TL;DR (research output)

- **Do this one thing: run a multi-iterative Noisy-Student self-training loop on the unlabeled
  `train_soundscapes`, using power-scaled SOFT pseudo-labels and an AUC-surrogate loss** — the exact
  lever that took the BirdCLEF+ 2025 1st-place solution from 0.898 → 0.930 ("Power Scaling +
  pseudo-labeling II (4 rounds) — 0.898 → 0.930"). It directly attacks the focal→soundscape domain
  shift capping the score. The prior "own-teacher pseudo-labels" attempt almost certainly failed because
  it was single-round with hard/binarized labels; the winning version is 4–5 rounds, soft targets,
  power-scaled, with SED-refined frame localization.
- **Second priority (complementary, cheap): add a Perch v2 LP-FT model** (linear-probe-then-fine-tune
  the EfficientNet-B3 Perch 2.0 backbone) as an ensemble member — strongest in-domain prior, backbone
  diversity uncorrelated with the HGNetV2 SED stack.
- **What NOT to do:** more backbone/head/loss sweeps (~30 done, exhausted); naive cold end-to-end
  fine-tuning of Perch (full FT distorts pretrained features OOD vs linear probing).

## Key findings

1. **Metric = ranking/domain-adaptation problem, not architecture.** Scoring is macro-averaged ROC-AUC
   skipping classes with no true positives, over 234 species, per non-overlapping 5s window of Pantanal
   soundscapes, CPU-only 90-min runtime. Threshold-free → gains come from better *relative ordering on the
   soundscape domain* and rare-class ranking, not prettier probabilities or bigger nets.
2. **The plateau is a domain-shift ceiling.** Train clips (XC/iNat) clean/single-species/clip-weak;
   test soundscapes continuous/multi-species/noisy/strongly-labeled per 5s. 2025 2nd-place (Sydorskyy
   et al., CEUR-WS Vol-4038, paper_256): Pearson(train counts, soundscape counts) ≈ 0.10; >90% of train
   clips single-species. Every team past ~0.90 used **semi-supervised use of the soundscapes**, not
   architecture.
3. **Highest-ROI technique in the 0.90→0.93 band = iterative Noisy-Student PL.** 2025 winner (Nikita
   Babych, "Multi-Iterative Noisy Student Is All You Need"): baseline 0.872 → +PL round I + MixUp +
   StochasticDepth 0.898 → **+Power Scaling + PL round II (4 rounds) 0.930** → +separate insect/amphibian
   pipeline 0.933. Final ensemble: EfficientNet-L0/B4/B3, RegNetY-016/008, EfficientNet-B0 (amph/insect).
   2nd place (0.928 priv / 0.925 pub): "Public scores consistently exceeded 0.9 once pseudo-labeling was
   applied," via model distillation. Most reproducible single jump in BirdCLEF history.
4. **We tried the weak version.** The winning recipe differs from a naive own-teacher pass in 5 ways:
   (a) **multiple rounds** (4–5, teacher→student→new teacher); (b) **soft probability targets, not
   hard/binarized** (binarized → overconfidence, explicitly flagged as a failure mode in the 2nd-place
   paper); (c) **power-scaling between rounds** (p → p^γ, softening exponent ~0.5, then soft target);
   (d) **~50/50 labeled/pseudo-only mix per batch**; (e) **SED used to refine PLs to frame/5s resolution**.
5. **Perch v2 = strongest prior, probe/LP-FT not naive FT.** Perch 2.0 (van Merriënboer et al.,
   arXiv:2508.04665): EfficientNet-B3, 12M params, 14,795 classes, 1.5M+ recordings, 1536-d embedding;
   authors never fine-tune (linear/prototypical probing). Kumar et al. (ICLR 2022, arXiv:2202.10054):
   on 10 distribution-shift datasets FT is +2% ID but −7% OOD vs linear probing; LP-FT is +1% ID, +10%
   OOD vs full FT. → Perch v2 as backbone only via LP-FT, not cold end-to-end FT.

## The recommended experiment, precisely specified

**Name:** Multi-iterative Noisy-Student self-training on `train_soundscapes` with power-scaled soft PLs.

**Why it breaks this specific plateau:** the 0.929 stack has a good focal-domain model + Perch-embedding
probe but lacks a student that has *trained on the test-domain distribution*. PL'ing the soundscapes turns
the 0.929 ensemble into a teacher and bakes its soundscape knowledge into a student trained directly on
soundscape-distributed inputs — closing the domain gap end-to-end, not patching it with an embedding probe.

- **Step 0 — Validation first.** Build val from the **labeled `train_soundscapes`** (2026 expert
  annotations). Above ~0.90, focal-clip CV decorrelates from LB (2nd place: Pearson dropped to ≈ −0.13
  above 0.9). Grouped 5-fold by site/session to avoid leakage.
- **Step 1 — Teacher = existing 0.929 ensemble** (HGNetV2-B0 SED + Perch probe). No retraining.
- **Step 2 — Soft PLs** on every 5s window of all unlabeled `train_soundscapes`. Store **probabilities,
  not argmax**. OOF where possible (teacher fold unseen) to reduce confirmation bias.
- **Step 3 — Power-scale** p → p^γ, γ≈0.5 (softening), then use as soft target (optionally renormalize).
  Prevents the loop collapsing into overconfidence. 2nd-place config string `PseudoF2PT05MT01` ⇒
  power-transform exponent ≈0.5, confidence floor ≈0.1.
- **Step 4 — (optional, high value) SED refinement.** Use the SED head's frame-level outputs to localize
  energy within each 5s window → sharpen clip-level PLs toward the true event (weak→strong labels).
- **Step 5 — Train the student.**
  - Backbone: keep HGNetV2-B0 (best) AND a 2nd student (EfficientNetV2-S or BirdSet-pretrained) for diversity.
  - Data mix: ~50% labeled (comp labels + their PLs), ~50% unlabeled (PLs only).
  - Loss: BCE → **SoftAUCLoss** (winner: pairwise differences + log-loss, resistant to overfit, supports
    soft labels — directly optimizes ROC-AUC). Safe fallback: soft-target BCE (distillation).
  - Noise on student (essential): waveform MixUp (target = element-wise max), SpecAugment time/freq masking,
    background mixing (prior-year soundscapes + ESC-50), RandomFiltering (random EQ), StochasticDepth.
  - Optimizer: AdamW LR ~1e-3 (RAdam 1e-3 NFNet-style; 1e-4 AdamW EffNetV2), cosine→~1e-6, ~50 ep/fold,
    batch 64, AMP. Class sampling w_i=(Σc_j/c_i)^γ, γ=−0.5..−1.0; duplicate classes <100 up to threshold.
  - Label smoothing α=0.05 (counter weak-label noise).
- **Step 6 — Iterate.** Student→teacher, repeat **4 rounds** (rounds 2–4 are where 0.898→0.930 happened).
  Re-power-scale each round. Stop when labeled-soundscape val AUC plateaus.
- **Step 7 — Ensemble** student(s) with the 0.929 stack via **rank-averaging** (more robust than mean for
  AUC). Winner used "Quantile-Mix" (mean + rank averages).

**Expected outcome:** self-training on soundscapes ≈ +1.5 to +3 AUC pts in the 0.90+ band (3 independent
2025 solutions). Realistic: single best student ~0.92–0.93; ensembled 0.935–0.945.

## Complementary: Perch v2 LP-FT ensemble member

Take Perch 2.0 (Kaggle Models / HF `cgeorgiaw/Perch`, via `perch_hoplite`): (1) linear-probe a 234-class
head on frozen 1536-d embeddings of comp data + soundscape PLs until converged, then (2) unfreeze + FT
end-to-end at low LR (LP-FT). Do NOT cold-FT (Kumar et al.: −7% OOD). TF backbone + EfficientNet-B3 ≠
HGNetV2 → genuine diversity. Perch 2.0 needs TF 2.20rc0 + GPU for embedding (CPU variant pending) → plan
TFLite/OpenVINO export for the 90-min CPU budget. Torch-native alternatives: BirdSet
`ConvNeXT-Base-BirdSet-XCL` or `EfficientNet-B1-BirdSet-XCL` (HF `DBD-research-group`, ~520k XC, ~10k species).

## Pitfalls
- Binarized/hard PLs → overconfidence and no gain (likely why the first attempt failed). Keep soft + power-scaled.
- Trusting focal-CV above 0.90 — validate on labeled soundscapes only.
- Leakage in PLs — generate OOF; never let a teacher label its own training fold.
- Adding generic external data indiscriminately — 2025 2nd place: "additional data did not yield substantial
  improvements," some sources slightly hurt (speech artifacts); filter human voice (Silero VAD).
- CPU-runtime trap — student must export to ONNX/OpenVINO and run all 234-class inference <90 min CPU.
  Keep small (HGNetV2-B0/EffNetV2-S fine; Perch-B3 needs TFLite/OpenVINO).

## Recommendations (staging)
- **Stage 1 (Day 1):** Build labeled-soundscape val (grouped 5-fold by site). Confirm 0.929 ensemble AUC on it.
- **Stage 2 (Day 1–2):** Soft, power-scaled (γ≈0.5) PLs on all `train_soundscapes` with the 0.929 teacher.
  Train round-1 student (HGNetV2-B0, 50/50 mix, SoftAUC or soft-BCE, full noise aug). Check val vs teacher.
- **Stage 3 (Day 2–3):** Rounds 2–4 (student→teacher). In parallel: 2nd diverse student + Perch LP-FT member.
  Rank-average into final ensemble.
- **Decision thresholds:** round-1 ≥ teacher → loop works, continue to round 4. round-1 < teacher → PLs too
  hard/overconfident; lower γ (softer), raise aug, verify OOF. ensemble val gain < +0.005 after round 2 →
  stop iterating, pivot to ensemble weighting + Perch LP-FT diversity. single student ~0.92 on labeled-SS
  val → on track for >0.94 ensembled.

## Caveats
- 1st-place exact recipe partly secondhand (Kaggle writeup behind bot-detection; power-scaling formula/exponent,
  thresholds, round count from a 3rd-party slide summary [tekkix.com] + analogous 2nd-place config). Treat
  γ≈0.5 and 50/50 as strong starting points to tune, not gospel.
- Deltas domain-specific: 0.898→0.930 is 2025 Colombia; 2026 is Pantanal, 234 classes, newly expert-labeled
  soundscapes. Mechanism transfers; magnitude differs.
- **Expert-labeled soundscapes are a 2026-specific asset** — far more reliable validation than prior years,
  and usable as a small strongly-labeled fine-tune set (no prior winner had this). Exploit for both val and
  a final fine-tune pass.
- Perch v2 end-to-end FT has no published head-to-head beating probing for soundscapes; LP-FT is the
  theory-supported safe path but still an unverified bet for this dataset.
- "234 species" / "90-min CPU" from the rules snapshot ~late May 2026 — re-confirm against the live data page.
