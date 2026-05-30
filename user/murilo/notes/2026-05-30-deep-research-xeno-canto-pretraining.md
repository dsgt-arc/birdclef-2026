# Deep Research — Xeno-Canto Pretraining Strategy (to push 0.929 → >0.94)

**Captured**: 2026-05-30 · **Source**: Claude deep research (run #2 of 2) · **Author**: Murilo Gustineli
**Companion**: [2026-05-30-deep-research-noisy-student-pseudo-labeling.md](2026-05-30-deep-research-noisy-student-pseudo-labeling.md)
**Discriminator review / our-evidence cross-check**: see session discussion 2026-05-30. Key tension:
exp 034/036 already fine-tuned a broad-XC-pretrained backbone (Sydorskyi's donated EffNetV2-S, bit-identical
to source) → capped at **~0.88 single-fold** = exactly the 2nd-place "pretrain-only 0.881" number below.
Our KD-from-Perch path already exceeds it (0.906 single / 0.915 5-fold / 0.929 ensemble). So for us,
pretraining is a *known ~0.88 quantity* and a backbone-diversity/paper lever, NOT the plateau-breaker.

---

## Prompt context

Follow-up to research run #1. Question: *XC data for pretraining and/or pseudo-labels, then fine-tune on
2026 — is this a better overall contribution?* My answers to the clarifications:
1. Pull more XC to build a strong pretraining model to fine-tune for 2026 (+ maybe PL rounds), with
   pretraining as the priority. Leaning toward 234-target-only but wanted the evidence on narrow-vs-broad.
2. Treat as additive (pretrain → then PL), but compare if that's best.
3. Open to whatever the evidence points to as highest-impact.

---

## TL;DR (research output)

- **Do broad, supervised, multi-thousand-species classification pretraining on a large XC dump (target +
  wider Neotropical/global avifauna, several thousand species), then fine-tune on the 234 Pantanal species,
  then iterative pseudo-label the soundscapes.** This is the recipe both 1st (0.933 priv) and 2nd (0.928 priv)
  2025 teams used; broad fine-grained pretraining transfers better than narrow target-only, and is *additive*
  to PL, not redundant.
- **But PL is the bigger, faster lever** (≈+0.036 pub from PL vs +0.046/+0.018 pub/priv for transfer learning
  in the 2nd-place ablation). With 3–5 days: PL on the critical path, XC-pretraining in parallel on PACE;
  fold pretraining in once checkpoints land. Don't bet the timeline on pretraining alone.
- **Highest-leverage shortcut: warm-start from an existing broad checkpoint** (released
  ConvNeXT-Base-BirdSet-XCL, or a Perch 2.0 backbone) rather than from-scratch on a fresh huge download.
  You almost certainly cannot reach >0.94 with pretraining alone — needs the full stack (broad pretrain +
  fine-tune + multi-round PL + SED ensemble + postprocessing + TTA).

## Key findings

1. **Broad beats narrow for pretraining.** 2025 2nd place pretrained on **7,489 species / 819,032 recordings**
   (taxonomy of 16,607 from prior BirdCLEFs, pruned to ≥10 recordings, **2025 targets removed** to prevent
   leakage) — not target-only. BirdCLEF 2023 overview: top teams "mostly restricted pre-training to the 250
   species appearing in previous" comps — but that was a data-availability constraint; modern winners go
   broader. Consensus (Perch 2.0, "the bittern lesson"): fine-grained supervised species classification
   across as many species as possible is the most robust bioacoustics pretraining task.
2. **Supervised classification pretraining beats SSL/MAE here.** Perch 2.0 (van Merriënboer et al., 2025):
   "Self-supervised methods like BirdMAE, BirdAVES, and SimCLR-style models generally struggle to outperform
   strong supervised baselines in bioacoustics." For a 3–5 day sprint, supervised classification converges
   faster and performs better than MAE SSL.
3. **Pretraining and PL are additive.** 2nd-place ablation: in-domain transfer learning lifted EffNetV2-S
   0.835 → 0.881 pub (+0.046; +0.018 priv); PL then → 0.917 pub (+0.036; to 0.910 priv); postprocessing
   +1–1.5%. 1st place: pretrain on entire XC archive (≈0.84→0.87) then PL 2 rounds (0.87→0.91).
4. **Order: pretrain → fine-tune (reinit head) → pseudo-label** (every winning solution).

## The narrow-vs-broad decision (CRITICAL)

Recommend **broad Neotropical-plus-global, NOT target-only.**
- 2nd place: 7,489 species / 819,032 recordings (50 ep, no class balancing, targets removed) "consistently
  improved local CV and Public/Private test."
- Perch 2.0 (arXiv:2508.04665): supervised on >1.5M recordings; fine-grained classification across many
  species uniquely robust as pretraining; increasing classification difficulty improves embedding quality.
  0.908 ROC-AUC / 0.431 cmAP / 0.665 top-1 on BirdSet.
- Hong et al. (TMLR 2024) "Why fine-grained labels in pretraining benefit generalization" — theory behind it.
- **Diminishing returns:** 2nd place found extending pretraining beyond their broad pool (CSA, later XC,
  public Kaggle sets) "did not yield further improvements," and extra *fine-tuning* data slightly *hurt*.
  Go broad but a few thousand species / few hundred k recordings is the sweet spot.
- Practical scope: Pantanal hosts ~656 bird species (WWF; 463 recorded per Tubelis & Tomas); comp targets
  234. Pretrain wide (all S. American/Neotropical XC + full prior-BirdCLEF taxonomy), ensure the 234 are
  well represented, then re-init head and fine-tune only on the 234.

## Method: supervised classification, warm-started

- Supervised multi-label classification pretraining (all pretraining species as outputs), NOT MAE
  (SSL underperforms supervised per Perch 2.0, Ghani et al. 2023 "Global birdsong embeddings…", AVES/BirdAVES).
- **Warm-start from an existing broad checkpoint to save days:** BirdSet XCL = 528,434 weakly-annotated clips,
  **9,734 species, ~7,200 hours** (Rauch et al. 2024), released as **ConvNeXT-Base-BirdSet-XCL** (HF
  DBD-research-group; 97.5M-param ConvNeXt-B, 10 ep on one A100-80GB). Perch 2.0 also available.
  Continue-pretrain/fine-tune from these. Bird-MAE (Rauch et al. 2025): "bridging the domain gap requires
  full-pipeline adaptation" → adapt the whole pipeline (aug, fine-tuning, head), not just swap data.

## XC data collection practicalities

- XC API v3 **requires an API key** (registered members, verified email); wrappers enforce ~1 req/s for
  metadata. Audio downloads parallelizable.
- Filter by quality (`q:A`, `q:B`) and geography (`cnt:Brazil`, `area:america`) for Neotropical efficiency.
- XC holds >1,000,000 recordings / >12,900 species; Brazil/Neotropics very well represented (XC founded
  2005 on Neotropical birds).
- Preprocess: resample 32 kHz; fast on-disk format (HDF5, as 2nd place); secondary/background species as
  extra multi-label targets; Silero VAD to strip speech artifacts.

## Historical evidence from BirdCLEF winners

- **2025 1st (priv 0.933):** baseline 0.872 → pretrain on entire XC archive (≈0.84→0.87) → PL 2 rounds
  (0.87→0.91) → power-scaling + more PL (→0.930) → separate insect/amphibian pipeline (→0.933). Ensemble
  EfficientNet-l0/B4/B3 + RegNetY-016/008. Supplement: +5,489 bird, +17,197 insect/amphibian entries.
- **2025 2nd (priv 0.928 / pub 0.925):** 7,489-species broad XC pretrain (+0.046 pub on EffNetV2-S) + 2–3
  round PL (best single +0.036 pub to 0.917) + TopN postprocessing (+1–1.5%). Backbones EfficientNetV2-S and
  eca_nfnet_l0; loss BCE+Focal, label smoothing 0.05.
- **2024 (Henkel/Puget/Viel, 3rd):** capped XC at 500 rec/species (most recent), + prior-year comp data, heavy
  PL + model distillation on unlabeled soundscapes.
- **2023 4th:** base 0.806 pub → +KD 0.821 → +XC data 0.829 → +pretraining 0.831 → +4-model ensemble 0.840
  (pretraining added only ≈+0.002 *on top of* the XC-data addition — data expansion and pretraining-on-it overlap).
- **2023 1st (Sydorskyy):** ConvNeXt + eca_nfnet_l0; pretrain on prior comps + XC, fine-tune on 2023 species.

## Implementation recipe (recommended)

1. **Data:** XC quality A/B for all Neotropical/S. American species + full prior-BirdCLEF taxonomy + 234
   targets. Cap ~500 rec/species; drop species <10. Keep secondary labels. Strip human voice (Silero VAD).
   Resample 32 kHz, store HDF5.
2. **Backbone:** EfficientNetV2-S or eca_nfnet_l0 (CPU-friendly proven winners); HGNetV2-B0 also a valid FT
   target. Mel: n_mels=128, n_fft=2048, hop=512, f_min=20, top_db=80.
3. **Pretraining:** 50 ep, **no class balancing**, AdamW LR 1e-4 (EffNetV2) or RAdam 1e-3 (nfnet), cosine→1e-6,
   no warmup, batch 64. Loss = BCE+Focal, label smoothing 0.05. **Discard pretraining head; keep backbone.**
4. **Fine-tune on 234:** same optimizer/schedule, + class balancing (sqrt/inverse-freq, γ −0.5..−1), waveform
   MixUp (element-wise max of targets), background mixing (ESC-50 + prior soundscapes), SpecAugment, random-EQ.
5. **Pseudo-label:** OOF-predict unlabeled/expert-labeled train_soundscapes, threshold (keep chunk if max
   prob >0.5, zero classes <0.1), soft targets, retrain on ~50/50 mix; 2–3 rounds (plateau by round 3).
6. **Postprocessing + TTA:** TopN postprocessing (+1–1.5%), 2.5s left/right shift TTA.

## Recommendations (staging)

- **Stage 0 (now):** Kick off XC metadata crawl immediately (~1 req/s is the binding constraint). In parallel
  pull ConvNeXT-Base-BirdSet-XCL and a Perch 2.0 backbone as warm-start fallbacks.
- **Stage 1 (highest priority, days 1–2):** Iterative PL on train_soundscapes with *current* best models
  (Perch-probe + HGNetV2 SED) — largest/fastest lift (≈+0.03–0.04), uses data we have. Threshold-and-soft, 2 rounds.
- **Stage 2 (parallel, days 1–3):** Broad XC supervised classification pretraining of a fresh EffNetV2-S /
  eca_nfnet_l0 — or continue-pretrain/fine-tune from ConvNeXT-Base-BirdSet-XCL to save 1–2 days. Then
  fine-tune on 234 with full domain-shift augmentation.
- **Stage 3 (days 3–5):** Fold the XC-pretrained, fine-tuned, PL'd model into the Perch-probe + HGNetV2 SED
  ensemble for diversity. + TopN postprocessing + shift-TTA. Quantile/rank-mix CNN and SED predictions.
- **Decision thresholds:** from-scratch XC pretrain not beating ImageNet-init baseline on CV by day 3 →
  abandon from-scratch, continue-pretrain from BirdSet/Perch instead, or drop pretraining and double down on
  PL + ensemble diversity. PL round 2 no pub-LB gain → stop (round 3 plateaus/declines). Extra data degrades
  CV → keep pretraining but DON'T add extra fine-tuning data.

## Caveats
- **Already near the ceiling.** 0.929 sits between 2025 2nd (0.928) and 1st (0.933) private. Marginal gains
  shrink above 0.90; local CV decorrelates from LB above 0.90 (2nd place: near-zero, slightly negative
  Pearson/Spearman CV–pub above 0.9). Validate on soundscapes, not focal CV.
- **Domain shift is the core problem, not species coverage.** XC focal (single/clean, >90% single-species) vs
  Pantanal passive soundscapes (polyphonic/noisy; 2025 ~54% no-call). Pretraining helps init; augmentation
  (MixUp, background mixing) + PL close the gap.
- **Label noise:** XC weakly labeled (whole-recording), background species, human speech. Filter; exploit
  secondary labels as multi-label targets.
- **Time risk:** huge download (~1 req/s metadata) + from-scratch 50-ep pretraining can eat 2–3 days → warm-start.
- **Diminishing returns on breadth:** beyond a few thousand species gave no measured 2025 gain. Don't over-collect.
- **Source quality:** 1st-place 2025 numbers from a 3rd-party recap (tekkix) — indicative; 2nd-place from the
  primary CEUR working note — reliable.
