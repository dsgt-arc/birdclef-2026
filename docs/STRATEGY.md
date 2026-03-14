# Strategy: BirdCLEF 2026

> **Last Updated**: 2026-03-14

## Two-Track Approach

The project pursues two parallel tracks: a **competitive baseline** using proven techniques, and a **novel research direction** that hasn't been attempted in BirdCLEF competitions.

```
Track 1 (SOTA Baseline)                    Track 2 (Novel Research)
━━━━━━━━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━━━━━━━━━━
EfficientNet-B1                            SSAMBA (Mamba)
  + SED head (AttBlockV2)                    + Prompted SED
  + iterative pseudo-labeling                + 30-60s long context
  + source separation augmentation           + episode-based training
  + ensemble 3-5 models                      + source separation as data engine

Target: 0.85+ (competitive)               Target: novel method + paper angle
Proven, reproducible                       Unexplored in competition
```

Both tracks share the same **data infrastructure**: source separation (custom MixIT at 32 kHz), teacher labels (exp 010), and the Kaggle submission pipeline.

## Competition Context: BirdCLEF 2026

> Data source: [exp 100 EDA results](../user/anthony/experiments/100-eda/results.md)

### This is an amphibian competition, not a bird competition

The Pantanal test soundscapes are **76.6% amphibian** by labeled window count, with birds at only 28.7%. Under cmAP (equal weight per species), the 35 amphibian + 28 insect + 8 mammal + 1 reptile species contribute **~31% of the score** despite having only **5.2h of focal data** (vs 338h for birds). The class imbalance is 499x (max 499, min 1 recording per species), and the rarest species are non-birds with class weights up to 17x. **Rare non-bird species drive the score.**

### Dataset at a glance

| | Focal | Soundscape |
|---|---|---|
| Files | 35,549 | 10,658 |
| Hours | 344.5h | 177.6h |
| Labeled | All (species tag) | **66 files (0.6%)** — 739 5s windows |
| Unlabeled | — | **10,592 files (176.5h)** |
| Sample rate | 32 kHz | 32 kHz |

### Critical data gaps

| Gap | Detail | Impact |
|---|---|---|
| **176.5h unlabeled soundscapes** | Only 0.6% of soundscapes are labeled. Pseudo-labeling this data is the single highest-value activity. | Pseudo-labeling is priority #1 |
| **25 insect song-type species** | `47158son01`–`47158son25` appear only in soundscape labels. Zero focal training data. Zero overlap with the 3 focal insect species. | Requires unsupervised clustering of `47158/` recordings or reference-conditioned detection |
| **3 zero-shot amphibian species** | 1491113, 25073, 517063 have no focal data — only in 66 labeled soundscapes | Prompted SED (Track 2) is naturally suited: use labeled soundscape segment as reference clip |
| **Severe geographic gap** | Only 2.4% of focal data from the Pantanal test region. Zero Insecta/Reptilia focal from Pantanal. Amphibia: 15.6% coverage. | Domain adaptation via soundscape pseudo-labeling and source separation augmentation is critical |
| **16 kHz limitation** | Biodenoising and Demucs dns48 operate at 16 kHz. Insect vocalizations (8-20+ kHz) are inaudible at this rate. | MixIT separator must train at **32 kHz** — this is non-negotiable for full taxa coverage |

### Taxonomic breakdown (focal training data)

| Class | Species | Recordings | Hours | % of data | Soundscape presence |
|---|---|---|---|---|---|
| Aves | 162 | 34,799 | 338.4h | 97.9% | 28 species (28.7% of windows) |
| Amphibia | 32 (+3 soundscape-only) | 451 | 3.4h | 1.3% | 17 species (**76.6%** of windows) |
| Insecta | 3 (+25 song-type splits) | 199 | 1.6h | 0.6% | 25 species (22.7% of windows) |
| Mammalia | 8 | 99 | 1.0h | 0.3% | 4 species (5.5% of windows) |
| Reptilia | 1 | 1 | 0.002h | 0.003% | 1 species (1.8% of windows) |

### Key implications for strategy

1. **Class-weighted training is essential** — use `class_weights.json` from exp 100. Rare non-bird species have 17x weight.
2. **Pseudo-labeling the 176.5h unlabeled soundscapes** is by far the highest-value single activity.
3. **Source separation at 32 kHz is mandatory** — 16 kHz tools (biodenoising) cannot serve insect species.
4. **Track 2 (prompted SED) has a natural advantage** for zero-shot species: provide a labeled soundscape clip as reference.
5. **The 47158 song-type problem** (25 insect species sharing one folder) needs a dedicated solution.
6. **Only 18% species overlap with 2025** — backbone features transfer but heads need full retraining.

**Source separation is unexplored in competition**: No top BirdCLEF solution (2021-2025) has used source separation for augmentation or inference. This is either an untapped advantage or a technique that top competitors tried and found not worth the complexity. We treat it as an opportunity but with appropriate caution.

## Track 1: Competitive Baseline

### Goal
Close the gap from 0.649 (exp 012) to 0.85+ using techniques proven in 2025 top solutions.

**Rough gap analysis** (estimated contributions, not additive):

| Technique | Estimated gain | Basis |
|---|---|---|
| Backbone unfreezing + SED head | +0.05-0.10 | Exp 015 showed frozen+SED fails; unfreezing is prerequisite |
| Iterative pseudo-labeling (3-4 rounds) | +0.10-0.15 | Dominant 2025 technique; 1st place attributed most gain here |
| Source separation augmentation | +0.02-0.05 | Speculative — Bird MixIT paper showed +0.7-6.3% cmAP |
| Temporal smoothing + power adjustment | +0.01-0.03 | Multiple 2025 top solutions |
| Ensemble (3-5 models) | +0.03-0.05 | Standard competition technique |

These estimates are rough and not strictly additive. The 0.85+ target requires most of these to deliver.

### Pipeline

```
BirdCLEF data ──→ source separation ──→ augmented training set
    │               (offline, sbatch)        │
    │                                        ▼
    │                              EfficientNet-B1 (unfreeze backbone)
    │                                + SED head (AttBlockV2)
    │                                + focal + augmented + pseudo-labeled data
    │                                        │
    │                                        ▼
    │                              iterative pseudo-labeling (3-4 rounds)
    │                                train → predict soundscapes → threshold
    │                                → add high-confidence clips → retrain
    │                                        │
    │                                        ▼
    └──────────────────────────────→ ensemble 3-5 models
                                     (different seeds, augmentation configs)
                                             │
                                             ▼
                                     ONNX export → Kaggle submission
```

### Key techniques (from 2025 Kaggle analysis)

| Technique | 2025 Evidence | Status |
|---|---|---|
| SED + AttBlockV2 | All top-5 used it, +0.03-0.05 gain | Not yet attempted (exp 015 used soft labels which confounded it) |
| Iterative pseudo-labeling | 1st place used 3-4 rounds, temp 0.55-0.6 | Not yet attempted |
| Backbone unfreezing | Required for SED to work (exp 015 showed frozen+SED fails) | Not yet attempted |
| Source separation augmentation | Not used by any 2025 solution — potential edge OR too expensive for marginal gain | Concept ready, separator training needed |
| Temporal smoothing [0.2,0.6,0.2] | Multiple top solutions | Not yet attempted |
| Power adjustment | Multiple top solutions | Not yet attempted |
| Ensemble (3-5 models) | All top solutions | Not yet attempted |

### What we learned NOT to do
- Don't replace hard labels with soft labels from weak supervision (exp 014: 0.238 vs 0.336)
- Don't add SED heads to frozen backbones (exp 015: no benefit)
- Don't use soft labels as direct training targets — use them for pseudo-labeling instead

### EDA-informed additions (exp 100)
- **Class-weighted loss is essential** — 499x imbalance, rare non-bird species have 17x weight. Use `class_weights.json`.
- **Pseudo-labeling the 176.5h unlabeled soundscapes** is the single highest-value activity — orders of magnitude more in-domain signal than the 1h of labeled soundscapes.
- **Amphibian-aware evaluation** — optimize for all taxa, not just birds. Pantanal soundscapes are 76.6% amphibian.
- **The 47158 song-type problem** — 25 insect submission targets share one species folder with no type-assignment metadata. Options: unsupervised clustering, uniform prediction, or reference-conditioned detection via Track 2.

### Remaining work
1. **Unfreeze EfficientNet-B1** + add SED head with hard labels + class-weighted focal loss → establish unfrozen baseline
2. **Pseudo-labeling** → the **highest priority** activity. Predict on 176.5h unlabeled soundscapes, threshold, retrain. 3-4 rounds. Optionally with separation for cleaner per-channel labels.
3. **Source separation** → train MixIT separator at 32 kHz (6-24 hours), pre-compute augmentations for domain adaptation
4. **Ensemble** → train 3-5 models with different seeds/augmentation, blend predictions
5. **Optional: inference-time separation + max-pooling** → run separator on test soundscapes, classify each channel + original, take max per species. This is the lowest-effort proven approach from the Bird MixIT paper (+0.7-6.3% cmAP), but multiplies inference cost by 5x. Only if Kaggle time budget allows.
6. **47158 song-type clustering** → unsupervised clustering of the 47158/ recordings to assign song-type labels, or fall back to uniform prediction across the 25 splits

## Track 2: Prompted SED with Long-Context SSAMBA

### Goal
Build a novel prompted SED system where a reference clip conditions a Mamba model to detect a specific species across an entire 30-60s soundscape. This is simultaneously a competitive submission and a research contribution.

### Why this is novel — and why the EDA data makes it more compelling

No BirdCLEF solution has done any of these:
1. **Prompted/conditional detection** — all solutions use fixed classification heads with one output per species. Prompted SED uses the reference clip to specify what to look for, enabling few-shot and zero-shot detection.
2. **30-60s full-soundscape context** — all solutions window into 5-10s clips and aggregate. A model that sees the whole soundscape can learn temporal patterns: species turn-taking, call repetition intervals, dawn chorus progression.
3. **Mamba for bioacoustic SED** — linear complexity makes long context feasible on CPU. Transformers can't process 6000 tokens at inference (O(n^2)); Mamba does it in O(n).

**The EDA makes Track 2 more compelling than we initially thought:**
- **3 zero-shot amphibian species** (1491113, 25073, 517063) have no focal data. A fixed-head classifier cannot learn them. Prompted SED can — use a labeled soundscape segment as the reference clip.
- **25 insect song-type species** (`47158sonXX`) have no direct focal data. Prompted SED can detect each type by providing a reference clip extracted from labeled soundscapes.
- **Amphibian chorus detection** benefits from long context. Amphibian choruses are continuous — a 30-60s window captures the chorus character far better than 5s windows. With amphibians at 76.6% of test windows, this matters.
- **Median 8 species per soundscape window** means heavy overlap. The prompted approach handles overlap naturally — each species gets its own forward pass with its own reference conditioning.

### Architecture

```
Input: [ref_call: ~500 tokens (5s)] [SEP] [soundscape: 3000-6000 tokens (30-60s)]

                              SSAMBA backbone (24 layers)
                              bidirectional Mamba, O(n+m)
                                        │
                              per-token SED: Linear(D, 1)
                              "is the prompted species here at this frame?"
                                        │
                              activations (B, 3000-6000, 1) → sigmoid

Inference: run K=206 species × 12 segments × ~13ms ≈ 32s per soundscape
           → feasible within Kaggle 120-minute CPU limit
```

### Training pipeline (3 stages)

```
Stage 1: SSL Pre-Training                Stage 2: Episode Training              Stage 3: Soundscape Adaptation
━━━━━━━━━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━━━━━━━━━━               ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All audio, no labels                     Focal recordings + teacher labels      Focal refs + soundscape targets

Masked token prediction                  Episode types:                         Episode:
→ backbone learns audio structure        • within-recording (self-supervised)     ref = focal clip (clean)
                                         • cross-recording (same species)         target = soundscape 30-60s
Exp 009 pretrain-full/warm               • negative (cross-species, silent)       labels = teacher pseudo-labels
already running                          • source-separated augmented episodes
                                                                                This IS the domain adaptation
                                         Episode mixup:                         — training directly on
                                         • target mixup (blend spectrograms)      focal→soundscape matching
                                         • reference mixup (same species)
                                         • cross-domain mixup (focal+soundscape)
```

### Why long context matters

Real soundscapes have temporal structure that 5s windows destroy:

| Signal | 5s window | 30-60s window |
|---|---|---|
| Repeated calls (every 8-12s) | Catches 0-1 calls | Catches 3-6 calls — cumulative evidence |
| Species turn-taking | Can't observe | Thrush→warbler→wren sequence is diagnostic |
| Dawn chorus progression | Invisible | Robin first, song thrush second — temporal order is a signal |
| Call-and-response | Missed | Duetting species detected as correlated pair |
| Insect/amphibian choruses | Ambiguous onset | Continuous chorus character is clear over 30s |

Mamba's recurrent state naturally accumulates evidence: token 3000 has more context than token 100. The reference clip conditions the initial state, and each matching call in the soundscape reinforces it.

### How source separation feeds Track 2

Source separation isn't just augmentation — it's the **data engine** for episode construction:

```
source-separation-training ──→ train PyTorch MixIT separator (6-24h)
         │
         ▼
Pre-compute separations on focal + soundscape recordings
         │
         ├──→ Cleaner reference clips (separated foreground)
         │     → better prompts for prompted SED
         │
         ├──→ Augmented training episodes
         │     → focal foreground + soundscape background at varied SNR
         │     → co-occurrence mixing with separated species
         │
         └──→ Better pseudo-labels on soundscapes
               → classify separated channels, max-pool per species
               → higher-confidence labels for Stage 3 adaptation
```

### Dependencies and current state

| Dependency | Status | Needed for |
|---|---|---|
| SSAMBA backbone (exp 004-005) | Complete | Architecture |
| SSL pre-training (exp 009) | Pretrain done, finetune pending (Kaggle 0.484 preliminary) | Stage 1 |
| Teacher labels (exp 010) | Complete (BirdNET + Perch on all data) | Stage 2-3 episode labels |
| Prompted SED phase 0 (exp 011) | Inconclusive — emerging signal | Feasibility validation |
| EfficientNet metric structure (exp 016) | Awaiting results | Informs prototypical approach |
| MixIT separator training | Concept ready, not started | Augmented episodes |

## Shared Infrastructure

Both tracks depend on common data and tooling:

| Component | Description | Status |
|---|---|---|
| **EDA artifacts** (exp 100) | Label mapping, class weights, train metadata, soundscape analysis | **Complete** ([results](../user/anthony/experiments/100-eda/results.md)) |
| **Teacher labels** (exp 010) | BirdNET + Perch predictions on all recordings | Complete (88 GB) |
| **Source separation** | PyTorch MixIT separator on BirdCLEF data (32 kHz) | Concept ready ([source-separation-training](concepts/source-separation-training.md)) |
| **Separation augmentation** | Foreground/background remix, co-occurrence mixing | Concept ready ([source-separation-augmentation](concepts/source-separation-augmentation.md)) |
| **Kaggle pipeline** | ONNX export + submission via CLI | Tested end-to-end (exp 012) |
| **TB metrics** | `scripts/tb_metrics.py` for experiment comparison | Working |

## Priority Sequencing

### Near-term (next 2 weeks)

1. **Track 1: Unfreeze EfficientNet + SED + class weights** — fork exp 012, unfreeze backbone, add AttBlockV2, class-weighted focal loss using exp 100 weights. This is the single highest-impact proven change.
2. **Track 1: Pseudo-labeling round 1** — predict on 176.5h unlabeled soundscapes with the unfrozen model. This is the **highest-value single activity** given the 0.6% label rate on soundscapes.
3. **Train MixIT separator** — run two paths in parallel: Path A (Demucs warm-start, 16 kHz, 3-8h) and Path B (ConvTasNet from scratch, 32 kHz, 6-24h). Path B is mandatory for insect species. Enables both tracks.

### Medium-term (weeks 3-4)

4. **Track 1: Iterate pseudo-labeling** — rounds 2-3, optionally with separation-enhanced per-channel labels on soundscapes
5. **Pre-compute source separations** — run trained separator on focal + soundscape data, store on scratch (~500-720 GB)
6. **Track 2: Episode training prototype** — within-recording episodes on focal data, 30-60s windows, SSL backbone from exp 009. Test with both bird and amphibian species.

### Longer-term (weeks 5+)

7. **Track 1: Ensemble** — train 3-5 models with different seeds/augmentation/class weighting, blend predictions
8. **Track 2: Soundscape adaptation** — Stage 3 with focal ref + soundscape target episodes. Prioritize amphibian and zero-shot species.
9. **Track 2: Long context experiments** — compare 10s vs 30s vs 60s windows. Amphibian choruses likely benefit most from long context.
10. **47158 song-type solution** — unsupervised clustering, reference-conditioned detection, or uniform prediction

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Track 2 doesn't converge | High | Track 1 provides competitive fallback. Evidence base is thin: exp 011 (prompted SED phase 0) was inconclusive, exp 008 (attention-pooled SED) was rejected. |
| Separation artifacts as training confound | Medium | Imperfect separation introduces spectral holes and fragments. If training data has these but test data doesn't, the model learns artifact-specific features. Mitigate by keeping original unaugmented data in training set. |
| Pseudo-labeling doesn't improve over baseline | Medium | Source separation augmentation is independent |
| MixIT separator quality insufficient | Low | Background-swap augmentation still works with biodenoising (proven tool, but 16 kHz only — loses insect species) |
| Kaggle CPU time limit too tight for prompted SED | Medium | Batch 206 species efficiently; fall back to Track 1 ensemble |
| SSAMBA backbone too weak (0.173 cmAP on HSN) | High | Prompted SED may overcome this by leveraging reference conditioning rather than raw classification quality. Exp 009 preliminary Kaggle score (0.484) suggests SSL pretraining helps substantially. |
| 47158 song-type species unsolved | Medium | 25 submission targets with no type-assignment metadata. Uniform prediction is a safe fallback (~4% of score). Clustering or prompted detection could recover some of this. |
| Zero-shot species (3 amphibians) | Medium | Track 1 (fixed heads) cannot learn these at all. Track 2 can use labeled soundscape segments as references. Even Track 1 can learn from pseudo-labeled soundscapes if teacher models detect them. |
| Non-bird taxa underrepresented in training | High | Only 5.2h focal data for 44 non-bird species (vs 338h for 162 birds). Class weighting helps but doesn't create training signal where none exists. Pseudo-labeling soundscapes is the main mitigation. |

## Research Contribution

**Caveat**: Track 2's evidence base is currently thin — exp 011 (prompted SED phase 0) was inconclusive and exp 008 (attention-pooled SED) was rejected. The ideas are well-motivated but unproven. If Track 2 succeeds, the contribution is:

1. **Prompted SED for bioacoustics** — reference-conditioned species detection, enabling few-shot and zero-shot scenarios
2. **Long-context sequence modeling** — 30-60s full-soundscape processing with Mamba, exploiting temporal co-occurrence patterns
3. **Source separation as training data engine** — PyTorch MixIT separator providing augmented episodes for prompted training
4. **CPU-viable architecture** — Mamba's O(n) complexity makes long-context SED feasible under Kaggle's inference constraints

## Quick Links

| Resource | Link |
|---|---|
| EDA results | [exp 100 results](../user/anthony/experiments/100-eda/results.md) |
| Current best submission | Exp 012: 0.649 Kaggle private (EfficientNet-B1 frozen) |
| Prompted SED concept | [concepts/prompted-sound-event-detection.md](concepts/prompted-sound-event-detection.md) |
| SSAMBA concept | [concepts/ssamba-audio-mamba.md](concepts/ssamba-audio-mamba.md) |
| Source separation training | [concepts/source-separation-training.md](concepts/source-separation-training.md) |
| Source separation augmentation | [concepts/source-separation-augmentation.md](concepts/source-separation-augmentation.md) |
| Soundscape domain shift | [concepts/soundscape-domain-shift.md](concepts/soundscape-domain-shift.md) |
| 2025 Kaggle landscape | [references/kaggle/birdclef-2025/README.md](references/kaggle/birdclef-2025/README.md) |
| All experiments | [STATUS.md](STATUS.md) |
