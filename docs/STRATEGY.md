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

**Key changes from 2025**: BirdCLEF 2026 provides **soundscape recordings for training** (not just testing). This is a major opportunity — soundscapes can be used directly for pseudo-labeling, domain adaptation, and source separation training.

**Multi-taxa scope**: 206 species = 146 Aves + 34 Amphibia + 17 Insecta + 9 Mammalia. The insect species are particularly relevant for tooling decisions: biodenoising operates at 16 kHz (Nyquist: 8 kHz), which makes insect vocalizations (often 8-20+ kHz) inaudible. The custom MixIT separator must train at 32 kHz to cover all taxa.

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

### Remaining work
1. **Unfreeze EfficientNet-B1** + add SED head with hard labels → establish unfrozen baseline
2. **Source separation** → train MixIT separator (6-24 hours), pre-compute augmentations
3. **Pseudo-labeling** → 3-4 rounds on soundscape data, optionally with separation for cleaner per-channel labels
4. **Ensemble** → train 3-5 models with different seeds/augmentation, blend predictions
5. **Optional: inference-time separation + max-pooling** → run separator on test soundscapes, classify each channel + original, take max per species. This is the lowest-effort proven approach from the Bird MixIT paper (+0.7-6.3% cmAP), but multiplies inference cost by 5x. Only if Kaggle time budget allows.

## Track 2: Prompted SED with Long-Context SSAMBA

### Goal
Build a novel prompted SED system where a reference clip conditions a Mamba model to detect a specific species across an entire 30-60s soundscape. This is simultaneously a competitive submission and a research contribution.

### Why this is novel

No BirdCLEF solution has done any of these:
1. **Prompted/conditional detection** — all solutions use fixed classification heads with one output per species. Prompted SED uses the reference clip to specify what to look for, enabling few-shot and zero-shot detection.
2. **30-60s full-soundscape context** — all solutions window into 5-10s clips and aggregate. A model that sees the whole soundscape can learn temporal patterns: species turn-taking, call repetition intervals, dawn chorus progression.
3. **Mamba for bioacoustic SED** — linear complexity makes long context feasible on CPU. Transformers can't process 6000 tokens at inference (O(n^2)); Mamba does it in O(n).

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
| **Teacher labels** (exp 010) | BirdNET + Perch predictions on all recordings | Complete (88 GB) |
| **Source separation** | PyTorch MixIT separator on BirdCLEF data | Concept ready ([source-separation-training](concepts/source-separation-training.md)) |
| **Separation augmentation** | Foreground/background remix, co-occurrence mixing | Concept ready ([source-separation-augmentation](concepts/source-separation-augmentation.md)) |
| **Kaggle pipeline** | ONNX export + submission via CLI | Tested end-to-end (exp 012) |
| **TB metrics** | `scripts/tb_metrics.py` for experiment comparison | Working |

## Priority Sequencing

### Near-term (next 2 weeks)

1. **Train MixIT separator** — run two paths in parallel (per [source-separation-training](concepts/source-separation-training.md)): Path A (Demucs from biodenoising weights, 16 kHz, 3-8h) and Path B (ConvTasNet from scratch, 32 kHz, 6-24h). Compare to determine best foundation. Enables both tracks.
2. **Evaluate exp 016 step 0** — does frozen EfficientNet have metric structure? Informs Track 1 pseudo-labeling and Track 2 prototypical design.
3. **Track 1: Unfreeze EfficientNet + SED** — fork exp 012, unfreeze backbone, add AttBlockV2 with hard labels. This is the single highest-impact change for Track 1.

### Medium-term (weeks 3-4)

4. **Pre-compute source separations** — run trained separator on focal + soundscape data, store on scratch
5. **Track 1: Pseudo-labeling round 1** — predict on soundscapes with unfrozen model, add high-confidence clips
6. **Track 2: Episode training prototype** — within-recording episodes on focal data, 30s windows, using SSL backbone from exp 009

### Longer-term (weeks 5+)

7. **Track 1: Iterate pseudo-labeling** — rounds 2-3, add separation-enhanced pseudo-labels
8. **Track 1: Ensemble** — train 3-5 models, blend
9. **Track 2: Soundscape adaptation** — Stage 3 with focal→soundscape episodes
10. **Track 2: Long context experiments** — compare 10s vs 30s vs 60s windows

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Track 2 doesn't converge | High | Track 1 provides competitive fallback. Evidence base is thin: exp 011 (prompted SED phase 0) was inconclusive, exp 008 (attention-pooled SED) was rejected. |
| Separation artifacts as training confound | Medium | Imperfect separation introduces spectral holes and fragments. If training data has these but test data doesn't, the model learns artifact-specific features. Mitigate by keeping original unaugmented data in training set. |
| Pseudo-labeling doesn't improve over baseline | Medium | Source separation augmentation is independent |
| MixIT separator quality insufficient | Low | Background-swap augmentation still works with biodenoising (proven tool, but 16 kHz only — loses insect species) |
| Kaggle CPU time limit too tight for prompted SED | Medium | Batch 206 species efficiently; fall back to Track 1 ensemble |
| SSAMBA backbone too weak (0.173 cmAP on HSN) | High | Prompted SED may overcome this by leveraging reference conditioning rather than raw classification quality. Exp 009 preliminary Kaggle score (0.484) suggests SSL pretraining helps substantially. |

## Research Contribution

**Caveat**: Track 2's evidence base is currently thin — exp 011 (prompted SED phase 0) was inconclusive and exp 008 (attention-pooled SED) was rejected. The ideas are well-motivated but unproven. If Track 2 succeeds, the contribution is:

1. **Prompted SED for bioacoustics** — reference-conditioned species detection, enabling few-shot and zero-shot scenarios
2. **Long-context sequence modeling** — 30-60s full-soundscape processing with Mamba, exploiting temporal co-occurrence patterns
3. **Source separation as training data engine** — PyTorch MixIT separator providing augmented episodes for prompted training
4. **CPU-viable architecture** — Mamba's O(n) complexity makes long-context SED feasible under Kaggle's inference constraints

## Quick Links

| Resource | Link |
|---|---|
| Current best submission | Exp 012: 0.649 Kaggle private (EfficientNet-B1 frozen) |
| Prompted SED concept | [concepts/prompted-sound-event-detection.md](concepts/prompted-sound-event-detection.md) |
| SSAMBA concept | [concepts/ssamba-audio-mamba.md](concepts/ssamba-audio-mamba.md) |
| Source separation training | [concepts/source-separation-training.md](concepts/source-separation-training.md) |
| Source separation augmentation | [concepts/source-separation-augmentation.md](concepts/source-separation-augmentation.md) |
| Soundscape domain shift | [concepts/soundscape-domain-shift.md](concepts/soundscape-domain-shift.md) |
| 2025 Kaggle landscape | [references/kaggle/birdclef-2025/README.md](references/kaggle/birdclef-2025/README.md) |
| All experiments | [STATUS.md](STATUS.md) |
