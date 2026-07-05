# Concept: Strong Baseline via Mel Overhaul + Training Recipe Alignment

**Status**: developing
**Created**: 2026-03-24
**Updated**: 2026-03-24

## Idea

Our SED v1 model (0.823 LB) has the right architecture (GeM + AttHead, matching 3/4 BirdCLEF winners) but is likely limited by its **mel spectrogram resolution** — the single largest untested gap. We use n_fft=1024 with 128 mel bins, while the 2025 1st place (0.937) uses n_fft=4096 with 224 mel bins. This is a 4× difference in frequency resolution and 1.75× in spectral detail. With 234 species including spectrally similar bird calls, amphibian harmonics, and high-frequency insect stridulation, our model literally cannot distinguish signals that a higher-resolution spectrogram would separate.

The proposal is a phased experiment that first establishes a **strong labeled-data-only baseline** by systematically closing the gap between our training recipe and proven winner recipes — starting with the mel overhaul (highest expected impact, never tested), then layering additional improvements one at a time. Only after establishing the strongest possible single-model baseline do we move to pseudo-labeling the unlabeled data, which is where the 2025 winner gained the final +0.06 (0.872 → 0.933).

## Motivation

### The mel resolution hypothesis

Two mel paradigms exist in BirdCLEF competitions:

| Paradigm | n_fft | n_mels | hop | Typical LB | Used by |
|----------|-------|--------|-----|------------|---------|
| **SED-native** | 1024 | 128 | 256-512 | 0.857-0.878 | Public SED notebooks |
| **High-res + SED** | 4096 | 224+ | 1252 | **0.937** | 2025 1st place |

Our v1 uses the SED-native paradigm. The 2025 1st place used high-res mel WITH SED head — a combination that most public notebooks don't use. This untested combination may account for a significant portion of our gap.

**Why frequency resolution matters for BirdCLEF 2026 specifically:**
- 234 species with many spectrally close pairs (e.g., sympatric warblers)
- Amphibian harmonics require fine frequency discrimination
- Insect stridulation occupies narrow frequency bands (8-20+ kHz)
- At n_fft=1024 (32kHz SR), frequency resolution is ~31 Hz per bin; at n_fft=4096 it's ~7.8 Hz per bin

### The recipe gap is cumulative

The v3 ablation proved that no single small change (weights, loss, mixup domain) improved v1. But v1 is missing **multiple** techniques that winners use simultaneously:

| Missing from v1 | Expected impact | Source |
|----------------|-----------------|--------|
| High-res mel (n_fft=4096, 224 mels) | Unknown (never tested) | 2025 1st, 20th place |
| StochasticDepth (drop_path=0.15) | Regularization | 2025 1st place |
| Larger backbone (B3/B4) | More capacity | 2025 1st place |
| Model soup (checkpoint averaging) | +0.005-0.01 | Competition consensus |
| Full-signal mel (compute once for 60s) | Phase coherence | 2025 1st place inference |
| ASL loss | +0.023 vs BCE | Competitor ablation |

The approach is to test the highest-impact change first (mel overhaul), then layer additional changes only if the mel change shows promise.

### Why labeled-data-only first

The 2025 1st place scored **0.872 before any pseudo-labeling** — their baseline with only labeled data was already competitive. Their pseudo-labeling then added +0.06. Our baseline (0.823) has room to grow before we touch unlabeled data. A stronger baseline produces better pseudo-labels, which produce a better student, compounding the advantage.

## Background

### 2025 1st Place Exact Recipe (from vendored inference notebook)

```
Backbone: EfficientNet-B0/B3/B4, RegNetY-008/016, ECA-NFNet-L0
Head: AttHead (GeM freq → dense 512 → dropout 0.5 → Conv1d framewise)
Mel: n_fft=4096, hop=1252, n_mels=224, fmin=0, fmax=16000, 32kHz
Window: 20s training, 5s inference step (overlap-average reconstruction)
Loss: CrossEntropyLoss (dual framewise + clipwise)
Augmentation: MixUp (0.5-1.0), StochasticDepth (drop_path=0.15)
Normalization: AmplitudeToDB (top_db=80) → standardize → min-max [0,1]
Inference: OpenVINO, delta-shifted TTA (±2 frames), Gaussian smoothing
Ensemble: 7 models, weights [0.133-0.166]
```

### Our v1 Recipe (exp 016, 0.823 LB)

```
Backbone: EfficientNet-B0 (ImageNet-RA pretrained)
Head: AttHead (GeM freq → dense 512 → dropout 0.5 → Conv1d framewise)
Mel: n_fft=1024, hop=320, n_mels=128, fmin=0, fmax=Nyquist, 32kHz
Window: 10s training, 5s inference (v1-10s: 10s inference, awaiting LB)
Loss: Soft CE (normalized multi-hot, dual framewise + clipwise)
Augmentation: Spectrogram MixUp (prob=0.5, alpha=0.5)
Normalization: log(mel+1e-6), AudioSet mean/std
Inference: ONNX CPU, TTA ±2.5s shifts, Gaussian smoothing, TopN
```

### Key Differences (Prioritized)

| # | Parameter | Ours | 2025 1st | Impact est. | Tested? |
|---|-----------|------|----------|-------------|---------|
| 1 | **n_fft** | 1024 | **4096** | High (4× freq res) | **No** |
| 2 | **n_mels** | 128 | **224** | High (1.75× detail) | **No** |
| 3 | **drop_path** | 0.0 | **0.15** | Medium (regularization) | **No** |
| 4 | **backbone** | B0 only | B0/B3/B4/RegNet | Medium (capacity) | **No** |
| 5 | **mel norm** | AudioSet log | **AmplitudeToDb+minmax** | Low-Medium | **No** |
| 6 | **hop_length** | 320 (10ms) | 1252 (39ms) | Tradeoff | **No** |
| 7 | **model soup** | None | Checkpoint avg | Free | **No** |
| 8 | **ASL loss** | Soft CE | CE / ASL | +0.01-0.02 | **No** |

### References

- [2025 1st place inference](../references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md)
- [SED improvement plan](../references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- [Winning solutions analysis](../references/kaggle/winning-solutions-analysis.md)
- [Exp 016 summary](../../user/murilo/experiments/016-sed-architecture/summary.md)
- ASL paper: Ridnik et al., "Asymmetric Loss For Multi-Label Classification" (2021, arXiv:2009.14119)

## Open Questions

1. **Does n_fft=4096 help our SED architecture?** Most public SED notebooks use 1024. Only the 1st place combined high-res mel with SED. The interaction is unknown.

2. **Should we keep our 10ms hop or switch to 39ms like the 1st place?** Our hop gives 1000 frames for 10s (more temporal resolution, more compute). The 1st place hop gives 256 frames for 20s (less temporal resolution, less compute, higher frequency resolution per frame). These are different tradeoffs.

3. **Does the mel normalization method matter?** We use log(mel+1e-6) with AudioSet mean/std. The 1st place uses AmplitudeToDB(top_db=80) → standardize → min-max [0,1]. These produce very different value distributions.

4. **Will the larger input (224 mels) cause OOM at batch_size=128?** 224×1000 is 1.75× more data than 128×1000. May need to reduce batch size.

5. **Is EfficientNet-B0 (3.6M params) sufficient for 224-mel input?** Higher-res input may need more model capacity to exploit. The 1st place paired 224 mels with B3/B4 (12-19M params).

6. **How much does StochasticDepth help?** It's standard regularization for deeper nets but untested with our small B0.

7. **Should we switch to AttBlockV2?** Competition standard uses tanh+softmax attention with sigmoid classification (bounded activations). Our AttHead uses raw logits. The dual-activation pattern is used by 5+ top solutions.

8. **How much do metadata priors help our SED model?** The Perch-based notebooks get huge gains from site/hour priors (especially for amphibians/insects which are 76.6% of labeled soundscape windows). This is inference-only, zero retraining.

## Testable Hypotheses

### Phase 1: Labeled-Data Baseline (Exp 017)

**H1 (Mel resolution)**: Changing from (n_fft=1024, n_mels=128) to (n_fft=4096, n_mels=224) while keeping all other v1 parameters identical will improve LB by ≥0.02. This is the **primary hypothesis** — if this fails, the remaining gap is likely not in input resolution.

**H2 (Mel normalization)**: Switching from log+AudioSet normalization to AmplitudeToDB(top_db=80)+standardize+min-max will improve LB by ≥0.01, as it better preserves dynamic range and matches the winning pipeline.

**H3 (Regularization)**: Adding StochasticDepth (drop_path=0.15) will improve LB by ≥0.005, reducing overfitting on the relatively small labeled dataset.

**H3b (AttBlockV2 head)**: Switching to dual-activation head (tanh+softmax attention, sigmoid classification) and using framewise-max at inference (instead of 0.75/0.25 blend) will improve LB by ≥0.005. This is the competition-standard SED head design.

**H4 (Model soup)**: Averaging the top-3 checkpoints by val/auroc will improve LB by ≥0.005 over single best checkpoint, at zero training cost.

**H5 (Backbone scaling)**: EfficientNet-B3 (12M params) with the high-res mel config will outperform B0 (3.6M) by ≥0.01 LB, as the larger model has capacity to exploit the richer input.

### Phase 2: Unlabeled Data (Exp 018, future)

**H6 (Pseudo-labeling)**: Using the Phase 1 best model as teacher to pseudo-label 10,592 unlabeled soundscapes, then retraining with PowerTransform (p^γ, γ=2.0) soft labels, will improve LB by ≥0.03 over the labeled-only baseline.

**H7 (Iterative refinement)**: 3 rounds of pseudo-labeling with temperature T=0.55 will outperform 1 round by ≥0.015 LB.

## Connections

- **Based on**: [Exp 016 SED Architecture](../../user/murilo/experiments/016-sed-architecture/summary.md), [2025 1st place analysis](../references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md), [SED improvement plan](../references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- **Related to**: [Soundscape domain shift](soundscape-domain-shift.md), [Working note framing](working-note.md)
- **Could lead to**: Exp 017 (strong baseline), Exp 018 (pseudo-labeling), multi-model ensemble

## Notes

### Proposed Experiment Structure (Exp 017)

**Step 0: Free wins (no retraining)**
- Model soup: average v1 checkpoints epochs 20-24
- v1-10s result (submitted, awaiting LB)
- Metadata priors: site/hour tables from 66 labeled soundscapes, multiply into predictions (H8)

**Step 1: Mel overhaul (single variable from v1)**
- v4a: n_fft=4096, n_mels=224, keep hop=320, keep everything else from v1
- If OOM: try n_mels=160 or increase hop to 640
- This tests H1 in isolation

**Step 2: Normalization + head (single variables from best of step 1)**
- v4b: Add AmplitudeToDB + min-max normalization — tests H2
- v4c: Switch to AttBlockV2 head (tanh+sigmoid dual activation, framewise-max inference) — tests H3b

**Step 3: Regularization + capacity (cumulative)**
- v4d: Add StochasticDepth (drop_path=0.15) — tests H3
- v4e: Switch to EfficientNet-B3 — tests H5
- v4f: Try ASL loss (γ⁻=4, γ⁺=0, clip=0.05)

**Step 4: Polish**
- Model soup on best config (H4)
- Full-signal mel at inference (compute once for 60s, slice)
- Ensemble 2-3 best variants if inference budget allows

### Key Decision Points

- If Step 1 (mel overhaul) shows ≥+0.02 LB: continue the phase plan
- If Step 1 shows <+0.01 LB: the gap is NOT in mel resolution. Pivot to:
  - Perch-based approach (ensemble our SED with Perch predictions)
  - Metadata priors (site/hour from filenames)
  - Direct pseudo-labeling with current model

### Inference Time Budget

| Config | Mel frames (10s) | ONNX time/call | 200 files × 12 seg × 3 TTA | Budget |
|--------|-------------------|----------------|------------------------------|--------|
| 128 mels, 1000 frames | 128K | ~315ms | ~38 min | OK |
| 224 mels, 1000 frames | 224K | ~550ms (est) | ~66 min | Tight |
| 224 mels, 500 frames | 112K | ~275ms (est) | ~33 min | OK |

If 224×1000 is too slow for Kaggle, we can either:
- Increase hop at inference only (train 10ms, infer 20ms → 500 frames)
- Use OpenVINO INT8 quantization for ~2× speedup

### Hop length tradeoff

Two options for the mel overhaul:

**Option A: Keep our hop (320 = 10ms)**
- Input: (224, 1000) for 10s
- Pro: Maximum temporal resolution, matches v1 training
- Con: 1.75× more data per sample, potentially slower
- This is what we'd try first

**Option B: Match 1st place hop (1252 = 39ms)**
- Input: (224, 256) for 10s or (224, 512) for 20s
- Pro: Proven to work, less compute, matches winner
- Con: Requires switching to 20s windows (different data pipeline)
- Try this if Option A is too slow/OOM

### Why not just use Perch?

Perch alone = 0.825 LB, roughly matching our SED v1+pp (0.823). But:
1. Perch is a frozen TF model — we can't fine-tune it on our GPU cluster
2. Perch covers 203/234 species (missing some amphibians/insects)
3. The Perch-based notebooks (0.907-0.917) build sophisticated temporal models on TOP of Perch — they're not just running Perch
4. Our trained SED model is **complementary** to Perch (different architecture, different training data, different error patterns)
5. The highest-value outcome is: strong SED baseline + Perch ensemble, where both contribute uniquely

A strong SED baseline (0.85+) ensembled with Perch predictions could potentially match or exceed the Perch+ProtoSSM approaches (0.913-0.917) by combining trained spectral features with foundation model generalization.
