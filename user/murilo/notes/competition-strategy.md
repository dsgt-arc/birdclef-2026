# Competition Strategy & Roadmap

Working notes on the overall BirdCLEF 2026 strategy, synthesized from experiments
and competitor analysis. Updated as new results come in.

Last updated: 2026-03-26

## Performance Landscape

Two distinct tiers on the leaderboard:

| Tier | Approach | LB Range | Our Best |
|------|----------|----------|----------|
| **Foundation model** | Perch v2 frozen + learned heads | 0.907–0.917 | — (not implemented) |
| **Trained SED** | EfficientNet-B0 SED from ImageNet init | 0.795–0.859 | **0.859** (v4b) |

The gap (~0.05–0.06) is driven by Perch v2's pretraining on 10K+ bird species vs our
ImageNet features. This is a **representation gap**, not an architecture gap.

### Foundation Model Leaderboard

| Notebook | LB | Approach | Key Innovation |
|----------|-----|---------|----------------|
| Pantanal Distill | **0.917** | ProtoSSM v3 + ResidualSSM | Metadata embeddings in SSM, two-pass boosting |
| Perch+ProtoSSM (zirach) | **0.913** | ProtoSSM v2 + MLP probes | Bidirectional SSM temporal modeling |
| Perch+Probes (chiranjithdharma) | **0.910** | LogReg probes only | Simplest approach, no temporal model |
| Perch+ResidualSSM (blamerx) | **0.907** | ResidualSSM error correction | Two-pass with zero-init residuals |

**Key observation**: The simplest Perch approach (LogReg probes, 0.910) is only 0.007
below the most complex (ProtoSSM v3 + ResidualSSM, 0.917). Temporal modeling adds modest
value. The foundation model itself is doing most of the work.

References:
- [ProtoSSM (zirach)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Pantanal Distill](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md)
- [Perch Probes (chiranjithdharma)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md)
- [Perch ResidualSSM (blamerx)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-blamerx.md)

### Our SED Progression

| Model | LB | Key Change | Exp |
|-------|-----|-----------|-----|
| ProtoCLR-2026 (focal) | 0.721 | First 2026 model | 011 |
| ProtoCLR-2026 (soundscape) | **0.767** | + soundscape fine-tuning | 012 |
| EfficientNet-B0 SED v1 | 0.795 | SED architecture switch | 016 |
| + post-processing (pp2) | **0.827** | TTA + smoothing + TopN + metadata | 016 |
| + mel overhaul (v4a) | **0.825** raw | n_fft=4096, n_mels=224 | 017 |
| + AmplitudeToDB (v4b) | **0.859** | AmplitudeToDB norm | 017 |

Reference: [MODEL-RUNS.md](../MODEL-RUNS.md)

## Training Paradigm Difference

An important confound in the ProtoCLR → EfficientNet comparison:

- **ProtoCLR/ConvNeXt/EAT** (exp 011→012): Two-stage training
  1. Train on focal data only (exp 011) → 0.721 LB
  2. Fine-tune on focal + labeled soundscapes (exp 012) → 0.767 LB

- **EfficientNet SED** (exp 016→017): Single-stage training
  - Train on focal + labeled soundscapes jointly from the start → 0.859 LB

The 0.092 LB gap (0.767 → 0.859) includes **both** the architecture change (SED vs
clip-level) **and** the training paradigm change (two-stage vs joint). We haven't
isolated which contributed more.

**Potential experiment**: Try two-stage training for EfficientNet SED:
1. Train on focal-only first (builds clean representations)
2. Fine-tune on focal + soundscapes with lower LR (domain adaptation)

This is essentially a curriculum learning approach. The 2023 1st place used a similar
two-stage pipeline (pretrain on 822 species → fine-tune on 264 scored species).

References:
- [BirdCLEF 2023 1st place](../../docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md) — two-stage pretrain→finetune
- [BirdCLEF 2024 1st place](../../docs/references/kaggle/birdclef-2024/birdclef2024-1st-place-solution.md) — noted multi-stage training **hurt** (different context)

## ProtoSSM vs ProtoCLR — Different Models

These are **completely different** despite the similar names:

| | ProtoCLR | ProtoSSM |
|--|----------|----------|
| **Type** | Contrastive-learning pretrained backbone | State Space Model on frozen embeddings |
| **Backbone** | CvT-13 (our LoRA fine-tuned) | Perch v2 (frozen, not trained) |
| **Training** | Fine-tuned on focal + soundscapes | Trained on labeled soundscapes only |
| **Parameters** | ~19M (backbone + LoRA) | ~200K (lightweight SSM) |
| **LB** | 0.767 (our best) | 0.913 (Kaggle reference) |
| **Pretraining** | ImageNet → contrastive audio | Google's 10K+ species bioacoustic |

ProtoSSM sits **on top of** Perch v2 features — it's a learned temporal model over
a frozen foundation, not a standalone audio encoder. The two could potentially be
combined (ProtoCLR as an additional feature source for ProtoSSM-style temporal modeling).

References:
- [ProtoSSM analysis](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach-analysis.md)

## Ensemble Opportunity

Our SED approach and the Perch-based approach see data fundamentally differently:

- **SED**: Learns from raw spectrograms, captures fine-grained acoustic patterns,
  benefits from high mel resolution and framewise detection
- **Perch-based**: Uses pretrained 10K-species embeddings, captures semantic bird
  identity from global features, benefits from temporal modeling across windows

These are **complementary** — errors are likely uncorrelated. An ensemble of:
1. Our best EfficientNet SED (v4b, 0.859)
2. A Perch-based pipeline (0.910+)

Could push well beyond either alone. Expected gain: +0.01–0.03 LB from the diversity.

## Techniques That Stack vs Don't

From exp 017 ablations, we learned that techniques don't always combine additively:

### What stacks
- Mel resolution (n_fft=4096, n_mels=224) ← foundational, always helps
- AmplitudeToDB normalization ← +0.034 LB on top of mel resolution
- Post-processing (TTA, smoothing, TopN) ← always helps, diminishing returns

### What doesn't stack
- **StochasticDepth + AmplitudeToDB** (v4f): 0.832 LB (−0.027 vs v4b alone)
  - Regularization counterproductive when normalization already preserves signal
- **ASL loss in any context** (v4d: 0.808, v4h: 0.858 val)
  - Converges fast, plateaus low — doesn't suit our data distribution
- **AttBlockV2 + soft CE** (v4c: 0.815 val) — loss/head mismatch

### Pending
- **AttBlockV2 + BCE** (v4g: 0.868 val, LB pending) — correct pairing, promising offline

Reference: [Exp 017 tasks.md](../experiments/017-mel-overhaul/tasks.md) — full config table

## Winning Solutions Patterns (2023–2025)

### Common themes across 3 years of winners

| Technique | 2023 1st | 2024 1st | 2025 1st |
|-----------|----------|----------|----------|
| **Data curation** | XC collection + noise filtering | Signal stats filtering + Google classifier | Pseudo-label power transform |
| **Pretraining** | 822 species (2020-2022 data) | — | Self-supervised + pseudo-labeling (4 rounds) |
| **Architecture** | ConvNeXt-S + ECA-NFNet-L0 | EfficientNet-B0 + RegNetY-008 | EfficientNet-B0/B4, RegNetY |
| **Head** | Attention SED (GeM + att) | Simple backbone (no SED) | Attention SED (GeM + att) |
| **Loss** | Focal (BCE-based) | **CE** (not BCE — big finding) | CE + SoftAUCLoss (later rounds) |
| **Augmentation** | Mixup + bg noise + SpecAugment | Cutmix + XY masking | Mixup + drop_path=0.15 |
| **Post-processing** | ONNX ensemble (3 models) | min() ensemble + OpenVINO | Overlap reconstruction + Gaussian smoothing |
| **Pseudo-labels** | Not used | Unlabeled soundscapes | 4-round power-transformed self-training (+0.032) |
| **Inference** | ONNX CPU | OpenVINO CPU | OpenVINO CPU |

**Key takeaways**:
1. **Data matters more than model** — every winner emphasizes curation/filtering
2. **CE loss works** — 2024 winner found CE significantly better than BCE for sparse multi-label
3. **EfficientNet-B0 is consistently competitive** — no winner needed large models
4. **Multi-round pseudo-labeling is the biggest single gain** (2025: +0.032 private LB)
5. **OpenVINO/ONNX for CPU inference** — universal among winners

References:
- [2023 1st place](../../docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md)
- [2024 1st place](../../docs/references/kaggle/birdclef-2024/birdclef2024-1st-place-solution.md)
- [2025 1st place inference](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md)

## What To Try Next

Roughly ordered by expected impact and feasibility:

### High Impact, Near-term
1. **Perch-based pipeline** — Even simple LogReg probes on Perch embeddings hit 0.910 LB.
   Could implement in 1-2 days. Would immediately become our best single model.
2. **Ensemble SED + Perch** — Combine v4b (0.859) with a Perch pipeline for diversity.
3. **Multi-round pseudo-labeling** with Perch as teacher — 2025 1st place gained +0.032
   from 4 rounds of power-transformed self-training.

### Medium Impact
4. **Two-stage training for SED** — Focal-only first, then soundscape fine-tune.
   Tests curriculum learning hypothesis.
5. **OpenVINO inference** — Replace ONNX Runtime with OpenVINO for faster CPU inference.
   Winners all use it. May allow larger models within 90min budget.
6. **CE loss for SED** — 2024 1st place found CE > BCE. We use soft CE which is
   similar but not identical. Worth testing standard CE with sigmoid at inference.

### Exploratory
7. **Perch knowledge distillation** — Tom's experiments show +0.030 val from soft KD.
   Train SED with Perch soft targets as auxiliary loss.
8. **Prior-year pretraining** — 2023 1st place pretrained on 822 species from 2020-2022.
   2025 2nd place gained +0.014 from 819K recordings.
9. **K-fold ensemble** — Tawara's HGNetV2-B0 4-fold hits 0.859 (matching our single model).
   4-fold of our v4b could be ~0.865-0.870.

## Open Questions

- Does the SED + Perch ensemble actually work? (Need to verify error correlation)
- Can we use Perch as a teacher for our SED models? (Perch KD)
- Is two-stage training better than joint for SED?
- Would a larger backbone (B3, B4) + AmplitudeToDB beat B0?
- Can we train ProtoSSM/MLP probes ourselves on Perch embeddings?
