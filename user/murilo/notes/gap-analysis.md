# Gap Analysis: v4b (0.859) vs Competition — 2026-03-26

Cross-reference of our best model against competitor approaches and winning solutions.
Identifies what we're missing, what's working, and a prioritized improvement path.

Last updated: 2026-03-26

## Current Leaderboard Context

Top 20 ranges from **0.927 to 0.938 LB**. Our best (0.859) would place roughly in the
top 50-100 range. The gap to top-1 is **0.079 LB** (0.859 → 0.938).

| Position | Score | Approach (estimated) |
|----------|-------|---------------------|
| 1st | 0.938 | Likely Perch ensemble + pseudo-labels |
| ~15th | 0.929 | Tom + Claude-Code (SED + Perch combo?) |
| ~20th | 0.927 | Ensemble approaches |
| Our best | **0.859** | Single EfficientNet-B0 SED (v4b) |

Note: Tom (from our reference docs) is at 0.929 LB — significantly above his reported
0.8139 soundscape val AUC, suggesting he has improved since those 28 experiments were
shared (likely added Perch ensemble, more pseudo-label rounds).

## What We Got Right

| Technique | Our Implementation | Competitor Validation |
|-----------|-------------------|----------------------|
| **n_fft=4096** | v4a: +0.030 LB | 2025 1st place uses same |
| **n_mels=224** | Standard across v4* | Tom, 2025 1st, all use 224 |
| **AmplitudeToDB + min-max** | v4b: +0.034 LB | Universal among top SED performers |
| **10s inference windows** | Since exp 016 | 2024 1st: +0.015 LB from context overlap |
| **GeM freq pooling** | In AttHead | Tom, Tawara, 2025 1st all use GeM |
| **SED dual loss (clip+frame)** | 0.5/0.5 weighting | Tom, Tawara, 2025 1st all use dual |
| **Full-signal mel inference** | v4a: >1h → <32min | 2025 1st uses same approach |
| **Soundscape training data** | Combined focal + soundscapes | Critical: exp 012 showed +0.046 LB |

## Critical Gaps (Ordered by Expected Impact)

### Gap 1: No Pseudo-Labeling (Expected: +0.02–0.03 LB)

**What competitors do:**
- 2025 1st: 4-round self-training with power transform (γ=2.0) → **+0.032 private LB**
- Tom: r1-r5 pseudo-labels with Perch soft KD → **+0.034 val** (cumulative with KD)
- 2024 1st: Unlabeled soundscape pseudo-labels → **+0.025 holdout**

**What we tried**: Exp 015 (single round, w=0.5, ProtoCLR+ConvNeXt teachers) → **failed**
(−0.005 to −0.024 LB regression)

**Why ours failed**: Weight too aggressive (0.5), single round (no refinement), weak
teachers (clip-level models, not SED). The winners use:
- Low weight (0.05–0.3 on pseudo-labels)
- Multiple rounds (4-5) with progressive sharpening
- Strong teachers (Perch, or previous best SED model)
- Power transform to suppress confident noise between rounds

**Action**: Re-attempt pseudo-labeling with v4b as the teacher (much stronger than
exp 012 models), lower weight (0.1–0.2), power transform between rounds.

References:
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md) — Phase 3

### Gap 2: No Ensemble / K-Fold (Expected: +0.01–0.03 LB)

**What competitors do:**
- Tawara: 4-fold MSGKF → 0.859 LB (matches our single model!)
- Tom: 4-fold GroupKFold
- 2025 1st: 7-model ensemble
- 2024 1st: 9-model ensemble (6 EffNet + 3 RegNet)

**Our status**: Single model, single 90/10 split.

**Action**: 4-fold of v4b could reach ~0.865–0.870. Free improvement at 4x training cost.
Our v4b takes ~4h per run → 16h total for 4-fold. Tawara matches our score with 4-fold
of a weaker backbone (HGNetV2-B0) — our 4-fold should exceed their result.

**Ensemble opportunity**: v4b (0.859) + a Perch-based pipeline (~0.910+) would combine
fundamentally different feature sources. The diversity should yield +0.01–0.03.

### Gap 3: No Perch Integration (Expected: +0.03–0.05 LB with ensemble)

**What competitors do:**
- Simple Perch probes: 0.910 LB (just LogReg on frozen Perch embeddings)
- ProtoSSM on Perch: 0.913–0.917 LB
- Tom at 0.929: likely SED + Perch ensemble

**Our status**: No Perch integration at all.

**Two paths:**
1. **Perch as inference ensemble member**: Run frozen Perch on test soundscapes alongside
   v4b, blend predictions. Can fit in 90min budget (v4b ~30min, Perch ~15min, blend ~5min).
2. **Perch as knowledge distillation teacher**: Generate soft targets from Perch for SED
   training. Tom gained +0.030 val from Perch KD.

**Action**: Path 1 is immediate (no retraining). Path 2 requires a training round.

References:
- [Perch ProtoSSM (zirach)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Perch Probes (chiranjithdharma)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md)

### Gap 4: LogSumExp vs Hard Max for Frame Aggregation (Expected: +0.005–0.01 LB)

**What Tawara does**: `temperature * logsumexp(timewise_logits / temperature, axis=1)`
with T=0.1. This is a smooth differentiable approximation to max.

**What we do**: `framewise_logits.max(dim=-1).values` — hard max, discontinuous gradient.

**Why it matters**: Hard max has zero gradient for all non-max frames, which limits
the loss signal to a single frame per class. LSE distributes gradient across nearby-max
frames, potentially producing more robust temporal representations.

**Action**: Low-effort change in SEDFinetuneModule. Test as a single-variable ablation.

### Gap 5: Pretrained Weights (Expected: uncertain, possibly +0.01 LB)

**Tom uses**: `tf_efficientnet_b0.ns_jft_in1k` (Noisy Student on JFT-300M → ImageNet-1K)
- 4.0M params, **1280-dim** features

**We use**: `efficientnet_b0` (standard ImageNet, RA pretraining)
- 3.6M params (similar), **320-dim** features after `features_only`

**Our exp 016 v2 tested NS-JFT** → **regressed** (−0.023 AUROC, −0.054 cmAP). But that
was with soft CE loss + old mel config (n_fft=1024, n_mels=128). The interaction with
AmplitudeToDB + BCE might be different.

**Action**: Low priority given v2 regression. But could re-test with v4b's recipe
(AmplitudeToDB norm) since the normalization change may interact differently with
NS-JFT features.

### Gap 6: MixUp Intensity (Expected: +0.005 LB)

**Tawara**: alpha=1.0 (uniform [0.5, 1.0] after max), theta=0.8 label binarization,
disabled during first 5 warmup epochs.

**We use**: alpha=0.5, enabled all epochs.

**2025 2nd place**: Raw waveform MixUp gave +0.036 private LB.

**Action**: Try alpha=1.0 with warmup disable. Also consider raw waveform MixUp
(infrastructure exists from v3b, but untested with AmplitudeToDB norm).

### Gap 7: OpenVINO Inference (Expected: 2-3x faster)

**Every winner (2023–2025) and every reference notebook** uses OpenVINO for CPU inference.
We use ONNX Runtime.

**Why it matters**: If inference is faster, we can:
- Fit larger models (B3, B4) in the 90min budget
- Run more TTA shifts
- Fit v4b + Perch ensemble comfortably

**Current timing**: v4b ~30min ONNX. With OpenVINO could be ~15min, leaving 75min for
Perch + blending.

**Action**: Convert ONNX → OpenVINO IR, benchmark on Kaggle.

### Gap 8: SpecAugment / Frequency Masking (Expected: +0.005 LB)

**2024 1st**: XY masking (SpecAugment-style) was in their recipe.
**2023 1st**: SpecAugment with freq mask (len=10, masks=3) + time mask (len=20, masks=3).

**We use**: Only MixUp. No frequency/time masking.

**Action**: Add SpecAugment (freq + time masking) as augmentation. Low-effort, well-established.

## What Doesn't Stack / Shouldn't Pursue

From our exp 017 ablations and competitor findings:

| Technique | Our Result | Competitor Finding | Verdict |
|-----------|-----------|-------------------|---------|
| StochasticDepth + AmplitudeToDB | v4f: −0.027 LB | 2025 1st uses drop_path=0.15 | Don't combine with a2db |
| AttBlockV2 + soft CE | v4c: 0.815 val | Works with BCE (Tawara) | Head needs matching loss |
| AttBlockV2 + BCE | v4g: 0.819 LB | Tawara: 0.859 (4-fold) | High val but poor LB generalization |
| ASL loss | v4d: 0.808, v4h: 0.800 | Tom: +0.023 over BCE | Doesn't work with our setup, even with BCE+AttBlockV2 |
| Background noise aug | — | Tom: −0.041 | Avoid (conflicts with pseudo-labels) |
| CutMix | — | Tom: −0.047 | Avoid (destroys spectral coherence) |
| Rating filter ≥3 | — | Tom: −0.128 | Avoid (lose too much data) |
| NS-JFT + soft CE | v2: −0.023 AUROC | Tom uses NS-JFT + ASL | Interaction effects, risky |

## Prioritized Improvement Roadmap

Ranked by expected impact × feasibility. v4b (0.859) is the baseline.

### Tier 1: High Impact, Near-Term (target: 0.87–0.90)

| # | Action | Expected | Effort | Dependencies |
|---|--------|----------|--------|-------------|
| 1 | **Perch ensemble** (v4b + frozen Perch probes) | +0.03–0.05 | 2-3 days | None — just inference pipeline |
| 2 | **4-fold v4b** (rank-averaged ensemble) | +0.01–0.02 | 16h training | v4b recipe locked |
| 3 | **Multi-round pseudo-labeling** (v4b teacher, 3 rounds) | +0.02–0.03 | 3-5 days | v4b fulldata model |

### Tier 2: Medium Impact (target: 0.90–0.92)

| # | Action | Expected | Effort | Dependencies |
|---|--------|----------|--------|-------------|
| 4 | **Perch soft KD** for SED training | +0.01–0.02 | 2 days | Perch inference pipeline |
| 5 | **OpenVINO conversion** (faster inference → fit more in 90min) | +0.005 indirect | 1 day | — |
| 6 | **LogSumExp frame aggregation** (replace hard max) | +0.005–0.01 | Few hours | — |
| 7 | **SpecAugment** (freq + time masking) | +0.005 | Few hours | — |
| 8 | **MixUp alpha=1.0** + warmup disable | +0.005 | Config change | — |

### Tier 3: Exploratory (diminishing returns)

| # | Action | Expected | Effort | Notes |
|---|--------|----------|--------|-------|
| 9 | 20s windows (matching 2025 1st) | uncertain | Medium | More context but slower inference |
| 10 | NS-JFT weights with AmplitudeToDB | uncertain | Low | v2 regressed, but different context |
| 11 | CE loss with sigmoid inference trick | uncertain | Low | 2024 1st's approach |
| 12 | Raw waveform MixUp | +0.005? | Low | Untested with AmplitudeToDB |

## Inference Budget Analysis

Current: v4b takes ~30min on Kaggle CPU. Budget: 90min.

| Configuration | Estimated Time | Fits in 90min? |
|---------------|---------------|----------------|
| v4b alone | 30min | Yes (33% of budget) |
| v4b + Perch frozen | 30 + 15 = 45min | Yes (50%) |
| v4b 4-fold ensemble | 30 × 4 = 120min | No — need OpenVINO or subset |
| v4b (OpenVINO) + Perch | 15 + 15 = 30min | Yes (33%) — room for more |
| v4b (OV) 4-fold + Perch | 15×4 + 15 = 75min | Yes (83%) |

**Key insight**: OpenVINO is a prerequisite for fitting ensembles within the budget.
Converting to OpenVINO should be done before scaling to multi-model inference.

## References

- [Tom's 28 experiments](../../docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md)
- [HGNetV2-B0 (Tawara)](../../docs/references/kaggle/birdclef-2026/discussions/hgnetv2-b0-baseline-tawara.md)
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md)
- [2025 1st place](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md)
- [2024 1st place](../../docs/references/kaggle/birdclef-2024/birdclef2024-1st-place-solution.md)
- [2023 1st place](../../docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md)
- [Perch ProtoSSM (0.913)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-protossm-zirach.md)
- [Pantanal Distill (0.917)](../../docs/references/kaggle/birdclef-2026/notebooks/pantanal-distill.md)
- [Perch Probes (0.910)](../../docs/references/kaggle/birdclef-2026/notebooks/perch-probes-chiranjithdharma.md)
