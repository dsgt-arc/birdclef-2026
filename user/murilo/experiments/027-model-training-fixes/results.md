# Experiment 027: Model Training Fixes — Results

## Context

Deep research (2026-04-11) identified two fundamental implementation errors
in our SED pipeline compared to BirdCLEF 2024-2025 winners:

1. **MixUp**: We used interpolative mixing (lambda*x1 + (1-lambda)*x2). Winners
   use additive waveform mixing (x1 + x2) with max labels. Impact: -0.036
   private LB without correct MixUp (BirdCLEF 2025 2nd place ablation).

2. **Loss**: We used BCE across all 234 species. Winners use CE on primary label
   only during training, sigmoid at inference. Impact: +0.044 private LB from
   this single change (BirdCLEF 2024 1st place).

See: `notes/2026-04-11-deep-research-claude.md`, `notes/2026-04-11-deep-research-gemini.md`

## Phase 1: v10a (Additive MixUp + CE loss)

**Config**: `configs/v10a_addmix_ce.yaml`
**Changes from v5j**: `mixup_mode=additive`, `raw_waveform_mixup=true`, `loss_type=ce`
**Everything else**: identical to v5j (5s window, LSEHead, batch 64, 20 epochs)

**SLURM**: Job 6439641, RTX 6000, 38m51s walltime
**Checkpoint**: epoch 11 (best by val_sc/auroc)

### Metrics

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.835 |
| val/cmap | 0.571 |
| val/f1_macro | 0.497 |
| val/top1_acc | 0.733 |
| val/top5_acc | 0.859 |
| val/loss | 1.597 |
| **val_sc/auroc (soundscape)** | **0.863** |
| LB | — (pending submission) |

### val_sc/auroc progression

Best checkpoint picked at epoch 11 (val_sc/auroc=0.863). Progression:

| Epoch | val_sc/auroc | Note |
|-------|-------------|------|
| 0 | 0.782 | Initial |
| 1 | 0.826 | +0.044 |
| 2 | 0.857 | +0.031 |
| 3–10 | <0.857 | Plateau (MixUp warmup ends at epoch 5) |
| **11** | **0.863** | **Best — checkpoint saved** |
| 12–19 | <0.863 | No improvement |

### Comparison with baselines

| Model | Loss | MixUp | val/auroc | val/cmap | val/top1 | val_sc/auroc | LB |
|-------|------|-------|-----------|----------|----------|-------------|-----|
| v5j (exp 019) | BCE | interp spec | 0.874 | — | — | — | **0.903** |
| v5j-dualval (exp 026) | BCE | interp spec | 0.849 | 0.596 | 0.752 | 0.956* | 0.897 |
| **v10a (exp 027)** | **CE** | **additive wfm** | **0.835** | **0.571** | **0.733** | **0.863** | **pending** |

*v5j-dualval val_sc/auroc=0.956 uses per-class filtered metric (only species with positives).
v10a val_sc/auroc=0.863 uses the same per-class filtered metric.

### Analysis

**Focal metrics (val/auroc) decreased**: 0.835 vs 0.849 (v5j-dualval) and 0.874
(v5j). This is expected with CE loss — CE forces competition between classes via
softmax, producing a different probability calibration than BCE. The sigmoid-based
val metrics (AUROC, cmap) measure sigmoid(logits), but CE optimizes softmax(logits).
The logit magnitudes are calibrated differently.

**val_sc/auroc dropped from 0.956 to 0.863**: This metric uses per-class AUROC
filtered to species with positives. The drop could indicate the CE+additive MixUp
combination is less calibrated for multi-label soundscape prediction, OR it could
be that the v5j-dualval metric was inflated by the small 86-segment val set. Only
the LB score can tell us whether this is a real regression or a calibration artifact.

### LB Result: 0.494 — CATASTROPHIC FAILURE

**LB = 0.494** (essentially random). The combined additive MixUp + CE loss
destroys soundscape performance completely.

### Root Cause Analysis

**ONNX output comparison** (random mel input):

| Metric | v5j (BCE) | v10a (CE) |
|--------|-----------|-----------|
| Logit range | ~-9 (absent) to +5 (present) | -0.49 to +0.48 |
| sigmoid(absent) | ~0.0001 | ~0.38 |
| sigmoid(present) | ~1.0 | ~0.62 |
| Species > 0.5 | 0 | 113 of 234 |

**Why CE logits are tiny**: `F.cross_entropy` optimizes softmax (relative ranking),
not absolute logit magnitudes. Weight decay (1e-4) keeps weights small. BCE has
direct per-species gradient pressure to push absent logits → -∞; CE has no such
force. The model achieves good softmax probabilities with logits near zero.

**Why val/auroc=0.835 but LB=0.494**: The val set is 97% focal (training domain).
On focal data, even 0.5-unit logit differences preserve correct rankings → decent
AUROC. On soundscapes (LB domain), the domain shift perturbation (~0.3 units)
exceeds the logit spread (~1 unit), scrambling rankings completely. BCE logits
at -9 ± 2 survive domain shift; CE logits at -0.2 ± 0.3 do not.

**Why competitors succeed with CE**:
1. BirdCLEF 2024 1st place explicitly notes CE+sigmoid is "noisier" — they
   compensated with **min-ensemble reduction** (element-wise minimum across
   models). The +0.044 gain was measured in an **ensemble**, not single model.
2. Salman (0.922) uses CE with 4-model ensemble + site/hour priors.
3. Boredom (#1) has 53 rounds of pseudo-labels bridging domain shift.
4. **No evidence of CE+sigmoid succeeding as a standalone single-model change
   with extreme focal→soundscape domain shift** — which is our setup.

**Conclusion**: CE loss is fundamentally incompatible with our single-model,
high-domain-shift pipeline. The small logit magnitudes have zero safety margin
for the focal→soundscape transition. This is **not an implementation bug** —
it's a precondition failure (CE needs ensembles, pseudo-labels, or calibration).

CE loss is now confirmed dead for our pipeline: **7 tests, 0 successes**.

### Implication for Phase 2

v10a combined TWO changes. CE was catastrophic and masks the MixUp contribution.
v10b (additive MixUp + BCE) is essential to isolate whether the corrected MixUp
formulation helps when paired with the working loss function.

## Phase 2: v10b (Additive MixUp + BCE loss — isolate MixUp fix)

**Config**: `configs/v10b_addmix_bce.yaml`
**Changes from v5j**: `mixup_mode=additive`, `raw_waveform_mixup=true` (BCE loss, same as v5j)
**Everything else**: identical to v5j (5s window, LSEHead, batch 64, 20 epochs)

**SLURM**: Job 6465883, V100, 44m32s walltime
**Checkpoint**: epoch 17 (best by val/auroc = 0.847)

### Metrics (epoch 19 / last)

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.844 |
| val/cmap | 0.592 |
| val/f1_macro | 0.539 |
| val/top1_acc | 0.747 |
| val/top5_acc | 0.854 |
| val/loss | 0.013 |
| **val_sc/auroc (soundscape)** | **0.966** |
| val_sc/cmap | 0.885 |
| LB | — (pending submission) |

### Comparison with baselines

| Model | Loss | MixUp | val/auroc | val/cmap | val/top1 | val_sc/auroc | LB |
|-------|------|-------|-----------|----------|----------|-------------|-----|
| v5j (exp 019) | BCE | interp spec | 0.874 | — | — | — | **0.906** |
| v5j-dualval (exp 026) | BCE | interp spec | 0.849 | 0.596 | 0.752 | 0.956 | 0.897 |
| v10a (exp 027) | CE | additive wfm | 0.835 | 0.571 | 0.733 | 0.863 | 0.494 |
| **v10b (exp 027)** | **BCE** | **additive wfm** | **0.844** | **0.592** | **0.747** | **0.966** | **0.875** |

### LB Result: 0.875 — REGRESSED

**LB = 0.875** (-0.031 from v5j baseline). Additive waveform MixUp hurts even
with BCE loss. The "corrected" formulation from deep research is actually worse
than our previous interpolative waveform MixUp test (v6b = 0.887, -0.019).

All waveform MixUp variants regress vs spectrogram MixUp:

| MixUp variant | LB | Delta vs v5j |
|--------------|-----|-------------|
| v5j (interpolative spectrogram) | **0.906** | baseline |
| v6b (interpolative waveform, exp 021) | 0.887 | -0.019 |
| **v10b (additive waveform)** | **0.875** | **-0.031** |

**Why waveform MixUp hurts**: Additive waveform mixing (x1+x2) doubles the
amplitude, shifting the entire spectrogram energy distribution. The model trained
on amplitude-doubled spectrograms sees different energy patterns at inference
(single-recording spectrograms). This creates a train/inference distribution
mismatch — the same kind of domain shift that plagues all our training changes.
Spectrogram MixUp avoids this because the lambda-weighted blend preserves the
energy scale.

**Offline metrics were again misleading**: val_sc/auroc improved +0.010 (0.966
vs 0.956), but LB regressed -0.031. The offline/LB anti-correlation continues.

### Experiment 027 Conclusion

All three hypotheses from deep research (Claude + Gemini, 2026-04-11) failed:

1. **CE loss + sigmoid inference**: 0.494 LB (catastrophic)
2. **Additive waveform MixUp**: 0.875 LB (-0.031 regression)
3. **Overlapping window inference**: 0.906 LB (flat, TTA already covers it)

The deep research correctly identified techniques used by winners, but the
techniques do not transfer to our specific pipeline. After 16+ failed training
recipe changes across 27 experiments, the evidence is overwhelming: **local
modifications to the v5j recipe cannot reach 0.92+**. The gap requires either
a fundamentally different approach or understanding the specific architectural/
data differences between our pipeline and Salman's 0.922.

## Phase 4: Salman-Derived Implementation Changes

Analyzed Salman Ahmed's BirdCLEF 2025 training notebook + Mario Lasseck's
BirdCLEF 2024 paper. Testing specific structural pipeline differences.
See: `notes/2026-04-12-salman-ahmed-birdclef2025.md`, `notes/2026-04-12-mario-lasseck-birdclef2024.md`

### v5j-control (W&B baseline)

**Config**: `configs/v5j_control.yaml` — identical to v5j (exp 019), logged to exp027 W&B
**SLURM**: Job 6468944, RTX 6000, 39m24s walltime
**Checkpoint**: epoch 14 (best by val/auroc = 0.852)

| Metric | Value |
|--------|-------|
| val/auroc (focal) | 0.849 |
| val/cmap | 0.596 |
| val/f1_macro | 0.504 |
| val/top1_acc | 0.752 |
| val/top5_acc | 0.866 |
| val/loss | 0.011 |
| val_sc/auroc (soundscape) | 0.955 |
| val_sc/cmap | 0.853 |

Matches v5j-dualval (exp 026) closely: val/auroc 0.849 vs 0.849, val_sc/auroc
0.955 vs 0.956. Pipeline verified — this is the correct baseline for Phase 4
W&B comparisons.
