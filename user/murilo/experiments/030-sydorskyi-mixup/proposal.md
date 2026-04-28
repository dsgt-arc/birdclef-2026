---
experiment: "030-sydorskyi-mixup"
status: proposed
date: 2026-04-26
author: Murilo Gustineli
---

# Experiment 030: Sydorskyi-faithful MixUp on v5j

## Hypothesis

**Phase A (load-bearing test)**: replacing v5j's Zhang-style interpolated
spectrogram MixUp with Sydorskyi 2025 2nd-place's faithful waveform MixUp
recipe — `wave = (w1+w2)/2` + peak-normalize + `target = max(y1, y2)` —
will move LB. Direction is uncertain; the prediction is *change* (positive
or negative), not specifically improvement.

Falsifiable: if `v5j_sydorskyi` lands within ±0.005 of `v5j_control`
(seed-noise band per Mahog), the mixup recipe isn't a single-variable lever
on v5j.

**Phase B (extension)**: add EfficientNetV2-B0 backbone (Salman's choice
for his single-model 0.937 LB) to the matrix, giving us a clean 2x2
ablation to attribute main effects of backbone, mixup, and their
interaction.

## Background

### Why now (the audit + simulation + D1 trail)

This experiment is the most direct test of three concrete findings from the
2026-04-26 diagnostic phase (PR #42):

1. **MixUp implementation audit**
   ([note](../../analysis/notes/2026-04-26-mixup-implementation-audit.md)):
   v5j and prior wave-mixup attempts (v4j/v6b/v10b) all use Zhang-style
   interpolated targets `lam·y1 + (1-lam)·y2`. Sydorskyi (LB 0.917 single
   fold, top-3 finishers all do this) uses `max(y1, y2)`. The prior
   conclusion "wave-mixup is dead" was based on three different recipes,
   none matching Sydorskyi's exact specification.

2. **MixUp target-distribution simulation**
   ([note](../../analysis/notes/2026-04-26-mixup-target-simulation-results.md)):
   80% of v5j's active training target values are sub-1; 59% of training
   samples have NO class at full presence after mixup. The soundscape test
   distribution (multi-label binary, both birds at 1.0) is barely seen
   during training.

3. **D1 train-side attention verdict**
   ([note](../../analysis/notes/2026-04-26-d1-train-side-attention-verdict.md)):
   84% top-1% failure overlap between v5j (LSEHead) and v6k (AttHead).
   Failure pattern is data-driven, not head-driven. Information-side
   levers have stronger evidence than modeling-side levers given current
   data. Target-distribution mismatch is a *training-distribution*
   intervention (between data and modeling).

### What this is NOT

- **NOT** a faithful reproduction of Sydorskyi's full pipeline. Single-axis
  change. We're only testing the MixUp recipe in isolation. Sydorskyi
  also uses BG noise mixing, SpecAugment, RandomFiltering, balanced
  sampler, label smoothing, AmpToDB-then-standardize-then-minmax (we
  match this part), focal loss, in-domain pretraining at 819K-recording
  scale, and pseudo-distillation. None of those changes here.
- **NOT** a faithful Salman replication. We test his backbone choice
  (EffNetV2-B0) in Phase B but not his loss (CE), inference recipe, or
  20s windows.

### What we KNOW from prior tests

| Run | Domain | Lambda | Target combo | Result |
|---|---|---|---|---|
| v5j (current best) | spec | Beta(1,1) max-flipped | interp + theta=0.8 cap | **LB 0.906** |
| v4j (exp 017) | wave | Beta(0.5,0.5) max-flipped, p=0.5 | interp, no theta | 0.853 (regressed) |
| v6b (exp 021) | wave | Beta(1,1) per-sample | interp + theta=0.8 | 0.887 (regressed) |
| v10b (exp 027) | wave | n/a (additive) | max | 0.875 (regressed, but `x1+x2` no /2) |
| **v5j_sydorskyi (THIS)** | **wave** | **deterministic 0.5** | **max + (w1+w2)/2 + peak-norm** | **?** |

We've never tested the specific Sydorskyi cell (deterministic 0.5 average +
peak-normalize + max targets). v10b is closest but had `x1+x2` without
`/2` or peak-normalize — input distribution shift.

## Method

### Phase A — single-variable mixup test on v5j (committed)

| Variant | Mixup recipe | Other |
|---|---|---|
| `v5j_control.yaml` | spec, Beta(1,1), interpolate, theta=0.8, p=1.0, warmup=5 | identical to v5j |
| `v5j_sydorskyi.yaml` | wave, deterministic 0.5, `(w1+w2)/2` + peak-norm, max(y1,y2), p=0.5, no theta, no warmup | identical to v5j |

Single-variable diff: only `mixup_*` and `raw_waveform_mixup` keys differ.
Same backbone (HGNetV2-B0), same head (LSEHead, hidden=1024), same loss
(BCE), same optimizer (AdamW, lr=5e-4 with OneCycleLR), same schedule
(20 epochs), same data (90/10 + 8 SS held out, seed=42), same precision
(16-mixed), same inference postproc (TTA + smoothing + TopN + priors).

### Phase B — EfficientNetV2-B0 backbone, 2x2 ablation (committed but lower priority)

| Variant | Backbone | Mixup |
|---|---|---|
| `v5j_control` | HGNetV2-B0 | interp |
| `v5j_sydorskyi` | HGNetV2-B0 | sydorskyi |
| `effnetv2b0_control` | EfficientNetV2-B0 (`tf_efficientnetv2_b0.in1k`) | interp |
| `effnetv2b0_sydorskyi` | EfficientNetV2-B0 | sydorskyi |

The 2x2 lets us cleanly attribute:

- **Main effect of mixup**: `v5j_sydorskyi - v5j_control` and
  `effnetv2b0_sydorskyi - effnetv2b0_control`. Two estimates of the
  same effect; if they agree, the mixup lever is robust to backbone.
- **Main effect of backbone**: `effnetv2b0_control - v5j_control`. Tests
  whether Salman's backbone choice helps in our pipeline (with our recipe).
- **Interaction**: any non-additive component is the interaction term.

Phase B can run in parallel with Phase A or after, depending on GPU budget.
If Phase A produces a clean signal (either direction), Phase B is still
worth running for the backbone main-effect test.

### Implementation

`mixup_mode="average_max"` added to `src/birdclef/lightning/sed_finetune.py`
in this experiment's PR. The new mode:

```python
if self.hparams.mixup_mode == "average_max":
    if x.dim() != 2:
        raise ValueError("requires waveform domain (2D); set raw_waveform_mixup: true")
    x = (x + x[indices]) / 2
    x = x / (x.abs().amax(dim=-1, keepdim=True) + 1e-6)  # peak-normalize per sample
    labels = torch.maximum(labels, labels[indices])
```

This matches Sydorskyi's `code_base/datasets/wave_dataset.py` lines 762-820
when `alpha=None`.

### Variables

- **Independent** (Phase A): mixup recipe (control vs sydorskyi)
- **Independent** (Phase B): backbone (HGNetV2-B0 vs EfficientNetV2-B0) ×
  mixup (control vs sydorskyi)
- **Dependent**: Public LB (row-wise macro-averaged ROC-AUC)
- **Controlled** (held identical across all 4 variants):
  - LSEHead, BCE loss, OneCycleLR, 20 epochs, BS=64
  - mel pipeline (NNAudio + AmplitudeToDB + standardize + min-max)
  - Inference postproc (TTA + smoothing + TopN + priors)
  - Data split (seed=42, 90/10 + 8 SS held out)

### Reproducibility check

`v5j_control` is also a reproducibility test: it should reproduce v5j
(LB 0.906, exp 019) ± seed noise. If `v5j_control` lands materially below
0.906, we have a bigger reproducibility problem to investigate before
trusting the comparison.

Expected `v5j_control` outcomes:
- LB 0.901-0.911: within seed noise (Mahog reports ±0.005 at this level)
- LB 0.895-0.900: borderline, would want to investigate
- LB <0.895: reproducibility failure, halt and debug

## Open scoping decisions (locked)

| Question | Decision | Rationale |
|---|---|---|
| Run all 4 variants or just Phase A first? | All 4, parallelizable | Phase B is committed, not optional. 4 hours per run on rtx6000; can run 2 in parallel given GPU availability. |
| Same seed across all 4? | Yes (seed=42 in config) | Want the data split, mixup pair selection, etc. to be deterministic across variants. |
| Same hidden_dim for v2-B0 head? | Yes (1024) | Single-variable comparison vs v5j. Salman uses 512, but we'd need to also test his other changes for that to be meaningful. |
| EffNetV2-B0 LR same as HGNetV2-B0? | Yes (5e-4) | Single-variable. EffNetV2-B0 may want different LR; if Phase B regresses we'll flag this as a possible confound. |
| Inference: same as v5j (TTA + smoothing + TopN + priors)? | Yes | Inference is downstream of training; we want training-side effect only. |
| Train all variants on a single fold? | Yes | 4-fold ensemble is a downstream concern. Single-fold is the cleanest comparison. |

## Evaluation

### Metrics

| Metric | Description |
|---|---|
| **Public LB** | **Primary** — single-fold submission per variant |
| `val/auroc` | Checkpoint selection signal (we know it's anti-correlated with LB; included for W&B comparison) |
| `val/loss` | W&B convergence comparison |
| Loss curves on W&B | Visual comparison (user explicitly requested this) |

### Decision matrix

| `v5j_control` LB | `v5j_sydorskyi` LB | Verdict |
|---|---|---|
| ≥ 0.901 | ≥ 0.911 | **Strong confirm**: Sydorskyi mixup recipe is a real lever on v5j (+0.005 or more vs control). Endorse for follow-up (e.g., add to 4-fold v5j-pp2 stack). |
| ≥ 0.901 | 0.901-0.911 | **Within seed noise**: not a clean signal. Possibly small positive effect; would want repeat with different seeds to confirm. |
| ≥ 0.901 | ≤ 0.895 | **Reject**: Sydorskyi mixup regresses on v5j. Target-distribution lever isn't load-bearing for our pipeline. |
| 0.895-0.900 | (any) | **Reproducibility issue**: v5j_control didn't reproduce v5j 0.906 within seed noise. Halt comparison, investigate before drawing mixup-lever conclusions. |
| < 0.895 | (any) | **Reproducibility failure**: as above, blocking. |

### Phase B verdict matrix

Read after Phase A is complete:

| Pattern | Reading |
|---|---|
| Both `effnetv2b0_*` align with `v5j_*` (mixup effect same direction & magnitude) | Mixup lever is backbone-agnostic. Sydorskyi finding generalizes. |
| `effnetv2b0_*` flat to v5j (backbone has no main effect) | EffNetV2-B0 isn't a backbone lever in our recipe. |
| `effnetv2b0_*` beats v5j by ≥0.005 in both mixup conditions | Backbone is a real lever. Possibly worth a deeper Salman replication. |
| Strong interaction (mixup helps on EffNetV2-B0 but not HGNet, or vice versa) | Interesting — would need follow-up to understand. |

## Limitations

- **23rd training-recipe change in this competition.** Strong prior:
  recipes don't help LB. The audit + simulation give us the cleanest
  mechanistic argument we've had, but mechanistic arguments have failed
  before.
- **Single fold per variant.** Adds variance. If a result is borderline
  we can extend to 4-fold but that costs 4× more.
- **Offline metrics still anti-correlated with LB**. `val/auroc`
  comparisons on W&B are for *convergence behavior* and *training
  dynamics*, not for predicting LB.
- **Inference is unchanged**. We don't test e.g. Sydorskyi's TopN N=1 PP
  here. That's a separate experiment.
- **Phase B EffNetV2-B0 has many possible confounds**. Different
  pretraining (.in1k vs HGNet's .ssld), different feature dim (1280 vs
  1024), different GPU memory profile. If Phase B regresses, the
  attribution to backbone vs. these confounds is muddier than Phase A.

## Related work / potential follow-ups

If Phase A confirms (Sydorskyi mixup helps):
- 4-fold version of `v5j_sydorskyi` for ensemble + pp2 (matches v5j-4fold-pp2 stack)
- Stack with other Sydorskyi components (BG noise, SpecAugment) one at a time

If Phase A rejects (Sydorskyi mixup hurts or no effect):
- The information-side lever (pseudo-distillation on 2026 unlabeled SS
  using our 0.929 ensemble as teacher) becomes the next experiment.
- Re-examine the audit's other dimensions (probability 0.5 vs 1.0,
  warmup 0 vs 5) as potential isolated changes.

If Phase B confirms (EffNetV2-B0 helps):
- Stack with Sydorskyi mixup + 4-fold + pp2.
- Consider deeper Salman replication (CE loss, inference recipe).

## References

- Sydorskyi 2025 reference doc:
  `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md`
- Mixup audit: `user/murilo/analysis/notes/2026-04-26-mixup-implementation-audit.md`
- Mixup target simulation: `user/murilo/analysis/notes/2026-04-26-mixup-target-simulation-results.md`
- Sydorskyi-vs-v5j comparison: `user/murilo/analysis/notes/2026-04-26-sydorskyi-2025-vs-v5j-comparison.md`
- D1 verdict: `user/murilo/analysis/notes/2026-04-26-d1-train-side-attention-verdict.md`
- D1 follow-up discussion: `user/murilo/analysis/notes/2026-04-26-d1-followup-discussion.md`
- Salman's posts on EffNetV2-B0 (LB 0.937 single):
  `user/murilo/notes/kaggle-discussion-best-single-model-lb.md`
- timm model card: https://huggingface.co/timm/tf_efficientnetv2_b0.in1k
