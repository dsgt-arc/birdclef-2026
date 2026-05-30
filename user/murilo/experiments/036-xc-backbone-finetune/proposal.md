# Proposal: Unlock the XC backbone — faithful fine-tune recipe for a >0.93 single model

**Status**: proposed
**Created**: 2026-05-29
**Author**: Murilo Gustineli

## Hypothesis

exp 034 proved the Sydorskyi EffNetV2-S backbone is **bit-identical** to source (Gate 1), yet our
fine-tune scored **0.879 — below our 0.906 ImageNet best**. The backbone is not the problem; the
**fine-tune is**. Two falsifiable sub-claims:

1. **(cheap, test first)** exp 034 under-trained the in-domain features via too-aggressive full-FT
   (50 ep @ lr 1e-4). A gentler schedule (lower / discriminative LR, gradual unfreeze) that
   *preserves* the XC features lifts LB over 0.879.
2. **(the main lever)** The one remaining divergence from the proven recipe is the **data/sampling
   pipeline** — exp 034 kept exp 032's datamodule (additive mixup + MixSamp, no class-balancing,
   no BG-noise). Swapping in Sydorskyi's faithful `WaveDataset` (class-balanced sampler + sum-clip
   mixup + BG-noise OneOf + RandomFiltering) unlocks the backbone toward the **0.92–0.94** that
   Antoine (0.936 supervised) and Ali Ozan (0.941, same yoinked backbone) reach.

**Target: a single supervised model > 0.906 (beat our best); stretch > 0.93.** Then PL (+~0.007) + PP.

## Background

- Our **0.906 is the ImageNet-backbone ceiling** — 30 experiments couldn't pass it. Every disclosed
  >0.93 single model uses an in-domain XC backbone (Antoine 0.889→0.936 from XC pretrain; Ali Ozan
  0.941). The XC backbone has ~0.03–0.05 more headroom *if fine-tuned right*.
- exp 034 made the **model** faithful (mel + backbone + AttHead bit-identical) but kept **our data
  pipeline** → 0.879. So this is a clean experiment with **one load-bearing divergence left: the
  data/sampling recipe** — and class-balanced sampling directly targets the long-tail / soundscape-only
  failure the cv_trust diagnostic localized ([[project_cv_measurement_failure]]).
- **No trustworthy CV** (us + Antoine, 14th, both confirm it's only a rough screen). So this is
  **LB-arbitrated** with a few high-conviction A/Bs — not offline-tuned, not a 30-cell sweep.

## Method

### Phase A — LR-sanity (cheap, first; sub-claim 1)

Hold model + data pipeline = exp 034 (the known-0.879 config). Vary **only** the fine-tune schedule,
fold 0, trained in parallel (PACE is not the bottleneck — [[feedback_pace_compute_not_bottleneck]]):
- **A0** exp 034 fullft (0.879) — reference, already have it.
- **A1** lower LR (5e-5).
- **A2** discriminative / layer-wise LR decay (backbone ~1e-5, head ~1e-3) — preserves in-domain features.
- **A3** gradual unfreeze (freeze backbone N epochs → unfreeze) — protects features early.

Rough-screen with the full metric suite + cv_trust honest macro-AUC (noisy → screen only); **submit
the top 1–2 to LB vs 0.879.** Outcome: if a gentler schedule clears ~0.90, 0.879 was a fine-tune
issue (cheap win, carry the winner into Phase B). If flat, the lever is the data pipeline → Phase B.

### Phase B — faithful Sydorskyi data pipeline (the main lever; sub-claim 2)

On the best Phase-A schedule + the (already-faithful) model, replace exp 032's datamodule with
Sydorskyi's **vendored `WaveDataset`** (`vendor/sydorskyi-2025/code_base/datasets/wave_dataset.py`)
adapted to 2026 data — class-balanced `WeightedRandomSampler` + sum-clip MixUp + BG-noise OneOf
(nocall-SS | ESC-50) + RandomFiltering. Reuses our already-built+tested components (exp 034 B.7/B.8
sampler/SpecAug/RandomFiltering). **LB A/B vs 0.879 + Phase-A best.** This is the documented
fidelity-escalation path (exp 034 §B.8.3 "Option C").

### Phase C — PL + PP (gated on supervised ≥ ~0.92)

Only if Phase A/B clears ~0.92: add the **real** Sydorskyi PL protocol (the one the exp-033 divergence
audit showed we never ran: sum-clip PL×focal fusion + trim<0.1 + chunk-discard<0.5 + 40% class-matched
swap) for the disclosed +~0.007, then PP (smoothing already +0.014; TTA untried). LB-arbitrated.

### Logging (all previous-experiment metrics)

Wire the **full exp 032/033 metric suite** that exp 034 omitted, on **both W&B and TensorBoard**:
`val/auroc` (TuckerMacroAUC) + `val/cmap` (TuckerMacroAP) + f1_macro/micro + top1/top5 + train-side
auroc/loss + lr + **logit-std** (hengck23 overfit proxy). Logging-only — does not affect training;
gives the diagnostics to pick the SWA window + watch overfitting.

### Checkpointing (chosen for an untrustworthy CV)

- `ModelCheckpoint(save_top_k=-1, monitor="val/auroc", save_last=True)` — save **every epoch** (true
  last + best-by-val identifiable + SWA-able; avoids the exp-034 save_last pitfall
  [[feedback_save_top_k_does_not_preserve_last]]).
- Derive 3 candidates post-hoc: **(a) last**, **(b) best-by-val/auroc**, **(c) SWA over the last K
  epochs** (Sydorskyi's lever; selection-free → robust to the noisy CV).
- **LB-arbitrate, leading with SWA**; A/B last vs SWA if budget allows. We bet on a robust *average*,
  not a noisy *pick*.

### Variables

- **Independent**: fine-tune schedule (Phase A); data/sampling pipeline (Phase B); PL/PP (Phase C).
- **Dependent**: **LB** (arbiter); cv_trust honest macro-AUC + full suite (rough screen only).
- **Controlled**: the faithful model (exp 034 mel+backbone+AttHead), the XC backbone, fold 0 for A/B.

## Evaluation

| Metric | Role |
|---|---|
| **LB** | the arbiter (offline can't rank — us + Antoine) |
| cv_trust honest macro-AUC + full suite | rough screen + overfit watch + SWA-window selection |

### Baseline
- exp 034 XC full-FT + PP = **0.879** (the config to beat).
- Our overall single-model best = **0.906** (ImageNet + KD). Beating it = real progress.
- Targets: Antoine 0.936 supervised; Ali Ozan 0.941 (same backbone) → stretch >0.93.

### Success Criteria
- **Confirm if**: a Phase-A or Phase-B config > **0.906** on LB (we beat our ImageNet ceiling on the
  XC backbone); strong confirm > 0.93.
- **Partial**: > 0.879 but ≤ 0.906 — the backbone fine-tune improved but didn't pass our best.
- **Reject if**: nothing clears 0.879 after the LR fix + faithful data pipeline → the XC backbone
  doesn't transfer for us; fall back to securing the 0.929 ensemble + PP/TTA final picks.

## Limitations
- **No trustworthy CV** → LB-arbitrated, conviction-driven, few A/Bs (not 30 tweaks).
- Sydorskyi's *published supervised* recipe was 0.881 in 2025 (pre-PL, different comp); the 0.94
  singles (Antoine/Ali Ozan) use *undisclosed* recipes + PL. Porting his data pipeline is the
  best-documented attempt, **not a guarantee** of Antoine's 0.936.
- Adapting his stateful `WaveDataset` to 2026 data is real port surface (mitigated: reuse his
  vendored code directly + our already-tested components).
- Don't over-fit the rough offline screen — validate every apparent win on the LB
  ([[feedback_offline_metrics_unreliable]]).
