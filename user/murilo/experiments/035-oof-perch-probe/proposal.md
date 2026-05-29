# Proposal: OOF-cross-validated Perch probe — a trustworthy CV + a stronger ensemble member

**Status**: proposed
**Created**: 2026-05-29
**Author**: Murilo Gustineli

## Hypothesis

Two falsifiable claims, in priority order:

1. **(Measurement — high confidence)** A properly out-of-fold (OOF) cross-validated Perch→234
   probe yields a **leakage-free offline macro-AUC** that ranks *probe/ensemble configs
   self-consistently* — the first offline signal in this project we can tune against without
   spending an LB submission. (Today's `perch_234_probs` 0.995 is leakage; the SED labeled-SS
   OOF is too noisy to rank similar models — see `analysis/notes/2026-05-29-cv-trust-and-failure-localization.md`.)

2. **(LB — the real bet)** A probe re-selected against that honest metric — extended coverage
   to more soundscape-only species + retuned alpha/threshold/training-data — strengthens the
   Perch member enough to lift the **Perch ⊕ SED ensemble above its current 0.929 LB**.

## Background

The CV-trust diagnostic (2026-05-29) found the 0.906/0.929 plateau is largely a **measurement
failure**: our offline metric can't rank similar models (the two best SEDs are statistically
indistinguishable on the labeled-SS OOF; bootstrap CIs fully overlap), so ~30 experiments were
steered by noise. The one clean failure it localized: our SED scores ~random (0.601 AUC, 11/28
worse-than-random) on the **28/75 present species that have zero focal training data** (25 insect
"sonotypes" + 3 amphibians). The ensemble's +0.014 LB (0.915→0.929) is **Perch covering exactly
that blind spot**.

But the Perch probe we rely on (`apply_perch_probes.py` + exp 018) has a leakage flaw: its
final probes are `fit()` on **all 66 labeled-soundscape files** (`train_probes.py:475-484`; CV is
metrics-only), so its 0.995 offline is meaningless and we can't tune it honestly. And the 25
insect sonotypes are **unmappable from Perch's bird taxonomy** (fixed at −10 in the base map),
so the per-species probe is their *only* signal — making clean, well-tuned probes the highest-
leverage place to work. The probe is also the **one lever we can measure without leakage** (we
control its train/val split) and it **composes** with the existing ensemble + the known PP lever.

## Method

### Approach

Rebuild the Perch probe with proper K-fold OOF (matching the CV-trust fold map so it's consistent
with our trusted eval), use the resulting leakage-free OOF macro-AUC to tune probe coverage /
alpha / training-data / model, then deploy the best config and submit the rebuilt ensemble to LB.
Pure CPU on materialized Perch embeddings — no GPU, no SED retraining.

### Setup

| Component | Details |
|-----------|---------|
| Data | Perch v2 embeddings (1536-d) + logits already on disk: `shared/inference/birdclef-2026/perch-v2-tf/{train_audio,train_soundscapes}/`; GT `train_soundscapes_labels.csv` (739 windows, 75 species); `train.csv` (focal labels, 206 species) |
| Compute | CPU only (LogReg/small-MLP probes); minutes, not hours. PACE not a bottleneck |
| Dependencies | sklearn, pandas, joblib (existing); the CV-trust OOF artifacts (`analysis/data/cv_trust/`) for the trusted eval + SED OOF |
| Code | REUSE `user/murilo/scripts/apply_perch_probes.py` (142-feat builder, taxonomic base map, temporal smoothing), `experiments/018-perch-probes/scripts/train_probes.py` (probe fitting), `analysis/cv_trust/{build_oof,score_oof}.py` (fold map + skip-rule macro-AUC). New: `experiments/035-oof-perch-probe/` scripts |

### Procedure

1. **OOF probe** — K-fold by file (reuse `build_sc_folds` GroupKFold-5, the same map `build_oof.py`
   uses, so the probe OOF aligns with the trusted SED OOF). Per fold: fit scaler+PCA+probes+priors
   on train folds only, predict the held-out fold. Concatenate → leakage-free OOF over 739 windows.
   Leak-check: OOF macro-AUC must be far below the leaked 0.995.
2. **Tune against the honest OOF** (reuse `score_oof`'s skip-rule macro-AUC): sweep
   (a) probe coverage threshold (≥8 positives → lower, to cover more soundscape-only species),
   (b) alpha base/probe blend, (c) training data (SS-only vs **focal+SS** Perch embeddings — focal
   adds 35K leakage-free samples for the 206 focal species), (d) probe model (LogReg vs small MLP),
   (e) priors on/off. Select by honest OOF macro-AUC + per-class AUC on the soundscape-only species.
3. **Honest OOF ensemble** — blend OOF probe ⊕ the existing SED OOF (`oof_distilled.parquet` /
   `oof_v5j.parquet`); tune the blend weight against honest OOF macro-AUC. First leakage-free
   ensemble tuning we've had.
4. **Deploy + LB** — re-fit the chosen config on all labeled data (full SS [+ focal]) for the
   kernel (full-data deploy is fine: applied to the *hidden* test, no leak there); rebuild the
   exp 023 ensemble kernel with the new probe; submit. Keep the known PP lever on.
5. **Decision** — compare LB vs 0.929 AND check whether the honest OOF predicted the direction
   (validates the instrument for future use).

### Variables

- **Independent**: probe coverage threshold, alpha, probe training data (SS vs focal+SS), probe
  model, priors; ensemble blend weight.
- **Dependent**: leakage-free OOF macro-AUC (proxy) + per-class AUC on soundscape-only species; **LB (arbiter)**.
- **Controlled**: OOF fold structure (matches CV-trust), the SED member (best existing, fixed),
  Perch embeddings, PP.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Honest OOF macro-AUC | Leakage-free competition skip-rule macro-AUC over the 739 labeled windows — the new trustworthy proxy |
| Soundscape-only per-class AUC | Did extended coverage improve the 28 insect/amphibian species (the blind spot)? |
| LB | The arbiter — Perch ⊕ SED ensemble public score |

### Baseline

- Ensemble: **exp 023 Perch ⊕ SED = 0.929 LB** (the number to beat).
- Probe offline: leaked 0.995 (meaningless); honest OOF probe = TBD (expected far lower).
- SED OOF: 0.76 (distilled) / 0.79 (v5j); SED LB 0.915 / 0.913.

### Success Criteria

- **Confirm if**: rebuilt ensemble LB **> 0.929**; stretch — the honest OOF *predicted* the gain
  (instrument validated).
- **Partial-confirm if**: LB flat (~0.929) but we now have a **leakage-free OOF that ranks probe
  configs self-consistently** — a reusable trustworthy CV (claim 1), even without an LB gain.
- **Reject if**: the OOF probe can't be improved over the existing probe AND the honest OOF still
  fails to rank configs usefully → probe lever exhausted; fall back to securing the 0.929 + PP
  final picks.

## Limitations

- The honest OOF is still computed on the labeled soundscapes (insect-heavy vs the Aves-heavier
  hidden test, S22-dominated, label-incomplete) → self-consistent but a **biased absolute proxy**;
  it ranks probe variants *relatively*, and may not predict LB magnitude (few LB anchors). Per the
  diagnostic, the blind-spot's LB leverage is likely smaller than it looks offline — don't over-bet.
- Soundscape-only insects have only ~12–24 labeled windows total → even OOF probes are data-starved
  on them; coverage extension has a hard data ceiling.
- The ensemble is already at 0.929 — probe tuning alone may have little headroom.
- ⚠️ Do not let the honest OOF become the next thing we overfit: with few LB anchors, treat it as a
  relative screen, validate any "win" on the LB ([[feedback_offline_metrics_unreliable]],
  [[project_cv_measurement_failure]]).
