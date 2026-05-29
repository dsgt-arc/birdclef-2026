---
date: 2026-05-29
author: Murilo Gustineli (analysis by Claude)
type: diagnostic
scripts: user/murilo/analysis/cv_trust/{build_oof,score_oof,diagnose_oof}.py
data: user/murilo/analysis/data/cv_trust/
---

# Why we're stuck at 0.906/0.929: a measurement failure + a clean blind spot

Read-only diagnostic (no retraining) built from materialized OOF soundscape predictions
(v5j-4fold, distilled-sed-kd-5fold, Perch-234) + the 739 labeled-soundscape windows.
Goal: stop running blind experiments and find where we actually fail.

## TL;DR

1. **Our offline metric cannot rank similar models — we've been optimizing noise.** The
   training `val/auroc` (single CV fold, ~156 windows, ~29 classes) is a seed-luck estimator.
   Even a *competition-faithful full-corpus OOF* can't separate our two best SEDs: bootstrap
   95% CIs **fully overlap** (v5j 0.797 [0.756, 0.839] vs distilled 0.796 [0.750, 0.849]).
   Their LB gap is 0.002. **No offline metric on the labeled soundscapes resolves sub-0.003
   gaps** → the ~30-experiment plateau is the expected result of steering by noise.
   ([[feedback_offline_metrics_unreliable]] now has a mechanism.)

2. **Our strongest "offline" signal was leakage.** `perch_234_probs` scores 0.995 OOF but its
   probes were trained on the labeled-soundscape labels (it scores 1.00 on insect sonotypes
   that exist *only* in the labeled soundscapes). The clean Perch LB is 0.893 (probe) / 0.929
   (ensemble). Prior analysis treated 0.995 as a real "Perch dominates" oracle — it isn't.

3. **The one clean, real failure we found:** our SED has a **catastrophic blind spot on the
   28/75 present species that have ZERO focal training data** — 25 insect "sonotypes" + 3
   amphibians. distilled mean AUC **0.601** there (11/28 *worse than random*), vs **0.860** on
   the 47 focal species. It is **not a smooth long-tail** (focal AUC doesn't track recording
   count, Spearman 0.15 n.s.) — it's a **binary cliff**: has-focal-data vs not.

4. **Why the ensemble (0.929) beats SED-alone (0.915):** Perch covers exactly that
   soundscape-only blind spot; the +0.014 LB is real generalization even though the offline
   number is leaked. v5j and distilled are *complementary* (v5j better on soundscape-only
   0.764 vs 0.601; distilled better on focal 0.860 vs 0.807) — which is why they tie on LB.

## The numbers

Competition skip-rule macro-AUC over the full 739-window labeled corpus (V1):

| Model | single-fold V0 (the metric we used) | full-corpus OOF V1 | LB |
|---|---:|---:|---:|
| v5j-4fold | 0.815 | 0.791 | ~0.913 |
| distilled-5fold | **0.912** | 0.763 | ~0.915 |
| Perch-234 (LEAKED) | — | 0.995 | 0.893 probe / 0.929 ens |

- **V0 (single-fold) is what we trained on** — it said distilled (0.912) ≫ v5j (0.815), a
  0.097 gap, on non-comparable fold sizes (distilled fold-0 = 14 files, v5j = 55). The true LB
  gap is 0.002. V0 exaggerated ~50× and is dominated by which species landed in one fold.
- **V1 (full-corpus OOF) can't even hold the sign** (v5j > distilled, opposite of LB) — and
  the bootstrap shows that's noise (CIs overlap).
- Variants V2 (present-only), V3 (non-S22), V4 (≥5-pos), V5 (per-site) all agree: SEDs
  indistinguishable; the only large, stable separation is Perch (leaked).

Per-class diagnosis (`per_class_auc.csv`): the entire top of "where Perch beats distilled" is
insect sonotypes (distilled 0.08–0.19, i.e. anti-correlated) + amphibians, all train_count=0.
Gap by taxonomic class: **Insecta 0.40**, Aves 0.16, Amphibia 0.15, Mammalia 0.07.

## What this means (and what it doesn't)

- **The plateau is a measurement failure, not (only) a modeling one.** We could not tell good
  changes from bad — most of the 30 experiments were inside the offline noise band, so the
  signal we acted on was fold-luck. This is the single biggest process problem.
- **We have no clean offline CV for the SEDs**, and the labeled soundscapes (66 files, 4.22
  labels/window with incomplete annotation, 40/66 = S22) are too thin/noisy to build one. The
  SED scores 0.76 offline but 0.915 LB — the labeled set *under-rates* SEDs (label
  incompleteness) and is insect-heavy relative to the (Aves-heavier) hidden test, so even a
  faithful OOF is a biased absolute proxy.
- **Caveat on the blind spot's LB leverage:** because the labeled SS are insect-heavy and the
  hidden test scores us at 0.915 (> our 0.76 offline), the insect cliff likely matters *less*
  on LB than offline. Don't over-bet on "fix insects → big LB jump." But it IS the clean
  mechanism behind the ensemble's +0.014, and the only concrete failure we can point to.

## Competitor-consensus cross-check

- **CV is the hardest part for everyone** (yukiZ has a tracking CV; Anil's GKF-OOF=0.608 with
  LB 0.879; D.M.: "CV on clean clips can't correlate"). Our finding (labeled-SS CV is
  noise-limited for similar models) is consistent — nobody has a clean labeled-SS CV; the ones
  who iterate fast (yukiZ) validate differently (focal SKF).
- **Perch/ensemble complementarity is the consensus lever** (tennogh: Perch-embedding and SED
  models "aren't very correlated"; sghwr: extra Perch member ≈ +0.001 only once you already
  have one). Matches our data: the ensemble's lift is Perch covering the SED blind spot.

## Recommended direction (ranked by evidence, screenable honestly)

We cannot keep iterating blind. Two evidence-backed moves, in priority order:

1. **Stop blind SED-recipe experiments.** Confirmed: they live inside the offline noise band.
   This is the most important change — it reframes the whole effort.
2. **Bet on the ensemble — the one robustly LB-positive lever — and make it cleanly
   measurable.** Specifically a **properly OOF-cross-validated, broader-coverage Perch probe**:
   - It targets the clean failure (soundscape-only species Perch already rescues).
   - It is the **one thing we can validate offline without leakage** — we control its train/val
     split, so an honest OOF probe score is buildable (unlike SED-on-labeled-SS).
   - It's data-cheap: Perch embeddings for focal + soundscapes are already materialized
     (`shared/inference/.../perch-v2-tf/`).
   - It composes with the existing 0.929 ensemble + the known PP lever (+0.004–0.014).
   - Honest risk: the current ensemble already uses a Perch probe, so headroom is uncertain;
     and a broader probe must be validated OOF, not on the leaked full set.
3. **Secure the floor:** confirm the full PP package (site/hour priors + 5-tap Gaussian smooth)
   is applied to the 0.929 ensemble — a known, composable, LB-positive lever; cheap insurance
   for the final picks.

Explicitly **de-prioritized** by this analysis: another SED backbone/head/loss/mixup sweep
(noise-band), and offline-metric-chasing on the labeled soundscapes (biased + leaked).

## Artifacts

`user/murilo/analysis/data/cv_trust/`: `oof_{v5j,distilled,perch}.parquet`, `gt_Y.npy`,
`gt_windows.parquet`, `cv_scores.csv`, `cv_lb_correlation.csv`, `bootstrap_ci.csv`,
`per_class_auc.csv`, `per_site_auc.csv`, `oof_fold_maps.json`.
Scripts: `user/murilo/analysis/cv_trust/{build_oof,score_oof,diagnose_oof}.py` (read-only).
Leak-check passed for both SED OOF reconstructions (OOF 0.76–0.79 ≪ leaked 0.999).
