# Results — Exp 035: OOF-cross-validated Perch probe

**Status**: completed — **claim 1 CONFIRMED; claim 2 not pursued.**
**Date**: 2026-05-29
**Author**: Murilo Gustineli

## Outcome

**Claim 1 (leakage-free offline metric): CONFIRMED.** Built a proper per-fold OOF Perch→234
probe (`build_oof_probe.py`) using the CV-trust by-file fold map so it aligns with the SED OOF.
Honest macro-AUC over the 739 labeled-soundscape windows (skip-rule):

| Metric | macro-AUC |
|---|---:|
| base-only (raw Perch taxonomic map) | 0.746 |
| **OOF probe (leakage-free)** | **0.853** |
| leaked probe (fit-all / predict-all) | 0.948 |
| deployed `perch_234_probs` (leaked) | 0.995 |
| SED OOF (distilled / v5j) | 0.763 / 0.791 |

Leak-check passes decisively (0.853 ≪ 0.948). The probe adds a **genuine +0.107** over base and
**honestly beats the SED** (0.76). Per-group: it **rescues the soundscape-only blind spot for
real** (insects/amphibians 0.531 base → **0.740** OOF; capped by the ~12–24-window ceiling) and
leads on focal species (0.920 vs SED 0.860). So the 0.995 was leakage; the probe's true
contribution is ~0.853, which is also a far better estimate of its LB contribution.

## Why claim 2 (LB lift from a tuned probe) was NOT pursued

1. **The honest metric under-rates the SED → it cannot tune the ensemble.** The honest
   probe⊕SED blend scores **0.77–0.78, below probe-alone (0.853)** — it says "drop the SED,"
   which **contradicts the LB** (ensemble 0.929 > probe/SED alone). The labeled-SS bias
   ([[project_cv_measurement_failure]]) persists: label-incompleteness depresses the SED
   (0.76 offline → 0.915 LB). So the OOF is a *probe screen*, not an ensemble selector.
2. **Independent confirmation (Antoine Masq, 14th):** even his more careful CV (2 sites fully
   held out + per-species-balanced file folds) is *"very noisy … can roughly indicate whether an
   idea works but I cannot select my models based on it."* The CV ceiling is **fundamental to this
   competition**, not a gap in our setup — so a leakage-free metric will only ever be a rough screen.
3. **The probe's honest headroom is small.** It already rescues insects to the data ceiling (0.74)
   and covers focal (0.92); the deployed 0.929 ensemble already captures this. Pushing the honest
   OOF higher risks overfitting the 739 labeled windows — the exact trap we flagged.

## Deliverables (the real, kept value)

- **A leakage-free OOF probe metric** (rough screen we control the splits for): `build_oof_probe.py`
  → `analysis/data/cv_trust/oof_probe.parquet`.
- **The leakage exposed**: `perch_234_probs` 0.995 is the probe evaluated on its own training
  labels ([[project_perch_probes_leaked_offline]]); true contribution ≈ 0.853.
- **The probe's contribution quantified**: insects 0.74 (capped), focal 0.92.

## Verdict + next

Claim 1 delivered a reusable instrument + corrected a leaked number we'd been trusting. Claim 2's
upside is small and unvalidatable offline — not worth burning LB submissions on probe tuning.
**Pivot to exp 036**: a strong single model via the in-domain (XC) backbone + a faithful,
co-tuned fine-tune recipe — the one evidence-backed path to >0.93 single (Antoine 0.936 supervised;
Ali Ozan 0.941 on the same yoinked backbone). The probe screen here can rough-check probe members
for the final ensemble.
