---
experiment: "037-noisy-student-pl"
status: proposed
date: 2026-05-30
author: Murilo Gustineli
branch: mg/037-noisy-student-pl
depends_on: ["032-tucker-distilled-sed", "023-perch-sed-ensemble", "033-own-teacher-pl"]
---

# Proposal — Multi-Iterative Noisy-Student Soft Pseudo-Labeling

## 1. Overview

Break the **0.929 LB plateau** (exp 023 Perch-MLP + SED v5j ensemble) before the **2026-06-03**
deadline by distilling the ensemble's *soundscape-domain* knowledge into a single student via
**multi-iterative Noisy-Student pseudo-labeling** — the one documented lever that moved every 2025
top solution across the 0.90→0.93 band (winner: 0.872 → 0.930 via PL).

The student is `distilled-sed-kd` (exp 032, **EfficientNet-B0** `tf_efficientnet_b0.ns_jft_in1k`
SED + KD-from-Perch, **LB 0.906 single / 0.915 5-fold**) — our best reproducible single model and
strongest PL starting point. (NOT the HGNet `distilled_sed_kd_hgnet.yaml` variant; and distinct
from the teacher's SED, which IS HGNet-B0 v5j.) The teacher is the
**0.929 ensemble** (Perch-MLP + SED v5j-4fold = HGNet-B0), whose per-window predictions over all 10,658
`train_soundscapes` are **already cached on disk** (`perch-v2-tf/.../perch_234_probs.parquet` +
`hgnet-b0-sed-v5j-4fold/.../fold{0..3}/logits.parquet`). PL generation is therefore a pure
blend+transform — no new model inference.

## 2. Why our 4 prior PL attempts failed — and what's different

| Exp | Teacher | Transform | Result |
|---|---|---|---|
| 015 | own student | T=0.6 (**sharpen**) | dead |
| 025 | Perch Tier-C (6% prec.) | hard rows + loss_weight=0.5 | −0.009 |
| 033 | **own SED** (0.906/0.915) | **T=0.6 + P^1.5 (sharpen)** + 0.5-discard/0.1-trim filters | REJECT (−0.007…−0.027) |

All four **sharpened** the labels (the documented *overconfidence* failure mode), used an
**own/weak teacher** (confirmation bias), and **never reached round 2**. exp 033's own post-mortem
blamed sharpening + over-filtering destroying the soft long-tail.

**This experiment inverts every knob:** (1) **strong, diverse teacher** = the 0.929 ensemble (carries
Perch foundation-model soundscape knowledge the SED-only student lacks); (2) **no sharpening / no
aggressive filtering** — soften or leave raw; (3) **~50/50 GT:PL** with **loss_weight=1.0** (exp 025's
0.5 was a confound); (4) **multiple rounds**, LB-gated.

## 3. Hypothesis (falsifiable)

A `distilled-sed-kd` student retrained on a ~50/50 mix of ground-truth + **soft, un-sharpened**
pseudo-labels from the 0.929 ensemble teacher will beat its 0.906/0.915 baselines, and blend into the
0.929 ensemble for a new best.

| Bucket | LB (single-fold vs 0.906) | Decision |
|---|---|---|
| **CONFIRM** | ≥ 0.916 (+0.010) | PL transfers → 5-fold + round 2 + ensemble blend |
| **PARTIAL** | 0.909–0.915 | weak lift → 5-fold, tune γ/mix, blend test |
| **WEAK** | 0.907–0.908 | within noise → one transform retry then stop |
| **REJECT** | ≤ 0.906 | PL does not transfer even done-right → secure 0.929 + PP/TTA picks |

LB is the arbiter (offline labeled-SS is a rough screen only — [[project_cv_measurement_failure]],
[[feedback_offline_metrics_unreliable]]).

## 4. Method

**Teacher = the 0.929 ensemble (cached, TF-Perch quality 0.925, even stronger than the deployed ONNX).**

- **Phase 0** — leakage-free labeled-SS val harness (`scripts/eval_labeled_ss.py`) reusing the
  `analysis/data/cv_trust/` OOF scaffold (exp 035) + `macro_auc_skip_empty`; record the teacher's
  held-out macro-AUC as the reference bar.
- **Phase 1** — `scripts/generate_ensemble_pl.py`: prob-space blend `0.5·perch_234_probs +
  0.5·sigmoid(mean_folds(sed_logits))`, transform, write parquet `(filename, chunk_idx, +234 soft-prob
  cols)` over all 10,658 files. **No filtering.** Variants `raw` (T=1, γ=1) + `t2` (T=2,
  temperature-soften). **g05 (γ=0.5) DROPPED** — empirically over-inflates our already-sparse blend
  (frac>0.5: 0.95% → 56%); power-softening assumes a *peaked* teacher, ours isn't → temperature is the
  principled soften. Generator GUARDS against sharpening (T≥1, γ≤1 only).
- **Phase 2** — round-1 **ablation factorial** (15 single-fold cells; PACE compute is not the
  bottleneck — [[feedback_pace_compute_not_bottleneck]]). Built on exp 032/033 wiring
  (`DistilledSEDFinetuneModule` soft-BCE / SoftAUC, `DistilledSEDDataModule` `pseudo_label_*`, `PLDS`,
  `MixSamp` 50/50). Axes:
  - **KD routing** {A on/detach (PL→head), B off/e2e (PL→whole student), C on/no-detach (PL→both)} —
    the structural lever exp 033 never tested (it was A only).
  - **PL transform** {raw, t2}; **PL aug** {False control, True noise-the-student}.
  - → A/B/C × {raw,t2} × {augF,augT} = **12 soft-BCE** + **3 SoftAUC** (raw·augT per KD) = **15**.
  - **Budget:** 25 ep (= 0.906 baseline) · `steps_per_epoch=500` (GT-only — excludes the 127K PL pool so
    it's not a 5× balloon) · `save_top_k=-1` → best-checkpoint; extend B/C per-cell if val still climbing.
  - **LB screening (LB-limited + offline can't rank):** train all 15, use offline OOF only to *eliminate
    broken cells + read per-axis direction*, then LB **~5 mechanism-diverse** cells (best per KD regime +
    best SoftAUC + faithful reference). 5-fold + ensemble-blend the LB winner.
- **Phase 3** — round 2 (only if round 1 LB-green): promote round-1 student into the teacher,
  regenerate softened PLs, retrain, LB. **Stop if ensemble gain < +0.005.**
- **Phase 4** — final picks: #1 (safe) = reproduced 0.929; #2 (upside) = best PL-student blend.

**Explicitly AVOID (exp 033 failure modes):** sharpening (T<1 / γ>1, generator-guarded); chunk-discard
0.5 + trim 0.1 filters; own-teacher (SED-only) PLs; loss_weight=0.5.

**New code (flag-gated, defaults = exp 032/033 behavior, all unit-tested):** `kd_detach_head` (cell C),
`loss_type`/`SoftAUCLoss` (+ `train/cls_loss` component log), `pseudo_label_aug` (aug control),
`train/logit_std` (overfit proxy), GT-only `steps_per_epoch` (via config). Full metric suite
(cmap/auroc/f1/top-k/loss + n_evaluable + LR + logit_std) logs to **W&B and TensorBoard**.

## 5. Reuse

`DistilledSEDFinetuneModule`, `DistilledSEDDataModule` (`pseudo_label_path` + `sampler=mixsamp`),
`PLDS`/`GTDatasetWrapper`/`MixSamp`/`ClassBalancedMixSamp`, exp 033 `train.py`/`_common.yaml`,
cached teacher parquets, exp 035 `cv_trust/` eval scaffold + `macro_auc`.

## 6. Provenance / references

- `user/murilo/notes/2026-05-30-deep-research-noisy-student-pseudo-labeling.md` (the recipe)
- `user/murilo/notes/2026-05-30-deep-research-xeno-canto-pretraining.md` (parallel/post-deadline track)
- Babych 2025 (1st, "Multi-Iterative Noisy Student"); Sydorskyy et al. CEUR Vol-4038 (2nd; soft +
  power-scaling + 50/50). exp 033 results (failure post-mortem). exp 023 (0.929 teacher).

## 7. Out of scope (post-deadline paper track — run in parallel, compute is not the bottleneck)

Rounds 3–4; broad XC-pretrained student (deep-research #2); SoftAUC loss (not in repo; soft-BCE is the
deadline path); SED frame-refinement of PLs; leakage-free OOF Perch probe as a cleaner teacher;
2nd diverse student backbone.
