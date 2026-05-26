---
experiment: "033-own-teacher-pl"
status: draft
date: 2026-05-25
author: Murilo Gustineli
---

# Proposal: Own-teacher pseudo-labels on distilled-sed-kd

**Status**: draft (2026-05-25)
**Created**: 2026-05-25
**Author**: Murilo Gustineli

## Provenance

This experiment implements **own-teacher pseudo-labeling** — using a strong SED model's predictions on unlabeled soundscapes as additional training supervision, treated as 1:1 equivalent to ground-truth labels. The technique is classical semi-supervised learning (noisy student, self-training) and is the modal recipe at the top of the BirdCLEF 2026 LB. Five independent competitor disclosures converge on this lever:

- **Antoine Masq (49th, LB 0.943)** — "I just incorporated [PLs] by using the unlabeled data just as the labeled one but the labels are predicted by the teacher models." Reports +0.007 LB from teacher iteration (0.936 → 0.943) on top of XC pretrain.
- **OverfitOracle (685th, LB 0.946)** — SED + teacher-generated pseudo frame/event labels on XC + competition data.
- **Boredom (5th, LB 0.938+ no PL, 0.950+ with 1-iter PL)** — confirms +0.012 LB from a single PL iteration.
- **chenwenqiang_001** — independently reports +0.013 LB from own-teacher PL on a 0.886 base.
- **Tucker Arrants** — references the same iterative pattern in his distilled-SED discussion.

The protocol's name is generic in the literature; we drop competitor-specific naming from artifacts. Author attribution is preserved here in references, not in directory or branch names.

Cited materials (vendored / captured):

| Doc | Content |
|---|---|
| `notes/kaggle-discussion-non-perch-limit.md` | Antoine + OverfitOracle + Arunodhayan + Boredom disclosures |
| `notes/kaggle-discussion-best-single-model-lb.md` | Cross-reference for the same recipe pattern |
| `notes/2026-05-22-strategic-direction-tracks-A-B.md` | Strategic context: this is "Track B (own-teacher PL)" on top of "Track A (Tucker)" |
| `experiments/025-iter-pseudo-labeling/results.md` | Prior PL attempt (REJECTED, but for a different protocol) |
| `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` | PL generation script — reused for T2 parquet (Step 0.3) |
| `experiments/032-tucker-distilled-sed/results.md` | Phase D distilled-sed-kd (LB 0.906) — the base student |
| `experiments/032-tucker-distilled-sed/configs/distilled_sed_kd.yaml` | Tucker recipe + KD config — basis for exp 033 configs |
| `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` | Tucker code audit; recipe foundation |
| `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet` | **T1**: pre-generated PL parquet, v5j-4fold teacher (ready to use; pre-flight 7/7 PASS) |
| `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` | **T2**: PL parquet to be generated in Step 0 (distilled-sed-kd-5fold teacher) |
| `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/` | T1 cached inference (already exists) |
| `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/` | T2 cached inference to be generated in Step 0 |
| `experiments/010-submission-pipeline/artifacts/weights/distilled-sed-kd-5fold/model_fold{0..4}.onnx` | T2 ONNX weights for the inference caching script |

## Hypothesis

**1:1 own-teacher PL on the `distilled-sed-kd` recipe lifts single-fold LB from 0.906 to ≥ 0.920.** Treating unlabeled soundscape teacher predictions as 1:1 equivalent training labels (no down-weighting, no waveform mixing) engages the lever that lifted Antoine 0.936 → 0.943, Boredom +0.012, and chenwenqiang_001 +0.013.

Mechanism: our prior PL attempts (exp 014/015/025) all used **down-weighted PLs** (loss_weight=0.5 in Phase B) or **waveform-mixing** (Phase C). Both regressed LB (−0.009 / −0.012). The five competitor disclosures uniformly use 1:1 weighting. This experiment isolates the **weighting + sampler protocol** as the variable, and additionally tests **teacher architecture diversity** (v5j-4fold vs `distilled-sed-kd-5fold`) as an orthogonal axis.

The two teachers are ~equal in LB quality (Δ = +0.002, within seed noise) but architecturally different (HGNet+LSEHead+BCE vs EffNet+AttHeadLogitSpace+KD). This is NOT a teacher-strength test — it's a **teacher-diversity test**: do PLs from a structurally different but equivalent-quality teacher produce a different student?

### Falsifiable success matrix

Read against the paired **same-codebase baseline** (`distilled-sed-kd` Phase D fold 0 = LB 0.906, reproducible at HEAD per the bit-deterministic Phase D retrain). Per the exp 031 drift investigation: cross-run LB has ~0.007 hardware variance, so paired-within-codebase is the only reliable methodology.

| Bucket | Single-fold LB lift over 0.906 | Reading |
|---|---|---|
| **CONFIRM** | ≥ +0.014 | Antoine-protocol PL transfers; explore Round 2 + Sydorskyi-yoink |
| **PARTIAL** | +0.007 to +0.013 | Smaller than Antoine's +0.007-0.013; still useful for ensemble |
| **WEAK** | +0.001 to +0.006 | Inside seed band; needs 5-fold + LB resubmission to disambiguate |
| **REJECT** | ≤ 0.000 | PL doesn't transfer on this base; reroute to Sydorskyi-yoink directly |

Secondary read across ablations: the **MixSamp ratio × initialization** matrix should expose which protocol variant is the lever. If all 5 ablations land within ±0.003 of each other, the protocol is robust; if they spread > 0.010, the sampler choice is itself a recipe parameter to tune.

## Background

### Why prior PL attempts failed and this one is different

Exp 014/015/025 ran what looked like "PL on unlabeled SS" but with two critical departures from the disclosed Antoine protocol:

| Variable | Prior failed attempts (exp 014/015/025) | This experiment (exp 033) |
|---|---|---|
| PL loss weight | **0.5** (down-weighted vs GT's 1.0) | **1.0** (equal to GT) |
| Batch composition | **Hardcoded 50/50** GT/PL via `StratifiedBatchSampler` (oversamples GT 5× given pool sizes) | **Natural proportions** (uniform sampler) OR ablation of MixSamp ratios that respect the pool-size mismatch |
| Teacher | Perch (cross-arch, mismatch with student) for exp 014/015; mostly v5j-SED (own-arch) for exp 025 with Perch fallback for 44 species | **Two own-arch SED teachers paired** (T1=v5j-4fold, T2=distilled-sed-kd-5fold); no Perch tier C, no cross-arch contamination |
| Base recipe | v5j HGNet+LSEHead (LB 0.906 in 4-fold; ~0.897 retrained single-fold) | **distilled-sed-kd** (LB 0.906 reproducible at HEAD; KD-validated stack matching Tucker recipe) |

The exp 025 verdict was scoped to *"PL with Phase B down-weighting OR Phase C waveform-mixing"*. The 1:1 weighting + uniform sampling protocol that 5 competitors disclose has **never been tested** in our codebase.

### Why we expect this to work

| Argument | Direction |
|---|---|
| 5 independent disclosed data points at 0.93+ all use this exact protocol | Positive |
| Antoine's +0.007 came from **teacher iteration**, not a stronger first-round teacher; suggests the protocol scales | Positive |
| Both our teachers (T1=v5j-4fold cached at LB 0.913, T2=distilled-sed-kd-5fold at LB 0.915) are strong by competitor standards; Antoine's first-round teacher was likely ~0.92-grade. Two own-arch teachers also enable a paired diversity test. | Positive |
| Pool-size math: 127K PL segments vs 26K focal → MixSamp 0.9/0.1 makes the PL pool invisible (each PL seen 0.001× per epoch). Sampler choice is structurally important and a documented variable. | Positive (we're now correcting for this) |
| distilled-sed-kd's `AttHeadLogitSpace` consumes spatial features in logit space — Tucker recipe is engineered for this. PL targets are clip-level soft probabilities → should integrate cleanly with the dual-supervision loss | Positive |
| Pre-flight checks (7/7 PASS) confirm parquet schema + audio + mel + ckpt all wired correctly | Positive |
| Our base lacks XC pretrain that Antoine has; absolute ceiling likely below Antoine's 0.943 | Uncertain |
| Offline-LB anti-correlation extends to single-fold variance (fold 4 result, exp 032); offline metrics during training may underread or overread true LB lift | Uncertain |

## Method

### Approach

Train **distilled-sed-kd recipe** (Tucker recipe + Perch v2 KD, exp 032 base) on **focal + labeled SS + pseudo-labeled unlabeled SS** using the **Antoine protocol** (1:1 loss weighting, no down-weighting, no waveform-mixing). Run 5 paired ablations exploring **initialization × sampler × focal/SC ratio**. Submit the top-performing ablations to LB.

### Setup

| Component | Detail |
|---|---|
| Base recipe | `distilled-sed-kd` (exp 032 Phase D): EffNet-B0 NS-JFT + AttHeadLogitSpace + dual-supervision BCE (0.5/0.5) + Perch v2 KD (α=1.0) + TuckerMelSpec + waveform mixup hard-max (Beta(0.4,0.4)) + SpecAugment (1×freq×10 + 2×time×10) + 25 ep AdamW + cosine + grad_clip=1.0 |
| Teachers (paired comparison) | **(T1) v5j-4fold** — LB-validated 0.913 4-fold via cached `v1/sed_pseudo_labels.parquet` (own-arch HGNet+LSEHead+BCE, generated 2026-04-10).<br>**(T2) `distilled-sed-kd-5fold`** — LB 0.915 5-fold; cached predictions to be generated to `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/`, then PL parquet generated to `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`. Same logical schema as v1 (filename, chunk_idx, 234 species cols) — pure own-arch SED, no tiered Perch fallback. |
| Student | New runs on exp 032 fold 0 split (StratifiedKFold focal + GroupKFold SS, seed=42). Optionally warm-start from `distilled-sed-kd` Phase D `last-v2.ckpt` (LB 0.906) for A3/A4 variants |
| PL parquets | T1: `shared/pseudo-labels/birdclef-2026/v1/sed_pseudo_labels.parquet` (exists, pre-flight verified).<br>T2: `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` (to be generated; ~3hr work — inference caching + PL generation). Both pure-SED own-arch (no Perch tier C). |
| PL filter | Drop the 14 fold-0 labeled SS files (168 segments) to prevent val-fold leak. Drop the 52 labeled-SS train files (624 segments) from PL pool to avoid GT/PL label conflict — those files appear in the labeled SS pool with GT labels. Effective PL pool = **127,104 unlabeled SS segments** (10,592 files × 12 windows) per teacher. |
| PL loss weight | **1.0** (1:1 protocol — equal to GT) |
| Mel | TuckerMelSpec (n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000, top_db=80, single per-sample z-score), applied to all 3 sources (focal / labeled SS / unlabeled SS) — verified path-invariant by pre-flight Check 3 |
| Folds | exp 032 5-fold (StratifiedKFold focal + GroupKFold SS, seed=42), fold 0 only for Round 1 |
| Compute | rtx6000 GPU, ~4.5h walltime per ablation (10 ablations × ~4.5h, parallel-submittable on PACE). Plus ~3hr for T2 inference cache + PL generation prerequisite. |
| Code | Branch `mg/exp033-own-teacher-pl`. Extends exp 032's `DistilledSEDFinetuneModule` + `DistilledSEDDataModule` with PL data ingestion (port from exp 025's `data/sed.py` PL machinery). New inference caching script for T2 teacher. |

### Procedure

#### Step 0 — Prerequisites (before any training)

| # | Task | Cost |
|---|---|---|
| 0.1 | Write inference caching script — port `predict_soundscape_tucker_ensemble` (5-fold logit-mean + smooth + sigmoid) to a PACE-GPU offline script that writes `logits.parquet` + `framewise_logits.parquet` per fold to `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/`. | ~1-2 hr coding |
| 0.2 | Run inference caching on all 10,658 SS files (5 folds × 128K windows) | ~30-60 min on rtx6000 |
| 0.3 | Run `experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py` with the new T2 cache (re-use the same temperature=0.6, power_transform=1.5, max-across-folds config; no Perch tier). Output `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`. | ~30 min |
| 0.4 | Re-run pre-flight checks (Checks 1, 4, 5, 6 on the new parquet) to confirm schema/leakage | ~5 min |

#### Step 1 — Ablation grid (10 paired runs)

Each ablation is the same Tucker recipe (KD on) + Antoine 1:1 PL protocol. Variables: init × sampler × focal/SC ratio × **teacher**.

| ID | Init | Sampler | Focal/SC | LR | Epochs | Teacher | Hypothesis tested |
|---|---|---|---|---|---|---|---|
| **A1.v5j** | random | MixSamp | 0.5 / 0.5 | 5e-4 | 25 | T1 (v5j-4fold) | "Moderate PL emphasis from scratch, T1 teacher." |
| **A2.v5j** | random | **Uniform** | natural ~17/83 | 5e-4 | 25 | T1 | "Pure 1:1 protocol from scratch, T1 teacher." |
| **A3.v5j** | **warm-start** (`last-v2.ckpt`) | MixSamp | 0.2 / 0.8 | **5e-5** | **10** | T1 | "Continued training, PL-heavy mix, T1 teacher." |
| **A4.v5j** | **warm-start** | **Uniform** | natural | **5e-5** | **10** | T1 | "Warm-start + pure protocol, T1 teacher (compound)." |
| **A5.v5j** | random | MixSamp | 0.7 / 0.3 | 5e-4 | 25 | T1 | "Conservative under-exposure (predicted to underperform A1)." |
| **A1.kd** | random | MixSamp | 0.5 / 0.5 | 5e-4 | 25 | **T2 (`distilled-sed-kd-5fold`)** | A1 paired across teachers. |
| **A2.kd** | random | **Uniform** | natural ~17/83 | 5e-4 | 25 | T2 | A2 paired across teachers. |
| **A3.kd** | **warm-start** | MixSamp | 0.2 / 0.8 | **5e-5** | **10** | T2 | A3 paired across teachers. |
| **A4.kd** | **warm-start** | **Uniform** | natural | **5e-5** | **10** | T2 | A4 paired across teachers. |
| **A5.kd** | random | MixSamp | 0.7 / 0.3 | 5e-4 | 25 | T2 | A5 paired across teachers. |

Stretch (decided after Round 1):
- **A0.ensemble**: best-performing protocol (TBD) using **mean(PL_T1, PL_T2)** as targets — tests ensemble-of-teachers.

#### Pair structure for attribution

| Comparison | Variable isolated |
|---|---|
| A1.v5j ↔ A2.v5j, A3.v5j ↔ A4.v5j, A1.kd ↔ A2.kd, A3.kd ↔ A4.kd | **Sampler** (MixSamp vs Uniform) — 4 paired tests |
| A1.v5j ↔ A3.v5j, A2.v5j ↔ A4.v5j, A1.kd ↔ A3.kd, A2.kd ↔ A4.kd | **Init** (random vs warm-start) — 4 paired tests |
| A1.v5j ↔ A1.kd, A2.v5j ↔ A2.kd, A3.v5j ↔ A3.kd, A4.v5j ↔ A4.kd, A5.v5j ↔ A5.kd | **Teacher** (T1 vs T2) — 5 paired tests |
| A1.v5j vs A5.v5j, A1.kd vs A5.kd | **Sanity** — over-exposure vs under-exposure on fixed teacher |

#### LB submission strategy under daily cap (5/day)

| Day | Action |
|---|---|
| Day 1 morning | Submit T2 inference caching + PL generation prereq (Step 0) |
| Day 1 afternoon | Submit all 10 training jobs to PACE (parallel) |
| Day 2 morning | All 10 jobs complete (~4.5h); run D.5 e2e validation on all 10 ckpts. Rank by e2e val/auroc. |
| Day 2 | Submit **top 5 by e2e** to LB. Read results when they land. |
| Day 3 | Submit remaining 5 (or stretch A0 if Day 2 results suggest ensemble-of-teachers is promising). |
| Day 4 | Apply decision matrix. Plan Round 2 if CONFIRM. |

### Pool-size sampler math (already validated in design discussion)

With 127K PL segments + 26K focal samples and `nst = (focal + sc) / batch_size = 2390 steps/epoch`:

| Setup | Focal exposure/epoch | PL exposure/epoch |
|---|---:|---:|
| Tucker 0.9/0.1 (original ratio, transplanted) | 5.3× | 0.001× |
| A5: 0.7/0.3 | 4.1× | 0.003× |
| A1: 0.5/0.5 | 2.9× | 0.006× |
| A3: 0.2/0.8 | 1.2× | 0.010× |
| A2 / A4: Uniform | 1.0× | 1.0× |

Tucker's published 0.9/0.1 was tuned for **792 SS segments**. Naive transplant with 127K SS is structurally broken — the PL pool becomes invisible. A2/A4 (uniform) is the only setup that achieves Antoine's "treat as labeled" semantics; A3 (warm-start + 0.2/0.8) tests whether MixSamp's focal anchor still has value when the model is already trained.

### Variables

- **Independent (between ablations within exp 033)**:
  - **Init**: random Tucker-init vs warm-start from `distilled-sed-kd` Phase D `last-v2.ckpt`
  - **Sampler**: MixSamp (focal/SC ratio knob) vs Uniform (natural proportions over ConcatDataset)
  - **Focal/SC ratio** (when MixSamp): 0.5/0.5, 0.7/0.3, 0.2/0.8
  - **Teacher**: T1 (v5j-4fold cached, LB 0.913 4-fold) vs T2 (`distilled-sed-kd-5fold`, LB 0.915 5-fold)
- **Independent (between exp 033 and exp 025)**: loss_weight (1.0 vs 0.5), sampler family (uniform/MixSamp vs StratifiedBatchSampler hardcoded 50/50), teacher purity (own-arch SED vs tiered SED+Perch). Intentional multi-variable correction.
- **Dependent**: paired LB lift, val/auroc (with caveats from offline-LB anti-correlation history), val/cmap, e2e val/auroc (D.5 pre-flight)
- **Controlled (within exp 033)**: same Tucker recipe, same fold 0 split, same seed=42, same SLURM partition. Teacher swaps in T1/T2; everything else identical within each pair.

## Evaluation

### Metrics

| Metric | Description | Role |
|---|---|---|
| Paired LB lift | LB(student) − LB(distilled-sed-kd Phase D 0.906) | **Primary read** — Antoine reports +0.007-0.013 from this lever |
| Cross-ablation LB spread | std(LB across A1-A5) | **Secondary read** — characterizes protocol robustness |
| val/auroc (Tucker macro AUC) | Training-side, on fold 0 SS val pool | Sanity check; not the LB predictor (offline-LB anti-correlation) |
| End-to-end val/auroc (D.5) | Kernel inference path on fold 0 val SS | **Mandatory pre-flight before LB**; gap to training-side must be < 0.02 |
| train/cls_loss, train/dist_loss | Training trajectory health | Sanity (KD still working) |

### Baseline

- **Primary baseline**: `distilled-sed-kd` Phase D fold 0 = **LB 0.906** (paired-within-codebase, reproducible at HEAD)
- **Secondary baseline (historical)**: distilled-sed-kd-5fold = 0.915 LB (Phase 2.A ensemble; informative but ensemble is a different scope from single-fold PL)
- **Optimistic target**: 0.920+ (Antoine's +0.014 single-fold lift band lower edge; reaches CONFIRM)

### Success Criteria

See § Falsifiable success matrix above. CONFIRM (≥ +0.014) → proceed to Round 2 (teacher iteration) and/or Sydorskyi-yoink layering. PARTIAL (+0.007-0.013) → useful for ensemble with exp 023 (0.929). WEAK (+0.001-0.006) → ambiguous, need 5-fold to disambiguate. REJECT (≤ 0) → reroute to Sydorskyi-yoink as primary.

## Limitations

- **No XC pretrain**: Antoine's recipe is XC pretrain + own-teacher PL. We have only the PL half. Without XC, our backbone's in-domain prior is weaker → expected ceiling 0.92-0.93 single-fold, not 0.94+. XC layering (Sydorskyi-yoink) is reserved for follow-up if exp 033 confirms.
- **Single-fold for Round 1**: 5-fold extension reserved for if Round 1 CONFIRMs. Per exp 030/031 drift investigation, single-fold has ~0.007-0.013 cross-node LB variance — WEAK outcomes can't be reliably disambiguated without 5-fold or seed sweep.
- **Single-round PL**: Antoine's lift came from iterating teacher (Round 1 student → Round 2 teacher). We're testing only Round 1 first. Round 2 deferred until LB lands.
- **Teacher LB delta is small (Δ = +0.002)**: T1 (v5j-4fold) and T2 (`distilled-sed-kd-5fold`) are ~equal in quality. This is a teacher-architecture-diversity test, NOT a teacher-strength test. Antoine's teacher iteration was a much larger quality delta — we don't expect to replicate that mechanism, only the diversity benefit.
- **Offline-LB anti-correlation history**: 23+ data points in this project show offline metrics don't predict LB rank. We mandatory-gate LB submission on D.5 end-to-end validation but expect the LB number to still be the only reliable read.
- **Daily LB cap (5/day)**: 10 ablations cannot all be submitted on a single day. Top-5-by-e2e ranking determines Day 2 submissions; remainder go on Day 3. Risk: the e2e ranking might not perfectly correlate with LB (consistent with offline-LB anti-correlation history). Mitigation: D.5 gap (training vs e2e) has been more LB-correlated than training-side val/auroc in exp 032 (0.004-0.011 gap matched 0.906 LB landing).
- **T1 cached PL parquet was generated 2026-04-10**: pre-flight Check 1 confirms filename + species set still align with current data. T2 parquet (to be generated in Step 0) inherits the same data setup.
- **T2 inference caching prereq adds ~3hr** before the first training. Sequential dependency for the 5 T2-teacher ablations; T1 ablations can start immediately.

## Next-step plan (post-Round-1)

| Round 1 outcome | Round 2 plan |
|---|---|
| **CONFIRM** (LB ≥ 0.920) | Regenerate PLs using Round-1 student as teacher → train Round-2 student → submit (Antoine's iteration). Parallel track: layer Sydorskyi-yoink for XC base. |
| **PARTIAL** (+0.007-0.013) | Ensemble with exp 023 (0.929) for combined LB. Consider XC pretrain as the next lever rather than another PL round. |
| **WEAK** (+0.001-0.006) | Extend to 5-fold to disambiguate from single-fold seed noise. |
| **REJECT** (≤ 0) | Pivot to Sydorskyi-yoink as primary remaining lever. Document the protocol findings for archival. |
