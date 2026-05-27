---
experiment: "033-own-teacher-pl"
status: reject-protocol-fails-on-base
date: 2026-05-27
author: Murilo Gustineli
---

# Results: Own-teacher pseudo-labels on distilled-sed-kd

**Date Updated**: 2026-05-27 PM (3/5 Day-1 LB submissions landed — all REJECT)
**Author**: Murilo Gustineli

## Status: REJECT — 1:1 own-teacher PL protocol does NOT lift LB on the distilled-sed-kd base

3 LB submissions across both teachers, including the canonical Antoine cell + the
paired teacher-axis test, all regressed −0.007 to −0.020 vs the 0.906 baseline.
The teacher-axis paired Δ on the warm+Uniform cell is effectively zero
(0.899 vs 0.897 LB), refuting the proposal's headline teacher-diversity claim
at LB scale. **The protocol does not transfer on this base regardless of teacher.**

### LB summary (2026-05-27 PM)

| Ablation | Cell | Teacher | LB | Δ vs 0.906 | Decision-matrix bucket |
|---|---|---|---:|---:|:---:|
| `2026-exp033-a4-kd` | warm + Uniform | T2 (distilled-sed-kd-5fold) | **0.899** | **−0.007** | REJECT (edge of cross-hardware noise) |
| `2026-exp033-a2-kd` ⭐ | random + Uniform (canonical Antoine) | T2 | **0.886** | **−0.020** | REJECT (firmly inside REJECT band) |
| `2026-exp033-a4-v5j` | warm + Uniform | T1 (v5j-4fold) | **0.897** | **−0.009** | REJECT |
| `2026-exp033-a3-kd` | warm + MixSamp 0.2/0.8 | T2 | (not submitted) | — | — |
| `2026-exp033-a3-v5j` | warm + MixSamp 0.2/0.8 | T1 | (not submitted) | — | — |

Per `proposal.md § Falsifiable`: CONFIRM ≥ +0.014, PARTIAL +0.007 to +0.013,
WEAK +0.001 to +0.006, REJECT ≤ 0. **All 3 landed in REJECT.** The remaining
2 Day-1 slots (a3-kd, a3-v5j) and the 5 Day-2 cells (a1, a5 × teachers) are
unlikely to surprise positive given the trajectory and were not submitted.

### Paired Δ readings

- **Teacher axis (a4-kd vs a4-v5j, same protocol cell)**: Δ = 0.899 − 0.897 = **+0.002 LB** (T2 vs T1). Inside cross-hardware noise. The proposal's headline teacher-diversity hypothesis (T2 wins offline by +0.011 auroc) does NOT transfer to LB. Refutes the "T2 is the teacher to keep iterating with" claim from this morning's read.
- **Protocol axis (a4-kd warm vs a2-kd random, same teacher)**: a2-kd canonical Antoine LB 0.886 vs warm-variant a4-kd LB 0.899. Δ = **−0.013** (canonical < warm-start). Inverts the offline reading (a4-kd e2e 0.955 vs a2-kd e2e 0.940 = warm > canonical). Another offline-LB anti-correlation data point.

### Decision-matrix read

| Bucket | LB threshold | Result |
|---|---|---|
| CONFIRM | ≥ +0.014 | ❌ |
| PARTIAL | +0.007 to +0.013 | ❌ |
| WEAK | +0.001 to +0.006 | ❌ |
| **REJECT** | **≤ 0** | **✅ all 3 cells** |

**Verdict per proposal**: pivot to Sydorskyi-yoink (XC pretrain) as the primary
remaining lever. The own-teacher PL protocol is refuted on the distilled-sed-kd
base.

### Offline-LB anti-correlation: data point #24

This adds 3 paired data points to the project's documented offline-LB
anti-correlation pattern (24+ total now):

| Cell | Offline e2e val/auroc | LB | Offline rank | LB rank |
|---|---:|---:|---:|---:|
| a4-kd | 0.955 (rank 1/10) | 0.899 (rank 2 of 3) | strongest | middle |
| a2-kd ⭐ | 0.940 (rank 3/10) | 0.886 (rank 3 of 3) | middle | worst |
| a4-v5j | 0.940 (rank 4/10) | 0.897 (rank 1 of 3) | middle | best |

**The offline-ranked best (a4-kd) was middle on LB; the offline-ranked weakest
of the three (a4-v5j) was the LB best.** Same pattern as exp 032 fold 4 (+0.023
offline → −0.003 LB) and 22+ prior recipe data points.

---

## Status: TRAINING COMPLETE (10/10 fold-0 ablations) — D.5 e2e validation + LB submission pending (HISTORICAL — superseded by LB results above)

All 10 ablations of the Antoine-style 1:1 own-teacher PL protocol on top of the
Phase D `distilled-sed-kd` recipe finished overnight 2026-05-26 → 2026-05-27 on
gpu-rtx6000. Wall-time matched predictions (25-ep runs ~6.5h, 10-ep runs ~2.6h
at `num_workers=8`). Every run reached its full `max_epochs` (early stopping was
disabled with `patience: 999`); every `last.ckpt` written; every `metrics.json`
backfilled.

The slurm FAIL emails are a false alarm — `_job.sh:127` called a missing
`save_metrics.py` that wasn't ported from exp 032 in Phase E. `Training
complete!` line is present in all 10 logs; `outputs/test_results/*.json` exist
for all 10. Fix landed in this session (commit pending): new
`save_metrics.py` with `_common.yaml` inheritance, plus backfilled metrics for
all 10 runs. Future runs will exit 0.

**Caveat going in**: offline val/auroc has been **anti-correlated with LB** for
23+ data points in this project (most recently exp 032 fold 4: best offline
0.931 → near-worst LB 0.903). The offline rankings below indicate *direction*
only; D.5 end-to-end validation + actual LB submission is what closes the loop.

## Offline metrics (val on fold-0 held-out SS pool)

Baseline (`distilled-sed-kd` Phase D fold 0, LB **0.906**): val/auroc 0.908,
val/cmap 0.692, best val/cmap 0.709 (at epoch 10).

### Ranked by training-side val/auroc (last epoch)

| Rank | Ablation | Init | Sampler | Teacher | val/auroc | val/cmap | best val/cmap | Best Ep | Walltime | SLURM | W&B |
|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| 1 | `2026-exp033-a3-kd` | warm-start | MixSamp 0.2/0.0/0.8 | T2 | **0.9486** | 0.8311 | 0.8350 | 7 | 2h40m | 9191660 | `ax6t5qdz` |
| 2 | `2026-exp033-a4-kd` | warm-start | Uniform | T2 | **0.9477** | **0.8323** | 0.8335 | 7 | 2h38m | 9191661 | `htfgco92` |
| 3 | `2026-exp033-a2-kd` ⭐ | random | Uniform | T2 | 0.9390 | 0.8182 | 0.8201 | 3 | 6h29m | 9191659 | `1ye9s2ni` |
| 4 | `2026-exp033-a2-v5j` ⭐ | random | Uniform | T1 | 0.9362 | 0.7923 | 0.7977 | 14 | 7h20m | 9190540 | `4y0h7h1a` |
| 5 | `2026-exp033-a1-kd` | random | MixSamp 0.5/0.0/0.5 | T2 | 0.9357 | 0.8242 | 0.8390 | 12 | 6h46m | 9191658 | `2cg0kcux` |
| 6 | `2026-exp033-a4-v5j` | warm-start | Uniform | T1 | 0.9352 | 0.8043 | 0.8058 | 7 | 2h39m | 9190544 | `dl7pz6kv` |
| 7 | `2026-exp033-a3-v5j` | warm-start | MixSamp 0.2/0.0/0.8 | T1 | 0.9345 | 0.8061 | 0.8095 | 7 | 2h38m | 9190542 | `f781pbyz` |
| 8 | `2026-exp033-a1-v5j` | random | MixSamp 0.5/0.0/0.5 | T1 | 0.9288 | 0.7821 | 0.7859 | 11 | 7h25m | 9190539 | `ct3by7ax` |
| 9 | `2026-exp033-a5-kd` | random | MixSamp 0.7/0.0/0.3 | T2 | 0.9285 | 0.7915 | 0.7954 | 10 | 6h45m | 9191662 | `vn3vyrp5` |
| 10 | `2026-exp033-a5-v5j` | random | MixSamp 0.7/0.0/0.3 | T1 | 0.9113 | 0.7586 | 0.7595 | 21 | 6h41m | 9190545 | `opv1942s` |

⭐ = canonical Antoine protocol (random init + uniform sampler over ConcatDataset).
T1 = `v5j-4fold` teacher (`v1/sed_pseudo_labels.parquet`, LB 0.913).
T2 = `distilled-sed-kd-5fold` teacher (`v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet`, LB 0.915).

All 10 ablations land above the baseline 0.908 val/auroc; even the worst
(a5.v5j 0.911) is +0.003 over baseline. **Offline rankings are NOT a reliable LB
predictor on this project** — direction only, D.5 + LB submission required.

### Paired Δ across the three axes

#### Teacher axis: T2 (`distilled-sed-kd-5fold`) vs T1 (`v5j-4fold`)

5 paired tests across the same protocol cell:

| Pair | T1 val/auroc | T2 val/auroc | Δ auroc | T1 val/cmap | T2 val/cmap | Δ cmap |
|---|---:|---:|---:|---:|---:|---:|
| A1 (random / MixSamp 0.5/0.5) | 0.9288 | 0.9357 | **+0.0069** | 0.7821 | 0.8242 | **+0.0421** |
| A2 ⭐ (random / Uniform) | 0.9362 | 0.9390 | +0.0028 | 0.7923 | 0.8182 | +0.0259 |
| A3 (warm / MixSamp 0.2/0.8) | 0.9345 | 0.9486 | **+0.0142** | 0.8061 | 0.8311 | +0.0250 |
| A4 (warm / Uniform) | 0.9352 | 0.9477 | **+0.0125** | 0.8043 | 0.8323 | +0.0280 |
| A5 (random / MixSamp 0.7/0.3) | 0.9113 | 0.9285 | **+0.0172** | 0.7586 | 0.7915 | **+0.0330** |
| **mean** | 0.9292 | 0.9399 | **+0.0107** | 0.7887 | 0.8195 | **+0.0308** |

**T2 wins all 5 pairs on auroc AND cmap.** Despite the two teachers having
~equal LB (0.913 vs 0.915 = +0.002), the T2 student is uniformly stronger
offline. This is consistent with a **teacher-diversity benefit** — T2's
distilled-sed-kd is architecturally + recipe-distinct from the student
(same backbone family, but different LB-validated soft-label distribution than
the v5j ensemble). If the LB tracks the offline ranking even directionally,
T2 is the teacher to keep iterating with.

#### Sampler axis: Uniform vs MixSamp (within init + teacher)

4 paired tests:

| Pair | MixSamp ratio | MixSamp val/auroc | Uniform val/auroc | Δ (Uniform − MixSamp) |
|---|---|---:|---:|---:|
| A1.v5j vs A2.v5j (random / T1) | 0.5/0.5 | 0.9288 | 0.9362 | **+0.0074** |
| A1.kd vs A2.kd (random / T2) | 0.5/0.5 | 0.9357 | 0.9390 | +0.0033 |
| A3.v5j vs A4.v5j (warm / T1) | 0.2/0.8 | 0.9345 | 0.9352 | +0.0007 |
| A3.kd vs A4.kd (warm / T2) | 0.2/0.8 | 0.9486 | 0.9477 | −0.0009 |

**Uniform wins on random init; flat on warm-start.** This is consistent with
the proposal's prediction that pool-size structure matters more than ratio
names — uniform sampling over `ConcatDataset(focal=29K, sc=624, pl=127K)`
gives natural proportions ~18/0.4/82 per batch, very close to A3/A4's hand-set
0.2/0.0/0.8, so the two converge on warm-start. From scratch, the implicit PL
emphasis of uniform (82% PL exposure) appears stronger than A1's 50% MixSamp.

#### Init axis: warm-start vs random (within sampler + teacher)

4 paired tests (note: warm-start runs use 10 ep at lr=5e-5; random runs use
25 ep at lr=5e-4 — confounded with epochs/lr, see Limitations):

| Pair | Random val/auroc | Warm-start val/auroc | Δ (Warm − Random) |
|---|---:|---:|---:|
| A1.v5j vs A3.v5j (MixSamp / T1) | 0.9288 | 0.9345 | **+0.0057** |
| A1.kd vs A3.kd (MixSamp / T2) | 0.9357 | 0.9486 | **+0.0130** |
| A2.v5j vs A4.v5j (Uniform / T1) | 0.9362 | 0.9352 | −0.0010 |
| A2.kd vs A4.kd (Uniform / T2) | 0.9390 | 0.9477 | **+0.0086** |

**Warm-start helps on 3/4 pairs; flat on T1 + uniform.** The strongest lift is
A1.kd → A3.kd (+0.013) — warm-starting from the Phase D ckpt + heavier PL share
under MixSamp + T2 teacher is the best stack offline. With early-stop-by-cmap
on the 10-ep runs (best at ep 7), the model converges quickly and doesn't
need 25 ep. This is the canonical Antoine "iterate teacher on existing
student" pattern.

#### PL-exposure sanity: A1 (50% PL) vs A5 (30% PL)

| Pair | A1 (PL 50%) | A5 (PL 30%) | Δ (A1 − A5) |
|---|---:|---:|---:|
| v5j (T1) | 0.9288 | 0.9113 | **+0.0175** |
| kd (T2) | 0.9357 | 0.9285 | +0.0072 |

**More PL exposure helps**, as predicted. A5 was the "conservative
under-exposure" sanity check and lands at the bottom of both teacher families.
Confirms the proposal's pool-size analysis: Tucker's published 0.9/0.1 ratio
(transplanted naively) would be even worse — PL becomes structurally invisible
to the student.

### Per-batch composition × teacher heat map (val/auroc)

|                                          | T1 (v5j-4fold) | T2 (distilled-sed-kd-5fold) | Δ T2−T1 |
|---|---:|---:|---:|
| **A5** — random / MixSamp 0.7/0.0/0.3 (45/0/19) | 0.9113 | 0.9285 | +0.0172 |
| **A1** — random / MixSamp 0.5/0.0/0.5 (32/0/32) | 0.9288 | 0.9357 | +0.0069 |
| **A2** ⭐ — random / Uniform (~12/~0/~52)        | 0.9362 | 0.9390 | +0.0028 |
| **A4** — warm-start / Uniform (~12/~0/~52)      | 0.9352 | 0.9477 | +0.0125 |
| **A3** — warm-start / MixSamp 0.2/0.0/0.8 (13/0/51) | 0.9345 | 0.9486 | +0.0142 |

Reading top-to-bottom: as PL-per-batch share grows from 30% → ~82% (under
random init) the student's val/auroc grows monotonically (+0.025 on T1, +0.011
on T2). Warm-start at the high-PL end (A3, A4) extracts another +0.009-0.013
over random-init at the same PL share. The protocol's two strongest levers are
(a) ensure PL-per-batch ≥ ~50% and (b) warm-start when possible.

## Training dynamics

### Best-checkpoint epoch distribution

| Epoch range | Count | Ablations |
|---|---:|---|
| ≤ 3 | 1 | a2.kd (ep 3) |
| 7 | 4 | a3.kd, a3.v5j, a4.kd, a4.v5j (all warm-start runs converged at ep 7) |
| 10–14 | 4 | a1.kd (12), a1.v5j (11), a2.v5j (14), a5.kd (10) |
| 21 | 1 | a5.v5j |

Warm-start runs (a3/a4 × T1/T2) converge by epoch 7 — past the 10-epoch
budget they would not improve (this is consistent with the proposal's
"10-epoch warm-start fine-tune" sizing). Random-init runs typically peak
around epoch 10-14. `a5.v5j` (the conservative under-PL'd config) peaks
latest at epoch 21, consistent with the under-PL signal being slower to
saturate the focal-rich training distribution.

A2.kd's epoch-3 best is a likely false positive (val/cmap monitor on a small
held-out SS pool is noisy at the start); the last-epoch val/cmap (0.8182)
matches the trend on this row well.

### Training-side losses (sanity)

Final-epoch `train/cls_loss` is in the ~0.02 range across all 10 runs (vs
Tucker's recipe published 0.02-0.03 band). `train/dist_loss` (Perch v2 KD MSE)
plateaus at 0.002-0.003 (Tucker's published floor). No NaN or instability
anywhere; the dual-supervision loss + per-sample loss_weight integration is
healthy at scale.

## Compute summary

| Cell | Walltime (per run) | × 5 runs | Notes |
|---|---|---|---|
| 25-ep ablations (A1/A2/A5) | 6.5–7.4h | ~33h | matches 20 min/epoch × 25 ep + ~10 min warmup |
| 10-ep ablations (A3/A4) | 2.6–2.7h | ~13h | matches 16 min/epoch × 10 ep at lr=5e-5 |
| **All 10 ablations (parallel)** | — | wall ~7.5h | gpu-rtx6000, 5 nodes (15 + 19 + 21 + 7 + 11/3) |

Total compute consumed (T1 wave was the cancelled `num_workers=0` jobs +
relaunched fix; sunk ~20 GPU-hours on the cancelled runs). Effective parallel
overnight: ~7.5h wall-clock for all 10 ablations.

`num_workers=8` regression fix (commits `ecc3ccb` + `87d668d`) was the key
performance win — `num_workers=0` inherited from Tucker's notebook would have
made every job 4× slower, missing the overnight window.

## Artifacts

| Type | Location |
|---|---|
| Configs | `configs/_common.yaml` + `configs/a{1..5}_{v5j,kd}.yaml` |
| Logs | `logs/slurm_a{1..5}_{v5j,kd}-fold0_91*.log` (10 files) |
| Test results JSON | `outputs/test_results/2026-exp033-a{1..5}-{v5j,kd}-fold0.json` (10 files) |
| Structured metrics | `artifacts/training/a{1..5}_{v5j,kd}/fold0/metrics.json` (10 files; backfilled 2026-05-27) |
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/033-own-teacher-pl/checkpoints/2026-exp033-a{1..5}-{v5j,kd}/fold0/{best-epochepoch=NN.ckpt, last.ckpt}` |
| T2 PL parquet | `shared/pseudo-labels/birdclef-2026/v2-distilled-sed-kd-5fold/sed_pseudo_labels.parquet` (210,375 PLs > 0.5; 166 species; this session) |
| T2 inference cache | `shared/inference/birdclef-2026/distilled-sed-kd-5fold/train_soundscapes/fold{0..4}/logits.parquet` (639,480 rows; this session) |
| W&B | `dsgt-arc/birdclef-2026-exp033/{ct3by7ax,2cg0kcux,4y0h7h1a,1ye9s2ni,f781pbyz,ax6t5qdz,dl7pz6kv,htfgco92,opv1942s,vn3vyrp5}` |
| Code | branch `mg/exp033-own-teacher-pl` |

### T1 ckpt name caveat

For the T1 (`v5j`) ablations, `last.ckpt` in each fold0 directory is from the
**cancelled `num_workers=0` job** (epoch 4–5 snapshot, sat there from
~21:35 on 2026-05-26). The **actual final checkpoint** is `last-v1.ckpt`
(epoch 24 or 9, from the relaunched run with `num_workers=8`). D.5 + ONNX
export must point at `last-v1.ckpt` for the 5 T1 runs; T2 runs use the
standard `last.ckpt`.

## Decision-matrix read (OFFLINE ONLY — LB closes the loop)

Per `proposal.md § Falsifiable` (LB-only buckets; baseline LB 0.906):

| Bucket | LB threshold | Interpretation |
|---|---|---|
| **CONFIRM** | LB ≥ 0.920 (+0.014) | Antoine-protocol PL transfers; explore Round 2 (teacher iteration) and/or Sydorskyi-yoink XC base |
| **PARTIAL** | LB +0.007 to +0.013 | Useful for ensemble with exp 023 (0.929) |
| **WEAK** | LB +0.001 to +0.006 | Inside seed band; need 5-fold or seed sweep to disambiguate |
| **REJECT** | LB ≤ 0.906 | Reroute to Sydorskyi-yoink as primary |

Offline val/auroc lifts over the Phase D baseline (0.908) are **+0.020 to
+0.041** across the 10 ablations. Per exp 032's anti-correlation evidence
(fold 4: +0.023 offline → −0.003 LB), this does NOT translate 1:1 to LB. The
LB submission itself is the only reliable read.

The pre-LB ranking for the 5/day cap is therefore best made on **D.5
end-to-end val/auroc**, not training-side val/auroc — D.5 was a tighter
LB predictor in exp 032 Phase D (gap of +0.004 to +0.011 between e2e and
training-side, consistent with the 5-tap smoothing trick).

## Next Steps

D.5 + 3-cell LB landed; verdict is REJECT. Per the proposal's pivot plan, next
session = scope Sydorskyi-yoink (XC pretrain as the remaining LB lever). The
remaining 7 ablations (2 Day-1 + 5 Day-2) are NOT worth submitting on the
current trajectory.

1. **Closeout**: ✅ results.md updated · MODEL-RUNS.md updated · tasks.md
   closeout phase pending PR.
2. **Pivot**: scope `experiments/034-sydorskyi-yoink-xc-pretrain/` (or
   similar). Goal = layer XC pretrain (Sydorskyi's recipe) on top of the
   distilled-sed-kd recipe to unlock the lever Antoine has but we don't.
3. **Don't run** Day-1 a3-kd + a3-v5j or the 5 Day-2 ablations — the LB cap
   slots are better spent on the pivot's first iteration.

### Historical (pre-LB plan)

The original ladder, kept here for archival:

1. D.5 end-to-end validation on all 10 ckpts (✅ done — all 10 PASS).
2. Rank by e2e val/auroc → top 5 to LB Day 1 (3 submitted, 2 abandoned).
3. LB submit, apply decision matrix → REJECT.
4. CONFIRM → Round 2 + Sydorskyi-yoink. (Not triggered.)
5. PARTIAL → ensemble with exp 023. (Not triggered.)
6. WEAK → 5-fold. (Not triggered.)
7. REJECT → pivot to Sydorskyi-yoink directly. **← active path**

## Methodology notes

- **Slurm FAIL email gotcha**: All 10 trained to completion (`Training
  complete!` + JSON + `last.ckpt` + W&B upload all present in every log).
  `save_metrics.py` was missing from exp 033 (Phase E omission), causing
  `_job.sh` to exit non-zero → `--mail-type=END,FAIL` fired. Fix landed this
  session: `save_metrics.py` ported with `_common.yaml` deep-merge logic;
  10 backfills run on the head node.
- **Early stopping disabled**: `_common.yaml` set `patience: 999` on `val/cmap`
  monitor. All runs reached `max_epochs` (25 or 10). The best ckpt selection
  via `save_top_k: -1` + monitor `val/cmap` correctly identifies the
  per-epoch best (see Best Ep column above).
- **Offline-LB anti-correlation**: per exp 032 fold 4 (+0.023 offline → −0.003
  LB) and 22+ prior data points, we don't expect the offline rankings to
  translate 1:1. The +0.030 offline lift over baseline across the canonical
  Antoine protocol (A2.kd / A4.kd) is encouraging but not guarantee.
- **Confounds in the init axis**: warm-start runs (A3/A4) also use lr=5e-5
  + 10 epochs + warmup_epochs=0; random-init runs use lr=5e-4 + 25 epochs +
  warmup_epochs=2. The "warm-start helps" reading is a compound observation,
  not a clean isolated lever. The proposal acknowledges this — the cell-grid
  was designed around competitor recipes, not orthogonal ablations.
