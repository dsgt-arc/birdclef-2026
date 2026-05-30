---
experiment: "037-noisy-student-pl"
status: in-progress — round-1 factorial trained + offline-screened; LB screening next
date: 2026-05-30
author: Murilo Gustineli
---

# Results — Exp 037: Multi-Iterative Noisy-Student Soft Pseudo-Labeling

**Status**: Round-1 **ablation factorial (15 cells, fold 0) trained** to completion; offline
labeled-SS metrics pulled + screened. **No LB yet** — next step is ~5 mechanism-diverse LB
submissions (offline can only eliminate/direction, not rank — it anti-correlates with LB).
**Date**: 2026-05-30 · **Author**: Murilo Gustineli

> **Metric note:** primary = **best `val/auroc`** (the competition metric = macro ROC-AUC, and the
> metric exp 021's v5j-0.917 monitored). The configs originally inherited `val/cmap` monitoring from
> the Tucker port (exp 032/033); **reverted `_common.yaml` to `val/auroc`** for the round-2 / 5-fold
> runs. Round 1 lost nothing: `save_top_k=-1` saved **all 25 epochs** per cell, so for LB export we
> pick each cell's **best-by-`val/auroc`** epoch. `val/cmap` shown as a secondary column.

## Round-1 offline results (fold 0, 25 ep, sorted by best `val/auroc`)

Student = `distilled-sed-kd` (EfficientNet-B0). Teacher = cached 0.929 ensemble
(`0.5·perch_234_probs(TF) + 0.5·sigmoid(mean SED v5j-4fold)`), softened per transform. All cells:
mixsamp 50/50 GT:PL, loss_weight=1.0, `steps_per_epoch=500` (GT-only), no filtering. Jobs 9346722–37.

| Cell | KD routing | transf | aug | loss | **best val/auroc** | val/cmap | logit_std | job |
|---|---|---|---|---|---:|---:|---:|---|
| **r1-B-raw-augF** | off / e2e | raw | F | bce | **0.9555** | 0.8041 | 3.86 | 9346728 |
| r1-A-raw-augT-softauc | on / detach | raw | T | softauc | 0.9523 | 0.7763 | 2.85 | 9346723 |
| r1-B-raw-augT | off / e2e | raw | T | bce | 0.9507 | 0.7759 | 3.80 | 9346730 |
| r1-C-raw-augF | on / no-detach | raw | F | bce | 0.9504 | 0.7980 | 3.83 | 9346733 |
| r1-C-t2-augF | on / no-detach | t2 | F | bce | 0.9484 | 0.7799 | 3.71 | 9346736 |
| r1-A-raw-augT | on / detach | raw | T | bce | 0.9472 | 0.7800 | 3.29 | 9346724 |
| r1-A-raw-augF | on / detach | raw | F | bce | 0.9433 | 0.7771 | 3.49 | 9346722 |
| r1-C-raw-augT | on / no-detach | raw | T | bce | 0.9426 | 0.7988 | 3.77 | 9346735 |
| r1-C-raw-augT-softauc | on / no-detach | raw | T | softauc | 0.9423 | 0.8014 | 3.46 | 9346734 |
| r1-A-t2-augF | on / detach | t2 | F | bce | 0.9391 | 0.7246 | 3.42 | 9346725 |
| r1-B-t2-augT | off / e2e | t2 | T | bce | 0.9368 | 0.7642 | 3.71 | 9346732 |
| r1-A-t2-augT | on / detach | t2 | T | bce | 0.9238 | 0.7173 | 3.10 | 9346726 |
| r1-B-t2-augF | off / e2e | t2 | F | bce | 0.9213 | 0.7176 | 3.77 | 9346731 |
| r1-B-raw-augT-softauc | off / e2e | raw | T | softauc | 0.9173 | 0.7740 | 3.47 | 9346729 |
| r1-C-t2-augT | on / no-detach | t2 | T | bce | 0.9163 | 0.7418 | 3.65 | 9346737 |

## Per-axis read (direction only — see caveat)

Means over best `val/auroc` (cmap in parens where it diverges):

| Axis | val/auroc means | Verdict |
|---|---|---|
| **KD routing** (raw, bce) | A **0.945** · B **0.953** · C **0.947** | **B (off) ≳ C (no-detach) > A (detach)** on auroc (cmap: C>B>A). **Both metrics agree A (head-only) is worst.** → **letting PL reach the backbone (B/C) beats head-only (A)** — the structural lever exp 033 never tested. |
| **PL transform** | raw **0.945** vs t2 **0.931** (Δ **+0.014**) | raw > t2, but **weaker on auroc than cmap (Δ+0.046)**: the worst cells are t2**+augT**, while `C-t2-augF` (0.948) beats several raw cells. raw still tops both metrics + is the faithful blend → **LB raw**, but t2 is not cleanly dead on auroc. |
| **PL aug** (raw, bce) | augF **0.950** vs augT **0.947** | weak — augF slightly ahead on both metrics. Noise-the-student no clear win. |
| **SoftAUC vs BCE** (raw, augT) | A: softauc **0.952** > bce 0.947; B: bce 0.951 ≫ softauc 0.917; C: ~tie | **inconclusive** — SoftAUC best on A, much worse on B. No consistent direction. |

**cmap↔auroc divergence is itself a signal:** the two offline metrics rank cells differently
(e.g. cmap loves the C-softauc cell, auroc the A-softauc cell) — reinforcing that offline labeled-SS
is noisy. The **robust** cross-metric conclusions: (1) **B/C > A** (PL→backbone helps); (2)
**`r1-B-raw-augF` is #1 on both**; (3) **t2+augT cells are the worst**; (4) **raw is the safe pick**.

## Caveat (load-bearing) — offline is NOT a verdict

All 15 cells beat the no-PL `distilled-sed-kd` baseline offline (baseline fold-0 val/auroc 0.908,
val/cmap 0.692 → PL cells 0.92–0.96 / 0.72–0.80). **This is exactly the pattern exp 033 showed
before it REJECTED on LB** (its own-teacher PL also improved offline labeled-SS, then lost −0.007
to −0.027 on LB). Offline labeled-SS **anti-correlates with LB** for us
([[project_cv_measurement_failure]], [[feedback_offline_metrics_unreliable]]); single-fold (168
held-out labeled-SS windows). So these numbers **eliminate broken cells + read direction** — they do
NOT rank, and do NOT conclude PL works. No cell collapsed; `logit_std` (2.85–3.86) shows no runaway.
**LB decides.**

## Next step — LB screening (~5 mechanism-diverse cells)

All `raw` (tops both metrics). Picks span the KD axis (the #1 structural question) + SoftAUC;
each cell's **best-by-`val/auroc`** epoch checkpoint is exported (not the cmap-best):

| # | Cell | best val/auroc | Why |
|---|---|---:|---|
| 1 | `r1-A-raw-augF` | 0.9433 | KD-on/detach, BCE — faithful "0.906 recipe + clean PL" reference; closest to exp 033's structure (corrected teacher) → did fixing the teacher fix A? |
| 2 | `r1-B-raw-augF` | 0.9555 | KD-off/e2e — **offline #1 on both metrics**; the true-Noisy-Student mechanism |
| 3 | `r1-C-raw-augF` | 0.9504 | KD-on/no-detach — strong on both; "best of both" |
| 4 | `r1-A-raw-augT-softauc` | 0.9523 | **offline #2 auroc** + SoftAUC |
| 5 | `r1-C-raw-augT-softauc` | 0.9423 | SoftAUC on a 2nd regime (best-C on cmap 0.8014); alt = `r1-B-raw-augT` (e2e+aug, auroc 0.9507) |

Cells 1–3 = a clean A/B/C KD comparison (raw/augF/BCE) → resolves the structural lever on LB.
**Falsifiable (single-fold vs 0.906):** CONFIRM ≥0.916 / PARTIAL 0.909–0.915 / WEAK 0.907–0.908 /
REJECT ≤0.906. LB-green → 5-fold the winner (monitor now val/auroc) + blend into the 0.929 ensemble;
round 2 if it lifts. **Risk**: exp 033 (structurally similar PL) rejected on LB; the new bets are the
ensemble teacher + raw blend + the B/C structural lever. ~5 of ~20 remaining LB pulls.

## Artifacts

| Type | Location |
|---|---|
| Configs | `configs/r1_*.yaml` (15; emitted by `scripts/gen_round1_configs.py`); monitor now `val/auroc` |
| Checkpoints | `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/037-noisy-student-pl/checkpoints/2026-exp037-<cell>/fold0/` (save_top_k=-1, all 25 epochs as best-epoch* + last) |
| Logs | `logs/slurm_r1_*-fold0_*.log` · TB `outputs/tensorboard/exp037-*-fold0/` |
| Result JSONs | `outputs/test_results/2026-exp037-r1-*-fold0.json` (`best_val_score` = best val/cmap; use TB for best val/auroc) |
| PL parquets | `shared/pseudo-labels/birdclef-2026/v037-ens-{raw,t2}/ensemble_pseudo_labels.parquet` |
| Eval harness | `scripts/eval_labeled_ss.py` (leakage-free OOF macro-AUC; Gate-A bar 0.7632) |

## Implementation (all flag-gated, defaults = exp 032/033; 43 unit + 290 regression PASS)

`kd_detach_head` (cell C; gradient-routing test), `SoftAUCLoss`+`loss_type` (+ `train/cls_loss`
component log), `pseudo_label_aug`, `train/logit_std`, GT-only `steps_per_epoch=500`, 25 ep,
**monitor `val/auroc`** (reverted from the Tucker port's `val/cmap`). Full metric suite
(cmap/auroc/f1/top-k/loss/n_evaluable/LR/logit_std) on W&B + TensorBoard.
