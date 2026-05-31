---
experiment: "037-noisy-student-pl"
status: REJECT — round-1 trio LB ≤0.832, far below the 0.906 no-PL baseline (5th PL failure)
date: 2026-05-30
author: Murilo Gustineli
---

# Results — Exp 037: Multi-Iterative Noisy-Student Soft Pseudo-Labeling

**Status**: **REJECT.** Round-1 ablation factorial (15 cells, fold 0) trained; the A/B/C-raw-augF
trio (the clean KD-routing comparison) submitted to LB and **all three land far below the 0.906
no-PL baseline** (best = C 0.832, −0.074). PL **poisons** the distilled-sed-kd student. The 2
SoftAUC cells were NOT submitted (trio is decisive). **5th PL failure** (014/015/025/033/037).
Pivot to securing the 0.929 ensemble + PP/TTA before 2026-06-03. **Date**: 2026-05-30 · Murilo

## LB verdict (2026-05-30) — REJECT, hard

| Cell | KD routing (PL reaches…) | LB | Δ vs 0.906 base | offline val/auroc(last) |
|---|---|---:|---:|---:|
| `r1-A-raw-augF` | head only (KD-on/detach) | **0.754** | **−0.152** | 0.942 |
| `r1-B-raw-augF` | whole student (KD-off/e2e) | **0.790** | **−0.116** | 0.928 |
| `r1-C-raw-augF` | head + backbone (KD-on/no-detach) | **0.832** | **−0.074** | 0.942 |

**Two findings:**
1. **The KD-routing axis TRANSFERRED to LB: C > B > A** — same order as the offline cmap direction
   (a rare offline→LB agreement for us). So "let PL reach the backbone" is a **real relative effect**:
   C (KD-MSE *and* PL-BCE both train the backbone) > B (PL-BCE trains backbone, no KD) > A (PL→head
   only, backbone frozen-from-PL = worst). This confirms exp 033's structural limitation (it ran cell
   A only) and is a genuine result worth the paper.
2. **But PL poisons regardless of routing** — even the best cell (C, 0.832) is −0.074 below the no-PL
   0.906, and the worst (A, 0.754) is −0.152. The lever is net-negative.

**Why PL poisons here (most likely):** the **raw ensemble blend is a diffuse soft target** — mean
0.239, frac>0.5 only 0.95%, i.e. ~233 of 234 classes sit at small-but-nonzero prob (0.1–0.3) per
window. At **50/50 GT:PL with loss_weight=1.0**, half of every batch trains the student (via BCE) to
emit moderate probabilities across *all* classes → calibration flattens → ranking/AUC collapses. This
is *worse* than exp 033's own-teacher PL (0.886): exp 033 **sharpened** (T=0.6, P^1.5), which
concentrated the targets; our "no sharpening / no filtering" likely **flooded the student with diffuse
weak-positive mass**. The goldilocks band between "diffuse soft" (poisons, here) and "aggressive
filter/sharpen" (HURT, exp 033 −0.027) was not found — and at 5 PL failures, the base rate says stop.

**Addressing the candidate causes (2026-05-30):**
- **Validation metric (cmap vs auroc)? NOT the cause.** We submitted **`last.ckpt`** (final epoch),
  not the monitor-selected "best" epoch — the checkpoint monitor is irrelevant to what shipped. (We
  did revert the monitor to val/auroc for any future runs, but it would not have changed these.)
- **Train longer? No.** val curves plateaued by ep10–15; A (worst) and C (best) both ran 25 ep. More
  epochs on a poisoning objective worsens it.
- **PLs just bad? Yes — primary cause** (diffuse soft targets, above).
- **Perch-only PLs / XC pretraining?** Speculative pivots — see Next steps.

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

## LB screening (the trio, submitted) + offline picks (for the record)

The 5 mechanism-diverse picks were all `raw`, `last.ckpt` (final epoch 24, NOT best-val — the
168-window val is noisy + anti-correlates with LB). The A/B/C-raw-augF **trio** (the clean KD
comparison) was submitted first per plan and **decisively REJECTED** (see LB verdict above), so the 2
SoftAUC picks (`r1-A-raw-augT-softauc` 0.952, `r1-C-raw-augT-softauc` 0.914 offline) were **NOT
submitted** — LB budget conserved. (SoftAUC is a ranking loss, less sensitive to the diffuse-target
poisoning, so it's the one untested mechanism that *might* differ — but at 5 PL failures it's a
low-EV, post-deadline question, not worth an LB pull now.)

## Next steps — PIVOT (PL is exhausted; deadline 2026-06-03)

PL on our models is now **0/5** (014/015/025/033/037). Combined with pretraining capping at ~0.88
(exp 034/036) and training-recipe base-rate≈0 (exp 027), the deadline-rational move is **secure the
final submission**, not another single-model lever:
1. **Re-confirm + lock the 0.929 ensemble (exp 023)** as final pick #1.
2. **PP/TTA + ensemble-diversity** A/B on it (the only LB-proven lever at this altitude) for final pick #2.

**Deferred to post-deadline paper track** (compute is not the bottleneck — can run after 06-03; results
count for the working note, not the LB): (a) **one SoftAUC PL submission** to close the ranking-loss
question; (b) **Perch-only PLs** (drop the SED half of the teacher — it may add the noise); (c) a
**PL-target fix** (mild trim of the diffuse <0.05 tail, between exp-037-raw and exp-033-sharpen); (d)
**broad XC-pretraining** (deep-research #2). None is deadline-EV given 0/5 + 4 days.

**Reusable result for the paper:** the KD-routing axis (C>B>A) transferred offline→LB — "let the PL
loss reach the backbone, don't detach" is a real effect; exp 033's detach-only design was the
structural limit. The remaining gap is that ensemble soft-PL **poisons** our SED student regardless.

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
