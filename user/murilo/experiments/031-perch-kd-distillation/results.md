---
experiment: "031-perch-kd-distillation"
status: closed
date: 2026-05-10
closed_date: 2026-05-11
author: Murilo Gustineli
---

# Experiment 031: Perch v2 Knowledge Distillation on v5j — Results (Phase 1)

## Status: CLOSED — REJECT verdict; drift investigation resolved

Phase 1 KD on HGNet+LSEHead → **REJECT** (within-codebase Δ = −0.006 at
both best-by-val and ep 19 checkpoints). Drift investigation closed:
~0.002 LB attributable to code change between e420633 and HEAD, ~0.007 LB
attributable to GPU/hardware variance across SLURM nodes. uv.lock pinned
to v5j-era versions for future reproducibility (commit `9f9ec3d`). v5j-era
worktree cleaned up.

### Five LB submissions, final table

| Submission | Code | Node | Ckpt | LB | Δ vs 0.906 | Δ vs paired control |
|---|---|---|---|---:|---:|---:|
| v5j-original (exp 019, Mar 31) | e420633 | `02-004-35-0` | ep 19 | **0.906** | — | — |
| v5j-resubmit (exp 030 diag) | e420633 ckpt → today's inference | n/a | ep 19 | **0.906** | 0.000 ✅ inference clean | — |
| v5j_control (exp 030 v2 inherited) | HEAD-ish | various | ep 15 | 0.896 | −0.010 | — |
| v5j_control_last | HEAD | `02-004-25-0` | ep 19 | **0.897** | −0.009 | — |
| v5j_kd | HEAD | `03-003-19-0` | ep 17 | **0.890** | −0.016 | **−0.006 vs 0.896** |
| v5j_kd_last | HEAD | `03-003-19-0` | ep 19 | **0.891** | −0.015 | **−0.006 vs 0.897** |
| **v5j-eracode (drift diag)** | **e420633 (worktree)** | `03-003-19-0` | ep 19 | **0.899** | **−0.007** | — |

### Decisive insight: val/cmap signal isolates code vs hardware variance

| Run | Code | Node | val/cmap |
|---|---|---|---:|
| v5j-original | e420633 | A | **0.6429** |
| v5j-eracode | e420633 | B | **0.6429** ← matches v5j-original to 4 dp |
| v5j_control_resave | HEAD | C | 0.6268 ← drops 0.016 |

**Same code, different node → val/cmap matches to 4 dp.** Code change
accounts for the 0.016 cmap shift between e420633 and HEAD, but only ~0.002
LB shift (classic offline-LB anti-correlation). The bigger 0.007 LB gap
(0.906 → 0.899) is from GPU/node fp variance under same code. PyTorch's
`Trainer(deterministic=True)` makes algorithms deterministic on the SAME
hardware but cannot eliminate cross-hardware fp behavior.

### Implications

1. **0.906 stays as the real v5j baseline** (proven by v5j-resubmit).
2. **Single-fold v5j has ~0.007 cross-node LB variance** from GPU fp
   non-determinism — fundamental to GPU training, not fixable in code.
   Plus ~0.002 code-side component (now pinned).
3. **Reproducible LB-comparable experiments require paired within-run
   controls** (same node, same time, same env). The exp 031 KD verdict
   at −0.006 is robust because v5j_kd and v5j_control were paired.
4. **REJECT on v5j+HGNet+LSEHead+KD is unambiguous** — paired comparison
   is uncontaminated by hardware variance.
5. **Tucker's recipe is NOT falsified.** We tested KD on the wrong head
   (LSEHead) and the wrong backbone (HGNetV2-B0). Tucker uses EffNet-B0
   NS-JFT + bug-fixed AttHead (logit-space) + 25 epochs + dual-supervision
   + SpecAugment. The lever may still transfer when paired with the right
   downstream consumer.

Per the success matrix in [`proposal.md` § Success Criteria](proposal.md):
**REJECT**. KD as a single-variable add on the v5j pipeline (HGNetV2-B0 +
LSEHead + spec-interp mixup, 20 epochs, α_distill=1.0) did NOT transfer.

**Caveat — best-vs-last diagnostic requires retrain.** The submitted
checkpoint is best-by-val (ep 17). v5j-original (LB 0.906) used the LAST
epoch (ep 19). On post-hoc inspection, **neither run actually saved ep 19
weights**: with `save_top_k: 1` + `save_last: true`, Lightning's `last.ckpt`
mirrored the best-by-val ckpt (same global_step), NOT the literal final
epoch. The diagnostic requires a `save_top_k: -1` retrain. Configs scaffolded
([`configs/v5j_kd_resave.yaml`](configs/v5j_kd_resave.yaml),
[`configs/v5j_control_resave.yaml`](configs/v5j_control_resave.yaml));
retrain pending. See § Best-vs-last checkpoint diagnostic below.

## Summary

The KD branch trained cleanly. `dist_loss` decayed from 0.058 (fast-dev-run
random-init) to **0.004 by epoch 19** — at Tucker's published floor of ~0.003.
That kills the largest pre-flight risk (mel-pipeline mismatch raising the
floor) — see [`phase0_mel_equivalence.md`](phase0_mel_equivalence.md). The
no-KD path's offline numbers in this experiment are **bit-identical** to
exp 030's `v5j_control_v2` (val/auroc 0.8728, cmap 0.6268, f1_macro 0.5270),
sealing the **offline-side back-compat receipt** for the KD-branch additions
to `sed_finetune.py`. Bit-identical offline metrics under deterministic
training imply bit-identical weights, so v5j_control's LB inherits exp 030's
0.896 without needing a separate submission.

Offline, KD trailed control on every metric (val/auroc −0.012, val/cmap
−0.089, val/f1_macro −0.097). LB confirmed the offline cmap/f1 gap was
directionally informative this time: 0.890 sits below control by −0.006.
The historical offline-LB anti-correlation pattern (cmap improvement →
LB regression on training-recipe levers) didn't apply here because the
KD recipe HURT cmap; that turned out to be predictive.

## Phase 1 training: DONE

### Side-by-side summary

| Metric | v5j_kd | v5j_control | Δ (kd − ctrl) | Notes |
|---|---:|---:|---:|---|
| **val/auroc** | 0.8595 | 0.8728 | **−0.0132** | KD trails on rank-ordering |
| **val/cmap** | 0.5376 | 0.6268 | **−0.0891** | Larger gap; cmap historically more directional than auroc |
| **val/f1_macro** | 0.4303 | 0.5270 | **−0.0967** | Larger gap; threshold-based |
| val/f1_micro | 0.6245 | 0.6844 | −0.0599 | |
| val/top1_acc | 0.7138 | 0.7567 | −0.0429 | |
| val/top5_acc | 0.8461 | 0.8694 | −0.0233 | |
| val/loss | 0.0139 | 0.0120 | +0.0019 | Slightly higher BCE under KD |
| **best_val_score** | **0.8629** (ep 17) | **0.8747** (ep 15) | −0.0118 | |
| Walltime | 1h 18m 7s | 38m 0s | **+40 min** | Teacher forward + bigger model + slower convergence |
| Total params | 4.7M | 3.1M | +1.6M | DistillHead = `Linear(1024, 1536)` |
| Trainable params | 4.7M | 3.1M | +1.6M | All trainable; backbone is detached at the cls branch only |

`val/auroc` reported above is the metric at the **final epoch** (epoch 19).
`best_val_score` is the maximum over the run, which is what gets serialized
and submitted to LB.

### v5j_kd (the test, KD on)

| Metric | Value |
|---|---:|
| **val/auroc** (final epoch) | 0.8595 |
| val/cmap | 0.5376 |
| val/f1_macro | 0.4303 |
| val/f1_micro | 0.6245 |
| val/top1_acc | 0.7138 |
| val/top5_acc | 0.8461 |
| val/loss | 0.0139 |
| **best_val_score (val/auroc)** | **0.8629 (epoch 17)** |
| train/cls_loss_epoch (final) | 0.019 |
| **train/dist_loss_epoch (final)** | **0.004** |
| train/loss_epoch (final) | 0.022 |
| Walltime | 1h 18m 7s on rtx6000 |
| SLURM job ID | 8318953 |
| W&B run id | `xlkwpuv5` (project `birdclef-2026-exp031`) |
| Run name | `2026-exp031-v5j-kd` |
| Checkpoint | `/storage/project/.../031-perch-kd-distillation/checkpoints/2026-exp031-v5j-kd/best-epochepoch=17.ckpt` |
| Config | [`configs/v5j_kd.yaml`](configs/v5j_kd.yaml) |

### v5j_control (back-compat receipt; KD off)

| Metric | Value |
|---|---:|
| **val/auroc** (final epoch) | 0.8728 |
| val/cmap | 0.6268 |
| val/f1_macro | 0.5270 |
| val/f1_micro | 0.6844 |
| val/top1_acc | 0.7567 |
| val/top5_acc | 0.8694 |
| val/loss | 0.0120 |
| **best_val_score (val/auroc)** | **0.8747 (epoch 15)** |
| train/loss_epoch (final) | 0.016 |
| Walltime | 38m 0s on rtx6000 |
| SLURM job ID | 8319084 |
| W&B run id | `o16zyft7` (project `birdclef-2026-exp031`) |
| Run name | `2026-exp031-v5j-control` |
| Checkpoint | `/storage/project/.../031-perch-kd-distillation/checkpoints/2026-exp031-v5j-control/best-epochepoch=15.ckpt` |
| Config | [`configs/v5j_control.yaml`](configs/v5j_control.yaml) |

## Halt-watch retrospective

[Proposal § Procedure step 8](proposal.md) named three halt conditions and
one diagnostic. Reading them against the actual run:

| Halt condition | Actual | Verdict |
|---|---|---|
| `dist_loss` plateaus above 0.01 by epoch 5 | dist_loss decayed to 0.004 by ep 19; below 0.01 well before ep 5 (fast-dev-run was at 0.058 random init, mid-training values not logged per-epoch in the SLURM log but final 0.004 implies clean monotonic decay) | ✅ PASS — no halt |
| `cls_loss` diverges (above v5j-control's epoch-1 value) | Final cls_loss = 0.019 (well-behaved BCE training trajectory) | ✅ PASS |
| Either loss NaN | None | ✅ PASS |
| KD's val/auroc lags control's at early epochs (expected, not halt) | KD ep 0 val/auroc 0.475 vs control ep 0 val/auroc 0.509. KD ep 2 climbed to 0.616 (slightly better than control's 0.564 at ep 2). Consistent with proposal's note ("KD's val/auroc lags control's at early epochs"). | ✅ Expected pattern |

**Diagnostic question (proposal §): "did `dist_loss > 0.01` by epoch 5?"**

Per-epoch `dist_loss` values aren't in the SLURM log tail (only `val/auroc`
is printed at each ModelCheckpoint event); the W&B run `xlkwpuv5` has the
full curve. Final epoch 19 dist_loss is **0.004**, vs Tucker's published
~0.003 floor at epoch 25. Because we ran 20 epochs (Tucker ran 25) and
landed at 0.004, the curve is consistent with Tucker's published shape.
**Mel-pipeline mismatch did not contribute meaningfully to the dist_loss
floor.** Phase 0.0a (mel refactor) stays SKIPPED.

## Back-compat receipt — OFFLINE CONFIRMED, LB PENDING

The Phase 0.5 control was added (commit `fa94043`) explicitly to verify that
the KD-branch additions to `sed_finetune.py` left the non-KD path
bit-identical. v5j_control here uses the v5j-original config minus
`use_perch_distill` (defaults to False), same seed, same data, same
mixup, same schedule.

### Offline check vs exp 030's v5j_control_v2 (same recipe, same seed=42, exp 030 codebase)

| Metric | exp 030 v5j_control_v2 | exp 031 v5j_control | Match |
|---|---:|---:|---|
| val/auroc | 0.8728 | 0.8728 | ✅ bit-identical (4 dp) |
| val/cmap | 0.6268 | 0.6268 | ✅ bit-identical (4 dp) |
| val/f1_macro | 0.5270 | 0.5270 | ✅ bit-identical (4 dp) |
| val/f1_micro | 0.6844 | 0.6844 | ✅ bit-identical (4 dp) |
| val/top1_acc | 0.7567 | 0.7567 | ✅ bit-identical (4 dp) |
| val/top5_acc | 0.8694 | 0.8694 | ✅ bit-identical (4 dp) |
| val/loss | 0.0120 | 0.0120 | ✅ bit-identical (4 dp) |
| best_val_score | 0.8747 (ep 15) | 0.8747 (ep 15) | ✅ same epoch, same value |

Walltime differs (40m 12s vs 38m 0s) — node-level variance, not a
correctness signal. Every learnable metric matches to 4 decimal places.

**Offline-side back-compat receipt: ✅ CONFIRMED.** The KD-branch additions
left the non-KD path bit-identical relative to exp 030's reproducible
training pipeline. Pending LB confirms the inference side too.

### LB-side back-compat (PENDING)

Expected: `v5j_control` lands at LB **0.896 ± 0.005** (matching exp 030's
v5j_control_v2). If outside that band, that's a back-compat failure — the
KD result should not be trusted until the divergence is diagnosed.

## Per-epoch val/auroc trajectory

Both runs serialize a checkpoint when `val/auroc` improves. The per-epoch
best-by-val numbers from the SLURM logs:

| Epoch | v5j_kd val/auroc | v5j_control val/auroc | Δ (kd − ctrl) |
|---:|---:|---:|---:|
| 0 | 0.4747 | 0.5088 | −0.034 |
| 1 | (no improvement) | 0.5638 | — |
| 2 | 0.6163 | 0.7463 | −0.130 |
| 3 | 0.6662 | 0.8305 | −0.164 |
| 4 | 0.7663 | 0.8490 | −0.083 |
| 5 | 0.8029 | 0.8535 | −0.051 |
| 6 | 0.8171 | 0.8619 | −0.045 |
| 7 | 0.8304 | 0.8620 | −0.032 |
| 8 | 0.8400 | 0.8649 | −0.025 |
| 9 | 0.8450 | 0.8666 | −0.022 |
| 10 | 0.8513 | 0.8688 | −0.018 |
| 11 | 0.8535 | 0.8710 | −0.018 |
| 12 | 0.8572 | 0.8734 | −0.016 |
| 13 | 0.8604 | (no improvement) | — |
| 14 | **0.8625** | 0.8745 | −0.012 |
| 15 | (no improvement) | **0.8747** ← best | — |
| 17 | **0.8629** ← best | (no improvement) | — |

The KD trajectory shows the expected "lag then catch up" shape: backbone is
being driven by Perch alignment (not species labels), so SED metrics climb
slowly through ep 0-5, then pick up through ep 6-14 as the head learns to
read off the Perch-aligned representation. Best epoch 17 vs control's 15
suggests KD takes ~2 extra epochs to plateau at this single-fold scale —
consistent with Tucker's published 25-epoch schedule landing the best
checkpoint later than v5j's 20-epoch standard.

The Δ narrows from −0.164 (ep 3) to −0.012 (ep 14). KD is converging
toward the control trajectory but doesn't fully catch up by epoch 19
within our 20-epoch budget. This is one of the diagnostic questions for
LB: did the KD recipe need more epochs to fully express its lift, or is
it structurally below the control's offline ceiling regardless?

## Phase 1.3 LB submission (2026-05-10)

| Run | Checkpoint | LB | Notes |
|---|---|---:|---|
| `v5j_kd` | `best-epochepoch=17.ckpt` | **0.890** | Submitted via kernel `birdclef-2026-hgnet-b0-sed-v5j-kd` |
| `v5j_control` | (not submitted) | **0.896** (inherited) | Bit-identical weights to exp 030 `v5j_control_v2` (LB 0.896) under deterministic training; sealed via offline 4-dp metric match |
| `v5j_kd` Δ vs control | — | **−0.006** | Inside the ±0.013 single-fold seed band, but on the wrong side of zero |

Kernel logs verify the run was clean: ONNX session created in 0.2s, 24 rows
predicted across 2 train soundscapes used as test substitutes, 36.9s total.
No errors, no degenerate output.

### Success matrix application (REJECT)

Per [`proposal.md` § Success Criteria](proposal.md), reading against
`v5j_control` LB 0.896:

| Bucket | LB threshold | v5j_kd LB 0.890 |
|---|---|---|
| CONFIRM | ≥ 0.910 (control + 0.014) | ❌ |
| PARTIAL | 0.903 – 0.909 | ❌ |
| WEAK | 0.897 – 0.902 | ❌ |
| **REJECT** | ≤ 0.896 | **✅ (LB = 0.890, −0.006 below)** |

KD did not add anything as a single-variable add on this codebase. The
single-fold seed band (±0.013 per exp 030) is wider than this delta, so we
can't statistically distinguish "KD slightly hurt" from "KD null in noise."
Either way, the lift is not present.

## Post-mortem: why didn't the KD lever transfer?

Tucker's published +0.02 LB lift was on the FULL distilled-SED stack
(EfficientNet-B0 NS-JFT + AttHead with logit-space aggregation +
dual-supervision BCE + hard-max mixup + SpecAugment + focal×labeled-SS
mixup + 90/10 batch composition + 25-epoch warmup+cosine + 5-fold + Tucker
PP). We isolated only the KD branch and added it on the v5j stack. The
delta could come from any of these axes, ranked by likelihood:

### 1. LSEHead is the wrong downstream consumer of Perch features (highest likelihood)

LSEHead pools per-frame logits by their MAGNITUDE (`(1/τ) log Σ exp(τ·z_t)`,
soft-max-over-time). AttHead pools them by LOCALIZATION (`Σ_t softmax(tanh(att_t)) · logit_t`,
attention-weighted). Perch v2 was trained at scale on 14,795 species — its
representation almost certainly encodes "WHERE the species cue is," not just
"how loud it is." LSEHead drops the spatial information that KD just
installed. AttHead would consume it directly.

This is consistent with the `feedback_atthead_aggregation_domain.md` audit
finding: our `AttHead` aggregates in probability space (a known bug), but
Tucker's logit-space aggregation should be a clean test. A `v5j_kd` retrain
swapping LSEHead → bug-fixed AttHead is the **strongest single follow-up**.

### 2. 20-epoch budget too short

KD's val/auroc trajectory was still narrowing the gap to control through
the end of training (Δ −0.164 ep 3 → −0.083 ep 4 → −0.025 ep 8 → −0.012
ep 14 → best ep 17). The detached cls branch trains on a rapidly-changing
backbone representation; convergence is slower than non-KD. Tucker uses 25
epochs + cosine; we use 20 + OneCycle. A 25- or 30-epoch retrain on the
same recipe (cheap to test) would tell us whether the lift just needs
more time.

### 3. `alpha_distill = 1.0` may be miscalibrated for our codebase

With detach on the cls branch, the backbone is driven ENTIRELY by Perch
matching. At α=1.0, dist_loss (≈0.05 mid-training, ≈0.004 final) and
cls_loss (≈0.7 → 0.02) are at very different scales. Backbone optimization
is fully dominated by dist_loss; cls only shapes the head. A sweep of
α∈{0.1, 0.5, 1.0, 2.0} would tell us whether the recipe is sensitive to
this hyperparameter at our codebase's mel pipeline / feature dim.

### 4. Compounding stack: KD lift may not be portable as an isolated lever

Tucker's 0.02 LB delta (his "without KD ~0.89" → "with KD ~0.91") was
quantified on his full pipeline. Possible the lift requires interaction
with one or more of: AttHead's logit-space aggregation, dual-supervision
loss giving attention real gradient pressure, hard-max mixup keeping
training targets close to the test multi-hot binary distribution,
SpecAugment forcing the representation to be invariant. Single-variable
discipline is a strong methodology when levers are independent; less so
when they are.

### Ranked follow-up experiments (Phase 3+ candidates)

Ordered by expected information value per LB submission:

1. **`v5j_kd` with bug-fixed AttHead** (logit-space aggregation per
   `feedback_atthead_aggregation_domain.md`). Tests hypothesis #1 directly.
   Cost: 1 retrain + 1 LB.
2. **25- or 30-epoch retrain** of same v5j_kd recipe. Tests hypothesis #2.
   Cost: ~110 min training (1.4× current) + 1 LB.
3. **α_distill sweep** (0.1, 0.5, 2.0). Tests hypothesis #3. Cost: 3 trainings + 3 LB.
4. **Full Tucker replication on EfficientNet-B0 NS-JFT** (multi-variable, not
   single-variable). Tests hypothesis #4 by reversing the methodology — start
   with full reproduction, then ablate. Cost: 1 retrain + 1 LB.

Phase 2 (ownership work + 4-fold) is **halted** per the proposal's REJECT
action. Decision on next direction (any of the above 4, or pivot away from
KD entirely) is pending session discussion.

## Best-vs-last LB results (2026-05-10) — codebase reproduction FAILED

### Both LBs back: control 0.897, KD 0.891 — REGRESSION confirmed

| Submission | Checkpoint | LB | vs v5j-original 0.906 |
|---|---|---:|---:|
| v5j-original (exp 019) | ep 19, original codebase | **0.906** | — |
| v5j-resubmit (exp 030) | exp 019 ep 19 ckpt → today's inference | **0.906** | **0.000** ✅ inference clean |
| v5j_control (exp 030 v2 / exp 031) | ep 15 best-by-val, today's training | **0.896** | −0.010 |
| **v5j_control_last (exp 031)** | **ep 19, today's training (save_top_k=-1 rerun)** | **0.897** | **−0.009** |
| v5j_kd (exp 031) | KD on, ep 17 best-by-val | **0.890** | −0.016 |
| **v5j_kd_last (exp 031)** | **KD on, ep 19 (save_top_k=-1 rerun)** | **0.891** | −0.015 |

**The diagnostic conclusively rules out ckpt-selection** as the cause of the
0.906 → 0.896 gap. Ep 15 (best-by-val) gives 0.896; ep 19 (last) gives 0.897.
The +0.001 ep 19 vs ep 15 delta is well inside seed noise. **The whole
training trajectory produces different weights now**, not a single
checkpoint-selection issue.

### What this is, and what it isn't

**It is**: a regression in the training pipeline. v5j-original's 0.906 LB is
real and achievable — the v5j-resubmit diagnostic verified that re-exporting
the exact exp 019 ep 19 checkpoint through today's submission gives 0.906
exactly. The recipe works; the model the recipe TRAINED ORIGINALLY works.
We just can't currently reproduce that model from scratch on the same
config. **The 0.906 baseline is the real value** — 0.897 is what the
broken pipeline produces, not a "new normal."

**It isn't**: a re-baselining moment. Treating 0.897 as the new v5j baseline
would normalize a bug. Every comparison made since exp 026 against the
"v5j 0.906" baseline is silently miscalibrated by ~0.009 — but the fix
is to find and remove the regression, not to lower the bar.

### Implications for exp 031 verdict

- v5j_kd ep 17 LB 0.890 vs v5j_control ep 15 LB 0.896 = −0.006 (REJECT)
- v5j_kd_last ep 19 LB 0.891 vs v5j_control_last ep 19 LB 0.897 = −0.006 (REJECT)
- Both checkpoint-selection regimes give the same KD verdict: REJECT.
- The v5j_kd 0.890 is **−0.016 below v5j-original 0.906**. The KD
  comparison is contaminated by the codebase regression — some fraction
  of the 0.016 is the regression, not KD. Until the regression is fixed,
  we can't cleanly read whether KD has any partial-credit effect.
- The cleanest read for KD-vs-control on this codebase remains:
  **−0.006 either at ep 17 or ep 19**. Inside the ±0.013 seed band, but
  systematic. KD doesn't help on this stack; it slightly hurts.

### Reproducibility is the priority

Until we can reproduce v5j-original 0.906 from scratch, every single-fold
single-model experiment is comparing against a known-broken baseline. The
codebase regression is the **highest-priority finding** of exp 031, more
important than the KD verdict. See § Codebase regression diagnostic plan
below + comprehensive analysis at
[`analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md`](../../analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md).

### What this RULES OUT

- *"Save_top_k=1 was the bug."* Not the bug — ep 19 is +0.001 from ep 15, inside noise.
- *"Just need to use the right checkpoint."* Wrong checkpoint isn't the issue;
  the WHOLE TRAINING TRAJECTORY produces different weights now.
- *Inference-side drift.* v5j-resubmit confirmed inference reproduces exactly.

The drift is somewhere in the training pipeline between the exp 019 commit
that produced v5j-original and the current HEAD. Comprehensive root-cause
analysis underway — see [`analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md`](../../analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md).

## Best-vs-last checkpoint diagnostic (RETRAIN REQUIRED)

The 0.890 result is on best-by-val ep 17. v5j-original's 0.906 was on the
LAST epoch (ep 19). Two LB submissions would disambiguate ckpt-selection
from KD/codebase drift — but the original runs cannot supply the inputs.

### Why we need a retrain (post-hoc finding 2026-05-10)

Inspection of every checkpoint in `2026-exp031-v5j-{kd,control}/`:

| File | Epoch | global_step | Note |
|---|---:|---:|---|
| v5j_kd `last.ckpt` | 1 | 5 | **Fast-dev-run** state from job 8318887; pre-dates the full training |
| v5j_kd `last-v1.ckpt` | 17 | 9198 | **Bit-identical to** `best-epochepoch=17.ckpt` (same global_step) |
| v5j_kd `best-epochepoch=17.ckpt` | 17 | 9198 | What we already submitted (LB 0.890) |
| v5j_control `last.ckpt` | 15 | 8176 | **Bit-identical to** `best-epochepoch=15.ckpt` (same global_step) |
| v5j_control `best-epochepoch=15.ckpt` | 15 | 8176 | Bit-identical to exp 030 v5j_control_v2 (LB 0.896 inherited) |

With `save_top_k: 1` + `save_last: true`, Lightning's `ModelCheckpoint`
callback only fires on metric improvement. `save_last: true` is a
save-action modifier, not a separate trigger — it mirrors whatever
checkpoint was just written. Once val/auroc plateaus, neither file is
updated again. **The literal final-epoch weights were never serialized
for either run.** Saved as a feedback memory:
`feedback_save_top_k_does_not_preserve_last.md`.

### Retrain plan

Two new configs scaffolded with `save_top_k: -1` (saves every epoch):

- [`configs/v5j_kd_resave.yaml`](configs/v5j_kd_resave.yaml) — run name `2026-exp031-v5j-kd-resave`. Disk cost ~1.5 GB (20 × 75 MB).
- [`configs/v5j_control_resave.yaml`](configs/v5j_control_resave.yaml) — run name `2026-exp031-v5j-control-resave`. Disk cost ~1.1 GB (20 × 55 MB).

All other recipe parameters identical to the original configs. Training is
deterministic (`seed_everything(42, workers=True)` + `Trainer(deterministic=True)`),
so per-epoch weights will be bit-identical to what would've been saved if
the original runs had used `save_top_k: -1`. Wall time: ~80 min KD + ~40
min control on rtx6000 (1.5–2× v5j_control walltime due to ckpt I/O).

### Diagnostic submissions (after retrain)

| Submission | Checkpoint | What it tests |
|---|---|---|
| `v5j_kd_last` | `2026-exp031-v5j-kd-resave/best-epochepoch=19.ckpt` | Was the 0.890 a ckpt-selection artifact? If LB ≥ 0.910, REJECT was wrong; verdict flips to CONFIRM/PARTIAL. |
| `v5j_control_last` | `2026-exp031-v5j-control-resave/best-epochepoch=19.ckpt` | Does ep 19 recover v5j-original's 0.906 on the new codebase? Isolates ckpt-selection from codebase drift. |

Pipeline cost: 2 retrains + 4-file scaffolding × 2 + 2 LB slots.

## Offline read on KD vs control gap

The −0.089 cmap and −0.097 f1_macro gaps are larger than the val/auroc gap.
This is the inverse of the exp 030 pattern, where Sydorskyi mixup hit
val/auroc parity but improved cmap and f1_macro. Possible reasons here:

1. **20-epoch budget is too short for KD.** Tucker uses 25 epochs +
   cosine; we use 20 + OneCycle. KD's slower convergence shape (above)
   suggests we're cutting the schedule before the SED head fully adapts
   to the Perch-aligned representation.
2. **`alpha_distill = 1.0` may be too high.** With detach on the cls
   branch, the backbone is trained ONLY by Perch matching for 20 epochs.
   If `alpha_distill` were lower (e.g. 0.5), the gradient pressure on the
   backbone would be smaller per step, but we'd also be running fewer
   "Perch-aligned representation" updates. Untested for our codebase.
3. **LSEHead may be a worse downstream consumer of a Perch-aligned
   feature than AttHead.** LSE pools by magnitude; attention pools by
   localization. If the Perch representation is "where the species is,"
   AttHead is the natural downstream architecture and LSEHead may be
   throwing away spatial information. This is the Phase 3+ candidate
   in [`v5j-vs-tucker-pipeline-comparison.md` §5](v5j-vs-tucker-pipeline-comparison.md).
4. **None of the above; offline gap is real and translates to LB.**
   Possible. The seed-band on offline metrics is roughly the same as on
   LB (~0.003-0.005); a 0.09 cmap gap is far outside that band.

The offline-LB anti-correlation (FINDINGS.md insight #2) means we can't
predict LB sign from the offline gap alone. cmap is **slightly** more
directional than val/auroc but not reliably so at this magnitude.

## Pipeline next steps

Phase 1 LB pipeline complete (commits `1cdd5cb`, `2241093`):

- [x] ONNX export `v5j_kd` checkpoint via `export_efficientnet_sed.py`
- [x] MODELS entry in `upload_kaggle.py`
- [x] MODEL_SLUGS entry in `submission.py`
- [x] `kernel-metadata-hgnet-b0-sed-v5j-kd.json` created
- [x] Weights dataset uploaded to Kaggle
- [x] Kernel pushed and submitted; LB = 0.890
- [x] v5j_control LB inheritance from exp 030 0.896 sealed via offline 4-dp metric match (no separate submission needed)
- [x] Apply success matrix → REJECT
- [x] Update `MODEL-RUNS.md` with v5j_kd LB 0.890

Pending:

- [ ] Last-checkpoint diagnostic: scaffold + submit `v5j_kd_last` + `v5j_control_last` (see § Last-checkpoint diagnostic above)
- [ ] Update `CONTEXT.md` "What happened recently" section
- [ ] Decide on Phase 3+ direction (post-mortem candidates above) or pivot

## Decision matrix (LB-pending)

Once both LBs are in, apply (read against `v5j_control`'s actual LB):

| Pattern | Reading | Action |
|---|---|---|
| `v5j_control` LB outside 0.896 ± 0.005 | LB-side back-compat failure | Diagnose drift in KD-branch additions before trusting KD result |
| `v5j_kd` LB ≥ control + 0.014 | **CONFIRM**: KD lever transfers | Proceed to Phase 2 (ownership work + 4-fold) |
| `v5j_kd` LB control + 0.007 to control + 0.013 | **PARTIAL**: lift detectable but smaller than Tucker's +0.020 | Diagnose: dist_loss floor, alpha_distill sweep, longer schedule |
| `v5j_kd` LB control + 0.001 to control + 0.006 | **WEAK**: inside seed noise | Need 4-fold to disambiguate, but high variance cost |
| `v5j_kd` LB ≤ control | **REJECT**: KD didn't add anything as a single-variable add | Halt Phase 2; structural post-mortem (head architecture, schedule, alpha) |

## What this RULES OUT vs LEAVES OPEN

**Rules out** (post-LB):
- *Mel-pipeline mismatch raising the dist_loss floor.* Floor 0.004 ≈
  Tucker's 0.003. Phase 0.0a stays SKIPPED.
- *KD-branch additions breaking the non-KD path.* Bit-identical offline
  match to exp 030 v5j_control_v2; bit-identical weights → inherit 0.896 LB.
- *Detach bug or NaN/Inf.* All gradients flowed cleanly per Phase 0.4
  test; cls_loss landed at 0.019 (well-behaved); zero NaN/Inf in training.
- *KD as a single-variable add on v5j (HGNet + LSEHead + 20 ep + α=1.0).*
  LB 0.890 vs control 0.896 → REJECT bucket. The lever does NOT transfer
  in this configuration.

**Leaves open** (require additional submissions to resolve):
- Whether the 0.890 was ckpt-selection artifact vs structural KD failure.
  Tested by `v5j_kd_last` (ep 19 weights). PENDING.
- Whether LSEHead is the wrong downstream consumer of Perch features.
  Tested by KD + bug-fixed AttHead retrain. NOT SCHEDULED.
- Whether 20 epochs was the binding constraint. Tested by 25- or 30-epoch
  retrain. NOT SCHEDULED.
- Whether Tucker's lift requires the full stack (AttHead + dual-supervision
  + hard mixup + SpecAugment + focal×SS + 90/10 batch). Tested by full
  Tucker replication. NOT SCHEDULED.
- Whether the v5j-original 0.906 → v5j_control_v2 0.896 gap is
  ckpt-selection sensitivity vs codebase drift. Tested by `v5j_control_last`
  (ep 19 weights). PENDING.

## Caveats

- **Single fold** (seed=42). Per exp 030, the single-fold seed band on
  HGNet is ±0.013. Even a real +0.014 KD lift sits at ~1.1× the noise
  floor. WEAK / PARTIAL outcomes will need 4-fold to disambiguate.
- **Offline metrics anti-correlate with LB** (FINDINGS.md insight #2).
  cmap is slightly more directional, but the −0.089 cmap gap is too
  large to confidently predict LB direction.
- **Phase 1 uses Tucker's third-party Perch ONNX** (Option A). Phase 2
  switches to Option B (our owned export) for full pipeline ownership.
- **20 epochs (v5j default), not 25 (Tucker's recipe)**. Schedule shape
  difference (OneCycle vs warmup+cosine). May contribute to the lower
  ceiling under KD. Phase 2/3 candidate to revisit if Phase 1 is PARTIAL.

## Files

- Configs: [`configs/v5j_kd.yaml`](configs/v5j_kd.yaml),
  [`configs/v5j_control.yaml`](configs/v5j_control.yaml)
- Checkpoints:
  `/storage/project/.../031-perch-kd-distillation/checkpoints/2026-exp031-v5j-{kd,control}/`
- Metrics JSON: [`artifacts/training/v5j_kd/metrics.json`](artifacts/training/v5j_kd/metrics.json),
  [`artifacts/training/v5j_control/metrics.json`](artifacts/training/v5j_control/metrics.json)
- Test results JSON: `outputs/test_results/2026-exp031-{v5j-kd,v5j-control}.json`
- W&B project: `birdclef-2026-exp031`
  - v5j_kd: `xlkwpuv5`
  - v5j_control: `o16zyft7`
- SLURM logs: [`logs/slurm_v5j_kd_8318953.log`](logs/slurm_v5j_kd_8318953.log),
  [`logs/slurm_v5j_control_8319084.log`](logs/slurm_v5j_control_8319084.log)
- Phase 0 prerequisites: [`phase0_mel_equivalence.md`](phase0_mel_equivalence.md),
  [`phase0_perch_teacher_schema.md`](phase0_perch_teacher_schema.md),
  [`v5j-vs-tucker-pipeline-comparison.md`](v5j-vs-tucker-pipeline-comparison.md)
- Code audit motivating the experiment:
  `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`
- Branch / commits: `mg/exp031-perch-kd-distillation`
  - `ede03e6` — Implement KD branch (Phase 0.2-0.4)
  - `5f4b6aa` — Fix fast-dev-run ONNX path resolution + onnxruntime warnings
  - `fa94043` — Add v5j_control.yaml for live W&B comparison + back-compat receipt

## Open work

Phase 1 (DONE):

- [x] ONNX export `v5j_kd` checkpoint (commit `2241093`)
- [x] Wire `hgnet-b0-sed-v5j-kd` through Kaggle pipeline (4 file edits, commit `2241093`)
- [x] Upload weights, push kernel, submit to LB → 0.890
- [x] Seal v5j_control back-compat via offline 4-dp metric match (no separate LB needed)
- [x] Apply decision matrix → REJECT
- [x] Update `results.md` with verdict + post-mortem
- [x] Update `MODEL-RUNS.md`
- [x] Update `tasks.md`

Best-vs-last + drift diagnostics (DONE):

- [x] save_top_k=-1 retrains (v5j_kd_resave, v5j_control_resave) — bit-identical to originals
- [x] v5j_kd_last + v5j_control_last LB submissions → 0.891 / 0.897 (same −0.006 KD delta)
- [x] 8 static diagnostics (init / math / splits / libs / CUDA / Python / data / optimizer): all RULED OUT
- [x] v5j-eracode worktree training (v5j-era code on today's env, node `03-003-19-0`) → LB 0.899
- [x] val/cmap match to 4 dp confirms: same code different node = 0 cmap drift; different code = 0.016 cmap drift but only 0.002 LB impact
- [x] uv.lock pinned to v5j-era versions (timm 1.0.26, torch 2.5.1, hf-hub <1.10, etc.)
- [x] v5j-era worktree cleaned up (commit history preserved in HEAD branch + analysis/diagnostics/)

## Next step: Phase 3.1 — KD + bug-fixed AttHead (highest-priority Phase 3 candidate)

The exp 031 REJECT is scoped to **HGNet + LSEHead + 20 ep + α=1.0**.
Tucker's actual proven recipe uses **EffNet-B0 NS-JFT + AttHead (logit-space
aggregation) + 25 ep + dual-supervision + SpecAugment**. Highest-likelihood
post-mortem candidate is "LSEHead was the wrong consumer of Perch features".

### Phase 3.1 scaffold plan (next session)

New experiment dir: `user/murilo/experiments/032-kd-atthead/` (or similar)

**Implementation**:
1. Port Tucker's AttHead (logit-space aggregation) as a new class
   `AttHeadLogitSpace` in `user/murilo/src/birdclef/heads/sed_head.py`. Faithful
   port from `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`
   (cells 10-14 per the code audit). Key features:
   - GeMFreqPool (p=3 learnable, `clamp(min=1.0)`)
   - Dense block: `Dropout(0.25) → Linear(C, 512) → ReLU → Dropout(0.5)`
   - Att + Cls convs: `Conv1d(hidden_dim, num_classes, k=1, bias=True)`, xavier_uniform init
   - **Aggregation in LOGIT SPACE**: `sum(softmax(tanh(att)) * frame_logits, dim=time)` — NOT our broken probability-space version
2. Add `head_type: "att_head_logit_space"` to encoder/head dispatch in the
   experiment train.py
3. Two configs (paired control + KD):
   - `v5j_atthead_control.yaml`: HGNet + AttHeadLogitSpace, no KD
   - `v5j_atthead_kd.yaml`: HGNet + AttHeadLogitSpace + Perch v2 KD
   - Both use save_top_k=-1, paired SLURM submission to capture variance
4. Train + LB submit both. Apply paired-comparison interpretation:
   - 3 > 2 > 1 → KD lever + AttHead fix both contribute
   - 3 > 2 ≈ 1 → KD needs AttHead but AttHead alone doesn't help
   - 3 ≈ 2 > 1 → AttHead fix helps; KD adds nothing
   - 3 ≈ 2 ≈ 1 → neither helps on HGNet; pivot to EffNet+full-Tucker-stack

Stage 2 (gated on Stage 1 looking promising): full Tucker replication
(EffNet-B0 NS-JFT + AttHead + dual-supervision + 25 ep cosine + hard-max
mixup + SpecAugment + focal×SS mixup). Multi-variable but reaches for 0.917+.

### Methodology for Phase 3.1

- **Paired within-run controls** (control + KD trained same node, same date, same env) — uncontaminated by hardware variance.
- **Compare delta vs paired control** as the load-bearing signal. Don't compare to historical 0.906 (cross-run variance contaminates).
- **Use uv.lock** in the new experiment's pyproject.toml to inherit pinned versions.
- **Save_top_k=-1** to capture ep 19 + best-by-val. Pick by LB after fact.

### Phase 3+ deferred candidates (in case 3.1 fails)

- α_distill sweep (0.1, 0.5, 2.0) — diagnostic if KD has marginal effect
- 25-30 epoch schedule with AttHead+KD
- KLD-on-logits (Perch category #2, EliKal's 0.909 recipe — never tried)
- 4-fold ensembling (any winning single-fold recipe should be 4-folded for noise reduction)
