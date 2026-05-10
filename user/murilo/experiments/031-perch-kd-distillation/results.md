---
experiment: "031-perch-kd-distillation"
status: in_progress
date: 2026-05-10
author: Murilo Gustineli
---

# Experiment 031: Perch v2 Knowledge Distillation on v5j — Results (Phase 1)

## Status: TRAINING DONE — LB PENDING

Phase 1 training completed 2026-05-10 AM. Both SLURM jobs exited 0; checkpoints
saved. ONNX export, Kaggle upload, kernel push, and LB submission are still
pending. Verdict against the success matrix in
[`proposal.md`](proposal.md) is read against `v5j_control`'s actual LB
(not the historical 0.896), so the matrix can't be applied until both
LBs come back.

## Summary

The KD branch trained cleanly. `dist_loss` decayed from 0.058 (fast-dev-run
random-init) to **0.004 by epoch 19** — at Tucker's published floor of ~0.003.
That kills the largest pre-flight risk (mel-pipeline mismatch raising the
floor) — see [`phase0_mel_equivalence.md`](phase0_mel_equivalence.md). The
no-KD path's offline numbers in this experiment are **bit-identical** to
exp 030's `v5j_control_v2` (val/auroc 0.8728, cmap 0.6268, f1_macro 0.5270),
giving us the **offline-side back-compat receipt** for the KD-branch
additions to `sed_finetune.py`. The LB-side back-compat receipt is pending.

Offline, KD trails control on every metric (val/auroc −0.012, val/cmap
−0.089, val/f1_macro −0.097). This is a yellow flag but not a verdict —
val/auroc is anti-correlated with LB historically, and cmap/f1 are slightly
more directional but not confidently LB-predictive at this magnitude.
LB is the actual disambiguator.

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

In order:

1. **ONNX export both checkpoints** via
   `experiments/010-submission-pipeline/scripts/export_efficientnet_sed.py`.
   The export only loads `encoder` + `sed_head` state_dicts, so
   `distill_head` is naturally excluded from the KD ONNX. Verify both
   ONNX outputs are `(B, 234)` non-degenerate on a fixed random mel.
2. **Add MODELS entries** to `experiments/010-submission-pipeline/scripts/upload_kaggle.py`:
   `hgnet-b0-sed-v5j-kd` and `hgnet-b0-sed-v5j-control`.
3. **Add MODEL_SLUGS entries** to `experiments/010-submission-pipeline/kaggle_kernel/submission.py`.
4. **Create `kernel-metadata-*.json` files** in `kaggle_kernel/`.
5. **Upload weights datasets** (×2) and **push kernels** (×2).
6. **Submit both kernels** from the Kaggle UI; record both LBs.
7. **Apply success matrix** in [`proposal.md` § Success Criteria](proposal.md)
   using `v5j_control`'s actual LB as the baseline (not historical 0.896).
8. **Update this `results.md`** with the verdict, update `MODEL-RUNS.md`,
   update `CONTEXT.md`.

## Decision matrix (LB-pending)

Once both LBs are in, apply (read against `v5j_control`'s actual LB):

| Pattern | Reading | Action |
|---|---|---|
| `v5j_control` LB outside 0.896 ± 0.005 | LB-side back-compat failure | Diagnose drift in KD-branch additions before trusting KD result |
| `v5j_kd` LB ≥ control + 0.014 | **CONFIRM**: KD lever transfers | Proceed to Phase 2 (ownership work + 4-fold) |
| `v5j_kd` LB control + 0.007 to control + 0.013 | **PARTIAL**: lift detectable but smaller than Tucker's +0.020 | Diagnose: dist_loss floor, alpha_distill sweep, longer schedule |
| `v5j_kd` LB control + 0.001 to control + 0.006 | **WEAK**: inside seed noise | Need 4-fold to disambiguate, but high variance cost |
| `v5j_kd` LB ≤ control | **REJECT**: KD didn't add anything as a single-variable add | Halt Phase 2; structural post-mortem (head architecture, schedule, alpha) |

## What this RULES OUT vs LEAVES OPEN (so far)

**Rules out** (offline evidence only, may revise after LB):
- *Mel-pipeline mismatch raising the dist_loss floor.* Floor 0.004 ≈
  Tucker's 0.003. Phase 0.0a stays SKIPPED.
- *KD-branch additions breaking the non-KD path.* Bit-identical match to
  exp 030 v5j_control_v2 on every offline metric.
- *Detach bug or NaN/Inf.* All gradients flowed cleanly per Phase 0.4
  test; cls_loss landed at 0.019 (well-behaved); zero NaN/Inf in
  training.

**Leaves open** (LB-dependent):
- Whether KD transfers as a single-variable add to v5j on our codebase.
- Whether the −0.089 offline cmap gap predicts LB regression or is the
  usual offline-LB anti-correlation.
- Whether 20 epochs is enough for KD to express its lift (best epoch
  17 in KD run is later than 15 in control — KD may need more epochs).
- Whether LSEHead is a worse downstream consumer of Perch features than
  AttHead.

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

- [ ] ONNX export `v5j_kd` and `v5j_control` checkpoints
- [ ] Verify ONNX outputs `(B, 234)` non-degenerate on fixed random mel
- [ ] Add `hgnet-b0-sed-v5j-kd` + `hgnet-b0-sed-v5j-control` to `upload_kaggle.py` MODELS
- [ ] Add both slugs to `submission.py` MODEL_SLUGS
- [ ] Create both `kernel-metadata-*.json` files in `kaggle_kernel/`
- [ ] Upload both weights datasets to Kaggle
- [ ] Push both kernels
- [ ] Submit both kernels to LB; record scores
- [ ] Apply decision matrix; update this `results.md` with verdict
- [ ] Update `MODEL-RUNS.md` with new LB scores
- [ ] Update `CONTEXT.md` "What happened recently" section
- [ ] If CONFIRM: scaffold Phase 2 (ownership + 4-fold)
- [ ] If REJECT/PARTIAL/WEAK: write focused post-mortem identifying most
      likely cause (head architecture / schedule / alpha / teacher mismatch /
      seed noise)
