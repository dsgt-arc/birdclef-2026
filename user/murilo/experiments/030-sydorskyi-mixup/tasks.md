# Experiment 030 — Tasks

## Phase A — v5j A/B comparison (training DONE, LB pending)

### A.0 — Sanity (fast-dev-run)

- [x] `bash sbatch/run.sh configs/v5j_sydorskyi.yaml --fast-dev-run`
  — DONE 2026-04-26. 5/5 batches in 4 min, exit 0. New
  `mixup_mode="average_max"` branch fired without errors (waveform domain
  confirmed via "GPU mel + waveform mixup" log line in SLURM output).
  v5j_control sanity skipped (low risk — same code path as existing v5j).

### A.1 — v5j control (reproducibility baseline)

- [x] `bash sbatch/run.sh configs/v5j_control.yaml` — DONE 2026-04-26
  - SLURM job 7452610, walltime 37m 44s on rtx6000
  - val/auroc 0.8490 (best-by-val ckpt at epoch 14: 0.8516)
  - **Reproducibility CONFIRMED**: 0.849 vs exp 027 v5j-control 0.851
    (Δ −0.002, within Mahog's seed-noise band ±0.005)
  - W&B run id `el93xunm`

### A.2 — v5j Sydorskyi (the test)

- [x] `bash sbatch/run.sh configs/v5j_sydorskyi.yaml` — DONE 2026-04-26
  - SLURM job 7452611, walltime 41m 45s on rtx6000 (~4m extra for peak-norm)
  - val/auroc 0.8472 (best-by-val ckpt at epoch 14: 0.8507)
  - `mixup_mode="average_max"` path confirmed firing (no errors, no NaN,
    SLURM log shows full config dump with all 4 expected changes)
  - W&B run id `3bjn6i3y`

### A.3 — Compare on W&B

- [x] Visual inspection — DONE
  - `train/loss_epoch` diverges as predicted (sydorskyi 0.0178 vs control
    0.0156): max-targets are structurally harder to fit than interpolated.
  - `val/auroc` tracks closely: Δ −0.002 (within seed noise).
  - `val/cmap` notably higher for sydorskyi: Δ +0.011.
  - `val/f1_macro` notably higher for sydorskyi: Δ +0.043.
  - Best epoch identical (epoch 14) — same convergence shape.
- [x] `val_sc/auroc` sanity (broken metric per exp 026 caveat): consistent
  pattern, sydorskyi +0.011. Don't read absolute values.

### A.4 — ONNX export + LB submission for both

- [ ] Export v5j_control to ONNX
- [ ] Export v5j_sydorskyi to ONNX
- [ ] Submit v5j_control to Kaggle LB
- [ ] Submit v5j_sydorskyi to Kaggle LB
- [ ] Update `MODEL-RUNS.md` with LB scores
- [ ] Apply the Phase A decision matrix from `proposal.md` and update
  `results.md` with verdict

## Phase B — EffNetV2-B0 backbone (committed; can run in parallel with Phase A)

### B.0 — Sanity (fast-dev-run)

- [ ] `bash sbatch/run.sh configs/effnetv2b0_control.yaml --fast-dev-run`.
  - First time using `tf_efficientnetv2_b0.in1k` in our pipeline; verifies
    the 1280-channel feature dim flows through to the head correctly.
- [ ] `bash sbatch/run.sh configs/effnetv2b0_sydorskyi.yaml --fast-dev-run`.

### B.1 — EffNetV2-B0 control

- [ ] `bash sbatch/run.sh configs/effnetv2b0_control.yaml`.
  - W&B run name `2026-exp030-effnetv2b0-control`.

### B.2 — EffNetV2-B0 Sydorskyi

- [ ] `bash sbatch/run.sh configs/effnetv2b0_sydorskyi.yaml`.
  - W&B run name `2026-exp030-effnetv2b0-sydorskyi`.

### B.3 — 2x2 ablation analysis

- [ ] Compute main effects + interaction on LB:
  - Mixup main effect: average of (sydorskyi - control) across both backbones
  - Backbone main effect: average of (effnetv2b0 - v5j) across both mixup conditions
  - Interaction: deviation from additivity
- [ ] Apply Phase B decision matrix from `proposal.md`.

## Documentation

- [ ] Write `results.md` with all 4 LB scores + decision-matrix verdict.
- [ ] Update `user/murilo/MODEL-RUNS.md` with the four new model entries.
- [ ] Update `user/murilo/CONTEXT.md` "What happened recently" section.
- [ ] Update `user/murilo/FINDINGS.md` if Phase A finds a non-flat result.
- [ ] If Phase A confirms: write follow-up note proposing 4-fold +
  ensemble extensions.
- [ ] If Phase A rejects: write follow-up note pivoting to
  pseudo-distillation as next experiment.

## Decision matrix (copied from proposal.md for in-place tracking)

| `v5j_control` LB | `v5j_sydorskyi` LB | Verdict | Action |
|---|---|---|---|
| ≥ 0.901 | ≥ 0.911 | Strong confirm | Endorse, queue 4-fold version |
| ≥ 0.901 | 0.901-0.911 | Within seed noise | Repeat with different seed before deciding |
| ≥ 0.901 | ≤ 0.895 | Reject | Pivot to pseudo-distillation |
| 0.895-0.900 | (any) | Reproducibility issue | Halt, investigate |
| < 0.895 | (any) | Reproducibility failure | Blocking, investigate |

## Estimated wall time

- 4 SLURM jobs × ~2-3h each on rtx6000/v100 = ~10-12h total GPU
- Can run 2 in parallel if GPU available → ~4-6h wall time
- ONNX export + LB submission: ~1h per variant × 4 = 4h
- Documentation: 1-2h
- **Total: ~half-day to 1 day end-to-end**, depending on parallelism

## Open questions (not blocking)

- Does `v5j_control` LB exactly reproduce v5j 0.906, or is there a small
  drift from changes in our codebase since exp 019? (We've added several
  features to `sed_finetune.py` since then.)
- If `effnetv2b0_*` regresses, is it the backbone or the feature-dim
  mismatch? (1280 vs 1024 may cascade through the head differently.)
- Should we also test `tf_efficientnetv2_b0.ns_jft_in1k` (noisy-student JFT
  pretrained variant) as a stretch goal? Skipping for this experiment.
