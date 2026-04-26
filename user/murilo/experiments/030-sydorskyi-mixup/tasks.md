# Experiment 030 — Tasks

## Phase A — v5j A/B comparison (committed)

### A.0 — Sanity (fast-dev-run)

- [ ] `bash sbatch/run.sh configs/v5j_control.yaml --fast-dev-run`
  (5 batches, ~5 min, no W&B). Verifies the config + train.py paths work.
- [ ] `bash sbatch/run.sh configs/v5j_sydorskyi.yaml --fast-dev-run`.
  Verifies the new `mixup_mode="average_max"` path doesn't crash.

### A.1 — v5j control (reproducibility baseline)

- [ ] `bash sbatch/run.sh configs/v5j_control.yaml`
  - Expected: ~2-3h on rtx6000, val/auroc ~0.851 (matching v5j-control
    from exp 027), best-by-val checkpoint saved.
  - W&B project: `birdclef-2026-exp030`, run name `2026-exp030-v5j-control`.

### A.2 — v5j Sydorskyi (the test)

- [ ] `bash sbatch/run.sh configs/v5j_sydorskyi.yaml`
  - Expected: ~2-3h, similar wall time to A.1 (mixup path is a small
    overhead difference).
  - W&B run name `2026-exp030-v5j-sydorskyi`.
  - **Sanity check during training**: `mixup_mode="average_max"` log line
    should appear in worker output. If it falls back to "interpolate"
    silently, the config didn't propagate (analogous to the
    `model.head` vs `model.sed_head` bug we hit during D1).

### A.3 — Compare on W&B

- [ ] Visual inspection: train/loss, val/loss, val/auroc, val/cmap curves
  side-by-side. Note divergence patterns, not just final values.
- [ ] Check val_sc/auroc (broken metric — see exp 026 caveat) just as a
  sanity check that runs match the same val structure.

### A.4 — ONNX export + LB submission for both

- [ ] Export v5j_control to ONNX + submit to LB.
- [ ] Export v5j_sydorskyi to ONNX + submit to LB.
- [ ] Update `MODEL-RUNS.md` with both LB scores.
- [ ] Apply the Phase A decision matrix from `proposal.md` and update
  `results.md`.

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
