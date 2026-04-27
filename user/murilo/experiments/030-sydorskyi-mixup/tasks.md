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

## Phase B — EffNetV2-B0 backbone (training DONE, LB pending)

### B.0 — Sanity (fast-dev-run)

- [x] `bash sbatch/run.sh configs/effnetv2b0_sydorskyi.yaml --fast-dev-run`
  — DONE 2026-04-26. EfficientNetV2-B0 backbone loaded successfully via
  `tf_efficientnetv2_b0.in1k`. Encoder log shows `embed_dim=192, stages=5`
  (uses stage-5 spatial features, not the 1280-d post-pool projection).
  control sanity skipped (low risk — same backbone, only mixup differs).

### B.1 — EffNetV2-B0 control

- [x] `bash sbatch/run.sh configs/effnetv2b0_control.yaml` — DONE 2026-04-26
  - SLURM job 7454128, walltime 41m 17s on rtx6000
  - val/auroc 0.8510 (best-by-val ckpt at epoch 19: same value)
  - W&B run id `jhfx75pb`

### B.2 — EffNetV2-B0 Sydorskyi (best offline cell)

- [x] `bash sbatch/run.sh configs/effnetv2b0_sydorskyi.yaml` — DONE 2026-04-26
  - SLURM job 7454131, walltime 45m 37s on rtx6000
  - val/auroc 0.8518 (best-by-val ckpt at epoch 19: same value)
  - val/cmap 0.6303, val/f1_macro 0.5751 — best on every offline metric
  - W&B run id `bampbee6`

### B.3 — 2x2 ablation analysis (offline DONE; LB pending)

- [x] Offline 2x2 main effects + interaction:
  - **Backbone** (v2-B0 − HGNet, avg over mixup): +0.003 auroc, **+0.020 cmap**,
    **+0.028 f1_macro**
  - **Mixup** (sydorskyi − control, avg over backbone): −0.001 auroc,
    **+0.014 cmap**, **+0.043 f1_macro**
  - **Interaction**: small positive on auroc/cmap (sydorskyi helps slightly
    more on v2-B0). Mostly additive.
  - Best cell: `effnetv2b0_sydorskyi`. +0.034 cmap and +0.071 f1_macro vs
    v5j (HGNet + spec interp).
- [ ] Compute LB main effects + interaction (after all 4 LBs in)
- [ ] Apply Phase B decision matrix from `proposal.md`

## v1 LB DONE — CONFOUND IDENTIFIED

| Slug (v1) | LB | Notes |
|---|---:|---|
| hgnet-b0-sed-v5j-control | 0.890 | dual_validation cost vs v5j 0.906 |
| hgnet-b0-sed-v5j-sydorskyi | 0.892 | mixup +0.002 (null) on HGNet |
| effnetv2b0-sed-control | 0.893 | backbone main effect tiny on control side |
| **effnetv2b0-sed-sydorskyi** | **0.901** | best v1 cell, +0.011 vs v5j_control |

`dual_validation: true` (added to all 4 v1 configs) is the confound:
changes `val/auroc` to focal-only, shifts checkpoint selection, costs
~0.009 LB per exp 026 v5j-dualval = 0.897 vs v5j = 0.906.

## v2 retraining — all 4 variants

### V2.0 — Configs updated

- [x] `configs/v5j_control.yaml`: removed `dual_validation: true`,
  run_name → `2026-exp030-v5j-control-v2`
- [x] `configs/v5j_sydorskyi.yaml`: removed `dual_validation: true`,
  run_name → `2026-exp030-v5j-sydorskyi-v2`
- [x] `configs/effnetv2b0_control.yaml`: removed `dual_validation: true`,
  run_name → `2026-exp030-effnetv2b0-control-v2`
- [x] `configs/effnetv2b0_sydorskyi.yaml`: removed `dual_validation: true`,
  run_name → `2026-exp030-effnetv2b0-sydorskyi-v2`

### V2.1 — Retrain all 4 — DONE 2026-04-26 → 2026-04-27 (overnight)

- [x] `bash sbatch/run.sh configs/v5j_control.yaml` — SLURM 7455688, walltime
  40m 12s. val/auroc **0.8728**, cmap 0.6268, f1_macro 0.5270, best_val
  0.8747 @ ep 15. W&B `586p423f`.
- [x] `bash sbatch/run.sh configs/v5j_sydorskyi.yaml` — SLURM 7455689, walltime
  1h 14m 54s. val/auroc **0.8700**, cmap 0.6339, f1_macro 0.5663, best_val
  0.8737 @ ep 14. W&B `e6hdxkg8`.
- [x] `bash sbatch/run.sh configs/effnetv2b0_control.yaml` — SLURM 7455690,
  walltime 41m 5s. val/auroc **0.8745**, cmap 0.6461, f1_macro 0.5545,
  best_val 0.8745 @ ep 19. W&B `fyqjktl5`.
- [x] `bash sbatch/run.sh configs/effnetv2b0_sydorskyi.yaml` — SLURM 7455691,
  walltime 41m 16s. val/auroc **0.8760**, cmap **0.6556**, f1_macro
  **0.5922**, best_val 0.8760 @ ep 19. W&B `cg9v7ksj`. **best v2 cell.**

**Reproducibility check (v5j_control_v2 vs v5j-original 0.874 val/auroc)**:
Δ −0.001 → ✓ PASS within ±0.005 seed noise band. Pipeline reproducibility
confirmed for v2 regime; +0.024 val/auroc shift across all 4 cells matches
the dual_validation hypothesis (focal-only → combined focal+SS).

**v2 2x2 offline structure replicates v1**: backbone +0.020 cmap (matches
v1), mixup +0.008 cmap (vs +0.014 v1), additive interaction. Best cell on
every metric: `effnetv2b0_sydorskyi_v2`.

### V2.2 — Re-export, re-upload, re-push, re-submit (NEXT)

Pipeline wiring is in place — `export.sbatch` points at `-v2` checkpoint
dirs (lines 678/691/704/717), `upload_kaggle.py` has the 4 MODELS entries
(lines 322-340), `submission.py` has the 4 weight slugs in MODEL_SLUGS
(lines 95-98), and 4 kernel-metadata-*.json files exist. Existing-model
re-export path: re-export overwrites `artifacts/weights/`, `--push-kernel`
works one-shot.

For each of the 4 models, after training completes:
- [ ] ONNX export: `bash exp010/sbatch/run.sh export --model {slug}`
- [ ] Upload + push: `python exp010/scripts/upload_kaggle.py --model {slug} --push-kernel`
  (replaces v1 weights at the same Kaggle slug; same kernel-metadata files)
- [ ] Submit to LB from Kaggle UI

Slugs: `hgnet-b0-sed-v5j-control`, `hgnet-b0-sed-v5j-sydorskyi`,
`effnetv2-b0-sed-control`, `effnetv2-b0-sed-sydorskyi`.

### V2.3 — Verdict (after v2 LBs in)

- [ ] **Reproducibility check on LB**: does v5j_control_v2 reproduce v5j 0.906
  (within ±0.005)? If yes → dual_val confound confirmed; pipeline reproduces.
  If 0.895-0.900 → partial recovery; investigate. If ≤ 0.895 → halt.
- [ ] **Mixup main effect on HGNet**: v5j_sydorskyi_v2 vs v5j_control_v2.
- [ ] **Backbone main effect**: effnetv2b0_control_v2 vs v5j_control_v2.
- [ ] **Combined effect**: effnetv2b0_sydorskyi_v2 vs v5j_control_v2.
  If ≥ +0.005 LB and v5j_control_v2 ≥ 0.901, this is the **first
  single-model improvement over v5j 0.906** for our pipeline.

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
