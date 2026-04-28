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

### V2.2 — Re-export, re-upload, re-push, re-submit (DONE 2026-04-27/28)

- [x] ONNX export for all 4 v2 cells (04-27 09:49–09:50)
- [x] Kaggle upload + kernel push for all 4 v2 cells (04-27 14:11–14:20)
- [x] Submitted v5j-resubmit kernel → **0.906 LB** (inference pipeline confirmed clean)
- [x] Submitted v5j_control_v2 → **0.896 LB**
- [x] Submitted v5j_sydorskyi_v2 → **0.892 LB**
- [x] Submitted effnetv2b0_control_v2 → **0.893 LB**
- [x] Submitted effnetv2b0_sydorskyi_v2 → **0.901 LB**

### V2.3 — Verdict (DONE 2026-04-28)

- [x] **Reproducibility on LB**: v5j-resubmit landed at 0.906 → original v5j
  reproduces today; inference pipeline clean.
- [x] **3/4 v2 LBs literally identical to v1** (0.892, 0.893, 0.901): root-
  caused to deterministic training (`pl.seed_everything` + `deterministic=True`)
  making v2 weights bit-identical to v1. `dual_validation: true` only changes
  val_dataloader, not training. Per-key state_dict audit confirmed:
  420/420 effnet learnable tensors and 320/320 v5j_sydorskyi tensors are
  bit-identical between v1 and v2. Only `_val_sc_pos_counts` non-trainable
  buffer differs (val-only, stripped from ONNX).
- [x] **2x2 LB reading**: HGNet mixup Δ −0.004 (null), EffNet mixup Δ +0.008
  (borderline architectural interaction). Both within ±0.005 seed-noise
  band — needs seed=43 retrain to confirm.
- [x] **EffNet ONNX sanity**: produces non-degenerate, valid output (std 0.006
  on random mel, top-3 stable). No export bug. 0.901 < 0.906 is offline-LB
  anti-correlation, not a pipeline issue.

## V3 — Seed=43 retrain (LAUNCHED 2026-04-28)

Diagnostic to disambiguate the EffNet × Sydorskyi mixup +0.008 interaction
from seed noise. Default `seed: 42` preserved in the existing 4 v2 configs.

### V3.0 — New configs

- [x] `configs/v5j_control_seed43.yaml` (`seed: 43`, run_name `-seed43`)
- [x] `configs/v5j_sydorskyi_seed43.yaml`
- [x] `configs/effnetv2b0_control_seed43.yaml`
- [x] `configs/effnetv2b0_sydorskyi_seed43.yaml`

### V3.1 — Train — DONE 2026-04-28 (~00:25)

All 4 SLURM jobs exit 0; ~40-44 min each on rtx6000/v100. Only `data.seed`
changes vs v2 (42 → 43); everything else identical.

| Cell | val/auroc | val/cmap | val/f1_macro | best_val | best epoch | SLURM |
|---|---:|---:|---:|---:|---:|---:|
| `v5j_control_seed43` | 0.8926 | 0.6426 | 0.5679 | 0.8933 | 16 | 7488877 |
| `v5j_sydorskyi_seed43` | 0.8967 | 0.6437 | 0.5854 | 0.8978 | 16 | 7488880 |
| `effnetv2b0_control_seed43` | 0.8890 | 0.6455 | 0.5604 | 0.8930 | 18 | 7488881 |
| **`effnetv2b0_sydorskyi_seed43`** | **0.8932** | **0.6490** | **0.5945** | **0.8982** | 18 | 7488884 |

**Caveat**: different seed → different val split → absolute val/auroc not
comparable across seed groups. Within-seed deltas are the diagnostic
(see `results.md` "Seed=43 within-seed main effects + interaction").

Within-seed signal (seed=43):
- **Mixup main effect on auroc**: +0.004 (vs seed=42 v2: −0.001) — both
  backbones now show small positive
- **Backbone main effect**: collapses to ±0.004 across all metrics (vs
  seed=42 v2: +0.020 cmap / +0.027 f1_macro)
- **f1_macro interaction reproduces**: EffNet mixup +0.034 vs HGNet +0.018
- Best cell on every offline metric: `effnetv2b0_sydorskyi_seed43`

### V3.2 — Pipeline plumbing — IN PROGRESS 2026-04-28

Four-file pattern per `experiments/010-submission-pipeline/adding-new-models.md`.
NEW kernel slugs (`*-seed43`) so the seed=42 v2 weights stay live.

Slug shortening: `birdclef-2026-{base}-seed43-weights` exceeds Kaggle's 50-char
dataset slug limit (53-55 chars). Drop `-2026-` from prefix → `birdclef-{base}-seed43-weights`
(47-50 chars). Kernel id keeps `birdclef-2026-` (no length issue there).

- [x] Add 4 new MODELS entries to `upload_kaggle.py` for seed43 slugs
- [x] Add 4 new export functions to `export.sbatch` pointing at seed43 ckpt dirs
  (`2026-exp030-{model}-seed43`)
- [x] Add 4 new kernel-metadata-*-seed43.json files
- [x] Add 4 new entries to submission.py MODEL_SLUGS
- [ ] ONNX export × 4 (`bash sbatch/run.sh export --model X-seed43`)
- [ ] Upload + push × 4 (two-step: `upload_kaggle.py --model X` then `--push-kernel`)
- [ ] Submit to LB × 4 (Kaggle 5/day cap)

### V3.3 — Verdict (after seed=43 LBs in)

Apply the matrix below; pick next experiment based on outcome:

| Pattern | Conclusion | Action |
|---|---|---|
| EffNet syd − control ≥ +0.005 AND HGNet syd − control near 0 | Architectural interaction is real | Endorse `effnetv2b0_sydorskyi`; 4-fold + Sydorskyi component staircase |
| Both 2x2 Δs flip sign or change > ±0.005 | All seed=42 LBs were noise | Pivot to pseudo-distillation |
| EffNet Δ shrinks but stays positive (≥+0.003) | Borderline | 4-fold disambiguator |
| v5j_control_seed43 LB ≥ 0.901 | 0.010 gap was seed/checkpoint sensitivity | Confirms pipeline isn't structurally broken |

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
