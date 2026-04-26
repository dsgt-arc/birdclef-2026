# Experiment 029: 2025 Pretrain → 2026 Fine-tune — Tasks

## Goal

Test whether sequential 2025-pretrain → 2026-fine-tune improves LB
over the v5j baseline (0.906). **Single-variable change from v5j**:
2025-pretrained init vs ImageNet init.

**Branch**: `mg/exp029-2025-pretrain`
**Baseline**: v5j single fold = 0.906 LB (exp 019)
**Reference**: [proposal.md](proposal.md), [per-frame analysis note](../../analysis/notes/2026-04-25-per-frame-attention-analysis.md)

---

## Stage 1: Pretrain on 2025 focal data — COMPLETE

### 1.1 Sanity check — done

- [x] 1.1.1 Verify 2025 train.csv loads cleanly via existing `SEDFocalDataset`
      (point `data_dir` at 2025 path)
- [x] 1.1.2 Fast-dev-run on `configs/v5j_pretrain_2025.yaml` (5 batches,
      no W&B). First attempt failed on missing `train_soundscapes_labels.csv`
      → fixed in `src/birdclef/data/soundscape.py:load_soundscape_labels`
      (graceful empty-DF fallback). Second fast-dev-run completed
      successfully (SLURM 7432783).

### 1.2 Train — done

- [x] 1.2.1 Submit pretrain SLURM job: `bash sbatch/run.sh configs/v5j_pretrain_2025.yaml`
- [x] 1.2.2 Walltime: **30m16s on RTX 6000** (much faster than the 1.5–2h estimate)
- [x] 1.2.3 Best-by-val/auroc checkpoint saved at epoch 15 (val/auroc 0.972)
      Path: `/storage/.../mgustineli3/experiments/029-2025-pretrain/checkpoints/2026-exp029-v5j-pretrain-2025/best-epochepoch=15.ckpt`
- [x] 1.2.4 Recorded stage-1 metrics in `results.md`. Headline:
      val/auroc 0.972, val/cmap 0.649, val/top1 0.761. Convergence
      trajectory and shape documented.

---

## Stage 2: Fine-tune on 2026 with classifier transfer — TRAINING DONE

### 2.1 Configure — done

- [x] 2.1.1 Updated `configs/v5j_finetune_2026.yaml` `init.from_checkpoint`
      to point at `best-epochepoch=15.ckpt` from Stage 1
- [x] 2.1.2 Confirm `init.source_taxonomy_csv` points to 2025 taxonomy.csv

### 2.2 Sanity check — done

- [x] 2.2.1 Fast-dev-run completed (SLURM 7433613). Transfer logs verified:
      - 41 of 234 target species mapped from source ✓
      - 318 params copied directly (backbone + non-classifier head) ✓
      - 2 partial transfers: `sed_head.cls_fc.3.weight`, `sed_head.cls_fc.3.bias` ✓
      - 0 skipped — no pretrained params lost ✓

### 2.3 Train — done

- [x] 2.3.1 Submitted fine-tune SLURM job (job 7433914)
- [x] 2.3.2 **Walltime 38m40s on RTX 6000** (within revised estimate)
- [x] 2.3.3 Best-by-val/auroc checkpoint saved at epoch 12 (val/auroc 0.850)
      Path: `/storage/.../mgustineli3/experiments/029-2025-pretrain/checkpoints/2026-exp029-v5j-finetune-2026/best-epochepoch=12.ckpt`

**Result**: val/auroc 0.850 — within ±0.001 of v5j-control (0.851).
**Offline result is neutral.** See `results.md` for full trajectory and
analysis.

---

## Stage 3: Evaluate — DONE

### 3.1 ONNX export + LB submission — done

- [x] 3.1.1 ONNX export via standard pipeline (`bash sbatch/run.sh export`)
- [x] 3.1.2 Submit to LB. **Result: 0.885 (−0.021 vs v5j baseline 0.906)**
- [x] 3.1.3 Recorded in `results.md` and `MODEL-RUNS.md`. Mid-flight slug
      fix: 52→47 chars (Kaggle's 50-char limit was silently failing).

### 3.2 Per-frame triage on the new model — pending (independent of LB)

- [ ] 3.2.1 Re-run `precompute_losses.py` on the stage-2 checkpoint,
      pointing at 2026 val data (need to point `--checkpoint` at
      `029-.../best-epochepoch=12.ckpt` and use 2026 v5j config)
- [ ] 3.2.2 Run `triage.py` + `per_frame_analysis.py` on new outputs
- [ ] 3.2.3 Compare attention entropy / logit_std on `mixed_uncertainty`
      vs v5j baseline — useful to understand *what* the failed pretrain
      changed structurally, even though LB regressed.

---

## Decision gate — landed on REJECT

| Outcome | LB | Triage shift | Action |
|---------|---|---|---|
| Confirm | > 0.906 | any | Record win, plan 4-fold scaleup |
| Strong confirm | ≥ 0.910 | any | Plan 4-fold + ensemble + Perch fusion |
| Retry | ≤ 0.906 | per-frame structure changed | Lever may be real; tune fine-tune LR / epochs / freeze schedule before giving up |
| **Reject** | **≤ 0.906 by ≥0.005 (got −0.021)** | **TBD (triage queued)** | **2025 pretrain at our scale doesn't transfer; pivot to Salman replication or silent-class.** |
| Pivot | Stage 1 val/auroc didn't converge | n/a | Debug 2025 pipeline rather than proceed to Stage 2 |

**Outcome: REJECTED.** Final LB 0.885. See `results.md` for full
interpretation. Cheap retry candidate (`head_lr: 5e-4` split LR)
exists, but strategic call is whether to keep investing in
training-side variations at all given the per-frame finding.

---

## Path A retry: split-LR fine-tune (queued)

Single-variable retry from the rejected Stage 2: keep `lr: 2e-4`
(backbone — preserve pretrained features), bump `head_lr: 5e-4`
(match v5j's full LR for the 193 random-init classifier rows).

**Hypothesis**: the original Stage 2 regression was caused by
undertraining the random-init head at 2e-4. Splitting the LR
should give random-init rows the same training signal they got in
v5j-from-scratch, while preserving pretrained backbone features.

**Config**: `configs/v5j_finetune_2026_split_lr.yaml`
(only changes from `v5j_finetune_2026.yaml`: `head_lr` 2e-4 → 5e-4,
`run_name` suffix `-split-lr`)

**Kaggle pipeline**: 4-file dance done in
`010-submission-pipeline/`. Slug: `hgnet-b0-sed-v5j-ft-splitlr`
→ Kaggle dataset `birdclef-2026-hgnet-b0-sed-v5j-ft-splitlr-weights`
(49 chars, just under 50-char limit).

**Decision logic**:

| Outcome | Action |
|---|---|
| LB ≈ 0.906 ± 0.005 | LR was the dominant cause; pretrain backbone is neutral. Stage 2 setup was wrong, not the lever. |
| LB regresses (<0.900) | 2025-pretrain at our scale is fully ruled out regardless of LR config. |
| LB > 0.910 | Surprise win; pretrain helps when head LR is correct. Plan 4-fold scaleup. |

**Pending tasks**:

- [ ] Submit pretrain SLURM job:
      `bash sbatch/run.sh configs/v5j_finetune_2026_split_lr.yaml`
- [ ] ONNX export:
      `bash sbatch/run.sh export --model hgnet-b0-sed-v5j-ft-splitlr` (in 010-submission-pipeline)
- [ ] Kaggle dataset upload + kernel push (per `010-submission-pipeline/adding-new-models.md`)
- [ ] LB submission
- [ ] Record result here and decide next experiment

---

## What is NOT in this experiment

For reference; each is a future experiment if pursued.

- **Joint training** (combine 2025 + 2026 in single training run with 234-class head)
- **Filtered XC pretrain** (closer to Ali Ozan's actual pretrain corpus, much heavier data engineering)
- **4-fold ensemble** (standard follow-up if Stage 2 wins)
- **Salman recipe replication on EffNetV2-B0** (different bet, separate experiment)
- **Exp 030 silent-class** (still queued, see per-frame analysis note)
