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

## Stage 2: Fine-tune on 2026 with classifier transfer — IN PROGRESS

### 2.1 Configure

- [ ] 2.1.1 Update `configs/v5j_finetune_2026.yaml` `init.from_checkpoint`
      to: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/029-2025-pretrain/checkpoints/2026-exp029-v5j-pretrain-2025/best-epochepoch=15.ckpt`
- [x] 2.1.2 Confirm `init.source_taxonomy_csv` points to 2025 taxonomy.csv
      (already correct in committed config)

### 2.2 Sanity check

- [ ] 2.2.1 Fast-dev-run on `configs/v5j_finetune_2026.yaml`. Verify
      classifier transfer logs report:
      - 41 overlap species mapped
      - ~140+ params copied directly (backbone + head except final classifier)
      - 2 partial transfers (`model.head.cls_fc.3.weight`, `model.head.cls_fc.3.bias`)

### 2.3 Train

- [ ] 2.3.1 Submit fine-tune SLURM job:
      `bash sbatch/run.sh configs/v5j_finetune_2026.yaml`
- [ ] 2.3.2 Monitor — expected ~1–2h on RTX 6000 (revised down based on
      Stage 1 walltime); 2026 dataset is slightly larger so expect a bit
      longer than Stage 1's 30 min
- [ ] 2.3.3 Verify best-by-val/auroc checkpoint saved

---

## Stage 3: Evaluate

### 3.1 ONNX export + LB submission

- [ ] 3.1.1 ONNX export via standard pipeline
      (use the v5j inference kernel's export script as reference)
- [ ] 3.1.2 Submit to LB
- [ ] 3.1.3 Record result in `results.md` and `MODEL-RUNS.md`

### 3.2 Per-frame triage on the new model

- [ ] 3.2.1 Re-run `precompute_losses.py` on the stage-2 checkpoint,
      pointing at 2026 val data
- [ ] 3.2.2 Run `triage.py` + `per_frame_analysis.py` on new outputs
- [ ] 3.2.3 Compare attention entropy / logit_std on `mixed_uncertainty`
      vs v5j baseline — did the head behavior change?

---

## Decision gate

After 3.1 and 3.2:

| Outcome | LB | Triage shift | Action |
|---------|---|---|---|
| Confirm | > 0.906 | any | Record win, plan 4-fold scaleup |
| Strong confirm | ≥ 0.910 | any | Plan 4-fold + ensemble + Perch fusion |
| Retry | ≤ 0.906 | per-frame structure changed (e.g., logit_std rose on mixed_uncertainty) | Lever may be real; tune fine-tune LR / epochs / freeze schedule before giving up |
| Reject | ≤ 0.906 | per-frame structure unchanged | 2025 pretrain at our scale doesn't transfer; consider Salman replication or silent-class as next exp |
| Pivot | Stage 1 val/auroc didn't converge | n/a | Debug 2025 pipeline rather than proceed to Stage 2 |

---

## What is NOT in this experiment

For reference; each is a future experiment if pursued.

- **Joint training** (combine 2025 + 2026 in single training run with 234-class head)
- **Filtered XC pretrain** (closer to Ali Ozan's actual pretrain corpus, much heavier data engineering)
- **4-fold ensemble** (standard follow-up if Stage 2 wins)
- **Salman recipe replication on EffNetV2-B0** (different bet, separate experiment)
- **Exp 030 silent-class** (still queued, see per-frame analysis note)
