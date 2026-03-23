# Experiment 015: Student Training — Tasks

## Goal

Train student models on focal recordings (hard labels) + labeled soundscapes
(hard labels) + pseudo-labeled unlabeled soundscapes (soft labels from exp 014).
Target: LB improvement ≥0.02 over exp 012 baseline.

**Baseline** (exp 012): ProtoCLR 0.767, ConvNeXt 0.755, EAT 0.720

---

## 1. Setup

- [x] 1.1 Create experiment directory structure
- [x] 1.2 Create `pyproject.toml` (same deps as exp 012 + scikit-learn)
- [x] 1.3 Implement `src/birdclef/data/student.py` (WeightedDatasetWrapper, PseudoLabeledDataset, StudentDataModule)
- [x] 1.4 Implement `src/birdclef/lightning/student_finetune.py` (StudentFinetuneModule)
- [x] 1.5 Update `__init__.py` package exports
- [x] 1.6 Create training configs (protoclr, convnext, eat)
- [x] 1.7 Create SLURM scripts (run.sh, _job.sh)
- [x] 1.8 Implement `sweep_temperature.py`
- [x] 1.9 Implement `train.py` (adapted from exp 012)
- [x] 1.10 Implement `save_metrics.py` (saves structured metrics to `artifacts/` after training)
- [x] 1.11 Add `--gpu-check` support (GPUUtilizationMonitor callback, outputs txt + png)

## 2. Temperature Scaling Sweep

- [x] 2.1 Run `python sweep_temperature.py` on login node
- [x] 2.2 Record results (see below)
- [x] 2.3 Decision: use T=1.0 (no scaling) for round 0

### Temperature Sweep Results (2026-03-23)

AUROC is rank-based and invariant to T. We measured log loss and Brier score
(calibration-sensitive metrics) instead.

| T    | Log Loss | Brier  | AUROC  |
|------|----------|--------|--------|
| 0.50 | 0.3163   | 0.0414 | 0.8771 |
| 0.55 | 0.2901   | 0.0410 | 0.8771 |
| 0.60 | 0.2678   | 0.0407 | 0.8771 |
| 0.65 | 0.2487   | 0.0403 | 0.8771 |
| 0.70 | 0.2323   | 0.0400 | 0.8771 |
| 0.80 | 0.2058   | 0.0393 | 0.8771 |
| 1.00 | 0.1699   | 0.0380 | 0.8771 |

**Finding**: Teachers are slightly over-confident — lower T (sharpening)
increases log loss. T=1.0 gives the best calibration. However, the 2025
1st place winner used T=0.55-0.6 for training benefit (stronger gradient
signal), not for calibration. The true test of T is downstream student
quality, which is expensive to evaluate.

**Decision**: Use raw T=1.0 pseudo-labels for round 0 to establish a clean
baseline. Defer T tuning to future work (round 1+) once we have a baseline
to compare against.

## 3. Pipeline Validation (Fast Dev Run)

- [x] 3.1 Submit fast-dev-run: `bash sbatch/run.sh config/protoclr_student.yaml --fast-dev-run`
      Job 5413719, completed in 5m19s on gpu-rtx6000 (Quadro RTX 6000)
- [x] 3.2 Verify dataset sizes: 212,725 focal + 570 soundscape + 127,104 pseudo = **340,399 train**
      Val: 27,116 | Test: 26,299 (no pseudo-labels in val/test)
- [x] 3.3 Verify loss: train/loss=0.011, val/loss=0.010, test/loss=0.013 (no NaN/inf)
- [x] 3.4 Verify metrics: val/auroc=0.222, val/cmap=0.142 (expected low for 5-batch run)
      All metrics logged: auroc, cmap, f1_macro, f1_micro, top1_acc, top5_acc

## 4. Full Training (Round 0)

- [x] 4.1 Submit all 3 models: `bash sbatch/run.sh --all`
      Jobs: ProtoCLR 5414046, ConvNeXt 5414044, EAT 5414045
- [x] 4.2 Monitor training via W&B dashboard
- [x] 4.3 ProtoCLR finished — record results below
- [ ] 4.3b ConvNeXt — in progress
- [ ] 4.3c EAT — in progress
- [ ] 4.4 Verify structured metrics saved to `artifacts/training/` (auto-runs via `save_metrics.py` in `_job.sh`)

### GPU Check (ProtoCLR, job 5415802)

| Metric | Min | Avg | Max |
|--------|-----|-----|-----|
| GPU Utilization | 43.0% | 67.9% | 84.0% |
| Data Loading % | 0.6% | 13.6% | 84.4% |
| Throughput | 95 s/s | 577 s/s | 684 s/s |
| Memory Allocated | 0.14 GB | 0.62 GB | 1.09 GB |
| Memory % | 0.5% | 2.4% | 4.3% |

**Bottleneck**: I/O spikes from loading 10,592 unlabeled soundscape files on
shared filesystem. Updated configs from 8 → 12 num_workers for future runs.

### Training Results

| Model | val/auroc | val/cmap | test/auroc | test/cmap | Best Epoch | Stopped |
|-------|-----------|----------|-----------|----------|------------|---------|
| ProtoCLR | 0.827 | 0.476 | 0.846 | 0.485 | 5 | 9 (early stop) |
| ConvNeXt | pending | | | | | |
| EAT | pending | | | | | |

### Comparison vs Exp 012 Baseline (offline test set)

| Model | Exp 012 AUROC | Exp 015 AUROC | Delta | Exp 012 cmAP | Exp 015 cmAP | Delta |
|-------|--------------|--------------|-------|-------------|-------------|-------|
| ProtoCLR | 0.845 | 0.846 | +0.001 | 0.481 | 0.485 | +0.004 |
| ConvNeXt | 0.844 | pending | | 0.470 | pending | |
| EAT | 0.838 | pending | | 0.452 | pending | |

**Note**: Offline metrics essentially flat for ProtoCLR. Model barely moved
from exp 012 starting point (early stopped at epoch 9, best at epoch 5).

## 5. Evaluation

- [x] 5.1 Export ProtoCLR to ONNX: `bash sbatch/run.sh export --model protoclr-2026-student-v1`
      Added to exp 010 pipeline: export.sbatch, upload_kaggle.py, kernel metadata, submission.py MODEL_SLUGS
- [x] 5.2 Upload + push kernel: `python scripts/upload_kaggle.py --model protoclr-2026-student-v1 --push-kernel`
- [ ] 5.3 ProtoCLR LB score — Kaggle inference running (~1hr)
- [ ] 5.4 Export ConvNeXt to ONNX + submit (after training completes)
- [ ] 5.5 Export EAT to ONNX + submit (after training completes)

### LB Results

| Model | Exp 012 LB | Exp 015 LB | Delta |
|-------|-----------|-----------|-------|
| ProtoCLR | 0.767 | pending | |
| ConvNeXt | 0.755 | pending | |
| EAT | 0.720 | pending | |

## 6. Documentation

- [x] 6.1 Update MODEL-RUNS.md with exp 015 results (ProtoCLR done, others pending)
- [ ] 6.2 Update docs/STATUS.md (after all LB scores in)

---

## Future Work

- **Weight sweep**: Test w ∈ [0.1, 0.25, 0.75, 1.0] if baseline w=0.5 shows promise
- **CE loss**: Switch labeled data from BCE to CE loss (2024/2025 winners found it better)
- **Iterative self-training**: Use exp 015 students as teachers → re-run exp 014 → retrain (round 1+)
- **Confidence thresholding**: Filter low-confidence pseudo-labels (threshold 0.9+)
- **Temperature tuning**: Train one model with T=0.6 vs T=1.0 to measure downstream effect (calibration metrics were uninformative — AUROC is rank-invariant, and log loss favored T=1.0 due to teacher over-confidence)
