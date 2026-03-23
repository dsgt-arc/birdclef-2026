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

- [ ] 4.1 Submit all 3 models: `bash sbatch/run.sh --all`
- [ ] 4.2 Monitor training via W&B dashboard
- [ ] 4.3 Record best val/auroc and stopped epoch for each model

### Results

| Model | val/auroc | val/cmap | Best Epoch | Stopped |
|-------|-----------|----------|------------|---------|
| ProtoCLR | | | | |
| ConvNeXt | | | | |
| EAT | | | | |

## 5. Evaluation

- [ ] 5.1 Export best checkpoints to ONNX (via exp 010 pipeline)
- [ ] 5.2 Submit to Kaggle LB
- [ ] 5.3 Record LB scores and compare to exp 012 baseline

### LB Results

| Model | Exp 012 LB | Exp 015 LB | Delta |
|-------|-----------|-----------|-------|
| ProtoCLR | 0.767 | | |
| ConvNeXt | 0.755 | | |
| EAT | 0.720 | | |

## 6. Documentation

- [ ] 6.1 Update MODEL-RUNS.md with exp 015 results
- [ ] 6.2 Update docs/STATUS.md

---

## Future Work

- **Weight sweep**: Test w ∈ [0.1, 0.25, 0.75, 1.0] if baseline w=0.5 shows promise
- **CE loss**: Switch labeled data from BCE to CE loss (2024/2025 winners found it better)
- **Iterative self-training**: Use exp 015 students as teachers → re-run exp 014 → retrain (round 1+)
- **Confidence thresholding**: Filter low-confidence pseudo-labels (threshold 0.9+)
- **Temperature tuning**: Train one model with T=0.6 vs T=1.0 to measure downstream effect (calibration metrics were uninformative — AUROC is rank-invariant, and log loss favored T=1.0 due to teacher over-confidence)
