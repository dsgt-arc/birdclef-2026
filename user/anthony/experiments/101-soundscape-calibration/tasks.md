# Tasks

## 1. Setup
- [ ] 1.1 Create experiment package (`src/calibration_101/`)
- [ ] 1.2 Load Murilo's `.npz` files (handle `seg_keys` or reconstruct)
- [ ] 1.3 Build ground truth binary matrix from `soundscape_labels.parquet`
- [ ] 1.4 Align predictions to labeled windows; exclude 53 unlabeled gaps
- [ ] 1.5 Identify ~47 evaluable species (exclude 28 with zero focal data)
- [ ] 1.6 Flag species with <2 positive windows (exclude from per-species AUROC)
- [ ] 1.7 Quantify species coverage across CV sites vs S22 (for Phase 3 if needed)

## 2. Raw Model Evaluation (primary deliverable)
- [ ] 2.1 Per-species AUROC + AP for ProtoCLR / ConvNeXt / EAT
- [ ] 2.2 Macro + per-taxa summaries
- [ ] 2.3 Per-site breakdown
- [ ] 2.4 Inter-model agreement
- [ ] 2.5 Reliability diagrams (raw probability calibration)
- [ ] 2.6 Assign species tiers: reliable (>0.85) / usable (0.70–0.85) / unreliable (<0.70)

**Decision gate**: check macro AUROC and reliability diagrams → Phase 3 or skip to Phase 4

## 3. Platt Scaling (conditional — only if miscalibrated)
- [ ] 3.1 Partition: S22 test, 8 sites CV (`docs/split-strategy.md`)
- [ ] 3.2 Fit per-taxa Platt scaling via leave-one-site-out CV
- [ ] 3.3 ECE + Brier before/after
- [ ] 3.4 Evaluate on S22 test set

## 4. Recommendation
- [ ] 4.1 Select pseudo-labeling strategy (model, threshold, species tiers)
- [ ] 4.2 Produce config for 176.5h unlabeled soundscapes

## 5. Documentation
- [ ] 5.1 Write `results.md`
- [ ] 5.2 Update `docs/STATUS.md`
