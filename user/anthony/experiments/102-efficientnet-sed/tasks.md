# Tasks

## 1. Setup
- [ ] 1.1 Fork exp 012 model.py, cli.py, pyproject.toml
- [ ] 1.2 Create config for BirdCLEF 2026 data paths
- [ ] 1.3 Verify data module works with 2026 data (fast-dev-run)

## 2. Implementation
- [ ] 2.1 Run A model: frozen backbone + linear head + GAP (port exp 012 to 2026)
- [ ] 2.2 Run B model: unfreeze backbone, add attention head, differential LR
- [ ] 2.3 Run C model: add soundscape data loading + focal-soundscape mixup
- [ ] 2.4 Verify forward pass shapes with fast-dev-run for each variant

## 3. Training
- [ ] 3.1 Run A: frozen + linear (establish 2026 baseline)
- [ ] 3.2 Run B: unfrozen + attention (main hypothesis)
- [ ] 3.3 Run C: unfrozen + attention + mixup (domain adaptation)
- [ ] 3.4 Monitor GPU utilization and training curves

## 4. Evaluation
- [ ] 4.1 Evaluate all runs on focal validation (AUROC, cmAP)
- [ ] 4.2 Evaluate on 739 labeled soundscape windows (exp 101 pipeline)
- [ ] 4.3 Export best to ONNX
- [ ] 4.4 Kaggle submission

## 5. Documentation
- [ ] 5.1 Write results.md
- [ ] 5.2 Update docs/STATUS.md
