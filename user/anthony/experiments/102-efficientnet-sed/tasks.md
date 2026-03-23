# Tasks

## 1. Setup
- [ ] 1.1 Fork exp 012 model.py, cli.py, pyproject.toml
- [ ] 1.2 Create config for BirdCLEF 2026 data paths
- [ ] 1.3 Verify data module works with 2026 data (fast-dev-run)

## 2. Implementation
- [ ] 2.1 Add SED head: frame_head + att_head (Linear(1280, C) each)
- [ ] 2.2 Add attention pooling (train) / framewise max (inference) switching
- [ ] 2.3 Remove backbone freeze, add differential LR (backbone lr/10)
- [ ] 2.4 Add partial unfreeze option (last N blocks only)
- [ ] 2.5 Verify forward pass shapes with fast-dev-run

## 3. Training
- [ ] 3.1 Run A: full unfreeze + SED head
- [ ] 3.2 Run B: partial unfreeze (last 3 blocks) + SED head
- [ ] 3.3 Run C: full unfreeze + linear head (ablation)
- [ ] 3.4 Monitor GPU utilization and training curves

## 4. Evaluation
- [ ] 4.1 Evaluate on focal validation (AUROC, cmAP)
- [ ] 4.2 Evaluate on 739 labeled soundscape windows (exp 101 pipeline)
- [ ] 4.3 Export best to ONNX
- [ ] 4.4 Kaggle submission

## 5. Documentation
- [ ] 5.1 Write results.md
- [ ] 5.2 Update docs/STATUS.md
