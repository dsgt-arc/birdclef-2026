# Tasks

## 1. Setup
- [ ] 1.1 Fork exp 012 model.py, cli.py into src/birdclef_102/
- [ ] 1.2 Create pyproject.toml (add peft dependency)
- [ ] 1.3 Create configs: run-a.yaml (frozen baseline), run-b.yaml (LoRA)
- [ ] 1.4 Update data paths to BirdCLEF 2026
- [ ] 1.5 Verify with fast-dev-run

## 2. Implementation
- [ ] 2.1 Add LoRA wrapper via PEFT (apply_lora method on EfficientNet backbone)
- [ ] 2.2 Add config flag: freeze_backbone (true for Run A, false for Run B/C)
- [ ] 2.3 Add activity score loading option (sample weighting from focal_activity.parquet)
- [ ] 2.4 Run C: add soundscape data loading + mixup

## 3. Training
- [ ] 3.1 Run A: frozen + linear (2026 baseline)
- [ ] 3.2 Run B: LoRA + linear (main hypothesis)
- [ ] 3.3 Run C: LoRA + linear + soundscape mixup
- [ ] 3.4 Monitor training curves

## 4. Evaluation
- [ ] 4.1 Focal validation (AUROC, cmAP)
- [ ] 4.2 Soundscape evaluation (exp 101 pipeline, 739 windows)
- [ ] 4.3 Export best to ONNX → Kaggle submission

## 5. Documentation
- [ ] 5.1 Write results.md
- [ ] 5.2 Update docs/STATUS.md
