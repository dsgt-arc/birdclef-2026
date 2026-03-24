# Tasks

## 1. Setup
- [x] 1.1 Fork exp 012 model.py, cli.py into src/birdclef_102/
- [x] 1.2 Create pyproject.toml
- [x] 1.3 Create configs for each run
- [x] 1.4 Update data paths to BirdCLEF 2026
- [x] 1.5 Verify with fast-dev-run
- [x] 1.6 Build activity scores parquet

## 2. Implementation
- [x] 2.1 Vendor data module with parquet-based loading
- [x] 2.2 Vendor training module, losses, SpecAugment, GPU monitor
- [x] 2.3 Add LoRA via PEFT
- [x] 2.4 Add attention pooling with standard SED formula (tanh + sigmoid)
- [x] 2.5 Add GEMM hidden projection layer
- [x] 2.6 Add ONNX export (handles frozen, LoRA merge, attention)
- [x] 2.7 Add Kaggle submission kernel

## 3. Training
- [ ] 3.1 Run A: frozen + avg pool (2026 baseline) — job 5436408
- [ ] 3.2 Run B: LoRA + avg pool — job 5436505
- [ ] 3.3 Run C: LoRA + attention + GEMM(512) — ready to submit
- [ ] 3.4 Run D: best pooling + soundscape mixup — needs data module

## 4. Evaluation
- [ ] 4.1 Focal validation (AUROC, cmAP) for each run
- [ ] 4.2 Soundscape evaluation (exp 101 pipeline, 739 windows)
- [ ] 4.3 Export best to ONNX
- [ ] 4.4 Kaggle submission

## 5. Documentation
- [ ] 5.1 Write results.md
- [ ] 5.2 Update docs/STATUS.md
