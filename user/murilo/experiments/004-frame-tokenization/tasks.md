# Experiment 004: Frame-Based Tokenization - Tasks

## 1. Setup

- [ ] 1.1 Create experiment directory structure and `pyproject.toml`
- [ ] 1.2 Verify exp 003 scratch baseline is reproducible (re-run ViT-Small 16×16 scratch with fixed seed)
- [ ] 1.3 Confirm timm supports non-square `patch_size` kwarg for `vit_small_patch16_224` (test in a notebook or script)

## 2. Implementation

- [ ] 2.1 Extend `AudioClassifier` to accept configurable `patch_size` tuple via config
  - Modify `timm.create_model(...)` call to pass `patch_size=(freq, time)` and `img_size=(128, 500)`
  - Ensure 1-channel input adaptation still works with non-square patch embedding kernel
- [ ] 2.2 Verify positional embeddings are correctly sized for each patch configuration
  - 16×16 → 248 positions, 128×2 → 250 positions, 128×4 → 125 positions, 128×1 → 500 positions
  - For scratch training: random init, just confirm tensor shapes are correct
- [ ] 2.3 Create config files for each configuration:
  - `config/baseline_16x16.yaml` — ViT-Small scratch, patch_size: [16, 16]
  - `config/vertical_128x2.yaml` — ViT-Small scratch, patch_size: [128, 2]
  - `config/vertical_128x4.yaml` — ViT-Small scratch, patch_size: [128, 4]
  - `config/vertical_128x1.yaml` — ViT-Small scratch, patch_size: [128, 1]
- [ ] 2.4 Adapt `train.py` from exp 003 (or reuse directly) with patch_size config support
- [ ] 2.5 Create SLURM scripts for batch training all configurations

## 3. Execution

- [ ] 3.1 Run Config A (128×2) — primary hypothesis test
- [ ] 3.2 Run Config B (128×4) — coarser temporal resolution comparison
- [ ] 3.3 Run Config C (128×1) — finest temporal resolution comparison
- [ ] 3.4 Run baseline (16×16) with same seed — controlled comparison
- [ ] 3.5 Monitor training via W&B, check for convergence issues

## 4. Analysis

- [ ] 4.1 Compare cmAP across all 4 configurations
- [ ] 4.2 Analyze training curves (convergence speed, overfitting patterns)
- [ ] 4.3 Compare walltime per epoch across configs (token count affects attention cost)
- [ ] 4.4 Document results in `results.md`
- [ ] 4.5 Update concept doc `temporal-resolution-sequence-length.md` with findings
