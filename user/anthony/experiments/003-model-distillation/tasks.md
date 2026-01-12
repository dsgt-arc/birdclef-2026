# Tasks

## 1. Setup
- [ ] 1.1 Download AudioSet-20k (pilot) from HuggingFace: `confit/audioset-16khz-wds`
- [ ] 1.2 Download AudioSet-500k (main) from HuggingFace: `confit/audioset-16khz-wds`
- [ ] 1.3 Create WebDataset loader for efficient streaming of AudioSet
- [ ] 1.4 Verify teacher (BEATs) inference pipeline on AudioSet shards
- [ ] 1.5 **Regenerate Static Embeddings Codebook**: Run Model2Vec extraction on AudioSet-20k to build a robust, universal codebook

## 2. Implementation
- [ ] 2.1 Implement `DistillationStudent` architectures:
  - [ ] 2.1a `Mamba2Student` (using HuggingFace transformers)
  - [ ] 2.1b `ConvNeXt1DStudent` (depthwise separable + dilated)
  - [ ] 2.1c `TCNStudent` (using pytorch-tcn)
  - [ ] 2.1d `MLPMixerStudent` (token & channel mixing)
  - [ ] 2.1e `TinyTransformerStudent` (2 layers, 256-dim, 4 heads)
- [ ] 2.2 Implement `DistillationTrainer` with MSE/Cosine loss
- [ ] 2.3 Create training script `run-distillation.py` (streaming mode)

## 3. Execution
- [ ] 3.1 Run pilot distillation (e.g., 10k samples) to verify convergence
- [ ] 3.2 Run full-scale distillation (AudioSet Unbalanced Train)
- [ ] 3.3 Save student checkpoints for all architectures

## 4. Evaluation
- [ ] 4.1 Create evaluation script `eval-audioset.py` (Freeze backbone -> Train Linear -> Calc mAP)
- [ ] 4.2 Run AudioSet mAP evaluation
- [ ] 4.3 Run ESC-50 5-fold CV on distilled student (Transfer check)
- [ ] 4.4 Benchmark inference speed of students on CPU (check Mamba CPU viability)

## 5. Analysis
- [ ] 5.1 Compare Mamba vs CNN vs Transformer (Accuracy vs Speed)
- [ ] 5.2 Analyze impact of distillation vs direct training (Exp 002 comparison)
- [ ] 5.3 Document findings in `results.md`
