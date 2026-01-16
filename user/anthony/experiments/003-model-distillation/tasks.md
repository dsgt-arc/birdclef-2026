# Tasks

## 1. Setup
- [x] 1.1 Download AudioSet-20k (pilot) from HuggingFace: `confit/audioset-16khz-wds`
  - SLURM job 3269541 submitted
- [x] 1.2 Download AudioSet-500k (main) from HuggingFace: `confit/audioset-16khz-wds`
  - 145 shards, 147 GB, ~500k samples
  - SLURM job 3279383 completed in 2.7 minutes
- [x] 1.3 Create WebDataset loader for efficient streaming of AudioSet
  - `birdclef/data/audioset.py` - AudioSetWebDataset, AudioSetDistillationDataset
- [x] 1.4 Verify teacher (BEATs) inference pipeline on AudioSet shards
  - Teacher caching completed: 7 shards, 20,550 samples cached
- [x] 1.5 Cache teacher embeddings for AudioSet-500k
  - SLURM array job 3279629 completed in ~25 minutes
  - 15 parallel V100 tasks, 10 shards each
  - **143 shards cached, 687 GB** (2 shards missing from source dataset)
  - Uses NVMe local storage for faster IO (~28ms/sample)
  - Idempotent: `sbatch --array=0-14 cache-teacher-array.sbatch`
- [ ] 1.6 **Regenerate Static Embeddings Codebook**: Run Model2Vec extraction on AudioSet-20k to build a robust, universal codebook

## 2. Implementation
- [x] 2.1 Implement `DistillationStudent` architectures:
  - [ ] 2.1a `Mamba2Student` (using HuggingFace transformers) - deferred for later
  - [x] 2.1b `ConvNeXt1DStudent` (depthwise separable + dilated) - `birdclef/students.py`
  - [ ] 2.1c `TCNStudent` (using pytorch-tcn) - deferred for later
  - [x] 2.1d `MLPMixerStudent` (token & channel mixing) - `birdclef/students.py`
  - [x] 2.1e `TinyTransformerStudent` (2 layers, 256-dim, 4 heads) - `birdclef/students.py`
- [x] 2.2 Implement `DistillationTrainer` with MSE/Cosine loss - `birdclef/distillation.py`
- [x] 2.3 Create training script `run-distillation.py` (streaming mode)
  - `sbatch/run-pilot-distillation.sbatch`

## 3. Execution
- [x] 3.1 Run pilot distillation (e.g., 20k samples) to verify convergence
  - SLURM job 3270511 completed (convnext-small)
  - 5 epochs, loss: 0.386 → 0.141 (MSE: 0.072, Cosine: 0.211)
  - Model saved: `pilot_convnext-small_epoch5.pt`
- [x] 3.2 Run extended training with cached teacher embeddings
  - SLURM job 3278762 completed (convnext-small)
  - 50 epochs in 27 minutes using local NVMe cache
  - Val loss: 0.402 → 0.126 (MSE: 0.065)
  - Model saved: `cached_convnext-small_best.pt`
- [ ] 3.3 Run full-scale distillation (AudioSet-500k)
- [ ] 3.4 Save student checkpoints for all architectures

## 4. Evaluation
- [ ] 4.1 Create evaluation script `eval-audioset.py` (Freeze backbone -> Train Linear -> Calc mAP)
- [ ] 4.2 Run AudioSet mAP evaluation
- [x] 4.3 Create ESC-50 evaluation script - `sbatch/run-esc50-eval.sbatch`
- [x] 4.4 Run ESC-50 evaluation on pilot model
  - convnext-small pilot: 51.95% accuracy (5-fold CV)
- [x] 4.5 Run ESC-50 evaluation on extended training model
  - **convnext-small extended: 55.90% accuracy** (5-fold CV)
  - Fold accuracies: 55.50%, 56.25%, 59.25%, 55.25%, 53.25%
  - Comparison: Static=36%, Pilot=52%, **Extended=56%**, LightCNN=56%, BEATs=84%
- [ ] 4.6 Benchmark inference speed of students on CPU

## 5. Analysis
- [ ] 5.1 Compare Mamba vs CNN vs Transformer (Accuracy vs Speed)
- [ ] 5.2 Analyze impact of distillation vs direct training (Exp 002 comparison)
- [ ] 5.3 Document findings in `results.md`
