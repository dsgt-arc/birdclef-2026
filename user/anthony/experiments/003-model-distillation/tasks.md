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
- [x] 3.3 Redesign for StreamingCacheDataset with smart prefetching
  - **Problem discovered**: Initial jobs 3280182, 3280685 failed
    - 687GB upfront copy took 30+ minutes before training started
    - FileNotFoundError during training (missing shard files)
    - /tmp space constraints on compute nodes
  - **Solution v1**: Best-effort caching with dual-path fallback
    - Local NVMe cache (fast) with source cache (reliable) fallback
    - Optional upfront copy (COPY_TO_LOCAL flag)
    - Graceful degradation when files missing
  - **Problem discovered v2**: Dataset initialization still too slow
    - Loading all 143 shards (687GB) just to count samples
    - Random shuffling defeated prefetching (forced all shards in memory)
  - **Solution v2**: Fast metadata + sequential processing
    - Use safetensors metadata API to read shapes without loading data
    - Cache metadata to `shard_metadata.json` (seconds instead of minutes)
    - Sequential shard processing (0→143) with within-shard shuffling
    - Background prefetching of next N shards (always predictable)
    - `reset_epoch()` method reshuffles within shards each epoch
  - Documentation: `docs/references/streaming-cache-implementation.md`
- [x] 3.4 Add GPU monitoring and SLURM best practices
  - Created `scripts/monitor-gpu.sh` for real-time GPU utilization tracking
  - Integrated into sbatch scripts with background monitoring + cleanup trap
  - Created `/slurm:sbatch-review` skill for idiomatic sbatch patterns
  - Added venv sync to TMPDIR for faster package loading
- [~] 3.5 Run full-scale distillation (AudioSet-500k) with optimized streaming cache
  - SLURM job 3284233 submitted (convnext-small) - **Currently running**
  - 5 epochs, patience 3, batch_size 64, num_workers 8
  - ~500k samples (436k train, 47k val), 90/10 train/val split
  - Streaming-only mode (COPY_TO_LOCAL=false, no upfront copy)
  - Fast metadata initialization + sequential shard processing
  - GPU monitoring active: `logs/gpu-3284233.csv`
- [ ] 3.6 Save student checkpoints for all architectures

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
