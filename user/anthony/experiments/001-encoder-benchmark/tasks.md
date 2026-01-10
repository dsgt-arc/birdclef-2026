# Tasks

## 1. Environment Setup
- [x] 1.1 Create scratch directory structure (`~/scratch/birdclef-2026/`)
- [x] 1.2 Create virtual environment in scratch
- [x] 1.3 Create pyproject.toml with dependencies
- [x] 1.4 Install project in editable mode

## 2. Data & Checkpoint Downloads
- [x] 2.1 Download BEATs checkpoint - from HuggingFace mirror `lpepino/beats_ckpts`
- [x] 2.2 Download EAT checkpoint - from HuggingFace `worstchan/EAT-base_epoch30_pretrain`
- [x] 2.3 Verify ESC-50 data available (vendor/ESC-50)
- [ ] 2.4 (Stretch) Download AudioSet bird clips

## 3. Implementation
- [x] 3.1 Create `src/birdclef/` package structure
- [x] 3.2 Implement audio loading utilities (resampling to 16kHz)
- [x] 3.3 Implement wav2vec2 feature extractor (via transformers)
- [x] 3.4 Implement BEATs feature extractor wrapper
- [x] 3.5 Implement EAT feature extractor wrapper
- [x] 3.6 Implement linear probe training loop
- [x] 3.7 Implement evaluation metrics (accuracy)
- [x] 3.8 Create unified benchmark runner script

## 4. Part A: ESC-50 Benchmark
- [x] 4.1 Prepare ESC-50 dataloader with 5-fold CV splits
- [x] 4.2 Run wav2vec2 + linear probe on ESC-50 (48.2%)
- [x] 4.3 Run BEATs + linear probe on ESC-50 (95.8%)
- [x] 4.4 Run EAT + linear probe on ESC-50 (77.9% with mean pooling fix)
- [x] 4.5 Compare results to published (BEATs 98.1%, EAT 95.9%) - see results.md

## 5. Part B: LoRA Finetuning EAT
- [x] 5.1 Document LoRA approach (docs/concepts/lora-finetuning-audio.md)
- [x] 5.2 Implement LoRA finetuning script (finetune_lora.py)
- [ ] 5.3 Run LoRA r=8 on ESC-50
- [ ] 5.4 Run LoRA r=16 on ESC-50 (if r=8 underfits)
- [ ] 5.5 Compare to linear probe and published results

## ~~6. Part C: AudioSet Birds~~ (Out of Scope)
*Removed from experiment scope - focus on ESC-50 benchmark*

## 7. Analysis & Documentation
- [x] 7.1 Compile results table with accuracy/mAP/inference time
- [x] 7.2 Analyze which encoder performs best (BEATs)
- [x] 7.3 Document findings in results.md
- [ ] 7.4 Update STATUS.md with conclusions

---

## Notes

### Checkpoints

**BEATs** - Downloaded from HuggingFace mirror:
- Repo: `lpepino/beats_ckpts`
- Uploader: Leonardo Pepino ([@mrpep](https://github.com/mrpep)) - audio/deep learning researcher
- This is a mirror of the official Microsoft checkpoints (OneDrive links are not programmatically accessible)
- Original source: https://github.com/microsoft/unilm/tree/master/beats
- License: MIT

**EAT** - Downloaded from HuggingFace:
- Repo: `worstchan/EAT-base_epoch30_pretrain`
- Official release from EAT authors

### Running the Benchmark
```bash
# Run all encoders (linear probe)
sbatch user/anthony/experiments/001-encoder-benchmark/sbatch/run-benchmark.sbatch

# Run specific encoder
ENCODER=wav2vec2 sbatch user/anthony/experiments/001-encoder-benchmark/sbatch/run-benchmark.sbatch
```

### Running LoRA Finetuning
```bash
# LoRA with rank 8 (default)
cd user/anthony/experiments/001-encoder-benchmark/sbatch
sbatch run-lora-finetune.sbatch

# LoRA with rank 16
LORA_RANK=16 sbatch run-lora-finetune.sbatch
```

### Package Structure
```
user/anthony/src/birdclef/
├── __init__.py
├── audio.py           # Audio loading utilities
├── train.py           # Linear probe training
├── benchmark.py       # Unified benchmark runner
├── finetune_lora.py   # LoRA finetuning for EAT
├── data/
│   ├── __init__.py
│   └── esc50.py       # ESC-50 dataset
└── encoders/
    ├── __init__.py
    ├── base.py        # AudioEncoder base class
    ├── wav2vec2.py    # Wav2Vec2 encoder
    ├── beats.py       # BEATs encoder
    └── eat.py         # EAT encoder
```
