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

## 5. Part B: LoRA Finetuning
- [x] 5.1 Document LoRA approach (docs/concepts/lora-finetuning-audio.md)
- [x] 5.2 Implement LoRA finetuning script (finetune_lora.py)
- [x] 5.3 Run EAT LoRA r=8 on ESC-50 (94.6%)
- [x] 5.4 Run wav2vec2 LoRA r=8 on ESC-50 (3.0% - failed, model didn't learn)
- ~~[ ] 5.5 Run LoRA r=16 on ESC-50 (if r=8 underfits)~~ (skipped - r=8 sufficient)
- [x] 5.6 Compare to linear probe and published results
- [ ] 5.7 Run SSAMBA-HF LoRA r=8 on ESC-50 (job 3447764, in progress)

## ~~6. Part B.5: AudioSet Birds~~ (Deferred)
*Deferred - focus on encoder comparison first*

## 6. Part C: AST/SSAST/SSAMBA Benchmark (New)

### 6.1 Checkpoint Downloads
- [x] 6.1.1 Verify AST checkpoint auto-downloads from HuggingFace (`MIT/ast-finetuned-audioset-10-10-0.4593`)
- [x] 6.1.2 Download SSAST checkpoint (`SSAST-Base-Patch-400.pth`) from Dropbox
- [x] 6.1.3 Download SSAMBA checkpoints from HuggingFace (tiny, small, base)
- [x] 6.1.4 Clone Vim (Vision Mamba) repo into vendor/ssamba/

### 6.2 Implementation
- [x] 6.2.1 Implement `ASTEncoder` in `encoders/ast.py`
- [x] 6.2.2 Implement `SSASTEncoder` in `encoders/ssast.py`
- [x] 6.2.3 Implement `SSAMBAEncoder` in `encoders/ssamba.py`
- [x] 6.2.4 Add mamba-ssm dependency (installed via sbatch script, not pyproject.toml)
- [x] 6.2.5 Verify preprocessing matches original papers (fbank settings)

### 6.3 Benchmark Execution
- [x] 6.3.1 Run AST + linear probe on ESC-50 → **95.50% accuracy**
- [ ] 6.3.2 Run SSAST + linear probe on ESC-50 → **BLOCKED** (see issues below)
- [ ] 6.3.3 Run SSAMBA-tiny + linear probe on ESC-50 → **BLOCKED** (see issues below)
- [ ] 6.3.4 Run SSAMBA-small + linear probe on ESC-50
- [ ] 6.3.5 Run SSAMBA-base + linear probe on ESC-50

### 6.4 Inference Time Benchmarks
- [ ] 6.4.1 Benchmark all encoders on GPU (batch=1, 16, 32)
- [ ] 6.4.2 Benchmark all encoders on CPU (batch=1, 4, 16)
- [ ] 6.4.3 Compare SSAMBA vs transformer speedup

### 6.5 Analysis
- [ ] 6.5.1 Create accuracy vs speed comparison table
- [ ] 6.5.2 Plot Pareto frontier (accuracy vs inference time)
- [ ] 6.5.3 Determine if SSAMBA meets BirdCLEF inference budget
- [ ] 6.5.4 Update results.md with Part C findings

### 6.6 Known Issues (2025-01-15)

**SSAST - Environment Conflict:**
- Vendor code requires `timm==0.4.5` (strict assertion in code)
- Downgrading timm breaks CUDA libraries: `ImportError: libcublasLt.so.12`
- **Fix needed:** Either port SSAST to work with newer timm, or run in separate venv

**SSAMBA - CUDA Version Mismatch:**
- `mamba-ssm` and `causal-conv1d` need to compile CUDA kernels
- Cluster CUDA (13.0) mismatches PyTorch CUDA (12.8)
- HuggingFace only hosts weights, no model implementation
- **Fix needed:**
  1. Port SSAMBA model code to standalone (remove Vim dependency)
  2. Use pre-built mamba-ssm wheel matching cluster CUDA
  3. Run in separate venv with matching CUDA/PyTorch versions

**Recommended approach:**
- Create separate conda/venv environments for SSAST and SSAMBA
- Port minimal model code to avoid vendor dependency pollution
- For SSAMBA: implement standalone model class that loads HuggingFace weights

## 7. Analysis & Documentation
- [x] 7.1 Compile results table with accuracy/mAP/inference time
- [x] 7.2 Analyze which encoder performs best (BEATs)
- [x] 7.3 Document findings in results.md
- [x] 7.4 Update STATUS.md with conclusions

---

## Running Jobs

| Job ID | Encoder | Config | Submitted | Status | Output |
|--------|---------|--------|-----------|--------|--------|
| 3234438 | EAT | LoRA r=8, epochs=50, lr=1e-4 | 2026-01-10 | ✅ Completed (94.6%) | `logs/lora-finetune-3234438.out` |
| 3234515 | wav2vec2 | LoRA r=8, epochs=50, lr=1e-4 | 2026-01-10 | ❌ Failed (NaN loss) | `logs/lora-finetune-3234515.out` |
| 3242824 | wav2vec2 | LoRA r=8, epochs=50, lr=1e-5, grad_clip=1.0 | 2026-01-10 | ❌ Failed (3.0% - no learning) | `logs/lora-finetune-3242824.out` |
| 3447764 | SSAMBA-HF | LoRA r=8, epochs=50, lr=1e-4, batch=4 | 2026-01-26 | 🔄 Running (RTX 6000) | `logs/lora-finetune-3447764.out` |

**Expected outputs:**
- `~/scratch/birdclef-2026/outputs/esc50-lora/eat_lora_r8_results.json` ✅
- `~/scratch/birdclef-2026/outputs/esc50-lora/wav2vec2_lora_r8_results.json`

**Estimated runtime:** ~2-3 hours per job (5 folds × 50 epochs)

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

### Running Part C Benchmarks (AST/SSAST/SSAMBA)

All scripts are idempotent - they skip downloads/benchmarks if outputs already exist.
Use `--force` flag with benchmark.py to re-run.

```bash
cd user/anthony/experiments/001-encoder-benchmark/sbatch

# Step 1: Download checkpoints (run first, wait for completion)
sbatch download-part-c-checkpoints.sbatch

# Step 2: Run benchmarks (can run in parallel after step 1)
sbatch run-ast-benchmark.sbatch           # AST (no checkpoint needed - HuggingFace)
sbatch run-ssast-benchmark.sbatch         # SSAST (needs checkpoint from step 1)
MODEL_SIZE=tiny sbatch run-ssamba-benchmark.sbatch   # SSAMBA variants (need step 1)
MODEL_SIZE=small sbatch run-ssamba-benchmark.sbatch
MODEL_SIZE=base sbatch run-ssamba-benchmark.sbatch
```

**Part C Dependencies:**
- AST: Uses HuggingFace, auto-downloads (can run immediately)
- SSAST: Requires `timm==0.4.5` and checkpoint from Dropbox
- SSAMBA: Requires `mamba-ssm`, `causal-conv1d`, Vim repo, and checkpoints from HuggingFace

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
    ├── eat.py         # EAT encoder
    ├── ast.py         # AST encoder (Part C) ✅
    ├── ssast.py       # SSAST encoder (Part C) ✅
    └── ssamba.py      # SSAMBA encoder (Part C) ✅
```
