# Experiment 013: MixIT Source Separation + Pseudo-Labeling — Tasks

## 1. Goal

Use Bird MixIT source separation + exp 012 teacher ensemble to pseudo-label 10,592
unlabeled soundscapes, then train on pseudo-labels to improve LB score.

## 2. Phase 1: MixIT Conversion (TF1 → ONNX)

- [x] 2.1 Download Bird MixIT TF1 checkpoint to `shared/models/bird-mixit/`
- [x] 2.2 Write `scripts/convert_mixit_onnx.py` (tf2onnx conversion + validation)
- [x] 2.3 Write `sbatch/convert.sh` SLURM job
- [x] 2.4 Run conversion: ONNX model saved (38.4 MB)
- [x] 2.5 Validate ONNX matches TF1: max diff 0.000013 (well under 1e-4 atol)
- [x] 2.6 Benchmark: ~7s/file on GPU (CUDAExecutionProvider), ~30s/file on CPU

## 3. Phase 2: Offline Separation

- [x] 3.1 Write `scripts/separate_soundscapes.py` (GPU default, distributed via `--num-workers`)
- [x] 3.2 Write `sbatch/run.sh` launcher with `--num-workers N` (SLURM job arrays)
- [x] 3.3 GPU smoke test: 10 files in 1.2 min (7s/file with CUDAExecutionProvider)
- [x] 3.4 Labeled separation: 66 files across 4 GPUs in ~2 min
  - Output: `shared/mixit/pretrained/soundscapes_labeled/` (66 NPZ files, 1.25 GB)
  - Shape: (4, 1323000) per file — 4 channels × 60s at 22050 Hz
  - Active channels: 1-2 per file (most signal in channel 0)

### 3.5 Quality validation (BLOCKING — do before full unlabeled run)

> **Anthony's feedback**: The base Bird MixIT model was "kind of crappy" in BirdCLEF 2023.
> We must validate separation quality before scaling to 10,592 files.

- [ ] 3.5.1 Listen to separated channel outputs from the 66 labeled files
- [ ] 3.5.2 Visualize spectrograms in `user/murilo/notebooks/004-mixit-separation-quality.ipynb`:
  original vs 4 separated channels for 3-5 files
  - Pick files with different characteristics: dense frog chorus, bird dawn chorus, quiet/sparse
- [ ] 3.5.3 Compare separated channels against expert labels — does channel 0 contain
  the dominant species? Do other channels isolate background species?
- [ ] 3.5.4 Decision: proceed with pretrained model OR prioritize domain-matched training

### 3.6 Storage format update

> **Anthony's feedback**: Save as OGG (lossy, 192kbps VBR) instead of raw NPZ.
> Use ffmpeg for batch conversion. Keep 22050 Hz sample rate.

- [ ] 3.6.1 Update `separate_soundscapes.py` to save channels as OGG (via ffmpeg or soundfile)
  - Option A: 4 separate OGG files per soundscape (e.g., `{filename}_ch0.ogg`, ..., `{filename}_ch3.ogg`)
  - Option B: Single 4-channel OGG file per soundscape
- [ ] 3.6.2 Re-run labeled separation with OGG output
- [ ] 3.6.3 Verify downstream pipeline can load OGG channels for pseudo-labeling

### 3.7 Full unlabeled separation (after quality validation)
- [ ] 3.7.1 Run: `bash sbatch/run.sh separate --num-workers 10` (~2 hours)
- [ ] 3.7.2 Verify output: 10,592 files in `shared/mixit/pretrained/soundscapes/`

## 4. Phase 3: Teacher Pseudo-Labeling (depends on exp 012 completion)

- [ ] 4.1 Write `scripts/pseudo_label.py` using exp 012 ONNX models
- [ ] 4.2 Implement aggregation: max-pool channels, average models
- [ ] 4.3 Validate on 66 labeled soundscapes: per-taxon precision/recall (bird vs amphibian vs insect)
- [ ] 4.4 Run on separated + original audio for all unlabeled soundscapes
- [ ] 4.5 Sweep confidence thresholds (0.3, 0.5, 0.7) and pick best

## 5. Phase 4: Student Training (depends on Phase 3)

- [ ] 5.1 Create `PseudoLabelDataset` for soft-label training
- [ ] 5.2 Implement soft BCE loss (teacher probabilities as targets)
- [ ] 5.3 Create `train.py` for mixed data: focal + labeled soundscape + pseudo-labeled
- [ ] 5.4 Train all 3 encoders
- [ ] 5.5 Export + submit to Kaggle LB

## 6. Results

- [ ] 6.1 Write results.md with pseudo-label quality metrics + LB scores
- [ ] 6.2 Compare vs exp 012 baseline

## Key Context

- **ONNX model**: `/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx` (38.4 MB)
- **Separated labeled output**: `/storage/project/ps-clef2026_birdclef-0/shared/mixit/pretrained/soundscapes_labeled/` (66 NPZ, 1.25 GB)
- **Separated unlabeled output**: `/storage/project/ps-clef2026_birdclef-0/shared/mixit/pretrained/soundscapes/` (pending quality validation)
- **GitHub reference**: https://github.com/google-research/sound-separation/tree/master/models/bird_mixit
- **GPU performance**: ~7s/file on RTX 6000, distributed via SLURM array jobs
- **Non-bird limitation**: Pretrained model is bird-only; quality may degrade for frogs/insects
- **Anthony's BirdCLEF 2023 experience**: Base model was "kind of crappy" — validate before scaling
