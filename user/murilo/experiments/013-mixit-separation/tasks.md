# Experiment 013: MixIT Source Separation — Tasks

## 1. Goal

Use Bird MixIT source separation to separate 10,592 unlabeled soundscapes into 4
channels per file, producing cleaner audio for downstream pseudo-labeling (exp 014)
and student training (exp 015).

## 2. Phase 1: MixIT Conversion (TF1 → ONNX) ✓

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

### 3.5 Quality validation results (2026-03-21)

Notebook: `user/murilo/notebooks/004-mixit-separation-quality.ipynb`
Figures: `experiments/013-mixit-separation/figures/`

**Channel activity across all 66 labeled files:**
```
Active channels (RMS > 0.01) distribution:
  0 active:  4 files ( 6%) — MixIT produced near-silence, no separable sources found
  1 active: 21 files (32%) — no meaningful separation, just original in channel 0
  2 active: 16 files (24%) — some separation
  3 active: 14 files (21%) — good separation
  4 active: 11 files (17%) — full 4-source separation

Mean RMS per channel: [0.0403, 0.0187, 0.0242, 0.0144]
Channel 0 dominates: 2.1x higher RMS than channels 1-3
```

**Qualitative assessment** (listening + spectrograms):
- Separation worked well for some sounds, less well for others
- ~38% of files (0-1 active channels) get no meaningful separation
- ~62% of files (2-4 active channels) show some degree of source isolation
- Dense frog choruses: spectrograms show partial separation but quality unclear
- Sparse recordings: MixIT correctly puts near-silence in unused channels

**Spectrograms saved:**
- `figures/dense_frog_chorus.jpg` — file 0039, 5 species nocturnal frog chorus
- `figures/bird_multi_species.jpg` — file 0057, multi-species with birds
- `figures/sparse_quiet.jpg` — file 0006, single species

**Decision**: Proceed with full unlabeled separation. Even with imperfect separation,
the max-pool-across-channels approach means we never do worse than the original.
Phase 3 validation will quantitatively measure whether separation improves pseudo-label
quality. If it doesn't, skip MixIT for pseudo-labeling and consider domain-matched
training (Asteroid/PyTorch) as a follow-up.

### 3.6 Storage format (pending)

> **Anthony's feedback**: Save as OGG (lossy, 192kbps VBR) instead of raw NPZ.
> Use ffmpeg for batch conversion. Keep 22050 Hz sample rate.

- [ ] 3.6.1 Update `separate_soundscapes.py` to save channels as OGG
- [ ] 3.6.2 Re-run labeled separation with OGG output
- [ ] 3.6.3 Verify downstream pipeline can load OGG channels for pseudo-labeling

**Note**: 200 GB (NPZ) is fine on 900+ GB available PACE storage. OGG conversion is
a nice-to-have optimization, not a blocker. Can do before or after the full run.

### 3.7 Full unlabeled separation
- [x] 3.7.1 Submitted: `bash sbatch/run.sh separate --num-workers 10` — job 5355342 (2026-03-22)
- [ ] 3.7.2 Verify output: ~10,592 files in `shared/mixit/pretrained/soundscapes/`

## 4. Documentation

- [x] 4.1 Write proposal.md
- [x] 4.2 Write README.md
- [x] 4.3 Write error-analysis.md (exp 012 model evaluation on labeled soundscapes)

## Key Context

### Outputs
- **ONNX model**: `/storage/project/ps-clef2026_birdclef-0/shared/models/bird-mixit/bird_mixit_4sources.onnx`
- **Separated labeled**: `/storage/project/ps-clef2026_birdclef-0/shared/mixit/pretrained/soundscapes_labeled/` (66 NPZ files)
- **Separated unlabeled**: `/storage/project/ps-clef2026_birdclef-0/shared/mixit/pretrained/soundscapes/` (~10,592 NPZ files)

### Downstream experiments
- **Exp 014 (pseudo-labeling)**: Consumes separated audio + teacher ONNX models → produces soft labels
- **Exp 015 (student training)**: Consumes pseudo-labels → trains student models

### Git
- **Branch**: `mg/012-soundscape-finetune`
- **PR**: #17
