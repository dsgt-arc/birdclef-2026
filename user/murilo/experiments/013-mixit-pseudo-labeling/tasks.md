# Experiment 013: MixIT Source Separation + Pseudo-Labeling — Tasks

## 1. Goal

Use Bird MixIT source separation + exp 012 teacher ensemble to pseudo-label 10,592
unlabeled soundscapes, then train on pseudo-labels to improve LB score.

## 2. Phase 1: MixIT Conversion (TF1 → ONNX)

- [ ] 2.1 Download Bird MixIT TF1 checkpoint from Google Cloud Storage
- [ ] 2.2 Write `scripts/convert_mixit_onnx.py` (tf2onnx conversion)
- [ ] 2.3 Validate ONNX output matches TF1 output on test audio (atol < 1e-4)
- [ ] 2.4 Benchmark ONNX inference speed (target: <1s per 1-min file on CPU)

## 3. Phase 2: Offline Separation

- [ ] 3.1 Write `scripts/separate_soundscapes.py` (batch separation)
- [ ] 3.2 Create SLURM job for batch processing (CPU, 64GB RAM)
- [ ] 3.3 Run on all 10,592 unlabeled soundscapes
- [ ] 3.4 Spot-check: visualize spectrograms of separated channels vs original

## 4. Phase 3: Teacher Pseudo-Labeling

- [ ] 4.1 Write `scripts/pseudo_label.py` using exp 012 ONNX models
- [ ] 4.2 Implement aggregation: max-pool channels, average models
- [ ] 4.3 Run on separated + original audio for all 10,592 soundscapes
- [ ] 4.4 Evaluate pseudo-label quality on 66 labeled soundscapes (precision/recall)
- [ ] 4.5 Sweep confidence thresholds (0.3, 0.5, 0.7) and pick best

## 5. Phase 4: Student Training

- [ ] 5.1 Create `PseudoLabelDataset` for soft-label training
- [ ] 5.2 Implement soft BCE loss (teacher probabilities as targets)
- [ ] 5.3 Create `train.py` for mixed data: focal + labeled soundscape + pseudo-labeled
- [ ] 5.4 Train all 3 encoders
- [ ] 5.5 Export + submit to Kaggle LB

## 6. Results

- [ ] 6.1 Write results.md with pseudo-label quality metrics + LB scores
- [ ] 6.2 Compare vs exp 012 baseline

## Key Context

- **Bird MixIT checkpoint**: `gs://gresearch/sound_separation/bird_mixit_model_checkpoints`
- **4-source model**: `output_sources4/model.ckpt-3223090` (used in ICASSP paper)
- **Input**: 22.05 kHz mono audio
- **Output**: 4 separated waveforms (sum = input)
- **Sample rate mismatch**: BirdCLEF data is 32 kHz, need to resample to 22.05 kHz
