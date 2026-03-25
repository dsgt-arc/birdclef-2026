# Experiment 017: Strong Baseline via Mel Overhaul — Tasks

## Goal

Close the 0.081 gap (0.827 → 0.908) by upgrading mel spectrogram resolution and
training recipe. Primary hypothesis: n_fft=4096 + 224 mels will improve LB by ≥0.02.

**Baseline**: Exp 016 v1-10s + pp2 = 0.827 LB
**Target**: ≥0.85 labeled-data-only baseline
**Concept**: [strong-baseline-mel-overhaul](../../docs/concepts/strong-baseline-mel-overhaul.md)

### Status: IN PROGRESS

- **Branch**: `mg/017-mel-overhaul` (PR #19)

### v1 Baseline (exp 016) for Comparison

| Metric | Value |
|--------|-------|
| test/auroc | 0.860 |
| test/cmap | 0.593 |
| LB (raw) | 0.795 |
| LB (+pp) | 0.823 |
| LB (+pp2 +10s) | **0.827** |

---

## Step 1: Mel Overhaul — v4a (n_fft=4096, n_mels=224)

### 1.1 Setup

- [x] 1.1.1 Make NNAudioMelSpectrogram params configurable (n_fft, n_mels, hop_length, fmax)
- [x] 1.1.2 Create experiment directory, pyproject.toml, config, SLURM scripts
- [x] 1.1.3 Create train.py (adapted from exp 016, reads mel config section)
- [x] 1.1.4 Create save_metrics.py (adds mel params to output JSON)

### 1.2 Pipeline Validation

- [ ] 1.2.1 Fast dev run: validate pipeline with 5 batches (no W&B)
- [ ] 1.2.2 Verify mel output shape: (B, 1, 224, 1000) for 10s at hop=320
- [ ] 1.2.3 Verify encoder handles (1, 224, 1000) input without errors

### 1.3 GPU Check

- [ ] 1.3.1 Submit `--gpu-check 1`: run 1 epoch to verify no OOM at batch_size=64
- [ ] 1.3.2 Check GPU utilization and memory usage
- [ ] 1.3.3 If OOM: reduce batch_size to 32 + accumulate=8 (same 256 effective)

### 1.4 Full Training

- [ ] 1.4.1 Submit full training run (40 epochs, early stopping patience=10)
- [ ] 1.4.2 Monitor via W&B: `dsgt-arc/birdclef-2026-mel-overhaul`
- [ ] 1.4.3 Record offline metrics (test/auroc, test/cmap)
- [ ] 1.4.4 Save structured metrics to artifacts/

### 1.5 ONNX Export & Kaggle Submission

- [ ] 1.5.1 Export to ONNX (update mel_config.json with new params)
- [ ] 1.5.2 Submit to Kaggle LB **without** post-processing (raw model score)
- [ ] 1.5.3 Submit to Kaggle LB **with** pp2 (TTA + smoothing + TopN + temp/taxa/priors)
- [ ] 1.5.4 Record LB results

### 1.6 Decision Gate

- If LB (raw) ≥ v1 raw (0.795) + 0.02 = **≥0.815**: mel overhaul helps → continue to Step 2
- If LB (+pp2) ≥ v1+pp2 (0.827) + 0.02 = **≥0.847**: strong signal → continue
- If LB gain < +0.01: mel resolution is NOT the gap → pivot (Perch ensemble, pseudo-labeling)
- If 0.01 ≤ gain < 0.02: evaluate whether to continue or pivot

---

## Step 2: Normalization + Head (single variables from best of Step 1) — FUTURE

Only proceed if Step 1 passes decision gate.

- [ ] 2.1 v4b config: AmplitudeToDB(top_db=80) + standardize + min-max [0,1] normalization
- [ ] 2.2 v4c config: AttBlockV2 head (tanh+softmax attention, sigmoid classification)
- [ ] 2.3 Run v4b and v4c in parallel on SLURM
- [ ] 2.4 Record results, compare to v4a

## Step 3: Regularization + Capacity (cumulative) — FUTURE

- [ ] 3.1 v4d: StochasticDepth (drop_path=0.15)
- [ ] 3.2 v4e: EfficientNet-B3 backbone
- [ ] 3.3 v4f: ASL loss (γ⁻=4, γ⁺=0, clip=0.05)

## Step 4: Polish — FUTURE

- [ ] 4.1 Model soup on best config
- [ ] 4.2 Full-signal mel at inference
- [ ] 4.3 Ensemble best variants

---

## Key Config: v4a vs v1

| Parameter | v1 (exp 016) | v4a (exp 017) | Change |
|-----------|-------------|---------------|--------|
| n_fft | 1024 (derived) | **4096** | 4× freq resolution |
| n_mels | 128 | **224** | 1.75× spectral detail |
| hop_length | 320 | 320 | unchanged |
| input shape | (1, 128, 1000) | **(1, 224, 1000)** | 1.75× data |
| batch_size | 128 | **64** | OOM safety |
| accumulate_grad_batches | 2 | **4** | same 256 effective |
| Everything else | v1 | v1 | identical |
