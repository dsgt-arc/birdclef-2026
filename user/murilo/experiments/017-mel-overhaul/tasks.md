# Experiment 017: Strong Baseline via Mel Overhaul — Tasks

## Goal

Close the 0.081 gap (0.827 → 0.908) by upgrading mel spectrogram resolution and
training recipe. Primary hypothesis: n_fft=4096 + 224 mels will improve LB by ≥0.02.

**Baseline**: Exp 016 v1-10s + pp2 = 0.827 LB
**Target**: ≥0.85 labeled-data-only baseline
**Concept**: [strong-baseline-mel-overhaul](../../docs/concepts/strong-baseline-mel-overhaul.md)
**Key references**:
- [Tom's SED model comparison (28 experiments)](../../docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md) — ASL +0.023, Perch KD +0.030, AmplitudeToDB+minmax norm, tanh+softmax head, B0 > B3/V2S
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md) — technique rankings and code snippets
- [2025 1st place inference](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md) — n_fft=4096, 224 mels, 7-model ensemble
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md) — phased plan to >0.900

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

- [x] 1.2.1 Fast dev run: validate pipeline with 5 batches (no W&B)
- [x] 1.2.2 Verify mel output shape: (B, 1, 224, 1000) for 10s at hop=320
- [x] 1.2.3 Verify encoder handles (1, 224, 1000) input without errors

### 1.3 GPU Check

- [x] 1.3.1 Submit `--gpu-check 1`: run 1 epoch to verify no OOM at batch_size=64
- [x] 1.3.2 Check GPU utilization and memory usage
- [x] 1.3.3 If OOM: reduce batch_size to 32 + accumulate=8 (same 256 effective) — N/A, no OOM

### 1.4 Full Training

- [x] 1.4.1 Submit full training run (40 epochs, early stopping patience=10)
- [x] 1.4.2 Monitor via W&B: `dsgt-arc/birdclef-2026-mel-overhaul/runs/05pylg1y`
- [x] 1.4.3 Record offline metrics: test/auroc=0.862, test/cmap=0.615
- [x] 1.4.4 Save structured metrics to artifacts/training/efficientnet_b0_sed_v4a/metrics.json

### 1.5 ONNX Export & Kaggle Submission

- [x] 1.5.1 Export to ONNX (mel_config.json: n_fft=4096, n_mels=224, target_frames=1000)
- [x] 1.5.2 Submit to Kaggle LB **without** post-processing → **0.825 LB** (+0.030 vs v1 raw)
- [x] 1.5.3 Submit to Kaggle LB **with** pp2 → **0.830 LB** (+0.003 vs v1+pp2)
- [x] 1.5.4 Record LB results: raw 0.825, pp2 0.830

**Note**: PP2 gained only +0.005 on v4a (vs +0.032 on v1). Temperature and TTA
hyperparameters were tuned for v1 — may need recalibration for v4a.

### 1.6 Decision Gate

- If LB (raw) ≥ v1 raw (0.795) + 0.02 = **≥0.815**: mel overhaul helps → continue to Step 2
- If LB (+pp2) ≥ v1+pp2 (0.827) + 0.02 = **≥0.847**: strong signal → continue
- If LB gain < +0.01: mel resolution is NOT the gap → pivot (Perch ensemble, pseudo-labeling)
- If 0.01 ≤ gain < 0.02: evaluate whether to continue or pivot

---

## Step 2: Parallel Ablations (v4b/v4c/v4d/v4e, all from v4a base) — IN PROGRESS

Step 1 passed decision gate → run 4 single-variable ablations in parallel from v4a.
**All runs use 90/10 train/val** (test_fraction=0.0), **50 epochs**, patience=10.

Each variant changes ONE thing from v4a to measure independent impact:

| Variant | Change | Rationale |
|---------|--------|-----------|
| v4b | AmplitudeToDB + standardize + min-max [0,1] | Tom uses across all 28 experiments, 2025 1st place |
| v4c | AttBlockV2 head (tanh+softmax att, sigmoid cls) | PANNs competition-standard SED head |
| v4d | ASL loss (γ⁻=4, γ⁺=0, clip=0.05) | Tom's biggest single gain: +0.023 AUC |
| v4e | StochasticDepth (drop_path=0.15) | 2025 1st place regularization |

### Implementation

- [x] 2.1 Add `norm_type="amplitude_to_db"` to NNAudioMelSpectrogram (AmplitudeToDB + standardize + min-max)
- [x] 2.2 Add `AttBlockV2` class to sed_head.py (tanh+softmax attention, sigmoid classification)
- [x] 2.3 Add `AsymmetricLoss` class + `loss_type="asl"` to SEDFinetuneModule
- [x] 2.4 Add `drop_path_rate` parameter to EfficientNetEncoder (timm native)
- [x] 2.5 Create config YAMLs: v4b, v4c, v4d, v4e
- [x] 2.6 Wire all new params through train.py
- [x] 2.7 Verify all imports and basic functionality

### Training

- [x] 2.8 Submit all 4 jobs to SLURM (v4b/v4c on gpu-a100, v4d/v4e on gpu-rtx6000)
- [ ] 2.9 Check training results for all 4 runs (W&B + offline metrics)
- [ ] 2.10 Record offline metrics for each variant

### ONNX Export & Kaggle

- [ ] 2.11 ONNX export for each variant that trained successfully
- [ ] 2.12 Submit to Kaggle LB (raw + pp2 for best variants)
- [ ] 2.13 Record LB results, compare to v4a (0.825/0.830)

### 2.14 Decision: Combine Winners

- Pick the variants that show LB improvement over v4a
- Create a v4f config combining all winning changes
- Run one final combined training run

## Step 3: Fulldata Training — FUTURE

Train the best combined recipe on 100% labeled data (no val split).
This adds ~10% more focal recordings + more soundscape files.
Lock in the recipe first, then scale.

- [ ] 3.1 Add `train_all` support to SEDDataModule (focal 100% + all soundscape files)
- [ ] 3.2 Run fulldata with epoch cap from best 90/10 run (~1.5× best epoch)
- [ ] 3.3 No early stopping — save checkpoints every 5 epochs + last
- [ ] 3.4 Export best + last checkpoints, submit both to LB
- [ ] 3.5 Record LB results vs 90/10 baseline

**Exp 011 reference**: ProtoCLR fulldata gained +0.015 LB, but ConvNeXt regressed
−0.025 from overfitting. Epoch cap is critical.

## Step 4: Polish — FUTURE

- [ ] 4.1 Model soup on best config (average top checkpoints)
- [ ] 4.2 Full-signal mel at inference (compute once for 60s, slice)
- [ ] 4.3 Ensemble best variants

---

## Key Config: v4a-v4e Comparison

| Parameter | v4a (baseline) | v4b | v4c | v4d | v4e |
|-----------|---------------|-----|-----|-----|-----|
| n_fft | 4096 | 4096 | 4096 | 4096 | 4096 |
| n_mels | 224 | 224 | 224 | 224 | 224 |
| norm_type | audioset | **amplitude_to_db** | audioset | audioset | audioset |
| sed_head | AttHead | AttHead | **AttBlockV2** | AttHead | AttHead |
| loss_type | soft_ce | soft_ce | soft_ce | **asl** | soft_ce |
| drop_path_rate | 0.0 | 0.0 | 0.0 | 0.0 | **0.15** |
| test_fraction | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| max_epochs | 40 | 50 | 50 | 50 | 50 |
