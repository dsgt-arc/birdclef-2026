# Experiment 022: PCEN Normalization — Tasks

## Goal

Replace AmplitudeToDB normalization with PCEN (Per-Channel Energy Normalization)
to close the domain gap between focal recordings and Pantanal soundscapes.

**Current Best SED**: 0.906 LB (v5j single), 0.917 LB (v5j-4fold-pp2)
**Target**: ≥0.910 LB single model
**Baseline**: v5j (0.906 LB) — HGNetV2-B0, AmplitudeToDB, LSEHead, BCE, OneCycleLR, 20 epochs

### Status: **REJECTED — PCEN regressed −0.026 (v10a 0.880 vs v5j 0.906). v5j-repro 0.903 confirmed pipeline correct.**

- **Proposal**: [`proposal.md`](proposal.md)
- **Pattern analysis**: [`notes/2026-04-06-experiment-pattern-analysis.md`](../../notes/2026-04-06-experiment-pattern-analysis.md)

---

## Config Comparison

All configs identical to v5j except mel normalization. Everything else controlled:
HGNetV2-B0, LSEHead, BCE, OneCycleLR, 20 epochs, 90/10 split, spectrogram MixUp.

| Config | Normalization | PCEN Trainable | PCEN Params | val/auroc | LB | Delta vs v5j |
|--------|---------------|----------------|-------------|-----------|-----|-------------|
| v5j (baseline) | AmplitudeToDB | — | — | 0.874 | **0.906** | — |
| v10a | PCEN → min-max [0,1] | yes (1024 params) | s=0.025 α=0.98 δ=2.0 r=0.5 | 0.871 | **0.880** | −0.026 |
| v10b | PCEN → min-max [0,1] | no (frozen) | s=0.025 α=0.98 δ=2.0 r=0.5 | 0.870 | — | not submitted |
| v10c | PCEN → AmpToDB → std → min-max | no (frozen) | s=0.025 α=0.98 δ=2.0 r=0.5 | 0.870 | — | not submitted |

---

## Phase A: Implementation — COMPLETE

- [x] A.1 Create `PCENLayer` nn.Module (`src/birdclef/preprocessing/pcen.py`)
  - IIR smoothing + PCEN formula, per-channel learnable/fixed params
- [x] A.2 Add `norm_type="pcen"` and `"pcen_amp2db"` to `NNAudioMelSpectrogram`
- [x] A.3 Modify `SEDFinetuneModule` to allow gradients through learnable mel_transform
  - `_mel_grad_context()` conditionally skips `no_grad` when PCEN is trainable
- [x] A.4 Create exp 022 training infrastructure
  - `train.py`, `pyproject.toml`, `sbatch/`, 3 YAML configs (v10a/v10b/v10c)
- [x] A.5 Update submission pipeline (4-file checklist)
  - `submission.py`: `compute_mel_pcen`, `_apply_pcen_numpy`, full-signal PCEN
  - `export_efficientnet_sed.py`: extract learned PCEN params → `mel_config.json`
  - `export.sbatch`: export functions + dispatch for v10a/v10b/v10c
  - `upload_kaggle.py`: MODELS dict entries
  - Kernel metadata JSONs for v10a/v10b/v10c

---

## Phase B: Training — COMPLETE

- [x] B.1 GPU check (1 epoch) for v10a to verify PCEN training works
  - Job 6317377 — **PASSED**: 1 epoch in 2:17, train/loss 0.027, val/loss 0.028
  - GPU util 69% avg (PCEN IIR loop is sequential — 511 autograd-tracked steps)
  - Memory 0.3 GB / 25 GB (1.2%) — model is small, not memory-bound
  - Fixed autograd in-place write bug (RuntimeError on backward pass)
  - Optimized: in-place + no_grad EMA for fixed PCEN (v10b/v10c), autograd-safe
    list+cat for trainable PCEN (v10a)
- [x] B.2 Train v10a (learnable PCEN, 20 epochs) — val/auroc 0.871, val/cmap 0.620, best epoch 18
- [x] B.3 Train v10b (fixed PCEN, 20 epochs) — val/auroc 0.870, val/cmap 0.610, best epoch 18
- [x] B.4 Train v10c (PCEN + AmplitudeToDB, 20 epochs) — val/auroc 0.870, val/cmap 0.602, best epoch 17
- [x] B.5 Compare val/auroc across v10a/v10b/v10c vs v5j baseline
  - All PCEN variants ~0.003–0.004 below v5j offline (0.870–0.871 vs 0.874)
  - v10a (learnable) best across all offline metrics
  - Offline metrics are anti-correlated with LB — must submit to evaluate

---

## Phase C: Export + LB Submission — v10a SUBMITTED

- [x] C.1 Export v10a to ONNX
- [x] C.2 Upload weights to Kaggle
- [x] C.3 Push kernel + submit to LB
- [x] C.4 **v10a LB: 0.880 (−0.026 vs v5j)** — PCEN regressed significantly
- [x] C.5 v10b/v10c not submitted (worse offline than v10a, would likely regress further)

---

## Phase D: Pipeline Verification — COMPLETE

- [x] D.1 Train v5j-repro (AmplitudeToDB baseline through exp 022 pipeline) — **val/auroc 0.873, best epoch 18**
- [x] D.2 Export v5j-repro → ONNX → Kaggle LB — **0.903 LB (−0.003 vs v5j 0.906)**
- [x] D.3 v5j-repro ≈ 0.906: **PCEN is the problem, experiment rejected**
- [x] ~~D.4 If v5j-repro << 0.906: pipeline bug, investigate~~ — N/A

---

## Results

| Model | val/auroc | val/cmap | LB | Normalization | Notes |
|-------|-----------|----------|-----|---------------|-------|
| v5j (baseline) | 0.874 | — | **0.906** | AmplitudeToDB | — |
| v10a | 0.871 | 0.620 | **0.880** | Learnable PCEN | −0.026 regression |
| v10b | 0.870 | 0.610 | — | Fixed PCEN | Not submitted |
| v10c | 0.870 | 0.602 | — | PCEN + AmplitudeToDB | Not submitted |
| v5j-repro | 0.873 | — | **0.903** | AmplitudeToDB | Pipeline verified (≈v5j 0.906, −0.003) |
