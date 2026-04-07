# Experiment 022: PCEN Normalization — Tasks

## Goal

Replace AmplitudeToDB normalization with PCEN (Per-Channel Energy Normalization)
to close the domain gap between focal recordings and Pantanal soundscapes.

**Current Best SED**: 0.906 LB (v5j single), 0.917 LB (v5j-4fold-pp2)
**Target**: ≥0.910 LB single model
**Baseline**: v5j (0.906 LB) — HGNetV2-B0, AmplitudeToDB, LSEHead, BCE, OneCycleLR, 20 epochs

### Status: **IN PROGRESS — Implementation complete, ready to train.**

- **Proposal**: [`proposal.md`](proposal.md)
- **Pattern analysis**: [`notes/2026-04-06-experiment-pattern-analysis.md`](../../notes/2026-04-06-experiment-pattern-analysis.md)

---

## Config Comparison

All configs identical to v5j except mel normalization. Everything else controlled:
HGNetV2-B0, LSEHead, BCE, OneCycleLR, 20 epochs, 90/10 split, spectrogram MixUp.

| Config | Normalization | PCEN Trainable | PCEN Params | val/auroc | LB | Delta vs v5j |
|--------|---------------|----------------|-------------|-----------|-----|-------------|
| v5j (baseline) | AmplitudeToDB | — | — | 0.874 | **0.906** | — |
| v10a | PCEN → min-max [0,1] | yes (1024 params) | s=0.025 α=0.98 δ=2.0 r=0.5 | — | — | — |
| v10b | PCEN → min-max [0,1] | no (frozen) | s=0.025 α=0.98 δ=2.0 r=0.5 | — | — | — |
| v10c | PCEN → AmpToDB → std → min-max | no (frozen) | s=0.025 α=0.98 δ=2.0 r=0.5 | — | — | — |

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

## Phase B: Training — PENDING

- [ ] B.1 GPU check (1 epoch) for v10a to verify PCEN training works
  ```bash
  cd user/murilo/experiments/022-pcen-normalization
  bash sbatch/run.sh config/hgnet_b0_sed_v10a.yaml --gpu-check
  ```
- [ ] B.2 Train v10a (learnable PCEN, 20 epochs)
  ```bash
  bash sbatch/run.sh config/hgnet_b0_sed_v10a.yaml
  ```
- [ ] B.3 Train v10b (fixed PCEN, 20 epochs)
  ```bash
  bash sbatch/run.sh config/hgnet_b0_sed_v10b.yaml
  ```
- [ ] B.4 Train v10c (PCEN + AmplitudeToDB, 20 epochs)
  ```bash
  bash sbatch/run.sh config/hgnet_b0_sed_v10c.yaml
  ```
- [ ] B.5 Compare val/auroc across v10a/v10b/v10c vs v5j baseline

---

## Phase C: Export + LB Submission — PENDING

- [ ] C.1 Export best PCEN variant to ONNX
  ```bash
  cd user/murilo/experiments/010-submission-pipeline
  bash sbatch/run.sh export --model hgnet-b0-sed-v10a
  ```
- [ ] C.2 Upload weights to Kaggle
  ```bash
  python scripts/upload_kaggle.py --model hgnet-b0-sed-v10a
  ```
- [ ] C.3 Push kernel + submit to LB
  ```bash
  python scripts/upload_kaggle.py --model hgnet-b0-sed-v10a --push-kernel
  ```
- [ ] C.4 Record LB scores in results table above
- [ ] C.5 If ≥0.910: submit remaining variants; if <0.906: reject hypothesis

---

## Phase D: Follow-Up (if successful) — PENDING

- [ ] D.1 4-fold ensemble of best PCEN variant
- [ ] D.2 PCEN + site/hour priors
- [ ] D.3 Update MODEL-RUNS.md with results

---

## Results

| Model | val/auroc | val/cmap | LB | Normalization | Notes |
|-------|-----------|----------|-----|---------------|-------|
| v5j (baseline) | 0.874 | — | 0.906 | AmplitudeToDB | — |
| v10a | — | — | — | Learnable PCEN | — |
| v10b | — | — | — | Fixed PCEN | — |
| v10c | — | — | — | PCEN + AmplitudeToDB | — |
