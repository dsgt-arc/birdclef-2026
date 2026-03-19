# Experiment 010: BirdCLEF 2026 Submission Pipeline — Tasks

## 1. Goal

Submit LoRA models to the BirdCLEF 2026 competition via ONNX export and a unified Kaggle CPU kernel. Supports models trained on both BirdCLEF 2025 data (exps 006/007/009) and BirdCLEF 2026 focal data (exp 011).

## 2. ONNX Export

- [x] 2.1 Create `scripts/export_eat.py` — merge LoRA, wrap encoder+head, export to ONNX
- [x] 2.2 Create `scripts/export_convnext.py` — merge LoRA, wrap encoder+head, export to ONNX
- [x] 2.3 Create `scripts/export_protoclr.py` — merge LoRA, wrap encoder+head, export to ONNX (with TorchScript fallback for einops)
- [x] 2.4 Create `scripts/build_species_mapping.py` — map model output indices to 234 BirdCLEF 2026 columns
- [x] 2.5 Export 2025-trained models: `bash sbatch/run.sh export --model all-2025`
- [x] 2.6 Export 2026-trained models: `bash sbatch/run.sh export --model all-2026`
- [x] 2.7 Add missing deps: einops, torchinfo, scipy, huggingface-hub for ProtoCLR CvT

## 3. Local Validation

- [x] 3.1 Create `scripts/local_validate.py` — timing test on train_soundscapes subset
- [x] 3.2 Validate 2025-trained models: `bash sbatch/run.sh validate --model all-2025`
- [x] 3.3 Validate 2026-trained models: `bash sbatch/run.sh validate --model all-2026`

## 4. Kaggle Kernel

- [x] 4.1 Create unified `kaggle_kernel/submission.py` — auto-detects model, dispatches mel preprocessing
- [x] 4.2 Create per-model `kernel-metadata-{model}-{year}.json` files
- [x] 4.3 Create `scripts/upload_kaggle.py` — upload weights + push kernels
- [x] 4.4 Implement 3 mel pipelines: kaldi_fbank (EAT), log-mel (ConvNeXt), protoclr_mel (ProtoCLR)

## 5. Kaggle Submission Debugging

- [x] 5.1 Fix Kaggle username: `mgustineli3` → `murilogustineli`
- [x] 5.2 Fix kernel metadata: `id` must match slugified `title`
- [x] 5.3 Bundle onnxruntime wheels as a separate Kaggle dataset (`birdclef-deps`) — no internet on Kaggle
- [x] 5.4 Add pip install fallback for onnxruntime in kernel
- [x] 5.5 Switch to Anthony's two-dataset pattern for Kaggle submission (deps + weights as separate datasets)
- [x] 5.6 Add debug logging for Kaggle input paths during scoring
- [x] 5.7 Add `_find_input()` helper to support both old and new Kaggle directory layouts
- [x] 5.8 Fix ONNX output shape bug: `logits[0]` failed when output was already 1D — fixed with `logits.flatten()`
- [x] 5.9 Add train_soundscapes fallback (process 2 train files when test is empty) to surface inference bugs during public run

## 6. Kaggle Submissions — BirdCLEF 2025-Trained Models

- [x] 6.1 Upload ProtoCLR weights to Kaggle (`birdclef-2025-protoclr-weights`)
- [x] 6.2 Upload deps dataset to Kaggle (`birdclef-deps`)
- [x] 6.3 Push ProtoCLR kernel — v10 had ONNX IndexError, v11 surfaced the bug via train fallback, v12 fixed with `flatten()`
- [x] 6.4 ProtoCLR-2025 scored **0.531 macro-AUROC** on public LB (2026-03-15)
- [ ] 6.5 Upload and submit EAT-2025: `python scripts/upload_kaggle.py --model eat-2025 --push-kernel`
- [ ] 6.6 Upload and submit ConvNeXt-2025: `python scripts/upload_kaggle.py --model convnext-2025 --push-kernel`

## 7. Kaggle Submissions — BirdCLEF 2026-Trained Models (exp 011)

- [x] 7.1 Export all 2026 models to ONNX: `bash sbatch/run.sh export --model all-2026`
- [x] 7.2 Upload + push ProtoCLR-2026: `python scripts/upload_kaggle.py --model protoclr-2026 --push-kernel`
- [x] 7.3 ProtoCLR-2026 scored **0.721 macro-AUROC** on public LB (2026-03-15)
- [x] 7.4 Upload + push ConvNeXt-2026: `python scripts/upload_kaggle.py --model convnext-2026 --push-kernel`
- [x] 7.5 ConvNeXt-2026 scored **0.720 macro-AUROC** on public LB (2026-03-15)
- [x] 7.6 Upload + push EAT-2026: `python scripts/upload_kaggle.py --model eat-2026 --push-kernel`
- [x] 7.7 EAT-2026 scored **0.701 macro-AUROC** on public LB (2026-03-15)

## 8. Next Steps — Improving Scores

### 8.1 Retrain on 100% focal data
Current models use 80/10/10 train/val/test split — 20% of data is unused. Training on all
35,549 recordings should improve scores, especially for tail species with few samples.
Implementation done in exp 011 (tasks 8.1–8.7).

- [x] 8.1.1 Add `--train-all` flag to exp 011 (done in exp 011 task 8.1)
- [x] 8.1.2 ProtoCLR fulldata: **0.736 LB** (+0.015 vs 80/10/10)
- [x] 8.1.3 EAT fulldata: **0.698 LB** (−0.003, overfitting — see exp 011 results)
- [ ] 8.1.4 ConvNeXt fulldata: resubmitted with 48hr walltime (2026-03-19), training in progress
- [ ] 8.1.5 Export + submit ConvNeXt fulldata once retraining completes
- [ ] 8.1.6 Add ConvNeXt fulldata LB score to results.md

> **Removed: Ensemble models (formerly 8.2)** — infeasible under 90-min Kaggle inference limit.
> ProtoCLR + ConvNeXt alone estimate ~120 min for ~600 soundscape files. Would require
> significant inference optimization (e.g., batched ONNX, shared mel computation) to fit.

> **Removed: Incorporate labeled soundscapes (formerly 8.3)** — deferred to a future experiment
> (e.g., exp 012). Requires multi-label training support, a different data pipeline for
> soundscape segments, and careful handling of the 28 soundscape-only insect sonotypes.

## 9. Documentation

- [x] 9.1 Write proposal.md
- [x] 9.2 Write README.md with quickstart guide
- [x] 9.3 Add .gitignore for generated kernel-metadata.json
- [x] 9.4 Write results.md with submission scores

### Key context for next session:
- **Model names always include training year**: `protoclr-2025`, `protoclr-2026`, `all-2025`, `all-2026`
- **Current LB scores**: ProtoCLR-2026 = 0.721, ConvNeXt-2026 = 0.720, EAT-2026 = 0.701 (all trained on 80% of focal data)
- **Next priority**: retrain on 100% of focal data (no val/test split) for higher scores
- **Check status**: `kaggle competitions submissions birdclef-2026`
- **Species mapping**: 2025 models map 41/234 species; 2026 models map 206/234 species (28 soundscape-only insect sonotypes have no training data)
- **Public LB uses ~34% of test data**; final standings use the remaining ~66%
