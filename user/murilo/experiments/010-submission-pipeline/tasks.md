# Experiment 010: BirdCLEF 2026 Submission Pipeline — Tasks

## 1. Goal

Submit LoRA models to the BirdCLEF 2026 competition via ONNX export and a unified Kaggle CPU kernel. Supports models trained on both BirdCLEF 2025 data (exps 006/007/009) and BirdCLEF 2026 focal data (exp 011).

## 2. ONNX Export

- [x] 2.1 Create `scripts/export_eat.py` — merge LoRA, wrap encoder+head, export to ONNX
- [x] 2.2 Create `scripts/export_convnext.py` — merge LoRA, wrap encoder+head, export to ONNX
- [x] 2.3 Create `scripts/export_protoclr.py` — merge LoRA, wrap encoder+head, export to ONNX (with TorchScript fallback for einops)
- [x] 2.4 Create `scripts/build_species_mapping.py` — map model output indices to 234 BirdCLEF 2026 columns
- [x] 2.5 Export all 3 models (2025-trained) via `sbatch/run.sh export` — verified ONNX outputs match PyTorch (atol < 1e-4)
- [x] 2.6 Add missing deps: einops, torchinfo, scipy, huggingface-hub for ProtoCLR CvT
- [x] 2.7 Add 2026-trained export functions in `sbatch/export.sbatch` (exp 011 checkpoints)

## 3. Local Validation

- [x] 3.1 Create `scripts/local_validate.py` — timing test on train_soundscapes subset
- [x] 3.2 Validate all 3 models via `sbatch/run.sh validate`
- [x] 3.3 Confirm all models fit within 90-minute Kaggle CPU limit

## 4. Kaggle Kernel

- [x] 4.1 Create unified `kaggle_kernel/submission.py` — auto-detects model, dispatches mel preprocessing
- [x] 4.2 Create per-model `kernel-metadata-*.json` files (2025 and 2026 variants)
- [x] 4.3 Create `scripts/upload_kaggle.py` — upload weights + push kernels (supports `--model protoclr-2026` etc.)
- [x] 4.4 Implement 3 mel pipelines: kaldi_fbank (EAT), log-mel (ConvNeXt), protoclr_mel (ProtoCLR)

## 5. Kaggle Submission Debugging

- [x] 5.1 Fix Kaggle username: `mgustineli3` → `murilogustineli`
- [x] 5.2 Fix kernel metadata: `id` must match slugified `title`
- [x] 5.3 Bundle onnxruntime wheels as a separate Kaggle dataset (`birdclef-2025-deps`) — no internet on Kaggle
- [x] 5.4 Add pip install fallback for onnxruntime in kernel
- [x] 5.5 Switch to Anthony's two-dataset pattern for Kaggle submission (deps + weights as separate datasets)
- [x] 5.6 Add debug logging for Kaggle input paths during scoring
- [x] 5.7 Add `_find_input()` helper to support both old and new Kaggle directory layouts
- [x] 5.8 Fix ONNX output shape bug: `logits[0]` failed when output was already 1D — fixed with `logits.flatten()`
- [x] 5.9 Add train_soundscapes fallback (process 2 train files when test is empty) to surface inference bugs during public run

## 6. Kaggle Submissions — BirdCLEF 2025-Trained Models

- [x] 6.1 Upload ProtoCLR weights to Kaggle (`birdclef-2025-protoclr-weights`)
- [x] 6.2 Upload deps dataset to Kaggle (`birdclef-2025-deps`)
- [x] 6.3 Push ProtoCLR kernel — v10 had ONNX IndexError, v11 surfaced the bug via train fallback, v12 fixed with `flatten()`
- [x] 6.4 ProtoCLR v12 scored successfully: **0.531 macro-AUROC** on public LB (2026-03-15)
- [ ] 6.5 Upload and submit EAT model (2025-trained)
- [ ] 6.6 Upload and submit ConvNeXt model (2025-trained)

## 7. Kaggle Submissions — BirdCLEF 2026-Trained Models (exp 011)

- [ ] 7.1 Export ProtoCLR 2026 to ONNX: `bash sbatch/run.sh export --model protoclr-2026`
- [ ] 7.2 Upload ProtoCLR 2026 weights to Kaggle: `python scripts/upload_kaggle.py --model protoclr-2026`
- [ ] 7.3 Push ProtoCLR 2026 kernel: `python scripts/upload_kaggle.py --model protoclr-2026 --push-kernel`
- [ ] 7.4 Verify ProtoCLR 2026 submission scores on leaderboard
- [ ] 7.5 Export and submit EAT 2026
- [ ] 7.6 Export and submit ConvNeXt 2026

## 8. Documentation

- [x] 8.1 Write proposal.md
- [x] 8.2 Write README.md with quickstart guide
- [x] 8.3 Add .gitignore for generated kernel-metadata.json
- [x] 8.4 Write results.md with submission scores

### Key context for next session:
- **Submit ProtoCLR 2026**: export → upload → push kernel (see section 7 commands)
- **Upload command**: `source ~/clef/birdclef-2026/.venv/bin/activate && cd user/murilo/experiments/010-submission-pipeline && python scripts/upload_kaggle.py --model protoclr-2026 --push-kernel`
- **Check status**: `kaggle competitions submissions birdclef-2026`
- **Species mapping**: 2025 models map 41/234 species; 2026 models map 206/234 species (28 soundscape-only insect sonotypes have no training data)
- **Deps dataset is shared**: `birdclef-2025-deps` works for both 2025 and 2026 model kernels
