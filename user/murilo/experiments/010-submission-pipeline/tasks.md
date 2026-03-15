# Experiment 010: BirdCLEF 2026 Submission Pipeline — Tasks

## 1. Goal

Submit BirdCLEF 2025-trained LoRA models (EAT, ConvNeXt, ProtoCLR) to the BirdCLEF 2026 competition via ONNX export and a unified Kaggle CPU kernel. Validate the end-to-end pipeline, measure CPU timing, and get baseline scores (expected low: only 41/234 species overlap).

## 2. ONNX Export

- [x] 2.1 Create `scripts/export_eat.py` — merge LoRA, wrap encoder+head, export to ONNX
- [x] 2.2 Create `scripts/export_convnext.py` — merge LoRA, wrap encoder+head, export to ONNX
- [x] 2.3 Create `scripts/export_protoclr.py` — merge LoRA, wrap encoder+head, export to ONNX (with TorchScript fallback for einops)
- [x] 2.4 Create `scripts/build_species_mapping.py` — map 206 BirdCLEF 2025 indices to 234 BirdCLEF 2026 columns (41 overlap)
- [x] 2.5 Export all 3 models via `sbatch/run.sh export` — verified ONNX outputs match PyTorch (atol < 1e-4)
- [x] 2.6 Add missing deps: einops, torchinfo, scipy, huggingface-hub for ProtoCLR CvT

## 3. Local Validation

- [x] 3.1 Create `scripts/local_validate.py` — timing test on train_soundscapes subset
- [x] 3.2 Validate all 3 models via `sbatch/run.sh validate`
- [x] 3.3 Confirm all models fit within 90-minute Kaggle CPU limit

## 4. Kaggle Kernel

- [x] 4.1 Create unified `kaggle_kernel/submission.py` — auto-detects model, dispatches mel preprocessing
- [x] 4.2 Create per-model `kernel-metadata-*.json` files (eat, convnext, protoclr)
- [x] 4.3 Create `scripts/upload_kaggle.py` — upload weights + push kernels
- [x] 4.4 Implement 3 mel pipelines: kaldi_fbank (EAT), log-mel (ConvNeXt), protoclr_mel (ProtoCLR)

## 5. Kaggle Submission Debugging

- [x] 5.1 Fix Kaggle username: `mgustineli3` → `murilogustineli`
- [x] 5.2 Fix kernel metadata: `id` must match slugified `title`
- [x] 5.3 Bundle onnxruntime wheels as a separate Kaggle dataset (`birdclef-2025-deps`) — no internet on Kaggle
- [x] 5.4 Add pip install fallback for onnxruntime in kernel
- [x] 5.5 Switch to Anthony's two-dataset pattern for Kaggle submission (deps + weights as separate datasets)
- [x] 5.6 Add debug logging for Kaggle input paths during scoring
- [x] 5.7 Add `_find_input()` helper to support both old and new Kaggle directory layouts

## 6. Kaggle Submission

- [x] 6.1 Upload ProtoCLR weights to Kaggle (`birdclef-2025-protoclr-weights`)
- [x] 6.2 Upload deps dataset to Kaggle (`birdclef-2025-deps`)
- [x] 6.3 Push ProtoCLR kernel (version 10 pending)
- [ ] 6.4 **WAITING**: 2hr cooldown to submit notebook version 10 to Kaggle
- [ ] 6.5 Verify ProtoCLR submission scores on leaderboard
- [ ] 6.6 Upload and submit EAT model
- [ ] 6.7 Upload and submit ConvNeXt model
- [ ] 6.8 Record baseline scores for all 3 models

## 7. Documentation

- [x] 7.1 Write proposal.md
- [x] 7.2 Write README.md with quickstart guide
- [x] 7.3 Add .gitignore for generated kernel-metadata.json
- [ ] 7.4 Write results.md with baseline scores, CPU timing, and pipeline lessons learned
