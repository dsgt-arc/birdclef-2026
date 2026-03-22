# Experiment 012: Labeled Soundscape Fine-Tuning — Tasks

## 1. Goal

Fine-tune exp 011 models on labeled soundscapes to expand from 206 → 234 species
and adapt to the soundscape domain. Mixed focal + soundscape training with head expansion.

## 2. Data Pipeline

- [x] 2.1 Implement `SoundscapeDataset` in `src/birdclef/data/soundscape.py`
- [x] 2.2 Implement `FocalDataset234` wrapper for 206→234 label remapping
- [x] 2.3 Implement `build_full_species_list()` from taxonomy.csv (234 species)
- [x] 2.4 Implement `MixedDataModule` in `src/birdclef/data/mixed.py`
- [x] 2.5 Update `src/birdclef/data/__init__.py` with new exports

## 3. Training Script

- [x] 3.1 Create `train.py` (adapted from exp 011) with:
  - `--checkpoint` arg for exp 011 checkpoint
  - `remap_head_weights()` for 206→234 head expansion
  - `MixedDataModule` integration
- [x] 3.2 Create 3 YAML configs (EAT, ConvNeXt, ProtoCLR)
- [x] 3.3 Create SLURM scripts (`sbatch/run.sh`, `sbatch/_job.sh`)
- [x] 3.4 Create `save_metrics.py` for structured artifact saving

## 4. Validation

- [x] 4.1 Fast-dev-run: `bash sbatch/run.sh --all --fast-dev-run` — all 3 passed
- [x] 4.2 Verify head expansion: 206/206 species remapped; copied mean norm ~1.2–2.5, new mean norm ~0.58
- [x] 4.3 Verify all datasets produce 234-dim labels (234 classes in model summary)
- [x] 4.4 Verify soundscape dedup: 1478 rows → 739 unique segments from 66 files

Fast-dev-run results (2026-03-21):
- **EAT**: 450K LoRA + 513K head trainable params, 90.9M total. train/loss 0.104, val/auroc 0.078
- **ConvNeXt**: similar profile, completed successfully
- **ProtoCLR**: similar profile, completed successfully
- Data split consistent across all 3: 213,295 train (212,725 focal + 570 soundscape), 27,116 val, 26,299 test

## 5. Full Training

- [x] 5.1 Submit all 3 configs: `bash sbatch/run.sh --all` (2026-03-21)
- [x] 5.2 ProtoCLR completed: best epoch 11, early stopped at 13, test AUROC 0.845, cmAP 0.481
- [ ] 5.3 EAT: training in progress
- [ ] 5.4 ConvNeXt: training in progress
- [ ] 5.5 Check focal test AUROC vs exp 011 baseline (forgetting check: flag if >0.02 drop)
- [ ] 5.6 Check 28 soundscape-only species have non-zero predictions

## 6. Kaggle Submission

### 6.1 Exp 010 pipeline updates (done)
- [x] Add soundscape model slugs to `submission.py` MODEL_SLUGS
- [x] Add export functions to `export.sbatch` for `{encoder}-2026-soundscape`
- [x] Add MODELS entries to `upload_kaggle.py` with `all-2026-soundscape` filter
- [x] Create kernel-metadata JSON for all 3 soundscape models

### 6.2 ProtoCLR soundscape submission
- [x] Export ProtoCLR to ONNX: `bash sbatch/run.sh export --model protoclr-2026-soundscape`
- [x] Upload weights to Kaggle: `python scripts/upload_kaggle.py --model protoclr-2026-soundscape`
- [x] Push kernel: `python scripts/upload_kaggle.py --model protoclr-2026-soundscape --push-kernel`
- [ ] Record ProtoCLR LB score (inference running on Kaggle)

### 6.3 EAT soundscape submission (pending training)
- [ ] Export EAT to ONNX: `bash sbatch/run.sh export --model eat-2026-soundscape`
- [ ] Upload + push: `python scripts/upload_kaggle.py --model eat-2026-soundscape --push-kernel`
- [ ] Record EAT LB score

### 6.4 ConvNeXt soundscape submission (pending training)
- [ ] Export ConvNeXt to ONNX: `bash sbatch/run.sh export --model convnext-2026-soundscape`
- [ ] Upload + push: `python scripts/upload_kaggle.py --model convnext-2026-soundscape --push-kernel`
- [ ] Record ConvNeXt LB score

## 7. Documentation

- [x] 7.1 Write proposal.md (includes exp 012→013 sequencing rationale)
- [x] 7.2 Write README.md
- [ ] 7.3 Write results.md (after all LB scores collected)

### Key context for next session:
- **ProtoCLR**: trained, exported, submitted to Kaggle — awaiting LB score
- **EAT / ConvNeXt**: training in progress on PACE
- **Checkpoint paths**: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/012-soundscape-finetune/checkpoints/`
- **W&B project**: `dsgt-arc/birdclef-2026-soundscape-finetune`
- **Model naming**: `{encoder}-2026-soundscape` (exp 012), `{encoder}-2026-pseudo` (future exp 013)
- **Baseline LB scores** (exp 011 fulldata): ProtoCLR 0.736, EAT 0.698, ConvNeXt 0.695
- **Note**: Exp 012 uses 80/10/10 split (not fulldata). Could do fulldata run later if results are promising.
