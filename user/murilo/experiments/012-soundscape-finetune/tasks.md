# Experiment 012: Labeled Soundscape Fine-Tuning — Tasks

## 1. Goal

Fine-tune exp 011 models on labeled soundscapes to expand from 206 → 234 species
and adapt to the soundscape domain. Mixed focal + soundscape training with head expansion.

## 2. Data Pipeline

- [ ] 2.1 Implement `SoundscapeDataset` in `src/birdclef/data/soundscape.py`
- [ ] 2.2 Implement `FocalDataset234` wrapper for 206→234 label remapping
- [ ] 2.3 Implement `build_full_species_list()` from taxonomy.csv (234 species)
- [ ] 2.4 Implement `MixedDataModule` in `src/birdclef/data/mixed.py`
- [ ] 2.5 Update `src/birdclef/data/__init__.py` with new exports

## 3. Training Script

- [ ] 3.1 Create `train.py` (adapted from exp 011) with:
  - `--checkpoint` arg for exp 011 checkpoint
  - `remap_head_weights()` for 206→234 head expansion
  - `MixedDataModule` integration
- [ ] 3.2 Create 3 YAML configs (EAT, ConvNeXt, ProtoCLR)
- [ ] 3.3 Create SLURM scripts (`sbatch/run.sh`, `sbatch/_job.sh`)

## 4. Validation

- [ ] 4.1 Fast-dev-run: `bash sbatch/run.sh --all --fast-dev-run`
- [ ] 4.2 Verify head expansion: 206 species have non-random weights, 28 new have random
- [ ] 4.3 Verify all datasets produce 234-dim labels
- [ ] 4.4 Verify soundscape dedup: 739 unique segments from 66 files

## 5. Full Training

- [ ] 5.1 Submit all 3 configs: `bash sbatch/run.sh --all`
- [ ] 5.2 Monitor on W&B
- [ ] 5.3 Check focal test AUROC vs exp 011 baseline (forgetting check)
- [ ] 5.4 Check soundscape test AUROC
- [ ] 5.5 Check 28 soundscape-only species have non-zero predictions

## 6. Kaggle Submission

- [ ] 6.1 Export 234-class models to ONNX (via exp 010 pipeline)
- [ ] 6.2 Upload + push kernels to Kaggle
- [ ] 6.3 Record LB scores in results.md

## 7. Documentation

- [x] 7.1 Write proposal.md
- [ ] 7.2 Write README.md
- [ ] 7.3 Write results.md (after training completes)
