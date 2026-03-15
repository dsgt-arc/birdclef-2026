# Experiment 011: LoRA Baseline on BirdCLEF 2026 — Tasks

## 1. Goal

Retrain the proven LoRA fine-tuning pipeline (EAT, ConvNeXt, ProtoCLR) on BirdCLEF 2026 focal training data (35,549 recordings, ~206 focal species) to get competitive baselines. Same hyperparameters as 2025 experiments (006/007/009) for fair comparison.

## 2. Setup

- [x] 2.1 Create experiment directory structure (`config/`, `sbatch/`)
- [x] 2.2 Create unified `train.py` — dispatches on `config["encoder"]["type"]` for all 3 encoders
  - `build_encoder()`: EATEncoder / ConvNeXtEncoder / ProtoCLREncoder
  - `build_audio_transform()`: waveform_to_fbank / NNAudioMelSpectrogram / ProtoCLRMelSpectrogram
  - `apply_lora_to_encoder()`: wraps `.model` (EAT/ProtoCLR) or `.backbone` (ConvNeXt)
- [x] 2.3 Create `pyproject.toml` — unified deps (peft, einops, torchinfo in main deps)
- [x] 2.4 Create config files pointing to BirdCLEF 2026 data:
  - `config/eat_lora_attn_mlphead.yaml`
  - `config/convnext_lora_mlp_mlphead.yaml`
  - `config/protoclr_lora_attn.yaml`
- [x] 2.5 Create `sbatch/run.sh` launcher + `sbatch/_job.sh` SLURM job script
- [x] 2.6 Write proposal.md and README.md

## 3. Refactoring

- [x] 3.1 Rename `BirdCLEF2025DataModule` → `BirdCLEFDataModule` across all `user/murilo/` files (14 files updated)
- [x] 3.2 Rename `BirdCLEF2025Dataset` → `BirdCLEFDataset` across all `user/murilo/` files
- [x] 3.3 Update docstring in `src/birdclef/data/birdclef2025.py` to reflect 2025+2026 compatibility

## 4. Validation

- [x] 4.1 Fast dev run: `sbatch/run.sh --all --fast-dev-run` — pipeline works end-to-end with 2026 data
- [x] 4.2 Confirm `build_species_list()` returns ~206 focal species (not 234)

## 5. GPU Check

- [x] 5.1 `sbatch/run.sh --all --gpu-check` — all 3 configs completed successfully
- [x] 5.2 Review GPU utilization reports in `outputs/gpu_check/` (text + plots)
  - EAT (bs=32): 2.7% VRAM, 88.9% GPU util, 33.6% data loading
  - ConvNeXt (bs=128): 2.8% VRAM, 99.9% GPU util, 4.3% data loading
  - ProtoCLR (bs=256): 1.7% VRAM, 66.0% GPU util, 4.4% data loading
- [x] 5.3 Confirm all 3 models fit in RTX 6000 (24GB) VRAM — all under 4% usage
- [x] 5.4 No batch size adjustments needed — utilization is healthy across the board

## 6. Full Training

- [ ] 6.1 Submit all 3 configs: `sbatch/run.sh --all`
- [ ] 6.2 Monitor on W&B (`dsgt-arc/birdclef-2026-lora-finetune`)
- [ ] 6.3 Verify convergence within 50 epochs for all 3 models
- [ ] 6.4 Record cmAP/AUROC for each model

## 7. Results

- [ ] 7.1 Compare 2026 results with 2025 baselines (AUROC ~0.94 for all 3)
- [ ] 7.2 Write results.md with cmAP/AUROC per model, training curves, and comparison to 2025
- [ ] 7.3 Update `EXPERIMENTS.md` with exp 011 findings
