# Experiment 009: ProtoCLR Fine-Tuning -- Tasks

## Setup

- [x] Create experiment directory structure
- [x] Write proposal.md
- [x] Create pyproject.toml
- [x] Create ProtoCLR encoder wrapper (src/birdclef/encoders/protoclr.py)
- [x] Create ProtoCLR mel preprocessing (src/birdclef/preprocessing/protoclr_mel.py)
- [x] Create train.py with PEFT integration
- [x] Create YAML configs (attention-only + all-layers)
- [x] Create SLURM scripts (sbatch/_job.sh, sbatch/run.sh)
- [x] Create save_metrics.py

## Validation

- [ ] Fast-dev-run: verify ProtoCLR loads, mel transform works, forward pass completes
- [ ] Fast-dev-run: verify LoRA wraps correctly, check trainable param count
- [ ] GPU check (2 epochs): measure GPU utilization and throughput

## Training

- [ ] Submit primary run: `protoclr_lora_attn.yaml`
- [ ] Submit ablation: `protoclr_lora_all.yaml`

## Analysis

- [ ] Compare test cmAP across both configs
- [ ] Compare with Exp 006 EAT (cmAP=0.536) and Exp 007 ConvNeXt (cmAP=0.527)
- [ ] Document results in results.md
