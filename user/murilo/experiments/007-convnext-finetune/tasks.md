# Experiment 007: ConvNeXt-Base LoRA Fine-Tuning — Tasks

## Setup

- [x] Create experiment directory structure
- [x] Write proposal.md
- [x] Create pyproject.toml (from 006 + timm, drop adapters)
- [x] Create train.py with ConvNeXtEncoder + PEFT LoRA + mel transform
- [x] Create YAML configs (primary + 1 ablation)
- [x] Create SLURM scripts (sbatch/_job.sh, sbatch/run.sh)
- [x] Create save_metrics.py (from 006)

## Validation

- [ ] Fast-dev-run: verify PEFT wraps correctly, check trainable param count (~1.6M)
- [ ] Fast-dev-run: forward pass works with mel transform pipeline
- [ ] GPU check (2 epochs): verify GPU utilization and memory fit on RTX 6000

## Training

- [ ] Submit primary run: `convnext_lora_mlp.yaml`
- [ ] Submit ablation: `convnext_lora_mlp_mlphead.yaml`

## Analysis

- [ ] Compare test cmAP across variants
- [ ] Compare with Exp 006 EAT baselines (cmAP=0.470–0.536)
- [ ] Document results in results.md
