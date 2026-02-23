# Experiment 006: LoRA Fine-Tuning — Tasks

## Setup

- [x] Create experiment directory structure
- [x] Write proposal.md
- [x] Create pyproject.toml (from 005 + peft)
- [x] Create train.py with PEFT integration and LoRAFinetuneModule
- [x] Create YAML configs (primary + 2 ablations)
- [x] Create SLURM scripts (sbatch/_job.sh, sbatch/run.sh)
- [x] Create save_metrics.py (from 005 + LoRA fields)

## Validation

- [ ] Fast-dev-run: verify PEFT wraps correctly, trainable params ~600K
- [ ] Fast-dev-run: forward pass works, loss decreases over 5 batches
- [ ] GPU check (2 epochs): batch_size=256 fits in 24GB VRAM

## Training

- [ ] Submit primary run: `eat_lora_attn.yaml`
- [ ] Submit ablation: `eat_lora_all.yaml`
- [ ] Submit ablation: `eat_lora_attn_mlphead.yaml`

## Analysis

- [ ] Compare test cmAP across variants
- [ ] Compare with Exp 005 frozen baseline (cmAP=0.301)
- [ ] Document results with `/experiment:result`
