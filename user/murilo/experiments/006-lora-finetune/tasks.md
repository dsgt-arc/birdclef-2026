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

- [x] Fast-dev-run: verify PEFT wraps correctly, trainable params ~608K
- [x] Fast-dev-run: forward pass works, loss starts at ~0.694 (expected for 206-class BCE)
- [ ] GPU check (2 epochs): batch_size=32 × accum 8 fits in 24GB VRAM

## Training

- [ ] Submit primary run: `eat_lora_attn.yaml`
- [ ] Submit ablation: `eat_lora_all.yaml`
- [ ] Submit ablation: `eat_lora_attn_mlphead.yaml`

## Adapters Library Integration (alternative backend)

- [x] Create train_adapters.py with adapter-hub/adapters library
- [x] Define EAT AdapterModelInterface (plugin interface)
- [x] Add adapters optional dependency to pyproject.toml
- [ ] Fast-dev-run: verify adapters.init() wraps EAT correctly
- [ ] Fast-dev-run: trainable params match PEFT (~608K for attn config)
- [ ] GPU check: confirm equivalent memory usage to PEFT
- [ ] (Optional) Open PR / issue on adapter-hub/adapters with EAT interface

## Analysis

- [ ] Compare test cmAP across variants
- [ ] Compare with Exp 005 frozen baseline (cmAP=0.301)
- [ ] Document results with `/experiment:result`
