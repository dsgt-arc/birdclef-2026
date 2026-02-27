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

- [x] Fast-dev-run: verify PEFT wraps correctly, forward pass works with mel transform
- [x] GPU check (interactive): bs=32 → ~50% util, bs=128 → ~65% util, 22.8/24GB VRAM
- [x] Tuned batch size: bs=128, accum=2 (CNN needs larger batches than EAT transformer)
- [x] Tested num_workers=12 vs 8 — no meaningful improvement (compute-bound, not data-bound)
- [ ] GPU check (sbatch, 2 epochs): awaiting results

## Training

- [ ] Submit primary run: `convnext_lora_mlp.yaml` (submitted)
- [ ] Submit ablation: `convnext_lora_mlp_mlphead.yaml` (submitted)

## Analysis

- [ ] Compare test cmAP across variants
- [ ] Compare with Exp 006 EAT baselines (cmAP=0.470–0.536)
- [ ] Document results in results.md
