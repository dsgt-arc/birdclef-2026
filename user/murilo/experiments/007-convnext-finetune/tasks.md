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
- [x] GPU check (sbatch, 2 epochs): 100% avg GPU util, ~113 samples/s, compute-bound at bs=128

## Training

- [x] Submit primary run: `convnext_lora_mlp.yaml` — cmAP=0.527, stopped epoch 26 (best at 15)
- [x] Submit ablation: `convnext_lora_mlp_mlphead.yaml` — cmAP=0.525, stopped epoch 23 (best at 12)

## Analysis

- [x] Compare test cmAP across variants: linear head (0.527) ≈ MLP head (0.525), no benefit from richer head
- [x] Compare with Exp 006 EAT baselines: ConvNeXt 0.527 vs EAT best 0.536 (only 1.7% gap)
- [x] Document results in results.md

## Upstream Contributions (adapter-hub/adapters)

See `upstream-prs.md` for full context, challenges, and implementation plan.

- [~] Prerequisite: [PR #831](https://github.com/adapter-hub/adapters/pull/831) — `get_input_embeddings()` fix for vision models (awaiting review)
- [ ] File issue proposing nested stage support for stage-based architectures (ConvNeXt, Swin, EfficientNet)
- [ ] Implement ConvNeXt `CUSTOM_INTERFACES` entry (after #831 merges + issue discussed)
- [ ] Add tests using `VisionAdapterTestBase` + tiny `ConvNextConfig`
- [ ] Submit PR: add ConvNeXt to supported models
