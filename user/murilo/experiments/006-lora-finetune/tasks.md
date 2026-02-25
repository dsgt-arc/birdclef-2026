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
- [x] GPU check (2 epochs): 90.6% GPU util, 2.6% memory, 182 samples/s throughput

## Training (PEFT)

- [x] Submit primary run: `eat_lora_attn.yaml`
- [x] Submit ablation: `eat_lora_all.yaml`
- [x] Submit ablation: `eat_lora_attn_mlphead.yaml`
- [ ] Monitor runs on W&B

## Adapters Library Integration (alternative backend)

- [x] Create train_adapters.py with adapter-hub/adapters library
- [x] Define EAT AdapterModelInterface (plugin interface)
- [x] Add adapters optional dependency to pyproject.toml
- [x] Fix base_model resolution (use EATModel as base, dotted paths for model-level fields)
- [x] Fix get_input_embeddings() NotImplementedError (monkey-patch for audio model)
- [x] Fix fused QKV LoRA shape mismatch (adapters bug: get_n_heads counts "o")
- [x] Write pytest suite: tests/006-lora-finetune/test_adapters_lora.py (10/10 pass)
- [x] Fast-dev-run: verify adapters.init() wraps EAT correctly (442K LoRA params, loss ~0.690)
- [x] Pass dropout config to adapters LoRAConfig
- [x] Add --all flag to run.sh for batch submission
- [ ] GPU check: confirm equivalent memory usage to PEFT
- [x] Submit all 3 adapters runs (`bash sbatch/run.sh --all --backend adapters`)
- [x] Write reference doc for upstream contribution (docs/references/adapters-lora-eat.md)

## Upstream Contributions (adapter-hub/adapters)

- [ ] Open issue/PR: fix fused QKV LoRA shape mismatch (`LoRAMergedLinear.get_n_heads` counts "o")
- [ ] Open PR: add EAT as a supported model via plugin interface (`AdapterModelInterface`)

## Analysis

- [ ] Compare test cmAP across variants
- [ ] Compare with Exp 005 frozen baseline (cmAP=0.301)
- [ ] Document results with `/experiment:result`
