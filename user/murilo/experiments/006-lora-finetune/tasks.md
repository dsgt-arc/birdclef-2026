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
- [x] All 3 PEFT runs completed with test evaluation

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
- [x] Submit all 3 adapters runs (`bash sbatch/run.sh --all --backend adapters`)
- [x] Write reference doc for upstream contribution (docs/references/adapters-lora-eat.md)
- [x] 2/3 adapters runs completed; `adapters-eat-lora-all` timed out (>12h SLURM limit)

## Analysis

- [x] Compare test cmAP across variants (5 completed runs)
- [x] Compare with Exp 005 frozen baseline (cmAP=0.301) — all runs beat it by 56-78%
- [x] Document results in results.md
- [x] Update proposal.md with dual-backend method and upstream contributions

## GPU Profiling

- [x] GPU check: PEFT bs=32 (90.6% util, 5.4 GB reserved, 182 samples/s)
- [x] GPU check: adapters bs=32 (88.3% util, 5.1 GB reserved, 178 samples/s)
- [x] GPU check: PEFT bs=64 (92.5% util, 9.8 GB reserved, 232 samples/s)
- [x] Tested batch_size=64 — only ~18% faster per epoch, not worth changing configs
- [x] Document profiling results in gpu-profiling.md
- [x] Conclusion: 9-11h training is encoder-bound, not data/overhead-bound

## Infrastructure Fixes

- [x] Fix save_metrics.py to support `--backend adapters` flag
- [x] Fix cross-filesystem uv cache in create_venv.sh (`UV_CACHE_DIR`)
- [x] Write PACE GPU workflows reference (docs/references/pace-gpu-workflows.md)

## Upstream Contributions (adapter-hub/adapters)

See `upstream-prs.md` for full context, blockers, and implementation plan.

- [x] PR #830: fix fused QKV LoRA shape mismatch (`LoRAMergedLinear.get_n_heads` counts "o") — merged
- [x] PR #831: handle `NotImplementedError` from `get_input_embeddings()` for audio/vision models — merged
- [x] Ping maintainers (@calpt, @lenglaender, @TimoImhof) on PRs #830 and #831 for review
- [x] Remove local workarounds from `birdclef.lora.adapters` (monkey-patches, manual interface)
- [ ] PR #832: add EAT to `CUSTOM_INTERFACES` for auto-detection — CI passing, awaiting review
