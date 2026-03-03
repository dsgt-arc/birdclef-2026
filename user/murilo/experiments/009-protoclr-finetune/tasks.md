# Experiment 009: ProtoCLR Fine-Tuning -- Tasks

## 1. Setup

- [x] Create experiment directory structure
- [x] Write proposal.md
- [x] Create pyproject.toml
- [x] Create ProtoCLR encoder wrapper (src/birdclef/encoders/protoclr.py)
- [x] Create ProtoCLR mel preprocessing (src/birdclef/preprocessing/protoclr_mel.py)
- [x] Create train.py with PEFT integration
- [x] Create YAML configs (attention-only + all-layers)
- [x] Create SLURM scripts (sbatch/_job.sh, sbatch/run.sh)
- [x] Create save_metrics.py

## 2. Validation

- [x] Fast-dev-run: verify ProtoCLR loads, mel transform works, forward pass completes
- [x] Fast-dev-run: verify LoRA wraps correctly, check trainable param count
- [x] GPU check (2 epochs): measure GPU utilization and throughput
  - GPU utilization is very spiky/inconsistent -- needs investigation before full training

### Bugs fixed during validation

- [x] Fix build backend: setuptools -> hatchling (matching exp 007)
- [x] Add `torchinfo` dependency (required by ProtoCLR's vendored cvt.py)
- [x] Fix non-contiguous tensor error: `view()` -> `reshape()` in protoclr_mel.py

## 3. GPU Utilization Debugging

- [x] Profile dataloader: CPU mel transform confirmed as bottleneck (data spikes up to 4651ms)
- [x] Try increasing batch_size -- tested bs=512 (accum=1, lr=6e-4), throughput unchanged (503→511 samples/s), confirming CPU-bound bottleneck
- [x] Try increasing num_workers -- using 12 workers for full run (SLURM max CPU:GPU = 12:1)
- [~] Consider pre-computing mel spectrograms offline to eliminate CPU bottleneck
  - Mel caching not viable for BirdCLEF 2025 (~135K recordings, would need ~50-65 GB RAM)
  - Streaming data module with GPU-based nnAudio mel (as in exp 003) is the right long-term fix
  - Deferred: one-off training run, not worth the refactor
- [x] Re-run gpu-check after tuning, confirm stable utilization before submitting full runs
  - GPU check run 1 (bs=128, accum=2): avg GPU util 71.1%, avg throughput 503 samples/s
  - GPU check run 2 (bs=512, accum=1): avg GPU util 99.8% (misleading — NVML 1s averaging), avg throughput 511 samples/s
  - Conclusion: throughput is CPU-bound regardless of batch size; settled on bs=256, accum=1, lr=3e-4

## 4. Training

- [x] Submit primary run: `protoclr_lora_attn.yaml` (bs=256, accum=1, lr=3e-4, 12 workers, 24h wall time)
- [x] Submit ablation: `protoclr_lora_all.yaml` (same settings)

## 4b. Project Storage Migration

- [x] Update `DEFAULT_PROJECT_NAME` across all experiments (002-009) from `ps-dsgt_clef2026-0` to `ps-clef2026_birdclef-0`
- [x] Update `project_dir` path from `.../shared/birdclef-2026` to `.../shared/murilo`
- [x] Update `HF_HOME`/`TORCH_HOME` cache paths in all sbatch scripts
- [x] Move `cache/` and experiments 002-008 checkpoints to `/storage/project/ps-clef2026_birdclef-0/shared/murilo/`
- [x] Move exp 009 checkpoints from old dir after training jobs complete
  - Source: `/storage/project/ps-dsgt_clef2026-0/shared/birdclef-2026/experiments/009-protoclr-finetune/`
  - Dest: `/storage/project/ps-clef2026_birdclef-0/shared/murilo/experiments/009-protoclr-finetune/`

## 4c. User Directory Migration (shared/murilo → mgustineli3)

- [x] Update `project_dir` path from `.../shared/murilo` to `.../mgustineli3` in all train.py files (002-009)
- [x] Update `HF_HOME`/`TORCH_HOME` cache paths from `.../shared/murilo/cache` to `.../mgustineli3/cache` in all sbatch scripts
- [x] Update checkpoint path in results.md
- [x] Move repo, cache, and experiments 001-008 to `/storage/project/ps-clef2026_birdclef-0/mgustineli3/`
- [ ] Move exp 009 outputs after full fine-tuning SLURM job completes
  - Source: `/storage/project/ps-clef2026_birdclef-0/shared/murilo/experiments/009-protoclr-finetune/`
  - Dest: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/009-protoclr-finetune/`

## 5. Analysis (LoRA runs)

- [x] Compare test AUROC/cmAP across both LoRA configs
- [x] Compare with Exp 006 EAT (AUROC=0.945) and Exp 007 ConvNeXt (AUROC=0.941)
- [x] Document results in results.md

## 6. Full Fine-Tuning

- [x] Add full fine-tuning support to train.py (config-driven: no `lora:` section → full FT)
- [x] Create `config/protoclr_full_finetune.yaml` (lr=1e-4, all 19M params trainable)
- [x] Update sbatch scripts for both modes (conditional PEFT verification, generalized headers)
- [x] Fast-dev-run: verify full FT pipeline works end-to-end
- [x] GPU check: verify memory usage with all 19M params trainable
- [ ] Submit full fine-tuning run: `protoclr_full_finetune.yaml` (bs=256, lr=1e-4, 12 workers, 24h wall time)

## 7. Analysis (Full Fine-Tuning)

- [ ] Collect test metrics (AUROC, cmAP, F1 macro, training time, memory)
- [ ] Compare full FT vs LoRA configs — does training all 19M params beat LoRA (<1%)?
- [ ] Update results.md with full fine-tuning results and LoRA vs full FT comparison
- [ ] Update EXPERIMENTS.md (Quick Reference, Section 3.9, master table, key insights)
