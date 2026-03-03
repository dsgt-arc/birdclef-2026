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

## 5. Analysis

- [ ] Compare test cmAP across both configs
- [ ] Compare with Exp 006 EAT (cmAP=0.536) and Exp 007 ConvNeXt (cmAP=0.527)
- [ ] Document results in results.md
