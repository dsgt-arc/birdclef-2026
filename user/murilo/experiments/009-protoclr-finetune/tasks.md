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

- [ ] Profile dataloader: check if mel transform on CPU is the bottleneck (ProtoCLR mel is heavier than NNAudio -- 16kHz, n_fft=1024, 128 mels, per-sample normalization + standardization)
- [ ] Try increasing batch_size (currently 128) -- CvT-13 is only 19M params, GPU memory may be underutilized
- [ ] Try increasing num_workers (currently 8) to keep GPU fed
- [ ] Consider pre-computing mel spectrograms offline to eliminate CPU bottleneck
- [ ] Re-run gpu-check after tuning, confirm stable utilization before submitting full runs

## 4. Training

- [ ] Submit primary run: `protoclr_lora_attn.yaml`
- [ ] Submit ablation: `protoclr_lora_all.yaml`

## 5. Analysis

- [ ] Compare test cmAP across both configs
- [ ] Compare with Exp 006 EAT (cmAP=0.536) and Exp 007 ConvNeXt (cmAP=0.527)
- [ ] Document results in results.md
