# Experiment 008: Frame Tokenization for EAT — Tasks

## 1. Setup

- [x] 1.1 Create experiment directory structure
- [x] 1.2 Write proposal.md
- [x] 1.3 Create pyproject.toml
- [x] 1.4 Create src symlink to shared `user/murilo/src/`
- [x] 1.5 Create SLURM scripts (sbatch/_job.sh, sbatch/run.sh)
- [x] 1.6 Create save_metrics.py

## 2. Implementation

- [x] 2.1 Implement `FramePatchEmbed` — Conv2d with configurable (H, W) kernel/stride
- [x] 2.2 Implement positional encoding regeneration for non-square grids (1×256, 1×128)
- [x] 2.3 Implement `build_eat_frame_model()` — swap PatchEmbed_new with FramePatchEmbed, configure freeze/train
- [x] 2.4 Implement partial unfreeze logic for C3 config (first 2 blocks fully trained, rest LoRA)
- [x] 2.5 Create train.py integrating frame patch embed + PEFT LoRA + existing data pipeline
- [x] 2.6 Create YAML configs for all 6 runs (A, B, C1a, C1b, C2, C3)

## 3. Validation

- [x] 3.1 Fast-dev-run: Config A (pretrained 16×16 baseline)
- [x] 3.2 Fast-dev-run: Config B (random 16×16 control) — job 4422202, 5 batches OK
- [x] 3.3 Fast-dev-run: Config C1a (2×128 frame) — job 4422203, 5 batches OK
- [x] 3.4 Fast-dev-run: Config C1b (4×128 frame) — job 4422204, 5 batches OK
- [x] 3.5 GPU utilization check (2 epochs, all 4 configs) — see `gpu-profiling.md`

## 4. Phase 1 — Core Question

- [x] 4.1 Submit Config A: pretrained 16×16 baseline
- [x] 4.2 Submit Config B: random 16×16 control
- [x] 4.3 Submit Config C1a: frame 2×128
- [x] 4.4 Submit Config C1b: frame 4×128
- [ ] 4.5 Analyze Phase 1 results, select best frame size

## 5. Phase 2 — Capacity Exploration

- [ ] 5.1 Submit Config C2: best frame size + LoRA r=16
- [ ] 5.2 Submit Config C3: best frame size + partial unfreeze (blocks 0-1)
- [ ] 5.3 Analyze Phase 2 results

## 6. Analysis

- [ ] 6.1 Compare all configs: A vs B vs C variants
- [ ] 6.2 Interpret results using the interpretation guide from proposal
- [ ] 6.3 Compare with exp 006/007 baselines
- [ ] 6.4 Document results in results.md
- [ ] 6.5 Determine best teacher model candidate for pseudo-label generation
