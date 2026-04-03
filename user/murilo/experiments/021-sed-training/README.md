# Experiment 021: SED Training — Closing the Gap to 0.920+

Push single SED model from 0.906 → 0.920+ LB by applying competitor insights
(site/hour priors, CE+sigmoid, waveform augmentations, backbone scaling).

See [proposal.md](proposal.md) for hypothesis, motivation, and design rationale.
See [tasks.md](tasks.md) for the full task checklist and phase breakdown.

**Branch**: `mg/021-sed-training`
**Current Best SED**: 0.906 LB (v5j single), 0.913 LB (v5j-4fold)
**Target**: 0.920+ single, 0.925+ 4-fold

## Phases

| Phase | Goal | LB Target | Dependencies |
|-------|------|-----------|-------------|
| A | Site/hour priors on SED output (no retraining) | ≥0.912 | None |
| B | Training recipe ablations (CE+sigmoid, waveform augs) | ≥0.915 | Phase A result |
| C | Backbone scaling (HGNetV2-B2, ConvNeXt-V2, EffNet-B3 NS) | ≥0.920 | Phase B recipe |

## Key References

- [Kaggle Discussion #683791](https://www.kaggle.com/competitions/birdclef-2026/discussion/683791) — competitor single-model scores
- [Exp 019 SED Ablations](../019-sed-ablations/tasks.md) — v5a-v5n results (predecessor)
- [Exp 020 Perch ONNX](../020-perch-onnx/tasks.md) — prior tables code for Phase A

## Directory Structure

```
021-sed-training/
├── README.md
├── proposal.md
├── tasks.md
├── train.py                  # Training script (from exp 019)
├── generate_folds.py         # 4-fold CV generation
├── fold_assignments.json     # Fold assignments (4-fold, shared with exp 019)
├── save_metrics.py           # Post-training metrics aggregation
├── pyproject.toml
├── config/
│   ├── hgnet_b0_sed_v6a.yaml       # Phase B: CE(primary_label) + sigmoid
│   ├── hgnet_b0_sed_v6b.yaml       # Phase B: Waveform MixUp
│   ├── hgnet_b0_sed_v6c.yaml       # Phase B: Waveform MixUp + noise + time stretch
│   ├── hgnet_b0_sed_v6d.yaml       # Phase B: 25 epochs
│   ├── hgnet_b2_sed_v6f.yaml       # Phase C: HGNetV2-B2 (12M params)
│   ├── convnextv2_nano_sed_v6g.yaml # Phase C: ConvNeXt-V2 Nano (15M params)
│   └── effnet_b3_ns_sed_v6h.yaml   # Phase C: EfficientNet-B3 NS (12M params)
└── sbatch/
    ├── _job.sh               # SLURM job template
    └── run.sh                # Job launcher with GPU/fold support
```

## Config Comparison

| Config | Backbone | Params | Embed | Loss | MixUp | Epochs | Change from v5j |
|--------|----------|--------|-------|------|-------|--------|-----------------|
| v5j (baseline) | HGNetV2-B0 | 6M | 1024 | BCE | spec | 20 | — |
| v6a | HGNetV2-B0 | 6M | 1024 | **CE** | spec | 20 | CE(primary_label) |
| v6b | HGNetV2-B0 | 6M | 1024 | BCE | **wave** | 20 | Waveform MixUp |
| v6c | HGNetV2-B0 | 6M | 1024 | BCE | **wave+noise** | 20 | Full waveform augs |
| v6d | HGNetV2-B0 | 6M | 1024 | BCE | spec | **25** | Longer training |
| v6f | **HGNetV2-B2** | **12M** | **1536** | BCE | spec | 20 | Larger HGNet |
| v6g | **ConvNeXt-V2 Nano** | **15M** | **640** | BCE | spec | 20 | Modern ConvNet |
| v6h | **EffNet-B3 NS** | **12M** | 384→1024 | BCE | spec | 20 | MBConv comparison |

## Commands

All commands run from the experiment directory:

```bash
cd user/murilo/experiments/021-sed-training
```

### Quick validation (no GPU needed for syntax check)

```bash
# Validate config loads correctly
python3 -c "import yaml; print(yaml.safe_load(open('config/hgnet_b0_sed_v6a.yaml'))['run_name'])"
```

### Training on PACE

```bash
# Single run (submits SLURM job)
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml

# With specific GPU
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --gpu a100
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --gpu rtx6000

# Fast dev run (5 batches, no W&B, quick sanity check)
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --fast-dev-run

# GPU utilization check (1 epoch)
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --gpu-check
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --gpu-check 3  # 3 epochs

# 4-fold training (4 parallel SLURM jobs)
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --all-folds

# Single fold
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --fold 0

# Dry run (preview sbatch command without submitting)
DRY_RUN=1 bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml

# Override number of workers
bash sbatch/run.sh config/hgnet_b0_sed_v6a.yaml --num-workers 4
```

### Direct training (interactive GPU session)

```bash
# Activate venv first (see CLAUDE.md for venv policy)
source "$JOB_VENV/bin/activate"
uv pip install -e .

# Run training directly
python train.py --config config/hgnet_b0_sed_v6a.yaml
python train.py --config config/hgnet_b0_sed_v6a.yaml --fold 0
python train.py --config config/hgnet_b0_sed_v6a.yaml --fast-dev-run
```

### Monitoring

```bash
# Check SLURM job status
squeue -u $USER --format="%.8i %.40j %.8T %.10M"

# Watch TensorBoard logs
tensorboard --logdir outputs/

# W&B: https://wandb.ai/dsgt-arc/birdclef-2026-exp021
```

### Post-training

```bash
# Save structured metrics after training completes
python save_metrics.py --config config/hgnet_b0_sed_v6a.yaml
python save_metrics.py --config config/hgnet_b0_sed_v6a.yaml --fold 0
```
