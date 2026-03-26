# Experiment 017: Strong Baseline via Mel Overhaul

Upgrade mel spectrogram resolution from (n_fft=1024, n_mels=128) to (n_fft=4096, n_mels=224) and systematically layer additional improvements. Goal: close the gap from 0.827 to 0.85+ labeled-data-only baseline.

**Branch**: `mg/017-mel-overhaul` (PR #19)
**Baseline**: Exp 016 v1+pp2 = 0.827 LB
**Architecture**: EfficientNet-B0 SED (GeM + AttHead), 234 classes

## Results Summary

| Variant | Change | best val/auroc | LB (raw) | LB (pp2) |
|---------|--------|---------------|----------|----------|
| v1 (exp 016) | baseline (n_fft=1024, 128 mels) | 0.845 | 0.795 | 0.827 |
| **v4a** | n_fft=4096, n_mels=224 | 0.846 | **0.825** | **0.830** |
| **v4b** | + AmplitudeToDB norm | **0.865** | **0.859** | **0.860** |
| v4c | + AttBlockV2 head | 0.815 | skipped | |
| v4d | + ASL loss | 0.858 | 0.808 | |
| **v4e** | + StochasticDepth 0.15 | **0.868** | 0.840 | |
| v4f | v4b + StochasticDepth 0.15 | 0.862 | 0.832 | |
| v4g | v4b + AttBlockV2 + BCE | pending | pending | |
| v4h | v4b + AttBlockV2 + ASL | 0.858 | pending | |

v4a passed the decision gate (+0.030 LB raw). v4b-v4e are single-variable ablations from v4a.
**v4b (AmplitudeToDB) is the clear winner**: +0.034 LB over v4a, new best at 0.859.
v4f (v4b + StochasticDepth) regressed to 0.832 — techniques don't stack.

## Experiment Design

### Step 1: Mel Resolution (v4a) -- DONE

Single-variable change: n_fft 1024->4096, n_mels 128->224. Everything else identical to v1.
- **Result**: LB raw 0.825 (+0.030), cmAP 0.615 (+0.022). Decision gate passed.
- **Key finding**: PP2 gained only +0.005 on v4a (vs +0.032 on v1) -- model is more confident.

### Step 2: Parallel Ablations (v4b-v4e) -- DONE

Each changes ONE thing from v4a. All use 90/10 split + 50 epochs (vs v4a's 80/10/10 + 40).

| Variant | Change | Rationale |
|---------|--------|-----------|
| v4b | `norm_type: amplitude_to_db` (AmplitudeToDB + standardize + min-max) | Tom's 28 experiments, 2025 1st place |
| v4c | `sed_head.type: att_block_v2` (tanh+softmax att, sigmoid cls) | PANNs standard SED head |
| v4d | `loss_type: asl` (gamma_neg=4, gamma_pos=0, clip=0.05) | Tom's biggest single gain: +0.023 |
| v4e | `drop_path_rate: 0.15` (stochastic depth) | 2025 1st place regularization |

### Step 3: Combine Winners (v4f) -- FUTURE

Combine winning ablations into one config. Also sweep PP2 temperature on v4a.

### Step 4: Fulldata Training -- FUTURE

Train best recipe on 100% data (no val split).

## Directory Structure

```
017-mel-overhaul/
├── README.md
├── proposal.md              # Hypothesis, motivation, references
├── tasks.md                 # Step-by-step checklist
├── results.md               # Offline + LB results
├── train.py                 # Training script (adapted from exp 016)
├── save_metrics.py          # Record structured metrics after training
├── pyproject.toml
├── config/
│   ├── efficientnet_b0_sed_v4a.yaml   # Step 1: mel resolution only
│   ├── efficientnet_b0_sed_v4b.yaml   # Step 2: AmplitudeToDB norm
│   ├── efficientnet_b0_sed_v4c.yaml   # Step 2: AttBlockV2 head
│   ├── efficientnet_b0_sed_v4d.yaml   # Step 2: ASL loss
│   └── efficientnet_b0_sed_v4e.yaml   # Step 2: StochasticDepth
├── sbatch/
│   ├── _job.sh              # SLURM job template
│   └── run.sh               # Training launcher
├── artifacts/training/      # Structured metrics JSON per variant
├── outputs/
│   ├── tensorboard/         # TB logs per run
│   ├── wandb/               # W&B artifacts
│   ├── test_results/        # Raw test result JSONs
│   └── gpu_check/           # GPU utilization snapshots
└── logs/                    # SLURM training logs
```

## Training

All training runs from this experiment directory:

```bash
cd user/murilo/experiments/017-mel-overhaul

# Single model
bash sbatch/run.sh --config config/efficientnet_b0_sed_v4a.yaml

# GPU check (1 epoch, no W&B)
bash sbatch/run.sh --config config/efficientnet_b0_sed_v4a.yaml --gpu-check 1
```

### Key Training Config

| Parameter | v4a | v4b-v4e |
|-----------|-----|---------|
| n_fft | 4096 | 4096 |
| n_mels | 224 | 224 |
| hop_length | 320 | 320 |
| batch_size | 64 | 64 |
| accumulate_grad_batches | 4 | 4 |
| effective batch | 256 | 256 |
| split | 80/10/10 | **90/10** |
| max_epochs | 40 | **50** |
| patience | 10 | 10 |
| mel.gpu | true | true |

## ONNX Export + Kaggle Submission

Exports and submissions are managed through **exp 010-submission-pipeline**.

### Export (SLURM GPU job)

```bash
cd user/murilo/experiments/010-submission-pipeline

# Export all three ablations in one job
bash sbatch/run.sh export --model all-017-ablations

# Or export individually
bash sbatch/run.sh export --model effnet-b0-sed-v4e   # StochasticDepth (best val 0.868)
bash sbatch/run.sh export --model effnet-b0-sed-v4b   # AmplitudeToDB (val 0.865)
bash sbatch/run.sh export --model effnet-b0-sed-v4d   # ASL loss (val 0.858)

# Also available
bash sbatch/run.sh export --model effnet-b0-sed-v4a   # Mel baseline
bash sbatch/run.sh export --model effnet-b0-sed-v4c   # AttBlockV2 (poor offline, optional)

# Dry run (preview without submitting)
bash sbatch/run.sh export --model effnet-b0-sed-v4e --dry-run
```

This submits a GPU job that:
1. Creates a venv in `$TMPDIR` with torch + dependencies
2. Loads the checkpoint and config
3. Builds encoder (with drop_path_rate if applicable) + SED head (AttHead or AttBlockV2)
4. Exports to ONNX with blended sigmoid output (0.75 clip + 0.25 frame)
5. Saves `model.onnx`, `mel_config.json`, `species_list.json` to `artifacts/weights/effnet-b0-sed-v4X/`

### Upload Weights + Push Kernel to Kaggle

After the export job completes:

```bash
cd user/murilo/experiments/010-submission-pipeline

# Upload weights dataset + push kernel (one command does both)
python scripts/upload_kaggle.py --model effnet-b0-sed-v4e --push-kernel
python scripts/upload_kaggle.py --model effnet-b0-sed-v4b --push-kernel
python scripts/upload_kaggle.py --model effnet-b0-sed-v4d --push-kernel

# Upload only (no kernel push)
python scripts/upload_kaggle.py --model effnet-b0-sed-v4e

# Dry run
python scripts/upload_kaggle.py --model effnet-b0-sed-v4e --push-kernel --dry-run
```

This:
1. Uploads `artifacts/weights/effnet-b0-sed-v4X/` as a Kaggle dataset (create or version)
2. Copies the model's species_mapping.json into the weights dir
3. If `--push-kernel`: copies the model-specific kernel metadata and pushes the unified `submission.py`

### Check Submission Results

```bash
# Kernel status
kaggle kernels status murilogustineli/birdclef-2026-effnet-b0-sed-v4e

# All submissions
kaggle competitions submissions birdclef-2026 | head -20
```

## Checkpoints

All checkpoints stored under: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/`

| Variant | Checkpoint | Best Epoch | SLURM Job |
|---------|-----------|------------|-----------|
| v4a | `2026-effnet-b0-sed-v4a/best-epochepoch=24.ckpt` | 24 | 5484980 |
| v4b | `2026-effnet-b0-sed-v4b/best-epochepoch=30.ckpt` | 30 | 5496315 |
| v4c | `2026-effnet-b0-sed-v4c/best-epochepoch=15.ckpt` | 15 | 5496316 |
| v4d | `2026-effnet-b0-sed-v4d/best-epochepoch=16.ckpt` | 16 | 5496367 |
| v4e | `2026-effnet-b0-sed-v4e/best-epochepoch=29.ckpt` | 29 | 5496368 |

## W&B Tracking

Project: `dsgt-arc/birdclef-2026-mel-overhaul`

| Variant | Run ID |
|---------|--------|
| v4a | `05pylg1y` |
| v4b | `q9jdljkt` |
| v4c | `4kwgh6qs` |
| v4d | `2b43diog` |
| v4e | `07o7zjha` |

## Key References

- [Concept: strong-baseline-mel-overhaul](../../docs/concepts/strong-baseline-mel-overhaul.md)
- [Tom's SED model comparison (28 experiments)](../../docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md)
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- [2025 1st place inference](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md)
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md)
- [Exp 016 SED summary](../016-sed-architecture/summary.md)

## Known Issues

- **v4c (AttBlockV2) underperformed** (0.815 val): soft CE loss + sigmoid mismatch. Retry with BCE before concluding.
- **v4b-v4e confound**: 90/10 + 50ep vs v4a's 80/10/10 + 40ep means ~11% more data. Small deltas may be the extra data.
- **PP2 diminishing returns**: T=1.10 was tuned for v1. Temperature sweep on v4a is a zero-cost opportunity.
