# Experiment 017: Strong Baseline via Mel Overhaul

Upgrade mel spectrogram resolution from (n_fft=1024, n_mels=128) to (n_fft=4096, n_mels=224) and systematically layer additional improvements. Goal: close the gap from 0.827 to 0.85+ labeled-data-only baseline.

**Branch**: `mg/017-mel-overhaul` (PR #19)
**Baseline**: Exp 016 v1+pp2 = 0.827 LB
**Architecture**: EfficientNet-B0 SED (GeM + AttHead), 234 classes
**Current best**: **v4b = 0.859 LB** (AmplitudeToDB norm)

## Results Summary

| Variant | Change | val/auroc | LB (raw) | LB (pp2) |
|---------|--------|-----------|----------|----------|
| v1 (exp 016) | baseline (n_fft=1024, 128 mels) | 0.845 | 0.795 | 0.827 |
| **v4a** | n_fft=4096, n_mels=224 | 0.846 | **0.825** | **0.830** |
| **v4b** | + AmplitudeToDB norm | **0.865** | **0.859** | **0.860** |
| v4c | + AttBlockV2 head (soft CE) | 0.815 | skipped | |
| v4d | + ASL loss | 0.858 | 0.808 | |
| v4e | + StochasticDepth 0.15 | **0.868** | 0.840 | |
| v4f | v4b + StochasticDepth 0.15 | 0.862 | 0.832 | |
| v4g | v4b + AttBlockV2 + BCE | **0.868** | 0.819 | |
| v4h | v4b + AttBlockV2 + ASL | 0.858 | 0.800 | |
| v4b-fulldata | v4b on 100% data | — | 0.853 | — |

**Key findings:**
- Mel resolution is a major lever: v4a raw (0.825) nearly matches v1+pp2 (0.827)
- AmplitudeToDB normalization is the single biggest gain: +0.034 LB
- Techniques don't stack: v4f, v4g, v4h all regressed vs v4b alone
- Offline val/auroc is a poor LB predictor: v4g had best val (0.868) but 0.819 LB
- PP2 diminishing returns: +0.032 on v1, +0.005 on v4a, +0.001 on v4b

## Experiment Design

### Step 1: Mel Resolution (v4a) -- DONE

Single-variable change: n_fft 1024->4096, n_mels 128->224. Everything else identical to v1.
- **Result**: LB raw 0.825 (+0.030), cmAP 0.615 (+0.022). Decision gate passed.
- **Key finding**: PP2 gained only +0.005 on v4a (vs +0.032 on v1) -- model is more confident.

### Step 2: Parallel Ablations (v4b-v4e) -- DONE

Each changes ONE thing from v4a. All use 90/10 split + 50 epochs (vs v4a's 80/10/10 + 40).

| Variant | Change | Rationale | LB |
|---------|--------|-----------|-----|
| **v4b** | `norm_type: amplitude_to_db` | Tom's 28 experiments, 2025 1st place | **0.859** |
| v4c | `sed_head.type: att_block_v2` | PANNs standard SED head | skipped (0.815 val) |
| v4d | `loss_type: asl` (γ⁻=4, γ⁺=0) | Tom's biggest single gain: +0.023 | 0.808 |
| v4e | `drop_path_rate: 0.15` | 2025 1st place regularization | 0.840 |

### Step 3: Combine Winners + Re-ablations -- DONE

Attempted to combine winning techniques and re-test AttBlockV2/ASL with correct loss pairing.
**None beat v4b.**

| Variant | Change from v4b | val/auroc | LB | vs v4b |
|---------|----------------|-----------|-----|--------|
| v4f | + StochasticDepth 0.15 | 0.862 | 0.832 | −0.027 |
| v4g | + AttBlockV2 + BCE | **0.868** | 0.819 | −0.040 |
| v4h | + AttBlockV2 + ASL | 0.858 | 0.800 | −0.059 |

**Conclusion**: v4b's simple recipe (AmplitudeToDB + AttHead + soft CE) generalizes best.
AttBlockV2 + BCE trains well (highest val) but overfits to training temporal patterns.

### Step 4: Fulldata Training -- DONE

Trained v4b recipe on 100% labeled data (val_fraction=0.0). Epoch cap: 45 (~1.5× v4b's
best epoch 30). Save checkpoints every 5 epochs + last. No early stopping.
- **Result**: LB 0.853 (−0.006 vs v4b 90/10). Fulldata regressed — likely overtrained without val monitoring.

## Directory Structure

```
017-mel-overhaul/
├── README.md
├── proposal.md                          # Hypothesis, motivation, references
├── tasks.md                             # Step-by-step checklist
├── results.md                           # Offline + LB results
├── onnx-inference-optimization.md       # Full-signal mel + batched ONNX docs
├── train.py                             # Training script (supports fulldata mode)
├── save_metrics.py                      # Record structured metrics after training
├── pyproject.toml
├── config/
│   ├── efficientnet_b0_sed_v4a.yaml     # Step 1: mel resolution only
│   ├── efficientnet_b0_sed_v4b.yaml     # Step 2: AmplitudeToDB norm (winner)
│   ├── efficientnet_b0_sed_v4c.yaml     # Step 2: AttBlockV2 head
│   ├── efficientnet_b0_sed_v4d.yaml     # Step 2: ASL loss
│   ├── efficientnet_b0_sed_v4e.yaml     # Step 2: StochasticDepth
│   ├── efficientnet_b0_sed_v4f.yaml     # Step 3: v4b + StochasticDepth
│   ├── efficientnet_b0_sed_v4g.yaml     # Step 3: v4b + AttBlockV2 + BCE
│   ├── efficientnet_b0_sed_v4h.yaml     # Step 3: v4b + AttBlockV2 + ASL
│   └── efficientnet_b0_sed_v4b_fulldata.yaml  # Step 4: v4b on 100% data
├── sbatch/
│   ├── _job.sh                          # SLURM job template
│   └── run.sh                           # Training launcher
├── artifacts/training/                  # Structured metrics JSON per variant
├── outputs/
│   ├── tensorboard/                     # TB logs per run
│   ├── wandb/                           # W&B artifacts
│   ├── test_results/                    # Raw test result JSONs
│   └── gpu_check/                       # GPU utilization snapshots
└── logs/                                # SLURM training logs
```

## Training

All training runs from this experiment directory:

```bash
cd user/murilo/experiments/017-mel-overhaul

# Single model
bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml

# Fulldata (no val set, epoch cap)
bash sbatch/run.sh config/efficientnet_b0_sed_v4b_fulldata.yaml

# GPU check (1 epoch, no W&B)
bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml --gpu-check 1

# Specify GPU partition
bash sbatch/run.sh config/efficientnet_b0_sed_v4b.yaml --gpu a100
```

## ONNX Export + Kaggle Submission

Exports and submissions are managed through **exp 010-submission-pipeline**.
Uses optimized inference: full-signal mel + batched ONNX + session options
(see [onnx-inference-optimization.md](onnx-inference-optimization.md)).

```bash
cd user/murilo/experiments/010-submission-pipeline

# Export
bash sbatch/run.sh export --model effnet-b0-sed-v4b
bash sbatch/run.sh export --model effnet-b0-sed-v4b-fulldata

# Upload weights + push kernel
python scripts/upload_kaggle.py --model effnet-b0-sed-v4b --push-kernel
python scripts/upload_kaggle.py --model effnet-b0-sed-v4b-fulldata --push-kernel

# Check results
kaggle kernels status murilogustineli/birdclef-2026-effnet-b0-sed-v4b
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
| v4f | `2026-effnet-b0-sed-v4f/best-epochepoch=17.ckpt` | 17 | 5506300 |
| v4g | `2026-effnet-b0-sed-v4g/best-epochepoch=28.ckpt` | 28 | 5506698 |
| v4h | `2026-effnet-b0-sed-v4h/best-epochepoch=12.ckpt` | 12 | 5506700 |
| v4b-fulldata | `2026-effnet-b0-sed-v4b-fulldata/last.ckpt` | — (no val) | 5514462 |

## W&B Tracking

Project: `dsgt-arc/birdclef-2026-mel-overhaul`

| Variant | Run ID |
|---------|--------|
| v4a | `05pylg1y` |
| v4b | `q9jdljkt` |
| v4c | `4kwgh6qs` |
| v4d | `2b43diog` |
| v4e | `07o7zjha` |
| v4f | `v9em250j` |
| v4g | `zklk0fv2` |
| v4h | `4ck222t9` |

## Key References

- [Concept: strong-baseline-mel-overhaul](../../docs/concepts/strong-baseline-mel-overhaul.md)
- [Tom's SED model comparison (28 experiments)](../../docs/references/kaggle/birdclef-2026/discussions/sed-model-comparison-tom.md)
- [HGNetV2-B0 baseline (Tawara)](../../docs/references/kaggle/birdclef-2026/discussions/hgnetv2-b0-baseline-tawara.md)
- [SED improvement plan](../../docs/references/kaggle/birdclef-2026/discussions/sed-improvement-plan-681146.md)
- [2025 1st place inference](../../docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md)
- [Gemini strategic framework](../../docs/references/deep-research/2026-03-24-gemini-strategic-framework.md)
- [Exp 016 SED summary](../016-sed-architecture/summary.md)
- [ONNX inference optimization](onnx-inference-optimization.md)

## Key Learnings

1. **AmplitudeToDB normalization is universal among top performers** — Tom, Tawara, 2025 1st place all use it. Our biggest single-technique gain (+0.034 LB).

2. **Techniques don't combine additively** — StochasticDepth helped alone (+0.015) but hurt with AmplitudeToDB (−0.027). AttBlockV2+BCE had best val (0.868) but worst LB after v4h (0.819).

3. **Offline val/auroc is a poor LB predictor** — v4g (0.868 val → 0.819 LB) vs v4b (0.865 val → 0.859 LB). The val set overestimates generalization for some architectures.

4. **ASL loss doesn't suit our pipeline** — Consistent early convergence + plateau in both v4d (0.808 LB) and v4h (0.800 LB). Our soft CE already handles multi-label effectively.

5. **Loss/head pairing matters** — AttBlockV2 + soft CE = 0.815 val (v4c, catastrophic). AttBlockV2 + BCE = 0.868 val (v4g, good offline). Same head, completely different results.

6. **PP2 returns shrink with better models** — v1: +0.032, v4a: +0.005, v4b: +0.001. As the model improves, post-processing adds less.
