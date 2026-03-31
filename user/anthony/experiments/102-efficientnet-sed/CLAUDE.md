# CLAUDE.md

## What this experiment does

Multi-backbone LoRA fine-tuning experiment ladder for BirdCLEF 2026. Tests
EfficientNet-B1 and ConvNeXt-Base with attention pooling, soundscape data,
class weighting, and soundscape validation. Best result: **0.794 Kaggle public LB**.

## Current status

- **Best Kaggle LB: 0.794** (EfficientNet LoRA + attention, 2 epochs)
- Experiment ladder runs A-K completed or running
- Key finding: focal overfitting — more training hurts Kaggle LB
- Soundscape validation peaks at epoch 2-3, matching Kaggle LB pattern
- ConvNeXt overfits worse than EfficientNet (0.603 LB despite 0.644 val/cmap)
- Next: clean focal data using trained attention model's frame-level predictions

## Experiment ladder

| Run | Config | What | Best cmap | Kaggle LB |
|-----|--------|------|-----------|-----------|
| A | run-a-frozen | EfficientNet frozen baseline | 0.387 | 0.672 |
| B (old) | — | EfficientNet LoRA avg (2ep, bf16) | 0.524 | 0.754 |
| **C (old)** | — | **EfficientNet LoRA+attention (2ep, bf16)** | 0.531 | **0.794** |
| B | run-b-lora-gpumel | EfficientNet LoRA avg (fp16+compile) | 0.619 | 0.618 |
| D | run-d-convnext-gpumel | ConvNeXt LoRA avg | 0.644 | 0.603 |
| E | run-e-convnext-attention | ConvNeXt LoRA attention | 0.585 | 0.571 |
| F | run-f-convnext-soundscape | ConvNeXt attention + SC data | 0.591 | 0.586 |
| G | run-g-convnext-weighted | ConvNeXt attention + SC + activity wt | 0.583 | — |
| H | run-h-convnext-classweight | ConvNeXt attention + SC + inv_sqrt | 0.596 | — |
| I | run-i-efficientnet-attention | EfficientNet attention + SC | 0.581 | — |
| J | run-j-efficientnet-classweight | EfficientNet attention + SC + inv_sqrt | 0.592 | — |
| K | run-k-efficientnet-scval | EfficientNet attention + SC val | 0.866 SC | 0.609 |
| L | run-l-efficientnet-cleaned | EfficientNet attention + teacher filter + SC val | 0.870 SC | 0.528 |
| M | run-m-efficientnet-sed | EfficientNet LoRA + SED head (GeM + dual Conv1d) | cmap 0.597 | 0.553 |
| M-v2 | re-export only | SED head with prob-space framewise max (match Run C inference) | — | pending |

**Best model: Run C (old) at 0.794 LB.** Every subsequent run has regressed on LB
despite improving offline metrics. However, Run M used logit-space blended inference
which dilutes confident predictions — different from Run C's probability-space
framewise max. Run M-v2 re-exports with Run C's inference strategy (sigmoid → max)
to test whether the SED head's frame predictions are actually better.

## Key findings

1. **Attention pooling critical**: framewise max at inference catches brief calls in soundscapes (0.794 vs 0.754 LB)
2. **Focal overfitting is the dominant problem**: val/cmap improves with more epochs but Kaggle LB degrades. Every run past epoch 2 has regressed on LB.
3. **SED head (Run M) regressed**: 0.553 LB despite cmap 0.597. The GeM + dual Conv1d + dual supervision architecture trained to epoch 4, but more parameters + more epochs = worse LB. Same pattern as K/L.
4. **Offline metrics are anti-correlated with LB**: higher cmap/auroc → lower Kaggle LB across runs C, K, L, M. The only model that generalizes is the 2-epoch attention model.
5. **EfficientNet > ConvNeXt on Kaggle**: 256 mels + smaller model generalizes better than 128 mels + 89M params
6. **bf16 emulated on RTX 6000**: fp16 gives 3x throughput (sm_75 has no native bf16)
7. **torch.compile works**: torch 2.11+cu126, needs setuptools+triton, compiles full LoRA backbone
8. **GPU mel > CPU mel**: sustained 96-100% GPU utilization, no idle gaps
9. **Kaggle metric**: macro AUROC excluding species with no positive labels (not standard torchmetrics)
10. **Activity scores predict calls**: 0.913 AUROC from logistic regression on acoustic indices
11. **ONNX export verified correct**: mel and output mismatch < 0.002

## Next improvements (from Murilo's exp 016-019 results)

Validated changes to port into this codebase, in priority order:

1. **AmplitudeToDB normalization** — replace `log(mel) + normalize` with
   `torchaudio.transforms.AmplitudeToDB` (decibel scale + dynamic range clamp).
   Floors quiet values to a flat silence instead of spreading noise across a wide
   negative range. The model gets cleaner contrast between calls and silence without
   having to learn where the noise floor is. Murilo measured +0.034 LB from this alone.
   One-line change in `compute_mel`.

2. **Mel resolution** — n_fft 1024→4096, n_mels 256→224. Higher n_fft = narrower
   frequency bins = better separation of bird calls. The backbone handles arbitrary
   input shapes (HF EfficientNet adapts internally, `adaptive_avg_pool2d` collapses
   spatial dims), so this is just a config change. Murilo measured +0.030 LB.

3. **SED head** — full SED formulation with GeM frequency pooling + attention +
   framewise max, rather than current attention pooling. This was the biggest
   architectural jump in Murilo's work (+0.060 LB over clip-level in exp 016).

4. **Soft CE loss** — replace focal BCE. Part of Murilo's best SED config (v4b, 0.859 LB).

Things Murilo tested that **did NOT help on LB** (skip these):
- MixUp (regressed LB twice despite improving offline metrics)
- Stacking multiple techniques (no additive gains)
- Fulldata training without epoch caps (regressed)
- Pseudo-labeling with weak teacher (exp 015, but revisit if teacher > student)

## Next step: Clean focal data with frame-level predictions

The trained attention model produces per-frame, per-species probabilities on focal
recordings. Use this to:
1. **Identify which frames contain the actual call** (vs silence/noise in labeled chunks)
2. **Hard-filter**: drop chunks where model says the labeled species isn't calling
3. **Soft-relabel**: replace binary clip labels with frame-level temporal predictions
4. Goal: cleaner training data → less overfitting → better soundscape generalization

This is complementary to pseudo-labeling soundscapes (exp 101 Strategy A), but
addresses the root cause: noisy focal labels that the model memorizes.

## Running

```bash
# Training
sbatch sbatch/train.sbatch configs/run-X.yaml --run-name my-run
sbatch sbatch/train.sbatch configs/run-X.yaml --run-name test --fast-dev-run

# ONNX export
sbatch sbatch/export-onnx.sbatch <checkpoint> <config> <run-name>

# Validation
sbatch sbatch/validate-onnx.sbatch <checkpoint> <config> <onnx-dir>

# Analysis
sbatch sbatch/analyze-activity.sbatch

# Kaggle submission
cd kaggle_kernel && kaggle kernels push -p .
kaggle competitions submit birdclef-2026 -k acmiyaguchi/birdclef-2026-efficientnet -f submission.csv -v N -m "message"
```

## Architecture

- **Backbone-agnostic**: auto-detects EfficientNet vs ConvNeXt from HF config
- **LoRA**: EfficientNet targets `[project_conv, expand_conv, reduce, expand]`, ConvNeXt targets `[pwconv1, pwconv2]`
- **Pooling**: avg (logits) or attention (SED-style, probabilities via framewise max at inference)
- **Loss**: focal BCE with optional class weights and activity sample weights
- **Data**: parquet-based chunk index + optional soundscape windows + no-call windows
- **Mel**: nnAudio on GPU with `@autocast(enabled=False)` fp32 wrapper for fp16 training
- **Validation**: focal split or soundscape split (50/50 by file)
- **Metric**: Kaggle-style macro AUROC excluding absent species

## sbatch environment

- `cuda/12.6.1`, `torch 2.11+cu126`, `triton`, `setuptools`
- `--gres=gpu:RTX_6000:1`, 6 CPUs, 60G mem
- Venv in `$TMPDIR/birdclef-$SLURM_JOB_ID/.venv`

## Key data

- Focal: 35,549 files, 206 species, 244,875 chunks × 5s (340h)
- Activity scores: `artifacts/focal_activity.parquet`
- Soundscape labels: 66 files, 739 labeled windows + 53 no-call
- No-call index: `artifacts/nocall_index.parquet`
- ONNX weights: `artifacts/weights/{model-name}/`
- Kaggle datasets: `acmiyaguchi/birdclef-2026-efficientnet-weights`, `birdclef-2026-deps`

## Directory structure

```
├── CLAUDE.md
├── proposal.md
├── configs/                    # YAML configs for each run
│   ├── run-a-frozen.yaml
│   ├── run-b-lora-gpumel.yaml
│   ├── run-d-convnext-gpumel.yaml
│   ├── run-e-convnext-attention.yaml
│   ├── run-f-convnext-soundscape.yaml
│   ├── run-g-convnext-weighted.yaml
│   ├── run-h-convnext-classweight.yaml
│   ├── run-i-efficientnet-attention.yaml
│   ├── run-j-efficientnet-classweight.yaml
│   └── run-k-efficientnet-scval.yaml
├── src/birdclef_102/
│   ├── model.py                # Backbone-agnostic classifier + LoRA
│   ├── data.py                 # Parquet loader + soundscape + no-call
│   ├── training.py             # Lightning module + Kaggle AUROC metric
│   ├── losses.py               # Focal BCE + class weights
│   ├── cli.py                  # Config-driven entry point
│   ├── spec_augment.py
│   └── gpu_monitor.py
├── scripts/
│   ├── export_onnx.py          # ONNX export (handles LoRA merge)
│   ├── validate_onnx.py        # Compare PyTorch vs ONNX predictions
│   ├── build_activity_scores.py
│   ├── build_nocall_index.py
│   ├── analyze_activity_vs_labels.py
│   ├── kaggle_metric.py        # Reference: official BirdCLEF metric
│   └── build_deps.sh           # Download Kaggle offline deps
├── sbatch/
│   ├── train.sbatch
│   ├── export-onnx.sbatch
│   ├── validate-onnx.sbatch
│   └── analyze-activity.sbatch
├── kaggle_kernel/
│   ├── submission.py           # MODEL_NAME selects model subdir
│   └── kernel-metadata.json
├── docs/
│   ├── architecture.md
│   ├── experiment-ladder.md
│   └── activity-scores.md
├── logs/                       # Per-run training logs + checkpoints
└── artifacts/
    ├── focal_activity.parquet
    ├── nocall_index.parquet
    ├── deps/                   # Kaggle offline wheels
    └── weights/                # ONNX exports per model
```
