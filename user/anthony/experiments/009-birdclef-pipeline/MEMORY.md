# Experiment 009 Memory

## Architecture Decisions

### Clip-level mean pooling (not SED)
- BirdCLEF focal recordings have species labels but NO temporal event annotations
- SED token-level loss was training on noisy uniform labels (every token labeled as species)
- Switched from `SSAMBASEDClassifier` (attention pooling) to `SSAMBAClassifier` (mean pooling)
- `forward()` returns `clip_logits (B, C)` — single return value, not tuple
- `forward_windows(waveforms, window_size=500)` for inference: splits tokens into 5s windows, mean-pools each, applies head → `(B, num_windows, C)` logits
- Loss: single focal BCE on logits (`from_logits=True`), no SED token loss

### Pretraining: MPG with span masking
- **MPC (NCE contrastive) was degenerate** with 128x1 tokenization — adjacent mel frames nearly identical, MPC acc stuck at random (0.001-0.006)
- **JiTTER block shuffle rejected** — shuffling fights Mamba's sequential state dynamics. Even with positional embeddings, hidden state is built from scanning garbled sequence.
- **Current approach**: L = L_MPG + jitter_weight × L_JiTTER (jitter_weight=0 by default → pure MPG)
- **MPG**: Reconstruction of masked tokens (validated by Yadav 2025 with Mamba)
- **Span masking**: Variable-length spans from Uniform[5, 15] tokens (50-150ms). Contiguous spans preserve Mamba's sequential state in visible regions. Variable lengths prevent rhythmic artifacts.
- **Grid jitter**: Random 0-5 frame (0-50ms) temporal offset of mel spectrogram. Translation augmentation, Mamba-compatible.
- **Per-patch normalization**: `(patch - mean) / std` per 128-dim patch → forces model to predict spectral shape, not loudness
- **forward() returns dict** with keys: `loss_mpg`, `loss_jitter`, `monitor_mse`, `monitor_l1`, `viz_mel`, `viz_mask`, `viz_pred`, `viz_target`

### Model Parameters
- `mask_span_min`/`mask_span_max`: Variable span lengths (default 5/15)
- `loss_fn`: "mse" or "l1" (default "mse")
- `decoder_depth`: 1 (linear, SimMIM style) or 2 (MLP with ReLU, default)
- `normalize_targets`: Per-patch normalization (default True)
- `grid_jitter`: Max temporal offset in mel frames (default 5, 0=disabled)
- `mask_mode`: "span" or "random" (default "span")

### Evaluation Methodology
- **Cannot compare MSE vs L1 loss by absolute values** — different units/scales. Always log both `monitor_mse` and `monitor_l1` as cross-metrics.
- **Cannot compare linear vs MLP decoder by loss** — linear is strictly less expressive. If H's loss is within 5-10% of A, pick H (encoder absorbed complexity).
- **Spectrogram visualization**: Log original/masked/reconstructed trios to TensorBoard every N epochs. Check for blurriness (MSE) vs sharpness (L1).
- Lightning module logs: `train/loss`, `train/loss_mpg`, `train/loss_jitter`, `train/monitor_mse`, `train/monitor_l1`, `train/grad_norm`

## Overfit Ablation Results (Job 3952874)
- **Condition A** (baseline: span, norm=true, mask=78%): Epoch 52, loss=0.448
- **Condition B** (no-norm: norm=false): Epoch 52, loss=0.038 — lower absolute value expected (raw mel patches have low variance for quiet regions)
- **Condition D** (low-mask: mask=50%): Epoch 54, loss=0.444 — nearly identical to A, 78% masking is fine
- Note: A vs B absolute values not comparable; B's lower loss doesn't mean better learning

## Mini-grid (real data, 10 epochs each)
- **A**: Baseline — variable spans [5,15], MSE, MLP decoder
- **G**: L1 loss — same as A but loss_fn=l1
- **H**: Linear decoder — same as A but decoder_depth=1 (SimMIM philosophy)
- Configs: `configs/minigrid-{a,g,h}.yaml`
- Sbatch: `sbatch/pretrain-minigrid.sbatch` (array 0-2, 2h wall time)

## Prior Overfit Results
- **Job 3952833**: Random masking + JiTTER. Loss 0.559→0.036 (epoch 154). JiTTER works mechanically but likely hurts representations.
- **Job 3952847**: Span masking + grid jitter. Loss 0.332→0.020 (epoch 129). Lower loss, faster convergence.

## Data Stats (BirdCLEF 2025)
- 206 species, 28,564 focal recordings → 103,738 train chunks (10s each)
- 9,726 soundscapes → 58,356 chunks (10s each)
- Train/val split: 25,761 / 2,803 recordings (90/10, stratified)

## Model
- SSAMBA Tiny: 192-dim, 24 layers, 128×1 patch size
- Pretrain model: 7.1M params (backbone + mask_embed + gpredlayer)
- Finetune model: 7.0M params (head: Dropout(0.3) → Linear(192, 206))

## Key Files
- Model: `src/birdclef_009/models/pretrain.py` (SSAMBAPretrainModel — MPG)
- Model: `src/birdclef_009/models/finetune.py` (SSAMBAClassifier)
- Training: `src/birdclef_009/training/pretrain.py` (PretrainLightningModule — cross-metrics + viz)
- Training: `src/birdclef_009/training/finetune.py` (FinetuneLightningModule)
- CLI: `src/birdclef_009/cli/pretrain.py`, `cli/finetune.py`
- Tests: 51 tests (job 3953043 pending)
- Reference: `docs/references/deep-research/2026-02-14-frame-level-ssl-pretraining.md`
