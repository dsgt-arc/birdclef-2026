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

## Mini-grid Results (Job 3953050, TIMED OUT at 2h — 3/10 epochs completed)
- **A** (MSE+MLP): loss 0.767→0.456→0.371, monitor_mse=0.371, monitor_l1=0.450
- **G** (L1+MLP): loss 0.664→0.511→0.446, monitor_mse=0.371, monitor_l1=0.446
- **H** (MSE+Linear): loss 0.777→0.440→0.356, monitor_mse=**0.356**, monitor_l1=**0.439**
- Throughput: ~28 samples/sec, GPU 100%, ~40 min/epoch (911 steps × batch 64)
- **H (linear decoder) leads on both cross-metrics** — encoder absorbs complexity (SimMIM philosophy confirmed)
- **A vs G tied** on cross-metrics after 3 epochs — too early to differentiate
- Configs: `configs/minigrid-{a,g,h}.yaml`

## Full Pretrain Runs (IN PROGRESS)

### pretrain-full — from scratch
- **Config**: L1 loss + linear decoder, random init
- `configs/pretrain-full.yaml`, `sbatch/pretrain-full.sbatch`
- 100 epochs, 24h wall time (~35 epochs/run), auto-resume via `--ckpt-path last`
- Jobs: 3957327 (epochs 0–34) → 3973084 (continuation, epochs 35+)
- Checkpoints: `logs/pretrain-full/checkpoints/`
- TensorBoard: `logs/pretrain-full/lightning_logs/run/`
- **Epoch 34 metrics**: loss=0.411, monitor_MSE=0.324, monitor_L1=0.411
- Curve shape: steep drop epochs 0–10 (0.67→0.45), then gradual (0.45→0.41 by epoch 34)
- Still decreasing — no plateau, worth continuing to 100
- Plot: `artifacts/plots/pretrain_curves.png` (via `scripts/plot_pretrain.py`)

### pretrain-warm — AudioSet backbone
- **Config**: L1 loss + linear decoder, initialized from `ssamba_tiny_400.pth`
- `configs/pretrain-warm.yaml`, `sbatch/pretrain-warm.sbatch`
- Mamba layers (96.9%) from AudioSet pretrained; patch_embed + pos_embed re-initialized (16x16→128x1)
- Original job 3957368 FAILED (28s) — `UnboundLocalError: resuming` unbound, now fixed
- **Resubmitted as Job 3972957** — running
- Checkpoints: `logs/pretrain-warm/checkpoints/`
- TensorBoard: `logs/pretrain-warm/lightning_logs/run/`

### Resume infrastructure
- CLI: `--ckpt-path last` auto-finds `last.ckpt`, `--ckpt-dir` sets checkpoint location
- `--pretrained` skipped on resume (Lightning checkpoint restores full state)
- `save_last=True` + `every_n_epochs=1` — at most 1 epoch lost on timeout
- Fixed TensorBoard version (`version="run"`) for continuous metrics across restarts
- Resubmit same sbatch on timeout to continue

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

## Kaggle Submission Pipeline (WORKING)
- ONNX export: `scripts/export_onnx.py` — `TokenLogitsWrapper` wraps backbone+head, outputs `(B, T, C)` per-token logits
- Inference: `scripts/kaggle_inference.py` — librosa mel, ONNX Runtime + mamba-onnx custom ops, windowed pooling
- Deps: `scripts/build_deps.sh` — cp312 wheels, `manylinux_2_28` + `manylinux_2_17` platforms
- Upload: `scripts/upload_kaggle.py` — creates/versions Kaggle datasets (detects "already in use" in stdout)
- Kernel: `kaggle_kernel/submission.py` + `kernel-metadata.json`
- **Kernel v6 completed** (13s total, 1.5s ONNX session init) — pipeline end-to-end verified
- Submission to BirdCLEF 2025 pending scoring (epoch 1 checkpoint, cmAP ~0.014 — pipeline test only)
- To re-submit with better model: re-export ONNX → upload weights → push kernel → submit

## Finetune Runs
- **Job 3960497**: AudioSet backbone + BirdCLEF finetune, epoch 1 best (cmAP 0.0135)
- **Finetune config updated**: added SpecAugment (freq_mask=20, time_mask=50, 2+2 masks)
- **Finetune sbatch updated**: uses `--pretrained-ckpt` for AudioSet backbone init
- Need better pretrained checkpoint or longer finetune to improve cmAP

## Key Files
- Model: `src/birdclef_009/models/pretrain.py` (SSAMBAPretrainModel — MPG)
- Model: `src/birdclef_009/models/finetune.py` (SSAMBAClassifier)
- Training: `src/birdclef_009/training/pretrain.py` (PretrainLightningModule — cross-metrics + viz)
- Training: `src/birdclef_009/training/finetune.py` (FinetuneLightningModule)
- CLI: `src/birdclef_009/cli/pretrain.py`, `cli/finetune.py`
- ONNX: `scripts/export_onnx.py`, `scripts/kaggle_inference.py`
- Upload: `scripts/upload_kaggle.py`, `scripts/build_deps.sh`
- Tests: 51 tests (job 3953043 pending)
- Reference: `docs/references/deep-research/2026-02-14-frame-level-ssl-pretraining.md`
