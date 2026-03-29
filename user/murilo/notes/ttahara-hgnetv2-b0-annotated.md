# ttahara HGNetV2-B0 Baseline — Annotated Reference

Source: `kaggle kernels pull ttahara/birdclef-2026-hgnetv2-b0-baseline-training`
Inference: `kaggle kernels pull ttahara/birdclef-2026-hgnetv2-b0-baseline-inference`
Score: **0.884 LB** (4-fold ensemble, OpenVINO CPU inference)
Extracted code: `user/murilo/notes/ttahara-hgnetv2-b0-training.py`, `*-inference.py`

---

## Architecture

### Backbone
- **HGNetV2-B0** (`hgnetv2_b0.ssld_stage2_ft_in1k` from timm)
- Input: 1 channel (mono mel spectrogram)
- `global_pool=""`, `num_classes=0` — backbone used as feature extractor only
- No LoRA/adapter — full fine-tuning

### Head: LSEHead (Log-Sum-Exp Pooling)
```
backbone(x)        → (B, C, Freq, Time)
mean(axis=Freq)    → (B, C, Time)       # average over frequency
transpose          → (B, Time, C)
Linear(C → C) + ReLU + Dropout(0.5) + Linear(C → 234)
                   → (B, Time, 234)     # per-frame logits
LSEPooling(axis=Time, T=1.0)
                   → (B, 234)           # clip-level logits
```

**LSE Pooling**: `T * (logsumexp(x/T, axis=time) - log(n_time_frames))`
- Smooth approximation to max pooling (T→0 = max, T→∞ = mean)
- T=1.0 → balanced between max and mean
- Better than hard max for detecting brief calls in long windows

### Alternative Head: AttnSEDHead (UPDATE1, not used in final)
- Pre-FC: Linear(C→C) + ReLU + Dropout(0.1)
- Attention: tanh(Linear(C→234)) → softmax over time
- Classification: Linear(C→234) per frame
- Output: attention-weighted sum
- Similar to our AttHead but with tanh activation (not sigmoid)

## Mel Spectrogram
```python
mel_spectrogram_params = dict(
    sample_rate=32_000,
    n_fft=2048,
    win_length=626,        # ~20ms window
    hop_length=313,        # ~10ms hop → 512 time frames for 5s
    f_min=20,
    # f_max not set → defaults to sr/2 = 16kHz
    n_mels=256,
    power=2.0,
    center=True,
    pad_mode="reflect",
    norm="slaney",
    mel_scale="htk",
)
top_db = 80
lms_shape = (256, 256)    # resize to square 256×256
```

**Normalization**: AmplitudeToDB(power, top_db=80) → resize to (256,256) → per-sample min-max to [0,1]

Key differences from our setup:
- n_fft=2048 (ours: 4096) — lower frequency resolution
- n_mels=256 (ours: 224) — more mel bands
- Resize to fixed (256,256) — we don't resize
- Per-sample min-max normalization — we use AmplitudeToDB + standardize

## Training

### Data
- **Focal recordings**: train_audio converted to .wav for fast loading
- **Soundscape segments**: train_soundscapes split by annotation boundaries
  - NOT fixed 5s windows — variable-length segments based on label spans
  - e.g., if a bird calls from 5s-15s, that's one 10s segment
- **Both concatenated** into a single training DataFrame
- **Labels**: multi-hot (primary_label + secondary_labels)
- **CV**: MultiLabel Stratified GroupKFold (4 folds, grouped by audio_id)

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch size | 64 |
| Optimizer | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=5e-4, min_lr=5e-6, warmup=5 epochs) |
| Loss | BCEWithLogitsLoss |
| MixUp | alpha=1.0, theta=0.8 (on spectrogram, after warmup) |
| AMP | True (mixed precision) |
| Segment | 5s (random crop train, head crop val) |
| Workers | 16 |

### MixUp Details
- Applied on log-mel spectrogram (not waveform)
- Beta(1.0, 1.0) = Uniform distribution for lambda
- `lambda = max(lambda, 1-lambda)` — dominant sample ≥50%
- Labels: `label_mix = lam * y1 + (1-lam) * y2; label_mix[>=0.8] = 1.0`
- **theta=0.8 clipping** — if mixed label ≥0.8, snap to 1.0
- Skipped during warmup (first 5 epochs)

### CustomBCEWithLogitsLoss (UPDATE1/2, not used in final LSEModel)
- Dual loss: `(1-w) * BCE(clip_logits, y) + w * BCE(LSE(frame_logits), y)`
- timewise_weight=0.5, temperature=1.0
- Only used with AttnSEDModel, not the final LSEModel

## Inference

### OpenVINO Conversion
- Torch → ONNX (opset 11, dynamic batch) → OpenVINO IR (.xml)
- CPU inference with `PERFORMANCE_HINT: THROUGHPUT`, 4 threads, 2 streams
- Async inference queue with 16 requests

### Ensemble
- 4-fold average of sigmoid(logits) — simple probability averaging
- Option for rank averaging (disabled in default: `RANK_AVG = False`)

### Pipeline
1. Read 60s soundscape → split into 12×5s segments
2. Batch 12 segments → log-mel spectrogram transform
3. OpenVINO async inference → logits
4. sigmoid(logits) → 4-fold average → submission

## OOF Scores (from notebook)
- GPU model (torch): reported but values not in extracted code
- OpenVINO model (CPU): reported but values not in extracted code

## Key Design Decisions

1. **HGNetV2-B0 over EfficientNet-B0**: Faster inference, similar accuracy.
   HGNet uses heavy re-parameterization and is optimized for deployment.

2. **LSEHead over attention**: Simpler, no learnable attention weights.
   LSE pooling handles sparse events well (approximates max for rare calls).

3. **Variable-length soundscape segments**: Segments follow annotation boundaries
   rather than fixed 5s windows. This means training examples can be 5-15s+
   depending on how long the species vocalizes.

4. **Per-sample min-max normalization**: Each spectrogram normalized independently
   to [0,1]. Robust to volume differences between recordings.

5. **OpenVINO for CPU inference**: ~2-4× faster than PyTorch on CPU.
   Critical for staying within Kaggle's 90-min kernel budget.

6. **4-fold CV**: More folds = better generalization signal, at cost of
   longer training (4×30min = 2h on Kaggle GPU).
