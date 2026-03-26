---
title: "An example of training process (HGNetV2-B0 Baseline) — Tawara"
source: "https://www.kaggle.com/competitions/birdclef-2026/discussion/683822"
training_notebook: "https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-training"
inference_notebook: "https://www.kaggle.com/code/ttahara/birdclef-2026-hgnetv2-b0-baseline-inference"
author: "Tawara (ttahara, Kaggle Grandmaster)"
date_posted: "2026-03-22"
date_updated: "2026-03-24"
date_captured: "2026-03-26"
votes: 26
related: ["sed-model-comparison-tom.md", "sed-improvement-plan-681146.md"]
---

# HGNetV2-B0 Baseline — Tawara (Kaggle Grandmaster)

Shared training + inference notebooks for a 4-fold HGNetV2-B0 SED baseline.
Updated on 2026-03-24 to add AttnSEDHead and CustomLoss (v2).

## Scores

| Ver | Head | Loss | OOF CV (AUROC) | Public LB |
|-----|------|------|---------------|-----------|
| 1 | Linear | BCEWithLogitsLoss | 0.9574 | 0.856 (fold avg) |
| **2** | **AttnSED** | **0.5 clip + 0.5 timewise-LSE BCE** | **0.9626** | **0.859 (fold avg)** |

## Architecture

### Backbone: HGNetV2-B0

- `hgnetv2_b0.ssld_stage2_ft_in1k` from timm
- SSLD = Simple Semi-supervised Label Distillation (PaddlePaddle → timm port)
- Input: (B, 1, 256, 256) grayscale mel spectrograms
- `global_pool=""`, `num_classes=0` — uses raw feature maps
- drop_path_rate=0.0

### Head: AttnSEDHead

```python
class AttnSEDHead(nn.Module):
    def __init__(self, num_features, num_classes, dropout=0.1):
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.att_fc = nn.Linear(num_features, num_classes)
        self.cls_fc = nn.Linear(num_features, num_classes)

    def forward(self, h):
        """h: (B, C, Time) from GeM-pooled backbone"""
        h = h.permute(0, 2, 1)               # (B, Time, C)
        h = self.pre_fc(h)                    # (B, Time, C)
        att_w = torch.tanh(self.att_fc(h))    # (B, Time, N_CLASSES)
        att_w = F.softmax(att_w, dim=1)       # softmax over Time
        timewise_logits = self.cls_fc(h)      # (B, Time, N_CLASSES)
        logits = (att_w * timewise_logits).sum(dim=1)  # (B, N_CLASSES)
        return logits, timewise_logits
```

Key differences from our AttBlockV2:
- **Attention**: `tanh(Linear) → softmax` (same concept, but single linear instead
  of shared conv for att and cls)
- **No sigmoid on framewise**: Returns raw logits for both clip and frame paths
  (no sigmoid→logit roundtrip like our AttBlockV2)
- **Classification**: separate `cls_fc` linear, not shared with attention conv

### GeM Pooling

```python
class GeMPooling(nn.Module):
    def __init__(self, init_p=3.0, mean_axis=(1, 2), eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(init_p))
        self.mean_axis = mean_axis
        self.eps = eps

    def forward(self, h):
        p = self.p.clip(min=1.0)
        h = h.clip(min=self.eps).pow(self.p)
        h = h.mean(dim=self.mean_axis)
        h = h.pow(1.0 / self.p)
        return h
```

Applied with `mean_axis=2` to pool frequency dimension: (B, C, F, T) → (B, C, T).

## Loss: CustomBCEWithLogitsLoss

```python
class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, timewise_weight=0.5, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.timewise_weight = timewise_weight
        self.temperature = temperature

    def forward(self, logits, timewise_logits, label):
        loss = super().forward(logits, label)
        logits_timeaxis_lse = self.temperature * torch.logsumexp(
            timewise_logits / self.temperature, axis=1)
        loss_timeaxis_lse = super().forward(logits_timeaxis_lse, label)
        return (1 - self.timewise_weight) * loss + self.timewise_weight * loss_timeaxis_lse
```

- **Clip loss** (weight 0.5): Standard BCE on attention-weighted logits
- **Timewise loss** (weight 0.5): BCE on LogSumExp of frame logits
  - LSE with temperature=0.1 acts as a smooth max — focuses on the frame with
    the strongest activation for each class
  - Different from our approach: we use `framewise_logits.max(dim=-1)` (hard max),
    they use `logsumexp` (smooth max)

## Mel Spectrogram

```python
mel_spectrogram_params = dict(
    sample_rate=32_000,
    n_fft=2048,
    win_length=626,
    hop_length=313,
    f_min=20,
    n_mels=256,
    power=2.0,
    center=True,
    pad_mode="reflect",
    norm="slaney",
    mel_scale='htk',
)
top_db = 80
lms_shape = (256, 256)  # Resize after mel computation
```

Normalization pipeline (same for training and inference):
1. MelSpectrogram → (B, 256, time_steps)
2. AmplitudeToDB (top_db=80) → dB scale
3. Resize to (256, 256) via torchvision
4. Per-sample min-max normalization to [0, 1]
5. Add channel dim → (B, 1, 256, 256)

### Comparison to our mel config

| Parameter | Tawara | Our v4b | Our v4a |
|-----------|--------|---------|---------|
| n_fft | 2048 | 4096 | 4096 |
| n_mels | 256 | 224 | 224 |
| hop_length | 313 | 320 | 320 |
| f_min | 20 | 0 | 0 |
| win_length | 626 | n_fft (default) | n_fft (default) |
| norm | slaney | — | — |
| Normalization | AmplitudeToDB + min-max | AmplitudeToDB + standardize + min-max | log + AudioSet stats |
| Resize | Yes (to 256×256) | No | No |
| Segment | 5s | 10s | 10s |
| Input shape | (1, 256, 256) | (1, 224, 1000) | (1, 224, 1000) |

Notable: Tawara uses n_fft=2048 (not 4096) with n_mels=256 (more than our 224) and
**resizes** to a fixed 256×256 shape. We preserve native temporal resolution (1000 frames
for 10s at hop=320).

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | hgnetv2_b0.ssld_stage2_ft_in1k |
| Epochs | 20 |
| Warmup | 5 epochs (MixUp disabled during warmup) |
| Batch size | 64 |
| Optimizer | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | OneCycleLR (max_lr=5e-4, min_lr=5e-6, pct_start=0.25) |
| AMP | Enabled (fp16) |
| MixUp | alpha=1.0, theta=0.8 (spectral domain, after mel) |
| Loss | CustomBCEWithLogitsLoss (timewise_weight=0.5, temperature=0.1) |
| head_dropout | 0.2 |
| CV | 4-fold Multi-Label Stratified Group K-Fold |
| Training time | ~30 min/fold, ~2 hours total |

### MixUp Implementation

```python
class MixUp(nn.Module):
    def __init__(self, alpha=0.5, theta=1):
        super().__init__()
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        self.theta = theta

    def forward(self, lms, label):
        lambda_tensor = self.beta_dist.sample((batch_size,)).to(device)
        lambda_tensor = torch.maximum(lambda_tensor, 1 - lambda_tensor)
        shuffle_idxs = torch.randperm(batch_size).to(device)

        lms = lambda_tensor[..., None, None, None] * lms + \
              (1 - lambda_tensor[..., None, None, None]) * lms[shuffle_idxs]
        label = lambda_tensor[..., None] * label + \
                (1 - lambda_tensor[..., None]) * label[shuffle_idxs]
        label[label >= self.theta] = 1
        return lms, label
```

- Applied in **spectral domain** (after mel transform), not raw waveform
- Beta(1.0, 1.0) = Uniform[0,1], then `max(λ, 1-λ)` → range [0.5, 1.0]
- Labels mixed proportionally, then binarized at θ=0.8
- Disabled during first 5 warmup epochs

## Data Pipeline

### Sources
- **train_audio**: Focal recordings (pre-converted OGG → WAV for speed)
- **train_soundscapes**: 60s recordings split into species-contiguous segments
- **Labels**: primary_label + secondary_labels (both used as multi-hot targets)

### Soundscape Handling
- Consecutive labels for same species are merged into single segments
- Each segment saved as separate .wav file: `{audio_id}_{start}-{end}.wav`
- Both train_audio and soundscape segments combined into unified training DataFrame

### CV Strategy: Multi-Label Stratified Group K-Fold (K=4)
- Groups by audio_id (same file always in same fold)
- Assigns groups to folds minimizing per-class label count variance
- Processes groups in order of decreasing label distribution variance

### Audio Loading Optimization
- **OGG → WAV conversion**: Pre-converts all training audio for faster I/O
- **Partial file reading**: Uses `soundfile.SoundFile.seek()` to load only 5s chunks
- **Training**: Random crop from longer files, random-position padding for short files
- **Validation**: Head crop (first 5s), front-padding for short files

## Inference Pipeline

### Segment Processing
- 60s soundscapes → 12 non-overlapping 5s segments
- Per-segment mel spectrogram (NOT full-signal)
- Batched in groups of 12 (one full soundscape)
- Mel computation parallelized via joblib (n_jobs=4)

### OpenVINO Inference
- Models converted: PyTorch → ONNX (opset 11) → OpenVINO XML
- CPU inference with PERFORMANCE_HINT="THROUGHPUT", NUM_STREAMS=2
- AsyncInferQueue with 16 concurrent requests
- Callback-based ordered result collection

### Ensemble (4-fold)
- Each fold produces sigmoid probabilities + rank-normalized scores
- **Rank normalization**: Per-class rank / n_samples → scale-invariant [0,1]
- Final prediction: mean of rank-normalized scores across 4 folds

### Post-Processing
- **None** — no TTA, no temporal smoothing, no TopN
- Pure model prediction with rank-averaged ensemble

## Key Insights

1. **AttnSEDHead + CustomLoss improves over Linear**: 0.856 → 0.859 LB (+0.003)
   with dual-path BCE loss using LogSumExp for the frame path.

2. **LogSumExp vs hard max for frame aggregation**: Temperature-scaled LSE (T=0.1)
   acts as a smooth maximum that's differentiable everywhere, potentially more stable
   than our hard `framewise_logits.max(dim=-1)`.

3. **HGNetV2-B0 is competitive with EfficientNet-B0**: 0.859 LB with a different
   backbone family. Our EfficientNet-B0 v4b achieves 0.859 with more mel resolution
   (n_fft=4096 vs 2048) but fewer mel bins (224 vs 256).

4. **No post-processing needed at 0.859**: Tawara achieves the same LB as our v4b
   without any TTA, smoothing, or TopN — pure model + ensemble.

5. **Rank normalization for ensemble**: Scale-invariant averaging avoids issues when
   different folds produce different logit scales.

6. **BCE loss works well with SED heads**: Confirms that BCE (not CE/soft_ce) is the
   natural loss for multi-label SED, supporting our v4g/v4h re-ablation direction.

7. **Fast training**: 20 epochs, 30 min/fold on Kaggle GPU → 2 hours total.
   We use 50 epochs with patience=10, taking ~4 hours per run.

## Relevance to Our Exp 017

| Tawara's approach | Our status | Implications |
|-------------------|-----------|--------------|
| BCE loss + SED head | Testing in v4g/v4h | Validates BCE as natural SED loss |
| AttnSEDHead (tanh+softmax, no sigmoid roundtrip) | Our AttBlockV2 has sigmoid→logit roundtrip | Consider removing the logit conversion |
| LogSumExp frame aggregation (T=0.1) | We use hard max | LSE may be more stable; worth testing |
| n_fft=2048, n_mels=256 | n_fft=4096, n_mels=224 | More mels might help; we have more freq resolution |
| 5s segments, resize to 256×256 | 10s segments, native 224×1000 | Our longer context + native resolution may be better |
| 4-fold ensemble + rank avg | Single model | Ensemble is a future opportunity |
| No post-processing | TTA + smoothing + TopN + metadata priors | Our pp2 adds very little on v4b (+0.001) |
| 0.859 LB (4-fold ensemble) | 0.859 LB (single model v4b) | We match their ensemble with a single model |
