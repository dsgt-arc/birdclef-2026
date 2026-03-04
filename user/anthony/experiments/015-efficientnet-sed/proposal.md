# Experiment 015: EfficientNet SED with Soft Labels

- **Status**: PROPOSED
- **Author**: Anthony
- **Date**: 2026-03-04
- **Dependencies**: exp 012 (baseline), exp 013 (soft labels), exp 014 (soft label training)

## Hypothesis

Adding Sound Event Detection (SED) with per-frame supervision from 1s-resolution
soft labels will improve over clip-level prediction (exp 012: 0.336 val/cmap,
0.649 Kaggle private) by:
1. Training frame-level predictions that capture temporal bird call patterns
2. Using attention pooling (train) / framewise max (inference) — the dominant
   pattern in top BirdCLEF 2025 solutions (+0.03-0.05 gain)
3. Leveraging temporal structure in soft labels rather than discarding it via averaging

## Architecture

### Training Pipeline

```
Raw Audio (B, 160000)  <- 5 seconds @ 32kHz
         |
         v
+-----------------+
|  nnAudio Mel     |  log-normalize, pad/trim
+--------+--------+
         | (B, 1, 256, 625)
         v
+-----------------+
|  SpecAugment     |  freq/time masking (train only)
+--------+--------+
         | (B, 1, 256, 625)
         v
+-----------------+
|  EfficientNet-B1 |  frozen, 6.5M params, no gradients
|  (pretrained)    |
+--------+--------+
         | (B, 1280, 8, 20)
         |        ^    ^
         |      freq  time
         v
    mean(dim=2)        <- collapse 8 freq bins
         | (B, 1280, 20)
         v
    AvgPool1d(4)       <- 20 time steps -> 5 frames
         | (B, 1280, 5)
         v
    transpose(1,2)
         | (B, 5, 1280)       5 frames = 5 x 1s
         |
    +----+----+
    v         v
+--------+ +--------+
|frame   | |att     |   Both are Linear(1280, 206)
|head    | |head    |   frame_head has Dropout(0.3)
+---+----+ +---+----+
    |          |
    | (B,5,C)  | (B,5,C)
    |          v
    |     tanh -> softmax(dim=1)     <- attention weights per class
    |          | (B,5,C)              sum to 1.0 across 5 frames
    |          |
    v          v
 frame      att_weights * sigmoid(frame_logits)
 logits        |
    |          sum(dim=1)
    |          |
    |          v
    |       clip_pred (B, C)  <- probabilities [0,1]
    |
    v
 frame_logits (B, 5, C)  <- raw logits
```

~528K trainable params (frame_head 264K + att_head 264K), backbone frozen.

### Why 20 -> 5 frames?

```
5s audio window
|<------------------------------------------->|

Backbone output: 20 time steps (~250ms each)
|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|

AvgPool1d(4): pool groups of 4 -> 5 frames (1s each)
|----1s----|----1s----|----1s----|----1s----|----1s----|

Soft labels: 5 tokens (1s each) -- exact match!
|----1s----|----1s----|----1s----|----1s----|----1s----|
   0.02       0.85       0.91       0.03       0.01
              ^ bird calling here ^
```

Soft labels have genuine 1s resolution from exp 013 fusion. Pooling 20->5
preserves this resolution cleanly (20/4 = 5). Expanding 5->20 would just
duplicate values.

## Training

### Dual Loss

```
 Soft Labels from exp 013          train.csv hard labels
 (1s resolution parquets)          (val only)
         |                                |
         v                                |
 +---------------+                        |
 | frame_labels  | (B, 5, C)              |
 | per-second    |------+                 |
 | soft targets  |      |                 |
 +-------+-------+      |                 |
         |              |                 |
    mean(dim=0)         |                 |
         |              |                 |
         v              v                 v
   clip_labels     frame_logits      clip_labels
    (B, C)          (B, 5, C)          (B, C)
         |              |                 |
         v              v                 v
   +----------+   +----------+     +----------+
   |clip_loss |   |frame_loss|     | val_loss  |
   |FocalBCE  |   |FocalBCE  |     |FocalBCE  |
   |(probs)   |   |(logits)  |     |(probs)   |
   +----+-----+   +----+-----+     +----------+
        |               | x alpha
        |               |
        v               v
      loss = clip_loss + alpha * frame_loss
```

- `frame_loss`: FocalBCE(frame_logits, frame_labels, mask=frame_mask) — logits mode
- `clip_loss`: FocalBCE(clip_pred, clip_labels) — probability mode
- `frame_labels`: (B, 5, C) per-second soft labels from fused-v2 parquets
- `clip_labels`: (B, C) mean of frame_labels
- alpha = 1.0

Val uses hard labels from train.csv (same split as exp 012/014).
Primary metric: val/auroc (both auroc and cmap are tracked).

## Inference (Kaggle)

```
 5s window -> mel -> backbone -> pool -> frame_head
                                         |
                                  frame_logits (1, 5, C)
                                         |
                                    sigmoid
                                         |
                                  frame_probs (1, 5, C)
                                         |
                                    max(dim=0)        <- NOT attention!
                                         |              framewise max is
                                   clip_prob (C,)       more robust for
                                         |              brief calls
                                         v
         +----------------------------------------------+
Window 1 |  clip_prob_1                                 |
Window 2 |  clip_prob_2     temporal smoothing           |
Window 3 |  clip_prob_3     [0.2, 0.6, 0.2]             |
  ...    |                                              |
         +----------------------------------------------+
                          |
                    smoothed predictions
```

- ONNX exports frame_logits only (no attention branch)
- Per 5s window: sigmoid(frame_logits) -> max over 5 frames -> clip prob
- Post-processing: temporal smoothing [0.2, 0.6, 0.2] across adjacent 5s windows

**Key insight**: Train with attention (gradients flow to all frames), infer with
framewise max (catches brief calls better). This train/infer divergence is the
dominant pattern from top BirdCLEF 2025 solutions.

## Success Criteria

1. val/auroc improves over exp 012 baseline
2. Kaggle private score > 0.649 (exp 012 baseline)
3. Both clip_loss and frame_loss decrease during training

## Key Design Decisions

1. **Pool 20->5 frames** (not expand 5->20 labels): soft labels have genuine 1s resolution
2. **Attention for training, framewise max for inference**: matches top Kaggle pattern
3. **Frozen backbone**: focused experiment on SED head value; LoRA is a follow-up
4. **No feature caching**: backbone is fast, SpecAugment must precede it
5. **AUROC as primary metric**: more stable than cmAP for soft-label training
