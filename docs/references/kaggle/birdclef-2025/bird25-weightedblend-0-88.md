---
slug: "kumarandatascientist/bird25-weightedblend-0-88"
type: "notebook"
competition: "birdclef-2025"
date_captured: "2026-02-17"
votes: 577
---

# Kaggle Summary: Bird25 WeightedBlend 0.88

**Source**: https://www.kaggle.com/code/kumarandatascientist/bird25-weightedblend-0-88
**Competition**: BirdCLEF 2025
**Score**: Public LB 0.878

## Summary

This submission achieves a strong public leaderboard score by blending predictions from two different Sound Event Detection (SED) models with a 75/25 weighted ensemble. The approach combines an NFNet-based model with a SEResNeXt-based model, applying temporal smoothing and low-rank power adjustment post-processing to both model outputs.

## Approach

### Architecture

**Model 1 (NFNet)**: Uses `eca_nfnet_l0` as the backbone with a custom SED head architecture featuring attention-based pooling. The architecture includes an AttBlockV2 module that computes attention weights across the temporal dimension and performs weighted aggregation of frame-level predictions to produce clip-level outputs.

**Model 2 (SEResNeXt)**: Uses `seresnext26t_32x4d` as the backbone with a similar SED architecture. This model is configured for 10-second input windows during training but uses 5-second segments during inference with test-time augmentation (TTA) by shifting the temporal window.

Both models share similar architectural patterns:
- BatchNorm layer before the encoder
- Frame-level feature extraction followed by temporal pooling
- Channel smoothing using both max and average pooling
- FC layer with ReLU activation before attention pooling
- Attention-based aggregation producing clip-wise outputs

### Data Pipeline

**NFNet Model**:
- 32kHz sample rate with 5-second segments
- Mel spectrogram: 1024 FFT, 1024 window, 512 hop length
- 128 mel bins covering 40Hz to 15kHz
- HTK mel scale with Slaney normalization
- Log-mel transformation with mean/std normalization per spectrogram
- Processes 60-second soundscapes as 12 consecutive 5-second chunks

**SEResNeXt Model**:
- Same 32kHz sample rate and mel spectrogram parameters
- Uses 10-second training windows with 5-second target duration
- Inference applies padding/cropping to create overlapping context windows
- Audio is triplicated and concatenated for boundary handling during chunk extraction

### Training

Training details are not included in this inference-only notebook. Both models load pre-trained weights from uploaded datasets. The NFNet model uses 3 fold checkpoints averaged during inference, while the SEResNeXt model uses a single checkpoint.

### Inference / Post-Processing

**NFNet Pipeline**:
1. Extract mel spectrograms for all 12 5-second chunks
2. Run ensemble inference across 3 folds
3. Apply power adjustment to low-ranked species (top_k=30, exponent=2)
4. Average predictions across folds
5. Apply temporal smoothing with neighbor weighting

**SEResNeXt Pipeline**:
1. Extract overlapping 10-second windows centered on each 5-second segment
2. Apply TTA by shifting temporal attention window (±2 frames)
3. Weighted TTA combination: 50% center, 25% left-shift, 25% right-shift
4. Use framewise maximum instead of attention-weighted sum for final predictions
5. Apply temporal smoothing

**Temporal Smoothing**: Both models apply weighted averaging across consecutive 5-second segments:
- First/last segments: 80% current + 20% neighbor
- Middle segments: 60% current + 20% previous + 20% next

**Power Adjustment**: Low-confidence species (those not in the top 30 by maximum prediction across time) have their probabilities raised to the power of 2, effectively boosting weak signals while suppressing noise.

**Final Ensemble**: Weighted blend of 75.3% SEResNeXt + 24.7% NFNet predictions.

## Key Ideas

- **Low-rank power adjustment**: Applying power transformation selectively to species with low maximum predictions helps surface weak but real signals while suppressing uniform noise patterns
- **Asymmetric ensemble weights**: Despite using two models, heavily weighting one model (75/25 split) suggests the SEResNeXt model provides substantially better base predictions
- **Multi-scale TTA for temporal models**: Shifting the attention window during inference provides robustness to boundary effects in temporal aggregation
- **Temporal smoothing as post-processing**: Enforcing temporal consistency through neighbor averaging captures the biological reality that bird presence changes gradually, not abruptly between 5-second segments
- **Attention pooling vs max pooling**: The SEResNeXt model uses framewise maximum rather than attention-weighted sum, suggesting simpler aggregation can outperform learned attention for this task
- **Extended context windows**: Training on 10-second windows while predicting 5-second segments provides temporal context that improves discrimination

## Results

| Metric | Value |
|--------|-------|
| CV     | Not reported |
| Public LB | 0.878 |
| Private LB | Not reported |

**Note**: The notebook author warns that this high public LB score may represent overfitting to the public test set, a pattern observed in BirdCLEF 2022-2024 competitions.

## Connections

- **Related concepts**: Ensemble methods, temporal smoothing, attention-based SED
- **Informs experiments**: Post-processing strategies (power adjustment, temporal smoothing), TTA for temporal models, ensemble weight selection
- **Similar approaches**: Standard SED architecture with attention pooling, mel spectrogram preprocessing pipelines
