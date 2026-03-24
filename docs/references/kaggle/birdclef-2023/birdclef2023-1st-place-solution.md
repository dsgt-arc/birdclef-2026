---
slug: "volodymyr-1st-place-solution"
type: "writeup"
competition: "birdclef-2023"
date_captured: "2026-03-22"
---

# Kaggle Summary: BirdCLEF 2023 1st Place Solution — "Correct Data is All You Need"

**Source**: https://www.kaggle.com/competitions/birdclef-2023/discussion/412808
**GitHub**: https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
**Author**: Volodymyr Sydorskyi (Kaggle Competitions GrandMaster)
**Competition**: BirdCLEF 2023 (264 bird species)

## Summary

The winning solution emphasizes data curation over model complexity: careful collection of
additional Xeno-Canto data, multi-year historical data pretraining (2020–2022), secondary
label incorporation, and realistic background noise augmentation using nocall soundscape
recordings. The final ensemble uses 3 CNN models with attention-based SED heads, trained
in a two-stage pretrain→finetune pipeline, exported to ONNX.

## Approach

### Architecture: WaveCNNAttenClassifier

Raw waveform → on-the-fly mel-spectrogram (nnAudio) → CNN backbone → attention-based SED head.

**Three backbone models**:
1. **ConvNeXt-Small** (ImageNet-22k → ImageNet-1k, 384px) — `convnext_small.fb_in22k_ft_in1k_384`
2. **ECA-NFNet-L0** (drop_path_rate=0.2) — `eca_nfnet_l0`
3. **ConvNeXtV2-Tiny** (FCMAE → ImageNet-22k → ImageNet-1k, 384px) — `convnextv2_tiny.fcmae_ft_in22k_in1k_384`

**Attention Head (AttHead)**:
- GeMFreq pooling (Generalized Mean on frequency axis only, preserving temporal structure)
- Dense: Dropout(p/2) → Linear(in, 512) → ReLU → Dropout(0.5)
- Attention Conv1D + classification Conv1D (sigmoid × softmax over time)
- **Inference blending**: 75% clipwise attention-weighted + 25% framewise max

### Mel-Spectrogram (nnAudio)

- Sample rate: 32,000 Hz
- n_mels: 128, n_fft: 2048, hop_length: 512
- f_min: 20 Hz, normalized: True
- ONNX-exportable via nnAudio's TracableMelspec

### Training: Two-Stage Pipeline

**Stage 1 — Pretrain on 822 species**:
- Data: BirdCLEF 2020 + 2021 + 2022 + additional Xeno-Canto data + 2023 scored species
- Audio stored as HDF5 (precomputed)
- 50 epochs, Adam, CosineAnnealingWarmRestarts (T_0=50)
- LR: 1e-4 (ConvNeXt/V2), 1e-3 (ECA-NFNet)
- Batch size: 64, segment length: 5.0s

**Stage 2 — Finetune on scored species**:
- Data: BirdCLEF 2023 train_audio (264 species) + additional scored XC data with secondary labels
- No validation set — train to last epoch (50)
- Load from pretrain checkpoint

**Loss**: Focal Loss (BCE-based, multi-label)

**Augmentation**:
- Mixup (p=0.5, uniform alpha)
- Background noise injection (p=0.5): nocall soundscape recordings OR ESC-50 environmental sounds
  (dog, rain, insects, hen, engine, hand_saw, pig, rooster, sea_waves, cat, crackling_fire,
  thunderstorm, chainsaw, train, sheep, wind, footsteps, frog, cow, crickets)
- RandomFiltering: random frequency filtering (min_db=-20)
- SpecAugment: freq mask (len=10, masks=3, p=0.3), time mask (len=20, masks=3, p=0.3)
- Balanced sampling: WeightedRandomSampler by primary_label

### Ensemble & Inference

- 3-model ONNX ensemble with Gaussian mean averaging
- For 822-class models: indices rearranged to match 264 competition species
- Output: 75% clipwise + 25% framewise max per model, then averaged

## Key Takeaways

1. **"Correct Data is All You Need"**: Data curation was the decisive factor — additional
   Xeno-Canto data carefully collected and scored, secondary labels incorporated, nocall
   soundscapes used as realistic noise augmentation.

2. **Multi-year pretraining**: Training on historical BirdCLEF data (2020–2022, 822 species)
   then finetuning on current year (264 species) provides strong initialization.

3. **Background noise from real soundscapes**: Using nocall soundscape segments as
   augmentation noise is more realistic than synthetic noise or ESC-50 alone.

4. **GeMFreq pooling**: Pools only the frequency axis while preserving temporal resolution —
   better than standard GeM for Sound Event Detection.

5. **Attention + framewise max blending**: 75% attention-weighted clipwise + 25% framewise max
   captures both sustained calls and brief vocalizations.

6. **No validation, train to completion**: For final submission, removed validation set and
   trained all data to epoch 50. Relies on pretrain regularization to avoid overfitting.

7. **Simple CNN backbones**: ConvNeXt variants and ECA-NFNet — no transformers, no complex
   SED pipelines. Strong ImageNet pretraining + proper finetuning is sufficient.
