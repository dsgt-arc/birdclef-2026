# Deep Research Prompt: Closing the BirdCLEF 2026 Single-Model Gap

Copy this entire prompt (including the sed-architecture-overview document at the
bottom) into Gemini Deep Research, ChatGPT, or another LLM for analysis.

---

## The Problem

I'm competing in BirdCLEF 2026 (bird/frog/insect sound identification). My best
single SED model scores **0.903 on the Kaggle public leaderboard**. Competitors
using the same training data (no external data, no knowledge distillation, no
Perch embeddings) achieve **0.922+ LB**. That's a 0.019 gap I can't close.

I've run **20+ experiments** modifying the training recipe and every single one
has regressed on LB — including longer windows, attention heads, loss functions,
augmentation changes, and pseudo-labeling. My offline validation metrics
consistently improve while LB scores get worse.

## What I Need

1. **Diagnose why my pipeline hits a ceiling at 0.903** while competitors reach
   0.922+ with the same data. What am I fundamentally doing wrong?

2. **Identify the most likely root causes** of the gap, ranked by probability.
   Focus on things I haven't tried or things I may have implemented incorrectly.

3. **Provide a concrete action plan** — what should I change first, second, third?
   Each change should be testable with a single LB submission.

## My Current Pipeline (Summary)

- **Data**: 35,549 focal recordings (single-species, clean) + 66 labeled
  soundscapes (multi-species, noisy) for 234 species
- **Architecture**: HGNetV2-B0 backbone + LSEHead (smooth-max temporal pooling)
- **Training**: 5s random crops, BCE loss, spectrogram MixUp, OneCycleLR, 20 epochs
- **Inference**: ONNX model processes 5s segments independently on Kaggle CPU
- **Score**: 0.903 LB (single fold), 0.913 (4-fold ensemble)

## What Competitors Report

**Salman (0.922 single, 0.932 ensemble, 49th place)**:
- Trains on **20s windows** with train_audio + labeled soundscapes
- Uses **attention head** (framewise max) not LSEHead
- **CE loss** during training, **sigmoid at inference**
- **Waveform MixUp** (mixes raw audio, not spectrograms)
- No knowledge distillation, no Perch, no priors
- Ensemble of 4 scores 0.932

**Boredom (#1 place)**:
- Single EfficientNet model can "easily reach 925+"
- Uses train_audio + labeled soundscapes (no unlabeled data)
- CE loss implied from comments

**hengck23 (0.896, CNN-only)**:
- 15s windows (3×5s chunks), HGNetV2-B0 + LSEHead per chunk
- Emphasizes "context and augmentation is the key"
- Waveform MixUp designed for multi-chunk input

## What I've Tried and Failed (ALL regressed on LB)

| Change | LB | Delta vs 0.903 |
|--------|-----|---------------|
| AttHead + 5s (7 variants) | 0.889-0.893 | -0.010 to -0.014 |
| AttHead + 10s | 0.868 | -0.035 |
| AttHead + 15s | 0.877 | -0.026 |
| AttHead + 20s | 0.878 | -0.025 |
| LSEHead multi-chunk 15s | 0.882 | -0.021 |
| LSEHead multi-chunk 20s | 0.896 | -0.007 |
| LSEHead naive 10s pool | 0.880 | -0.023 |
| CE loss (6 variants) | — | -0.015 to -0.030 (offline) |
| Waveform MixUp | — | -0.003 to -0.019 (offline, never submitted to LB) |
| Pseudo-labels (347K PLs) | 0.897 | -0.006 |
| Soundscape checkpoint sel. | 0.897 | -0.006 |
| PCEN normalization | 0.880 | -0.023 |
| Various other changes | — | all regressed |

## Critical Observation: Train/Inference Mismatch

My Kaggle inference kernel **always processes 5s segments independently**,
regardless of the training window size. When I train on 20s:

- Training: 20s audio → backbone → 64 temporal positions → head
- Inference: 5s audio → backbone → 16 temporal positions → head

For LSEHead (fixed smooth-max), this causes mild regression (-0.007).
For AttHead (learned attention calibrated for 64 positions), this causes
severe regression (-0.025). The attention weights don't transfer to a
different sequence length.

**Competitors training on 20s likely also infer on 20s** — but I don't know
how they handle this on Kaggle where test files are 60s split into 12×5s
segments. Do they load wider context around each 5s segment? Use overlapping
sliding windows?

## Specific Questions

1. **How do BirdCLEF competitors handle inference with models trained on >5s
   windows?** Do they modify the Kaggle kernel to load 20s context around each
   5s test segment? Use overlapping sliding windows? Or something else?

2. **Why does CE loss work for Salman/Boredom but regress for me?** Is it
   because CE + AttHead + 20s interact differently than CE + LSEHead + 5s?
   Is there a specific CE implementation for multi-label SED that I'm missing
   (e.g., applying CE only to the primary label, not as multi-class softmax)?

3. **What does waveform MixUp look like in a winning BirdCLEF pipeline?** My
   implementation regressed offline, but competitors report it as effective.
   Are there specific implementation details (normalization before mixing,
   padding alignment, SNR balancing) that matter?

4. **Is there a standard BirdCLEF SED training pipeline** that reliably produces
   0.920+ scores? What are the critical components that separate 0.90 from 0.92?

5. **What about the crop selection strategy?** I use random crops from focal
   recordings. With 20s windows and a 5s bird call in a 45s recording, ~43% of
   crops contain only silence with a positive label. Do competitors use
   energy-based selection (RMS) or some other strategy?

6. **Are there data augmentation techniques** specifically effective for
   BirdCLEF that I'm not using? (Beyond MixUp — e.g., SpecAugment, frequency
   masking, time warping, background noise mixing from soundscapes)

## Full Pipeline Documentation

Below is my complete end-to-end pipeline documentation. It covers every step
from raw .ogg files to Kaggle submission, including all architectural details,
training recipe, loss computation, inference pipeline, and 8 identified gaps.

--- BEGIN SED ARCHITECTURE OVERVIEW ---

[PASTE THE ENTIRE CONTENTS OF user/murilo/notes/sed-architecture-overview.md HERE]

--- END SED ARCHITECTURE OVERVIEW ---
