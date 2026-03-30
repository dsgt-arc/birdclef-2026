# Proposal: SSAMBA SSL Pretraining on BirdCLEF Soundscapes

**Status**: proposed
**Created**: 2026-03-30
**Author**: anthony

## Hypothesis

Domain-specific masked patch pretraining on BirdCLEF 2025+2026 audio at 32kHz will produce better
downstream representations than AudioSet-pretrained SSAMBA weights, because:

1. Bird vocalizations extend to 16kHz — capturing this requires 32kHz audio; SSAMBA's original
   pretraining used 16kHz (8kHz ceiling), cutting off the upper octave of bird calls
2. Soundscape data is the target domain; AudioSet is biased toward clean single-source clips that
   don't reflect the overlapping multi-species recordings in BirdCLEF evaluation
3. SSAMBA-small (26M) and SSAMBA-base (99M) have untested capacity — prior experiments used tiny
   (7M) with AudioSet weights; these larger models may benefit more from in-domain SSL

## Background

Prior SSAMBA experiments established:
- Exp 004: Pure-PyTorch HF port works; AudioSet weights transfer perfectly
- Exp 005: Rectangular patches (128×2, 128×1) are O(n) linear, confirmed on RTX 6000
- Exp 006: Supervised from scratch → 0.086 cmAP (BirdSet HSN); pretraining is essential
- Exp 007: AudioSet pretrained → 0.173 cmAP; 2× lift but large gap vs SOTA (0.55)
- Exp 008: SED attention pooling → 0.164 cmAP; bottleneck is representation, not pooling
- Exp 009: SSL pretrain pipeline — PAUSED (backbone training done, finetune pending)

Exp 006 was *supervised* training from random init, not masked self-supervised pretraining.
We have never run the SSAMBA masked prediction objective on in-domain bird audio. This experiment
tests whether domain-specific SSL closes the representation gap.

See: `docs/concepts/ssamba-audio-mamba.md`, `docs/concepts/pretrained-mamba-transfer.md`

## Method

### Approach

Train SSAMBA-small and SSAMBA-base from random init using the original masked patch prediction
objective (masking 40% of spectrogram patches, predicting masked patches via MSE). Input is a
32kHz mel spectrogram with rectangular 128×2 patches (full-frequency-band, 20ms per token).
Train on all available BirdCLEF 2025/2026 audio — soundscapes + focal recordings (~1000+ hours).
Evaluate via reconstruction loss (SSL quality) and a linear probe on BirdCLEF 2026 focal cmAP
(downstream quality).

### Setup

| Component | Details |
|-----------|---------|
| Data | BirdCLEF 2026: 10,658 soundscapes + 35,549 focal recordings; BirdCLEF 2025: soundscapes; total ~1000h |
| Data paths | `/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2025/`, `birdclef-2026/` |
| Compute | RTX 6000 (24GB VRAM), fp16 training (bf16 emulated on sm_75), torch.compile |
| Dependencies | mamba-ssm (CUDA), causal-conv1d, torchaudio, nnAudio |
| Backbone | `vendor/ssamba` (original CUDA implementation — HF port from exp 004 too slow for training) |
| Windows | 10s clips, 50% overlap for SSL data augmentation |

### Procedure

1. **Build audio index**: scan all 2025/2026 focal + soundscape files, build a parquet index of
   (file_path, offset_sec, duration_sec) for 10s windows with 50% overlap. Hold out 5% of
   soundscape files for validation (stratified by recording site).

2. **Configure mel frontend**: 32kHz SR, n_fft=4096, n_mels=128, hop_length=320 (10ms), fmin=50,
   fmax=16000. Patch size 128×2 → 500 tokens per 10s window.

3. **Implement SSL training loop**: use SSAMBA's masked patch prediction head. Mask 40% of patches
   randomly, encode unmasked tokens, reconstruct masked patches via a lightweight decoder head
   (linear projection back to patch dimension). Loss: MSE on masked patches.

4. **Train SSAMBA-small** (embed_dim=384): warm up for 5 epochs, train to convergence (monitor
   val reconstruction loss, early stop at patience=10). Log to TensorBoard.

5. **Train SSAMBA-base** (embed_dim=768): same config, likely needs more epochs to converge given
   larger capacity. Expect ~2× wall time vs small.

6. **Linear probe** (Phase 2): freeze SSL backbone, add `Linear(embed_dim, 206)`, train on
   BirdCLEF 2026 focal chunks (parquet index from exp 102). Report cmAP and Kaggle-style AUROC.
   Compare against exp 007 baseline (0.173 cmAP with AudioSet weights + finetune).

7. **(Optional) Full finetune**: if linear probe beats 0.20, unfreeze backbone and finetune with
   LoRA (from exp 102 codebase). Submit to Kaggle for LB score.

### Variables

- **Independent**: SSL pretraining domain (BirdCLEF soundscapes vs AudioSet), model size (small vs base)
- **Dependent**: SSL val reconstruction loss; linear probe cmAP; Kaggle LB after finetune
- **Controlled**: patch size (128×2), masking ratio (40%), mel config (n_fft=4096, n_mels=128),
  training window (10s), optimizer (AdamW, lr=1e-4, warmup)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| SSL val loss (MSE) | Reconstruction quality on held-out soundscapes; lower = better representation |
| Linear probe cmAP | Downstream classification quality with frozen backbone on BirdCLEF 2026 focal |
| Linear probe AUROC | Kaggle-style macro AUROC excluding absent species |
| Kaggle LB (optional) | Public leaderboard score after LoRA finetune + submission |

### Baseline

| Baseline | cmAP | Source |
|----------|------|--------|
| SSAMBA-tiny, AudioSet pretrained (exp 007) | 0.173 | Full finetune on BirdSet HSN |
| SSAMBA-small, AudioSet pretrained (exp 007) | 0.165 | Full finetune on BirdSet HSN |
| EfficientNet LoRA + attention (exp 102 run C) | — | 0.794 Kaggle LB |

Note: exp 007 baselines are full-finetune cmAP on BirdSet HSN. Our Phase 2 linear probe uses
BirdCLEF 2026 focal, so the comparison is directional, not exact.

### Success Criteria

- **Confirm (SSL quality)**: val reconstruction loss decreases monotonically and falls below the
  "mask all patches → predict mean" trivial baseline
- **Confirm (representation)**: linear probe cmAP > 0.20 on BirdCLEF 2026 focal for either
  small or base (above exp 007's 0.173 ceiling with AudioSet weights + full finetune)
- **Reject**: linear probe cmAP ≤ 0.10 after 50 epochs (worse than supervised from scratch,
  suggests SSL objective is not converging or 32kHz mel config is broken)
- **Stretch**: Kaggle LB > 0.794 after LoRA finetune

## Limitations

- MSE reconstruction loss is an imperfect proxy for downstream quality — a model can
  reconstruct patches well via low-level statistics without learning useful features
- Exp 007 comparison is confounded: different data (BirdSet HSN vs BirdCLEF 2026), different
  protocol (full finetune vs linear probe). The comparison is directional, not exact.
- BirdCLEF 2025 soundscapes share recording sites with 2026 — some site-specific patterns may
  leak into SSL representations (not a label leak, but a domain overlap)
- SSAMBA-base (99M) on RTX 6000 (24GB) with 500 tokens/batch: memory budget needs verification
  before committing to this size; may need gradient checkpointing
- Starting from random init isolates the domain SSL contribution but gives up AudioSet
  pretraining for free — a continued-pretraining variant (start from AudioSet weights) is
  a natural follow-up if this underperforms
