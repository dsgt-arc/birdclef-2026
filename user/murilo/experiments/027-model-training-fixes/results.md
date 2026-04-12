# Experiment 027: Model Training Fixes — Results

## Context

Deep research (2026-04-11) identified two fundamental implementation errors
in our SED pipeline compared to BirdCLEF 2024-2025 winners:

1. **MixUp**: We used interpolative mixing (lambda*x1 + (1-lambda)*x2). Winners
   use additive waveform mixing (x1 + x2) with max labels. Impact: -0.036
   private LB without correct MixUp (BirdCLEF 2025 2nd place ablation).

2. **Loss**: We used BCE across all 234 species. Winners use CE on primary label
   only during training, sigmoid at inference. Impact: +0.044 private LB from
   this single change (BirdCLEF 2024 1st place).

See: `notes/2026-04-11-deep-research-claude.md`, `notes/2026-04-11-deep-research-gemini.md`

## Phase 1: v10a (Additive MixUp + CE loss)

**Config**: `configs/v10a_addmix_ce.yaml`
**Changes from v5j**: `mixup_mode=additive`, `raw_waveform_mixup=true`, `loss_type=ce`
**Everything else**: identical to v5j (5s window, LSEHead, batch 64, 20 epochs)

*Results pending — awaiting training run.*
