---
experiment: "012-soundscape-finetune"
status: ready
date: 2026-03-21
author: Murilo Gustineli
---

# Experiment 012: Labeled Soundscape Fine-Tuning

## Hypothesis

Fine-tuning exp 011 LoRA models on labeled soundscape data will (a) expand species coverage
from 206 to 234 (adding 28 soundscape-only species) and (b) improve LB scores by adapting
the models to the soundscape domain where they are evaluated.

## Motivation

Exp 011 models are trained only on focal recordings (single-species, high-SNR). The Kaggle
evaluation is on soundscapes (multi-species, noisy, overlapping). Two problems:

1. **28 species get 0.0 predictions** — 25 insect sonotypes + 3 amphibians exist only in
   labeled soundscape segments, with no focal recordings at all.
2. **Focal → soundscape domain gap** — models never saw multi-species overlapping audio
   during training, leading to LB scores (0.698–0.736) well below test-split AUROC (0.929–0.938).

BirdCLEF 2026 provides 66 expert-labeled soundscape files with 739 unique 5s segments
(75 species annotated, avg 4.2 species/segment). This is the only supervised path to
cover the 28 soundscape-only species.

## Approach

**Option A: Head expansion + mixed training** (selected)

1. Load exp 011 best checkpoint (80/10/10 split, not fulldata) for each encoder
2. Expand classification head from 206 → 234 classes:
   - Copy weights for 206 existing species (remapped to new sorted indices)
   - Initialize 28 new species neurons randomly
3. Fine-tune on mixed focal + soundscape data with multi-label BCE loss
4. Lower LR (1e-4 vs 3e-4) to preserve learned representations

### Variables

- **Independent**: Training data (focal-only vs focal+soundscape), number of classes (206 vs 234)
- **Dependent**: LB score, per-species AUROC, focal test AUROC (forgetting check)
- **Controlled**: Same encoders, same LoRA config, same preprocessing

## Data

| Source | Recordings | Segments | Species | Labels |
|--------|-----------|----------|---------|--------|
| Focal (`train_audio/`) | 35,549 | ~200k chunks | 206 | Single-label + secondaries |
| Soundscapes (`train_soundscapes/`) | 66 labeled | 739 unique 5s | 75 (incl. 28 new) | Multi-label (avg 4.2/segment) |

Soundscape segments are split **by file** (not by segment) to prevent data leakage:
~50 train / ~8 val / ~8 test files.

## Evaluation

### Metrics
- **Primary**: Kaggle LB macro-AUROC (all 234 species)
- **Secondary**: Focal test AUROC (catastrophic forgetting check), soundscape test AUROC,
  per-class AUROC for 28 soundscape-only species

### Baseline
- Exp 011 fulldata LB scores: ProtoCLR 0.736, EAT 0.698, ConvNeXt (pending)

### Success Criteria
- **Confirm if**: LB improves over exp 011 AND 28 new species get non-zero predictions
- **Reject if**: LB drops AND focal test AUROC drops >0.02 (catastrophic forgetting)

## Limitations

- Only 739 soundscape segments (vs ~200k focal chunks) — severe class imbalance
- 28 soundscape-only species have 6–626 labeled segments each, most in dense mixtures
- No data augmentation explored yet (deferred)
- Single seed (42)

## Leads to

Exp 013 (MixIT pseudo-labeling) will use these 234-class models as teachers to pseudo-label
10,592 unlabeled soundscapes.
