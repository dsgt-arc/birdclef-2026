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
- Exp 011 fulldata LB scores: ProtoCLR 0.736, EAT 0.698, ConvNeXt 0.695

### Success Criteria
- **Confirm if**: LB improves over exp 011 AND 28 new species get non-zero predictions
- **Reject if**: LB drops AND focal test AUROC drops >0.02 (catastrophic forgetting)

## Limitations

- Only 739 soundscape segments (vs ~200k focal chunks) — severe class imbalance
- 28 soundscape-only species have 6–626 labeled segments each, most in dense mixtures
- No data augmentation explored yet (deferred)
- Single seed (42)

## Why Soundscape Fine-Tuning Before MixIT (Exp 012 → 013 Sequencing)

Exp 013 will apply MixIT source separation to pseudo-label 10,592 unlabeled soundscapes.
We deliberately do labeled soundscape fine-tuning *first* for four reasons:

### 1. Ablation baseline — isolate MixIT's contribution

Without a soundscape-only baseline, we cannot measure what MixIT actually adds:

```
Focal-only baseline (exp 011):       0.695–0.736 LB
+ Labeled soundscape fine-tuning:     ??? LB        ← exp 012 measures this
+ MixIT-enhanced pseudo-labeling:     ??? LB        ← exp 013 measures this
```

If we skip exp 012 and go straight to MixIT + pseudo-labeling, any improvement could
be from the soundscape data itself, the pseudo-labels, or the separation — impossible
to disentangle. Exp 012 gives us a clean ablation point.

### 2. Multi-label infrastructure is needed regardless

Whether we use MixIT-separated sources or raw labeled soundscapes, we need the same
building blocks: BCE multi-label loss, soundscape segment data loading, 234-class head
expansion, and handling of the 28 soundscape-only species. Building and validating this
infrastructure on labeled soundscapes first means it's ready and tested when pseudo-labeled
data arrives from MixIT.

### 3. Stronger teachers produce better pseudo-labels

The teacher ensemble for pseudo-labeling in exp 013 will be better if the teachers have
already seen soundscape data. A focal-only teacher (exp 011) gives 0.0 for 28 species and
has a domain gap on the rest. A soundscape-fine-tuned teacher (exp 012) produces more
accurate pseudo-labels — MixIT + better teachers = compounding improvement.

### 4. Immediate coverage for 28 soundscape-only species

The labeled soundscapes have expert annotations for all 28 species. Even with dense
co-occurrence (avg 12.9 partners per species), multi-label BCE training will learn
*something* — going from 0.0 to any non-zero signal is a large jump in macro-averaged
AUROC. MixIT can refine these representations later with cleaner isolated signals.

**Counter-argument considered**: MixIT separation would give cleaner training signals
for the 28 species since they are never isolated in raw soundscapes. True — but even
noisy multi-label training is a massive step up from zero predictions, and we can always
re-fine-tune with MixIT-cleaned data in exp 013.

## Leads to

Exp 013 (MixIT pseudo-labeling) will use these 234-class models as teachers to pseudo-label
10,592 unlabeled soundscapes. The exp 012 LB score serves as the baseline against which
MixIT's contribution is measured.
