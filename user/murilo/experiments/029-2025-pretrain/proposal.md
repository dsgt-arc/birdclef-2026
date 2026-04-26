# Proposal: 2025 Pretrain → 2026 Fine-tune (Ali Ozan Path)

**Status**: proposed
**Created**: 2026-04-25
**Author**: Murilo

## Hypothesis

Pretraining v5j on BirdCLEF 2025 focal data, then fine-tuning on BirdCLEF
2026 with the 41 overlap species' classifier weights transferred and the
remaining 165 (+28 soundscape-only) species randomly initialized, will
improve LB over the v5j baseline (0.906) trained from ImageNet
initialization on 2026-only data.

**Single-variable change from v5j**: presence of 2025-pretrained
initialization vs ImageNet initialization. All other components held at
v5j values (HGNetV2-B0 + LSEHead, BCE, NNAudio + AmplitudeToDB mel,
spectrogram MixUp α=1.0, OneCycleLR, 5s window, 90/10 split).

Falsifiable: if the fine-tuned model's LB ≤ 0.906, 2025 data does not
help v5j on 2026 in the form we tested (joint or sequential, our choice
of stage 2 LR / epochs).

## Background

### Why this experiment now

After the [per-frame analysis on 2026-04-25](../../analysis/notes/2026-04-25-per-frame-attention-analysis.md),
the strategic picture changed in a specific way: v5j's behavior on hard
crops is *honest signal-matching* (correctly diffuse on low-content
crops), and the BCE+random-crop training objective penalizes that
honesty. This mechanistically explains the 21+ failed training-recipe
changes: they were trying to make the model more confident on low-SNR
crops, which is exactly the false-positive direction LB punishes.

The implication: the gap to Salman (0.937) and Ali Ozan (0.941) is
*not* something we can close by tweaking loss / augmentation /
attention head on 2026 data alone. We need a different lever.

Adding 2025 data is one such lever — it adds *information* (more bird
audio variety, richer backbone representations) rather than fighting
the training objective. Ali Ozan's 0.941 was achieved this way.

### What Ali Ozan did vs what we're doing

Ali Ozan used a publicly available checkpoint that someone else
pretrained on the **filtered Xeno-Canto dataset** (much larger than
BirdCLEF 2025 alone). He didn't pretrain himself. Discussion #683791:

> "They used the entire Xeno-Canto dataset by filtering it and fixing a
> bug during pretraining, but I didn't use anything beyond the
> checkpoint." — Ali Ozan, 10 days ago

We don't have that checkpoint. But we *do* have BirdCLEF 2025 train
audio (28,564 recordings, 206 species) locally. That's a smaller
analogue: pretrain v5j on 2025, fine-tune on 2026. Different scale than
Ali Ozan's pretrain corpus, same structural idea.

### Data we have

- 2025 train.csv: 28,564 rows, 206 species
- 2026 train.csv: 35,549 rows, 206 focal species (234 with
  soundscape-only species included)
- **Species overlap: 41** species in both 2025 and 2026
- **2025-only species: 165** — provide useful general bird-sound
  representation during pretraining even though they cannot be mapped
  to 2026 classifier indices
- **2026-only species: 165** + **28 soundscape-only** — these classifier
  rows must be randomly initialized at fine-tune time
- File format: 32 kHz mono OGG, identical to 2026 → drop-in compatible

Top species that gain meaningful data from 2025 (overlap species):

| Species | n_2026 | n_2025 added | New total |
|---|---:|---:|---:|
| grekis | 482 | +990 | 1,472 |
| compau | 493 | +808 | 1,301 |
| trokin | 483 | +787 | 1,270 |
| **roahaw** | **489** | **+709** | **1,198** |
| banana | 498 | +610 | 1,108 |
| whtdov | 491 | +572 | 1,063 |
| socfly1 | 494 | +543 | 1,037 |
| yeofly1 | 493 | +525 | 1,018 |
| bobfly1 | 492 | +514 | 1,006 |
| soulap1 | 497 | +487 | 984 |

The 41 overlap species more than double their training data via the
sequential approach. (Note: in the sequential approach the 2025 rows
are not all retained at fine-tune time; the *backbone* learns from
them during pretrain, and the classifier's overlap rows are warm-
started from 2025's representations.)

### Why sequential, not joint

Joint training (combine 2025 + 2026 into one DataFrame, train with a
234-class head): simpler but discards the 165 2025-only species'
worth of representation learning, since they have no place to map to.

Sequential pretrain → fine-tune: pretrains on the *full* 2025 corpus
(206 species head, 28K rows including the 165 2025-only species). The
backbone learns general bird-sound features from all of it. The
fine-tune stage replaces the classifier and copies overlap rows.

Sequential is closer to Ali Ozan's structural approach and uses all
2025 data. We chose sequential.

### References

- [`analysis/notes/2026-04-25-per-frame-attention-analysis.md`](../../analysis/notes/2026-04-25-per-frame-attention-analysis.md)
- [`analysis/notes/2026-04-25-audio-validation-low-snr-crops.md`](../../analysis/notes/2026-04-25-audio-validation-low-snr-crops.md)
- [`notes/kaggle-discussion-best-single-model-lb.md`](../../notes/kaggle-discussion-best-single-model-lb.md) — Ali Ozan's path
- [`MODEL-RUNS.md`](../../MODEL-RUNS.md) — exp 009 prior context (2025
  ProtoCLR checkpoint applied to 2026 → 0.531 LB; that was a different
  architecture, no fine-tuning, full classifier replacement)

## Method

### Setup

| Component | Details |
|-----------|---------|
| Architecture | HGNetV2-B0 + LSEHead (hidden_dim=1024, freq_pool=mean, lse_temperature=1.0) — identical to v5j |
| Loss | BCE — identical to v5j |
| Mel | NNAudio + AmplitudeToDB, n_fft=2048, n_mels=256, hop=313 — identical to v5j |
| Optimizer | AdamW + OneCycleLR — identical to v5j |
| Stage 1 (pretrain) data | 2025 focal (28,564 rows, 206 species, 90/10 split) |
| Stage 2 (fine-tune) data | 2026 focal + 66 labeled soundscapes (90/10 split, 8 SS files held out for val) — identical to v5j |
| Compute | 1× RTX 6000 (PACE), ~1.5–2h pretrain + ~1.5–2h fine-tune ≈ 3.5–4h GPU total |
| Code | Branch `mg/exp029-2025-pretrain`. Train script copied from 027 + classifier-transfer logic added. Configs copied from `027/configs/v5j_control.yaml` and adapted |

### Procedure

**Stage 1 — Pretrain on 2025 (config `v5j_pretrain_2025.yaml`)**

1. Train v5j architecture on 2025 focal data (28K rows, 206 species).
   Same recipe as v5j (BCE, MixUp, OneCycleLR, 20 epochs, 90/10 split).
2. Save best-by-val/auroc checkpoint.
3. Optional: fast-dev-run sanity check first.

**Stage 2 — Fine-tune on 2026 from pretrained init (config `v5j_finetune_2026.yaml`)**

4. Load Stage 1 checkpoint.
5. Build 2026 model with 234-class head (same config as v5j).
6. **Classifier weight transfer**: for every parameter in the 2025
   state dict whose shape matches the 2026 model, copy directly. For
   the final classifier weight (shape `[206, hidden_dim]`) and bias
   (shape `[206]`), do partial copy: for each species in the 41-species
   overlap, copy that row from 2025 to the corresponding 2026 index.
   Other 193 rows (165 2026-only + 28 soundscape-only) keep their
   random init.
7. Train v5j architecture on 2026 focal + 66 labeled soundscapes (same
   data as v5j, identical recipe except lower LR for warm start).
8. Save best-by-val/auroc checkpoint.

**Stage 3 — Evaluate**

9. ONNX export.
10. LB submission (single fold, no PP beyond v5j-default TTA + smoothing).
11. Re-run triage on the new model. Compare per-frame metrics
    (entropy, logit_std) to v5j baseline. Did the head behavior change?

### Variables

- **Independent**: 2025-pretrained init vs ImageNet init (single
  variable from v5j)
- **Dependent**: Public LB (row-wise macro-averaged ROC-AUC)
- **Controlled** (held at v5j values):
  - Architecture, head, loss, mel pipeline, optimizer family
  - Window duration (5s), MixUp, OneCycleLR shape
  - Validation split structure (90/10)
  - Training data composition at Stage 2 (2026 focal + labeled SS)

### Open scoping decisions (locked, recording for reference)

| Question | Decision | Rationale |
|---|---|---|
| Joint vs sequential | Sequential | Uses all 2025 data; closer to Ali Ozan |
| Pretrain on full 2025 (206 cls) or filter to overlap (41 cls) | Full 2025 | The 165 non-overlap species teach general features |
| Pretrain epochs | 20 | Same as v5j; full training shape for representation learning |
| Pretrain validation | 90/10 random | Best-by-val/auroc checkpoint selection |
| Fine-tune LR | 2e-4 | Reduced from v5j's 5e-4 for warm start (2× lower) |
| Fine-tune epochs | 20 | Best checkpoint selection from earlier epochs is fine; not constrained |
| Fine-tune backbone freeze stages | None | End-to-end fine-tune with reduced LR is cleaner single-variable change |
| Classifier transfer logic | 41 overlap rows from 2025 → 2026 indices, rest random | Standard transfer-learning practice |

## Evaluation

### Metrics

| Metric | Description |
|---|---|
| **Public LB** | **Primary** — row-wise macro-averaged ROC-AUC on test soundscapes |
| Stage 1 val/auroc | Pretrain checkpoint selection only (informational for LB) |
| Stage 2 val/auroc | Fine-tune checkpoint selection only (informational for LB) |
| Triage on stage-2 model | Sanity check: does the per-frame structure change? Compare attention entropy / logit_std on `mixed_uncertainty` vs v5j baseline |

### Baselines

- v5j single fold: **0.906 LB**
- v5j + 4-fold ensemble + pp2: 0.917 LB (separate question, future)
- Reference: Ali Ozan single model with 2025-pretrain (different
  pretrain corpus): 0.941 LB

### Success Criteria

- **Confirm**: stage-2 LB > 0.906
- **Strong confirm**: stage-2 LB ≥ 0.910
- **Reject**: stage-2 LB ≤ 0.906 AND triage shows no per-frame
  structural change vs v5j
- **Retry**: stage-2 LB ≤ 0.906 BUT triage shows the head behavior
  changed (e.g., logit_std on mixed_uncertainty increased) — possibly
  the LR / epoch tuning is off, retry with different fine-tune
  parameters before giving up
- **Pivot**: pretrain stage 1 fails to converge (Stage 1 val/auroc
  doesn't reach reasonable level for 2025 alone) — would suggest the
  pipeline doesn't work on 2025 for some reason; debug rather than
  proceed to Stage 2

## Limitations

- **Smaller pretrain corpus than Ali Ozan**. He used filtered XC,
  estimated to be 100K+ recordings. Our 2025 alone is 28K.
  Representation-learning gain may be smaller than his.
- **Different sites**. 2025 and 2026 are different geographic
  collections. The acoustic distribution may differ enough that 2025-
  pretrained features don't transfer cleanly. This is a real risk.
- **Only 41 species in overlap**. 165 + 28 = 193 of the 234 2026
  classes get random-init classifier rows at stage 2. Most of the
  representation benefit is in the backbone + the 41 overlap-species
  classifier rows.
- **Offline metrics still anti-correlated with LB**. Stage 1 and
  Stage 2 val/auroc are checkpoint-selection signals only, not
  evaluation metrics.
- **Single fold**. Adds variance vs 4-fold.
- **Fine-tune LR / epoch sensitivity**. We chose 2e-4 / 20 ep based on
  standard transfer-learning practice. If the wrong call, stage-2
  results may not reflect the underlying lever.

## Related work and potential follow-ups

Out of scope for this experiment, listed for reference:

- **Joint training** (combine 2025 + 2026 in single training run with
  234-class head): if sequential helps, joint may or may not help on
  top. Different experiment.
- **Filtered XC pretrain** (matching Ali Ozan more faithfully):
  significantly more data engineering; defer until we see whether
  our smaller-scale 2025 pretrain helps at all.
- **4-fold ensemble after stage 2**: standard follow-up if stage 2
  produces a winner. Maps to v5j's path: single-fold first, then
  4-fold.
- **Salman recipe replication on EffNetV2-B0** (formerly exp 029
  candidate): different bet, parallel future experiment.
- **Exp 030 (silent-class)**: still queued from per-frame analysis
  note. Independent of this experiment.

If exp 029 succeeds, the next experiment could be joint training, or
a 4-fold scaleup of the same recipe. If exp 029 fails, exp 030
(silent-class) or Salman replication become the candidates.
