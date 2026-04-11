# Proposal: Pseudo-Labeling v2

**Status**: proposed
**Created**: 2026-04-10
**Author**: Murilo Gustineli
**Branch**: TBD
**Depends on**: Exp 024 (data-centric exploration findings)

---

## 1. Overview

This experiment adds **pseudo-labeled soundscape data** to our training pipeline. We have
10,658 unlabeled soundscape files (127,896 5-second segments) that are the same domain as
the test set. Our best models (SED v5j and Perch v2) can predict which species are present
in these segments — those predictions become "pseudo-labels" that we use as additional
training data.

The goal is to close the gap between our current best (0.929 LB) and the #1 position
(0.948 LB) by providing the model with soundscape-domain training data for the 159 species
that currently only have focal recording data.

**Why this should work**: The LB gap is a data coverage problem, not a model quality
problem. Our model achieves 0.9958 AUC on the 75 species it has soundscape data for,
but the remaining 159 species have never been seen in the noisy, multi-species soundscape
context that the test set consists of. Pseudo-labeling bridges this gap.

---

## 2. Background

### 2.1 The data coverage problem (from exp 024)

Our training data has two sources:
- **Focal recordings** (35,549 files): Clean, single-species recordings from xeno-canto.
  Cover 206 species. High SNR, typically one bird per recording.
- **Labeled soundscapes** (66 files, 792 segments): Noisy, multi-species field recordings
  from Pantanal. Cover only 75 species. Low SNR, multiple species per segment.

The test set consists entirely of **soundscape recordings** — same domain as the labeled
soundscapes. The problem: 159 species have focal data but zero soundscape data. The model
trained on focal recordings must generalize "blindly" to soundscapes for these species.
This domain shift (clean→noisy, single→multi-species) is the core bottleneck.

Evidence from exp 024:
- v5j achieves **0.9958 macro-AUC** on the 75 species with soundscape data
- For the 159 unseen species, mean max-prob on soundscapes is **0.036** — the model
  confidently says "no" for species it hasn't seen in soundscapes
- SED max-across-folds covers **109/159** unseen species at >0.5 probability
- Perch (pretrained, no domain shift) covers **149/159** but with severe miscalibration
- Only **6 species** are truly invisible to both models (1-5 focal recordings each)

### 2.2 Why pseudo-labeling (and why it failed before)

**Pseudo-labeling** (also called self-training or noisy student) uses a trained model's
predictions on unlabeled data as training labels for a new model. The idea: even imperfect
predictions teach the student something the teacher learned from the training data.

We tried this in **exp 015** and it failed (-0.005 to -0.024 LB). But the conditions
were fundamentally different:

| Factor | Exp 015 (failed) | Exp 025 (this) |
|--------|------------------|----------------|
| Teacher LB | 0.767 | **0.929** (5× stronger) |
| Architecture | Clip-level (no temporal resolution) | **SED** (frame-level predictions) |
| Domain | Focal teacher → soundscape labels | **Soundscape teacher → soundscape labels** |
| Filtering | None — raw predictions | **Power transform + temperature scaling** |
| Label format | Single-label (one species per segment) | **Multi-label** (multiple species per segment) |
| Species coverage | Unknown | **109/159** unseen species at >0.5 |

The key insight from section 11.3 of our EDA: *"Even imperfect pseudo-labels for 80 species
are better than the current zero soundscape representation. The model literally has no
soundscape signal for 68% of species — any signal, even noisy, is an improvement."*

### 2.3 What the 1st place BirdCLEF 2025 solution did

Nikita's winning solution used iterative pseudo-labeling for **5 of 7 models**:
- 3-4 iterations of pseudo-labeling
- Temperature scaling (T=0.55-0.6) to sharpen predictions
- Mixup augmentation (ratio 0.5-1.0)
- 7 diverse models across 5 different backbones
- Used unlabeled soundscape data as same-domain training signal

Key takeaway: pseudo-labeling was central to the winning approach, not an afterthought.
The iteration (student-becomes-teacher) and model diversity were critical for improving
pseudo-label quality over multiple rounds.

---

## 3. Key Concepts

### 3.1 Pseudo-labeling / noisy student

Standard supervised learning: train on (input, true_label) pairs.
Pseudo-labeling: train on (input, predicted_label) pairs, where predicted_label comes
from a pre-trained "teacher" model run on unlabeled data.

The process:
1. Train teacher model on labeled data → get a good model
2. Run teacher on unlabeled data → get predictions (pseudo-labels)
3. Train student model on labeled data + pseudo-labeled data
4. (Optional) Student becomes new teacher → repeat

**Why it works**: The teacher has learned acoustic patterns from focal data. When it
"recognizes" a species in a soundscape, it's transferring that knowledge to the soundscape
domain. The student then learns directly in the soundscape domain.

**Risk**: If the teacher's predictions are wrong, the student learns wrong patterns. This
is why filtering (threshold, power transform, temperature) is critical.

### 3.2 Temperature scaling

Temperature scaling adjusts the "sharpness" of a probability distribution. Given a
model's raw logit `z` (the pre-sigmoid output), the normal probability is:

```
p = sigmoid(z) = 1 / (1 + exp(-z))
```

With temperature T:

```
p_T = sigmoid(z / T)
```

- **T = 1.0**: No change (standard sigmoid)
- **T < 1.0** (e.g., 0.5): **Sharpens** — confident predictions become more confident,
  uncertain predictions become more uncertain. A logit of 2.0 gives p=0.88 normally,
  but p=0.98 at T=0.5.
- **T > 1.0**: **Softens** — all predictions move toward 0.5.

**Why it matters for pseudo-labeling**: We want to amplify the teacher's confidence signal.
When the teacher is fairly sure (p=0.7) vs uncertain (p=0.55), temperature sharpening
makes this distinction clearer: at T=0.5, these become ~0.88 vs ~0.65. This helps
separate genuine detections from noise.

Nikita used T=0.55-0.6 for pseudo-label generation.

### 3.3 Power transform

Power transform raises probabilities to an exponent: `p_new = p ^ α`

With α = 1.5:
- p = 0.9 → 0.85 (high confidence preserved)
- p = 0.7 → 0.59 (moderate confidence reduced)
- p = 0.5 → 0.35 (borderline killed)
- p = 0.3 → 0.16 (low confidence crushed)

**Why it matters**: Power transform selectively suppresses uncertain predictions while
preserving confident ones. It's a non-linear filter that pushes the probability
distribution toward the extremes (very confident or very unlikely).

Combined with a threshold (e.g., keep only p > 0.5 after transform), this effectively
requires the original prediction to be quite confident — e.g., with P^1.5, the original
probability must be >0.63 to survive a >0.5 threshold after transform.

### 3.4 Soft labels and loss weighting

**Hard labels**: Species is present (1) or absent (0). Binary.
**Soft labels**: Species has probability 0.73 of being present. Continuous.

Soft labels preserve the teacher's uncertainty — a prediction of 0.95 carries more weight
than 0.55. When training with BCE (binary cross-entropy) loss, soft labels naturally
downweight uncertain predictions.

**Loss weighting** adds a second layer of control:

```
total_loss = w_gt * BCE(pred, ground_truth) + w_pl * BCE(pred, pseudo_label)
```

With `w_gt=1.0` and `w_pl=0.3-0.5`, pseudo-labeled data contributes less to the gradient
than ground truth. This limits damage from incorrect pseudo-labels while still allowing
the model to learn from them.

### 3.5 Max-across-folds

Our SED model was trained with 4-fold cross-validation, giving 4 models with slightly
different training sets. For pseudo-labeling, we take the **maximum** prediction across
all 4 folds per species per segment:

```
pseudo_prob[species, segment] = max(fold0, fold1, fold2, fold3)
```

This dramatically outperforms single-fold or averaged predictions for rare species
(exp 024 Phase A finding):

| Strategy | Unseen species covered @>0.5 |
|----------|------------------------------|
| Single-fold | 80 / 159 |
| Average-4fold | 68 / 159 (worse!) |
| **Max-4fold** | **109 / 159** |

**Why max works**: Each fold sees a slightly different training set, so different folds
catch different rare species. If fold 2 detects a rare bird at p=0.7 but the other
3 folds say p=0.01, averaging gives 0.18 (detection lost), but max gives 0.7 (detection
preserved).

**Why averaging hurts**: It smooths out confident individual-fold detections for rare
species. Averaging is good for calibration on common species but bad for recall on
rare ones.

### 3.6 Per-species normalization (new technique for this experiment)

This addresses the Perch calibration problem. Perch's raw probabilities are permanently
high for many species (e.g., predicting Social Flycatcher in 99.9% of segments). Standard
thresholding fails because all segments are above any reasonable threshold.

**The insight**: Even though absolute probabilities are wrong, the **relative ranking
within a species** may still carry signal. Segments where Perch is MOST confident for
species X are more likely to actually contain species X than segments where it's less
confident.

**Per-species z-normalization**:

```
For each species independently:
  mean_prob = mean(perch_prob across all 127K segments)
  std_prob  = std(perch_prob across all 127K segments)
  z_score   = (perch_prob - mean_prob) / std_prob
```

This transforms each species' distribution to mean=0, std=1. A segment with z=2.0 is
2 standard deviations above the species' average — genuinely unusual for that species.
A segment with z=-0.5 is below average — probably not a real detection.

After z-normalization, we can apply standard thresholding (e.g., keep z > 1.5-2.0) and
the selected segments are the ones where Perch is most confident **relative to that
species' baseline**, not in absolute terms.

---

## 4. Teacher Strategy

### 4.1 Species tiers

Based on exp 024 findings, species fall into 4 tiers based on what teacher signal is
available:

| Tier | Description | N | Teacher | Quality |
|------|-------------|---|---------|---------|
| **A** | Seen in labeled soundscapes | 75 | SED (validated 0.9987 AUC) | Excellent |
| **B** | Unseen, SED-covered at >0.5 | 109 | SED max-folds (94% precision extrapolated) | Good |
| **C** | Unseen, SED-zero, Perch-covered | 44 | Perch with per-species normalization | Uncertain |
| **D** | Truly invisible (neither at >0.3) | 6 | None — focal only | N/A |

Total: 75 + 109 + 44 + 6 = 234 species.

Note: Tier C is 44 (not 50) because 6 species are in Tier D. These 44 species are the
**highest-value targets** — they have zero soundscape representation today, and Perch is
the only source of soundscape-domain signal for them.

### 4.2 Tier A+B: SED teacher (184 species)

SED v5j max-across-folds is the teacher. Validated at 94% precision on labeled soundscapes.

Pipeline:
1. Load 4-fold logits → sigmoid → `max(fold0..fold3)` per species per segment
2. Apply temperature scaling with T=0.6 (sharpening — Nikita's range)
3. Apply power transform P^1.5 (suppress borderline predictions)
4. Keep as **soft labels** (no hard thresholding for Tiers A+B — let the power transform
   and loss weighting manage quality)

Expected yield: ~5K species×segment pairs above 0.5 for unseen species, plus expanded
site/hour coverage for 75 seen species.

### 4.3 Tier C: Perch teacher with per-species normalization (44 species)

These species have **zero SED coverage** — Perch is the only option. But Perch raw probs
are catastrophically miscalibrated (predicting 119 species per segment). We solve this
with per-species normalization:

Pipeline:
1. Load Perch raw probabilities (`perch_234_raw_probs.parquet`)
2. For each of the 44 Tier C species independently:
   a. Compute mean and std of Perch prob across all 127K segments
   b. Z-normalize: `z = (prob - mean) / std`
   c. Keep segments where z > threshold (e.g., z > 2.0 ≈ top 2.5%)
3. Convert z-scores back to soft labels via sigmoid: `label = sigmoid(z)`
4. Apply power transform P^1.5 on the sigmoid output

Expected yield per species: 127K × 2.5% ≈ 3,200 segments per species at z>2.0
(tunable — z>1.5 gives ~6.5K, z>2.5 gives ~600).

**Why this should work**: Perch scored 0.925 LB — it CAN detect these species. The raw
logit-to-probability mapping is broken, but the relative ordering of segments within a
species should reflect genuine acoustic signal. The segments where Perch is 3 standard
deviations above its own mean for species X are far more likely to contain species X
than segments at the mean.

**Risk**: If Perch's signal for a species is truly constant (zero variation), the
z-scores will be noise. We mitigate by checking per-species std — if std < threshold
(e.g., <0.01), skip that species.

### 4.4 Tier D: No teacher (6 species)

These 6 species have 1-5 focal recordings and are invisible to both SED and Perch.
They cannot be helped by pseudo-labeling. Ecological assessment (Phase E) confirmed
4 are genuinely present in Pantanal but data-scarce, 1 is out of range, 1 is diurnal.

For these species, the model relies solely on focal training data. Future work could
source external recordings from xeno-canto or Macaulay Library.

---

## 5. Step-by-Step Walkthrough

This section walks through the entire pipeline with concrete examples, so someone
unfamiliar with the project can follow every transformation.

### Step 1: Categorize species into tiers

First, we classify all 234 competition species by what teacher signal is available:

| Tier | Which species | N | Teacher | Why |
|------|--------------|---|---------|-----|
| **A** | 75 species that appear in labeled soundscapes | 75 | SED | SED validated at 0.9987 AUC on these |
| **B** | 109 unseen species where SED max-folds predicts >0.5 somewhere | 109 | SED | SED has sparse but precise predictions (94% precision) |
| **C** | 44 unseen species where SED predicts <0.5 everywhere, but Perch has signal | 44 | Perch | SED gives literally zero; Perch is the only option |
| **D** | 6 species invisible to both models | 6 | None | 1-5 focal recordings each; can't help |

**Tier C (44 species) is the highest-value target** — they have zero soundscape
representation today and account for a large share of the LB gap.

### Step 2: Generate SED pseudo-labels (Tiers A+B — 184 species)

Starting from the 4-fold SED logits we already have:

```
Raw logits (4 folds × 127K segments × 234 species)
    ↓
Sigmoid → probabilities per fold
    ↓
Max across folds (take the MOST confident fold per species per segment)
    ↓
Temperature scaling T=0.6 (sharpen: confident → more confident, uncertain → more uncertain)
    ↓
Power transform P^1.5 (crush borderline predictions, preserve confident ones)
    ↓
Result: soft probability per species per segment
```

**Example**: What happens to a probability through this pipeline:

| Original prob | After max-folds | After T=0.6 | After P^1.5 | Survives >0.5? |
|--------------|-----------------|-------------|-------------|----------------|
| 0.95 | 0.95 | 0.98 | 0.97 | Yes (strong signal) |
| 0.80 | 0.80 | 0.87 | 0.81 | Yes (good signal) |
| 0.60 | 0.60 | 0.65 | 0.52 | Barely (weak signal) |
| 0.50 | 0.50 | 0.53 | 0.38 | No — killed by P^1.5 |
| 0.30 | 0.30 | 0.25 | 0.13 | No — strongly suppressed |

The pipeline keeps high-confidence predictions and kills borderline ones. The output is
**soft** — we don't force 0/1, we keep the actual probability as the training target.

Note: We don't apply a hard >0.5 threshold. The power transform naturally suppresses low
probabilities, and the soft label format means weak predictions (e.g., 0.38) contribute
very little to the training gradient anyway.

**Expected yield**: ~5K species×segment predictions above 0.5 for the 109 unseen Tier B
species, plus expanded site/hour diversity for the 75 Tier A species.

### Step 3: Generate Perch pseudo-labels (Tier C — 44 species)

This is the hardest part. The problem: Perch raw probabilities are miscalibrated. Example:
Social Flycatcher (`socfly1`) has Perch prob >0.5 in **127,017 out of 127,104 segments**
(99.9%). Standard thresholding is useless — everything passes.

**The insight**: Even though absolute probabilities are wrong, the **relative ranking
within a species** carries signal. If Perch gives species X probability 0.98 in one
segment and 0.72 in another, the 0.98 segment is genuinely more likely to contain
species X.

**Per-species z-normalization**:

```
For species X across all 127K segments:
  mean = 0.85  (average Perch prob for this species)
  std  = 0.08  (standard deviation)

  Segment A: prob = 0.98  →  z = (0.98 - 0.85) / 0.08 = +1.6  (above average)
  Segment B: prob = 0.85  →  z = (0.85 - 0.85) / 0.08 =  0.0  (at average)
  Segment C: prob = 0.72  →  z = (0.72 - 0.85) / 0.08 = -1.6  (below average)
  Segment D: prob = 0.99  →  z = (0.99 - 0.85) / 0.08 = +1.75 (well above)
  Segment E: prob = 1.00  →  z = (1.00 - 0.85) / 0.08 = +1.88 (peak confidence)
```

Keep only segments where z > 2.0 (top ~2.3% for that species). These are the segments
where Perch is **unusually confident** relative to its own baseline.

Then convert z-scores to soft labels: `sigmoid(2.0) = 0.88`, `sigmoid(3.0) = 0.95`.
Apply P^1.5 after sigmoid.

**Safety check**: If a species has std < 0.01 (Perch gives essentially identical
predictions for all segments), the z-scores are noise → skip that species.

**Expected yield per species**: 127K × 2.3% ≈ ~3,000 segments at z > 2.0.
Total for Tier C: ~3,000 × 44 species ≈ ~130K pseudo-labels.

### Step 4: Merge into one pseudo-label file

```
For each of the 127K segments × 234 species:
  If Tier A or B species → use SED pseudo-label (from Step 2)
  If Tier C species     → use Perch pseudo-label (from Step 3)
  If Tier D species     → 0 (no pseudo-label available)
```

Save as one parquet: `merged_pseudo_labels.parquet` (127,896 × 234).

### Step 5: Train the student model

Use the **exact same v5j recipe** (HGNetV2-B0, LSEHead, BCE, 5s windows, 20 epochs).
The ONLY change is adding pseudo-labeled soundscapes to the training data.

| Data source | Segments | Label type | Loss weight |
|-------------|----------|------------|-------------|
| Focal recordings | ~265K | Hard ground truth (primary label = 1, rest = 0) | 1.0 |
| Labeled soundscapes | ~792 | Hard ground truth (multi-label, 0/1 per species) | 1.0 |
| **Pseudo-labeled soundscapes** | **~127K** | **Soft probabilities (0.0 to ~0.97)** | **0.5** |

Each training batch has:
- 50% from focal recordings + labeled soundscapes (hard ground truth, loss weight 1.0)
- 50% from pseudo-labeled soundscapes (soft pseudo-labels, loss weight 0.5)

The reduced loss weight (0.5) means pseudo-labels influence the model half as much as
ground truth per sample — limiting damage from incorrect labels while still allowing the
model to learn soundscape patterns.

**Why train from scratch, not fine-tune**: We train a new v5j model on ALL data (focal +
labeled + pseudo-labeled), not fine-tune the existing v5j on pseudo-labels only. This
ensures the model maintains its focal recording knowledge while adding soundscape knowledge.

### Step 6: Evaluate

1. Train fold 0 as sanity check → submit to LB → compare vs v5j fold 0 (0.906)
2. **Decision gate**: If fold 0 > 0.906 → proceed. If regression → diagnose and adjust.
3. If promising: train folds 1-3, submit 4-fold ensemble (compare vs 0.913)
4. Submit student + Perch ensemble (compare vs 0.929 current best)

### Step 7: Iterate (if round 1 works)

The round 1 student was trained on soundscape data, so its predictions on soundscapes
should be **better calibrated** than the original SED/Perch teacher.

1. Run student inference on all 127K soundscapes
2. Generate new pseudo-labels (student predictions should be cleaner)
3. Particularly for Tier C: student may now have genuine signal for those 44 species
   (it learned from Perch pseudo-labels), so its predictions are more trustworthy
4. Train round 2 student → submit → compare
5. Consider relaxing Perch z-threshold (2.0 → 1.5) for more coverage in round 2

---

## 6. Implementation Phases

### Phase A: Generate pseudo-labels

**A.1: SED pseudo-labels (Tiers A+B)**

```
Input:  4-fold SED logits (shared/inference/.../hgnet-b0-sed-v5j-4fold/train_soundscapes/)
Step 1: sigmoid(logits) per fold → 4 probability matrices (127,896 × 234)
Step 2: max(fold0, fold1, fold2, fold3) per species per segment
Step 3: temperature scaling: logit = log(p/(1-p)), logit_T = logit / 0.6, p_T = sigmoid(logit_T)
Step 4: power transform: p_final = p_T ^ 1.5
Output: sed_pseudo_labels.parquet (127,896 × 234, soft probabilities)
```

**A.2: Perch pseudo-labels (Tier C only)**

```
Input:  Perch raw probs (shared/inference/.../perch-v2-tf/train_soundscapes/perch_234_raw_probs.parquet)
Step 1: For each of the 44 Tier C species:
          - Compute mean, std across all 127K segments
          - Skip if std < 0.01 (no signal variation)
          - z = (prob - mean) / std
          - Keep segments where z > 2.0 (top ~2.3%)
          - Soft label = sigmoid(z) ^ 1.5
Step 2: For all non-Tier-C species: set to 0 (Perch doesn't contribute)
Output: perch_pseudo_labels.parquet (127,896 × 234, soft probabilities, mostly zeros)
```

**A.3: Merge and validate**

```
For each species:
  if Tier A or B: use SED pseudo-label
  if Tier C:      use Perch pseudo-label
  if Tier D:      no pseudo-label (zeros)
Output: merged_pseudo_labels.parquet (127,896 × 234)
```

### Phase B: Train student (Round 1)

Extend v5j training pipeline. Recipe unchanged — only data changes.

### Phase C: Evaluate and iterate

1. Fold 0 → LB → decision gate
2. If promising → full 4-fold + ensemble
3. If round 1 improves → student-as-teacher for round 2

---

## 7. Hyperparameters and Tuning Plan

All hyperparameters for round 1, with the rationale for each starting value:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **SED temperature T** | **0.6** | Nikita used 0.55-0.6 and won. Our teacher is strong (0.929 LB, 94% precision) — we can trust its confidence signal, so sharper is appropriate. |
| **Power transform α** | **1.5** | Nikita's technique. After T=0.6, a probability must be >0.63 originally to survive above 0.5 after P^1.5. This kills borderline predictions while preserving confident ones. |
| **Perch z-threshold** | **2.0** | Selects top ~2.3% of segments per species (≈3,000 per species). Enough data to learn from (~130K total for 44 species), but selective enough to exclude the 97.7% of segments where Perch's signal is average or below. |
| **PL loss weight** | **0.5** | Pseudo-labels get half the per-sample influence of ground truth. Combined with soft labels (uncertain predictions already contribute less gradient) and P^1.5 (borderline already suppressed), this lets good pseudo-labels teach while limiting damage from bad ones. Too low (0.2-0.3) and we waste the soundscape data; too high (0.7+) and noise from pseudo-labels can override ground truth. |
| **Batch PL fraction** | **0.5** | 50% focal/labeled, 50% pseudo-labeled per batch. Ensures every batch trains on both domains. |
| **Hard threshold** | **None** | Soft labels — P^1.5 + loss weight manage quality without discarding information. A prediction of 0.38 after P^1.5 contributes almost nothing to the gradient anyway. |
| **Training epochs** | **20** | Same as v5j baseline. More data does not require more epochs — the model sees the same total gradient steps. |

**Tuning priority** (if round 1 needs adjustment):
1. PL loss weight (0.3 → 0.5 → 0.7) — biggest lever for controlling pseudo-label influence
2. Perch z-threshold (1.5 → 2.0 → 2.5) — controls Tier C noise level
3. Power transform (1.5 → 2.0) — more aggressive noise suppression
4. SED temperature (0.5 → 0.6 → 0.8) — sharpening intensity

---

## 8. Evaluation

### Metrics

| Metric | Purpose | Where |
|--------|---------|-------|
| **Public LB** | Primary — only reliable evaluation | Kaggle submission |
| Training loss convergence | Sanity — should look similar to v5j | TensorBoard |
| Labeled soundscape AUC | Sanity — should remain >0.99 | Offline (66 files) |
| Pseudo-label coverage | Diagnostic — species×segments at each threshold | Phase A output |

### Baselines

| Model | LB | Notes |
|-------|-----|-------|
| v5j fold 0 | 0.906 | Single-fold SED baseline |
| v5j 4-fold | 0.913 | 4-fold SED ensemble |
| v5j 4-fold + PP | 0.917 | + post-processing |
| Perch+SED ensemble | **0.929** | Current best |

### Success criteria

| Round | Target | Meaning |
|-------|--------|---------|
| Round 1 (fold 0) | **>0.910** | Pseudo-labels help vs v5j fold 0 (0.906) |
| Round 1 (4-fold) | **>0.920** | Ensemble benefits compound |
| Round 1 (+ Perch) | **>0.935** | Closing the gap to #1 |
| Round 2+ | Any further gain | Iteration adds value |

### Failure modes and response

| Observation | Diagnosis | Response |
|-------------|-----------|----------|
| LB < 0.906 (regression) | Pseudo-labels too noisy | Reduce PL weight to 0.3, increase P^α to 2.0 |
| LB ≈ 0.906 (flat) | Pseudo-labels neutral | Check: are unseen species improving? Seen species regressing? |
| Labeled AUC drops below 0.98 | Pseudo-labels corrupting seen species | Reduce PL weight, increase Tier A+B temp sharpening |
| Fold 0 OK but ensemble doesn't improve | Pseudo-labels redundant across folds | Expected — each fold sees similar PL data |

---

## 9. Iteration Strategy

| Round | Teacher | What changes | Goal |
|-------|---------|--------------|------|
| **1** | SED max-folds + Perch (per-species normalized) | Add pseudo-labels to v5j training | Close the data coverage gap |
| **2** | Round 1 student (4-fold) | Better-calibrated teacher (trained on soundscapes) | Improve PL quality for Tier C species |
| **3** | Round 2 student + backbone diversity | Add EfficientNet-B3 NS student | Diverse ensemble, better teacher |

**Stop conditions**:
- Stop if LB plateaus for 2 consecutive rounds
- Stop if round 1 regresses and diagnosis variants don't help
- Backbone diversity (round 3) is a separate experiment, not a PL iteration

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pseudo-labels too noisy → student regresses | Medium | High | P^1.5, soft labels, w=0.5, start conservative |
| Perch Tier C z-normalization doesn't extract signal | Medium | Medium | Per-species std check; can drop Tier C in round 1 if needed |
| v5j recipe breaks with new data distribution | Low | High | Recipe unchanged; only data changes. Monitor train curves |
| Exp 015 repeats (student worse than teacher) | Low | High | Fundamentally different: 5× teacher, SED arch, domain match |
| 6 invisible species hurt LB | Low | Low | <2.5% of species, ~0.5% of LB. Accept the loss |
| Overfitting to pseudo-labels | Medium | Medium | w=0.5 loss weight, batch mixing, monitor labeled AUC |

---

## 11. Key References

| Document | Location |
|----------|----------|
| Exp 024 results (all findings) | `experiments/024-data-centric-exploration/results.md` |
| Inference EDA deep dive (14 sections) | `notes/2026-04-09-inference-eda-deep-dive.md` |
| Data-centric pivot plan | `notes/2026-04-09-data-centric-pivot.md` |
| Exp 015 failure analysis | `FINDINGS.md` (pseudo-labels section) |
| Nikita 1st place BirdCLEF 2025 | `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` |
| v5j training recipe | `experiments/021-sed-training/` |
| SED 4-fold inference outputs | `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/` |
| Perch raw probs | `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_raw_probs.parquet` |
| Perch full pipeline probs | `shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet` |
