# Concept: Weak Supervision Label Fusion for Token-Level SED

**Status**: developing
**Created**: 2026-02-08
**Updated**: 2026-03-14

## Idea

Combine multiple noisy label sources (bambird events, BirdNET/ConvNeXT predictions, energy detection, recording-level tags) into calibrated per-token soft labels using principled weak supervision techniques. Instead of treating any single source as ground truth, model each source's accuracy and produce probabilistic training targets with quantified uncertainty.

## Motivation

Each label source we have for token-level SED is noisy in a different way:

| Source | Strengths | Weaknesses |
|---|---|---|
| bambird events | Good temporal precision (10ms) | No species ID, misses quiet calls, false positives from non-target sounds |
| ConvNeXT predictions | Good species ID (0.48 cmAP) | 5s temporal resolution, domain-dependent accuracy |
| BirdNET predictions | Species ID for 6500+ species | 3s resolution, CC BY-NC-SA license |
| Energy detection | Simple, fast, decent recall | High false positive rate (any loud sound triggers) |
| Recording-level label | Guaranteed correct species | No temporal info — labels all tokens positive |

Using any single source naively introduces systematic bias. bambird alone misses quiet calls. ConvNeXT alone smears temporal boundaries across 5s windows. Recording labels alone produce zero negatives.

A principled label fusion approach:
- Produces **calibrated soft labels** (P=0.92 vs P=0.15 vs P=0.03)
- Estimates **each source's accuracy** without ground truth
- Enables **uncertainty-weighted training** (confident tokens contribute more to gradient)
- Is a **one-time CPU computation** cached alongside spectrograms

## Background

### Data Programming / Snorkel (Ratner et al., 2017)

The foundational framework: define labeling functions (LFs), fit a generative label model that learns LF accuracies and correlations, output calibrated probabilistic labels. The label model uses a factor graph over LF agreement/disagreement patterns.

**Status**: The Snorkel open-source library is effectively in maintenance mode. Dependency pins (`torch<2.0`, `numpy<1.20`) conflict with modern PyTorch stacks. The concept is sound but the library is not practically usable without vendoring.

### FlyingSquid (Fu et al., 2020)

Replaces Snorkel's iterative SGD with a closed-form triplet method — 170x faster, fewer hyperparameters, same quality. Requires minimum 3 conditionally independent LFs.

**Status**: Research artifact, no updates since 2020. The algorithm (~200 lines) is extractable but the library is unmaintained.

### Cleanlab / Confident Learning (Northcutt et al., 2021)

Different paradigm: train a model on noisy labels, then use out-of-sample predictions to identify and correct label errors. Estimates the noise transition matrix P(noisy_label | true_label). Well-maintained library (~11k GitHub stars, active through 2026).

**Status**: Actively maintained. Has a `multiannotator` module for combining multiple label sources. However, requires a trained model first (two-pass: train → clean → retrain).

### Dawid-Skene (1979) via Crowd-Kit

Classic EM-based crowdsourcing aggregation: models each annotator's confusion matrix, iteratively estimates true labels. Available through Toloka's `crowd-kit` library (actively maintained, released Oct 2025).

**Status**: Well-maintained library, proven algorithm, lightweight dependencies.

## Core Algorithm: FlyingSquid Triplet Method

The recommended label fusion approach. Closed-form, ~30 lines of numpy, no dependencies, no hyperparameters.

### Why FlyingSquid over Dawid-Skene

| | FlyingSquid (reimplement) | Dawid-Skene (Crowd-Kit) |
|---|---|---|
| **Computation** | Closed-form, instant | Iterative EM, 100+ iterations |
| **Dependencies** | numpy only | pandas, sklearn, crowd-kit |
| **Data format** | `(n_tokens, m_lfs)` array | Long-format DataFrame (millions of rows) |
| **Hyperparameters** | None | n_iter, convergence threshold |
| **Binary labels** | Natural fit | Models full confusion matrix (overkill) |
| **Code** | ~30 lines | Library install + data reshaping |
| **Correctness** | Provably recovers true parameters | Local optima possible |

For our use case — binary per-token labels, 3-5 LFs, millions of tokens — FlyingSquid is the clear choice.

### How it works

The core insight: with 3 conditionally independent labeling functions, their observable agreement statistics **uniquely determine** each LF's accuracy. No ground truth needed.

```
Given: m labeling functions, each outputting {-1, 0, +1} per token
       -1 = negative, 0 = abstain, +1 = positive

Step 1: Compute observable statistics
══════════════════════════════════════

  For each pair (i, j):    μ_ij  = mean(Li * Lj)      ← pairwise agreement
  For each triple (i,j,k): μ_ijk = mean(Li * Lj * Lk)  ← triple agreement

  These are just averages over all tokens. Trivial to compute.


Step 2: Solve for each LF's accuracy (closed form)
═══════════════════════════════════════════════════

  For LF k, using any other pair (i, j):

                μ_ijk
  accuracy_k = ─────── + 1
                μ_ij
               ─────────────
                    2

  Why this works (intuition):
  ┌──────────────────────────────────────────────────────────┐
  │ μ_ij  = E[Li·Lj] depends on accuracies of i AND j       │
  │ μ_ijk = E[Li·Lj·Lk] depends on accuracies of i, j, AND k│
  │                                                          │
  │ Dividing cancels out i and j:                            │
  │   μ_ijk / μ_ij  ≈  (2·accuracy_k - 1)                   │
  │                                                          │
  │ Solving: accuracy_k = (μ_ijk/μ_ij + 1) / 2              │
  └──────────────────────────────────────────────────────────┘


Step 3: With m > 3 LFs, average over all triplets
══════════════════════════════════════════════════

  For 5 LFs: (5 choose 3) = 10 triplets
  Each triplet gives an independent estimate of each LF's accuracy
  Take median across estimates → robust to outlier triplets

  Example with 5 LFs:
    LF         Estimated accuracies from each triplet    Median
    bambird    [0.82, 0.85, 0.79, 0.83, 0.81, ...]     0.82
    energy     [0.61, 0.58, 0.63, 0.60, 0.59, ...]     0.60
    convnext   [0.88, 0.91, 0.87, 0.89, 0.90, ...]     0.89
    birdnet    [0.85, 0.83, 0.86, 0.84, 0.85, ...]     0.85
    rec_label  [0.52, 0.51, 0.53, 0.52, 0.51, ...]     0.52


Step 4: Compute soft labels via Bayes rule
══════════════════════════════════════════

  With known accuracies a1...am, for each token t:

  log_odds(t) = Σ  Li(t) · log(ai / (1 - ai))
                i

  P(active | all LFs) = sigmoid(log_odds)

  Each LF's vote is weighted by its log-odds accuracy.
  A high-accuracy LF's vote counts more.
  An LF that abstains (Li=0) contributes nothing.
```

### Reference implementation

```python
import numpy as np
from itertools import combinations

def flying_squid(L):
    """Triplet method for label fusion.

    Args:
        L: (n_tokens, m_lfs) array of {-1, 0, +1}
           -1 = negative, 0 = abstain, +1 = positive

    Returns:
        soft_labels: (n_tokens,) probabilities in [0, 1]
        accuracies: (m_lfs,) estimated accuracy per LF
    """
    n, m = L.shape

    # Estimate accuracy for each LF via triplets
    accuracies = np.zeros(m)
    for k in range(m):
        estimates = []
        others = [j for j in range(m) if j != k]
        for i, j in combinations(others, 2):
            mask = (L[:, i] != 0) & (L[:, j] != 0) & (L[:, k] != 0)
            if mask.sum() < 10:
                continue
            Lm = L[mask]
            mu_ij = (Lm[:, i] * Lm[:, j]).mean()
            mu_ijk = (Lm[:, i] * Lm[:, j] * Lm[:, k]).mean()
            if abs(mu_ij) > 1e-6:
                estimates.append(mu_ijk / mu_ij)
        if estimates:
            accuracies[k] = np.clip((np.median(estimates) + 1) / 2, 0.01, 0.99)
        else:
            accuracies[k] = 0.5

    # Bayes rule: combine LF votes weighted by log-odds accuracy
    log_odds = np.zeros(n)
    for k in range(m):
        a = accuracies[k]
        log_ratio = np.log(a / (1 - a))
        log_odds += L[:, k] * log_ratio

    soft_labels = 1 / (1 + np.exp(-log_odds))
    return soft_labels, accuracies
```

## Full Pipeline: Teachers + FlyingSquid + Weakly Supervised Training

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ONE-TIME PRE-COMPUTATION (CPU/GPU)                │
│                                                                     │
│  XenoCanto recordings                                               │
│       │                                                             │
│       ├──→ bambird ──────────→ event timestamps ──→ LF_bambird      │
│       │    (CPU, unsupervised)   per recording       {-1, 0, +1}    │
│       │                                              per token      │
│       ├──→ ConvNeXT teacher ─→ species confidence ─→ LF_convnext    │
│       │    (GPU, 5s windows)     per 5s window       {-1, 0, +1}    │
│       │                                              per token      │
│       ├──→ BirdNET teacher ──→ species confidence ─→ LF_birdnet     │
│       │    (CPU, 3s windows)     per 3s window       {-1, 0, +1}    │
│       │                                              per token      │
│       ├──→ energy detector ──→ frame-level energy ─→ LF_energy      │
│       │    (CPU, per-frame)      per frame           {-1, 0, +1}    │
│       │                                              per token      │
│       └──→ recording label ──→ species tag ────────→ LF_recording   │
│            (metadata)            per recording       {+1} always    │
│                                                                     │
│       All LFs combined into label matrix L: (n_tokens, 5)           │
│                         │                                           │
│                         ▼                                           │
│                   FlyingSquid                                       │
│            (closed-form, ~30 lines numpy)                           │
│                         │                                           │
│                         ▼                                           │
│              soft_labels: (n_tokens,) ∈ [0, 1]                      │
│              accuracies: [0.82, 0.60, 0.89, 0.85, 0.52]            │
│                         │                                           │
│                         ▼                                           │
│            Cache in .pt files alongside mel spectrograms            │
│            {mel, soft_labels, padding_mask, accuracies}             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    GPU TRAINING                                      │
│                                                                     │
│  Load cached:  mel (1, 128, 3000)                                   │
│                soft_labels (3000,) per species                       │
│                padding_mask (3000,)                                  │
│                         │                                           │
│                         ▼                                           │
│                   SSAMBA backbone                                    │
│                         │                                           │
│                         ▼                                           │
│               token embeddings (B, 3000, D)                         │
│                         │                                           │
│                    SED head                                          │
│               Linear(D, 1) per token                                │
│                         │                                           │
│                         ▼                                           │
│              logits (B, 3000, 1)                                    │
│                         │                                           │
│                         ▼                                           │
│         BCE(logits, soft_labels) * padding_mask                     │
│         Optionally: * confidence_weight                             │
│              (|soft_label - 0.5| * 2)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    POST-TRAINING REFINEMENT (optional)               │
│                                                                     │
│  Trained model predictions (cross-validated)                        │
│                         │                                           │
│                         ▼                                           │
│              Cleanlab: find label issues                             │
│              Identify tokens where model disagrees                  │
│              strongly with fused labels                             │
│                         │                                           │
│                         ▼                                           │
│              Corrected soft labels                                  │
│                         │                                           │
│                         ▼                                           │
│              Retrain with cleaned labels                            │
│              (repeat 1-2 cycles)                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-step: constructing the label matrix

Each teacher/detector produces continuous outputs. We discretize to {-1, 0, +1} for FlyingSquid:

```python
def build_label_matrix(recording, species_idx, target_sr=32000):
    """Build (n_tokens, m_lfs) label matrix for one 30s chunk.

    Each LF votes: +1 (species active), -1 (species not active), 0 (abstain).
    """
    n_tokens = 3000  # 30s at 10ms/token
    L = np.zeros((n_tokens, 5), dtype=np.int8)

    # LF 0: bambird — vote +1 where dominant-cluster event overlaps, -1 elsewhere
    for start_s, end_s in bambird_events:
        t_start = int(start_s / 0.01)
        t_end = int(end_s / 0.01)
        L[t_start:t_end, 0] = +1
    L[L[:, 0] == 0, 0] = -1  # no event = vote negative

    # LF 1: energy — vote +1 above threshold, -1 below, 0 near threshold
    for t in range(n_tokens):
        if energy[t] > high_threshold:
            L[t, 1] = +1
        elif energy[t] < low_threshold:
            L[t, 1] = -1
        # else: abstain (0)

    # LF 2: ConvNeXT — 5s windows, vote based on species confidence
    for win_start in range(0, n_tokens, 500):  # 500 tokens = 5s
        win_end = min(win_start + 500, n_tokens)
        conf = convnext_confidence[species_idx]  # for this 5s window
        if conf > 0.5:
            L[win_start:win_end, 2] = +1
        elif conf < 0.1:
            L[win_start:win_end, 2] = -1
        # else: abstain

    # LF 3: BirdNET — 3s windows, same logic as ConvNeXT
    for win_start in range(0, n_tokens, 300):  # 300 tokens = 3s
        win_end = min(win_start + 300, n_tokens)
        conf = birdnet_confidence[species_idx]
        if conf > 0.5:
            L[win_start:win_end, 3] = +1
        elif conf < 0.1:
            L[win_start:win_end, 3] = -1

    # LF 4: recording label — always votes +1 (weak positive prior)
    L[:, 4] = +1

    return L  # (3000, 5)
```

Then fuse all recordings:

```python
# Stack label matrices for all tokens across all recordings
all_L = np.concatenate([build_label_matrix(rec) for rec in recordings])  # (N_total_tokens, 5)

# Run FlyingSquid once
soft_labels, accuracies = flying_squid(all_L)

# accuracies might look like:
# bambird=0.82, energy=0.60, convnext=0.89, birdnet=0.85, recording=0.52
# → ConvNeXT and BirdNET are the most reliable teachers
# → recording label is barely above chance (as expected — it's always +1)
# → energy detection is noisy but contributes some signal
```

### What the teachers contribute

```
Time →  0s    5s    10s   15s   20s   25s   30s
        │     │     │     │     │     │     │

bambird:  ──██──────██████───────────████──────
          precise timing, but no species ID

ConvNeXT: ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓░░░░░
          5s blocks: "species X likely" / "unlikely"

BirdNET:  ▓▓▓▓▓▓░░░▓▓▓▓▓▓░░░░░▓▓▓▓▓▓░░░░░░░░░
          3s blocks: finer than ConvNeXT

Energy:   ─███─────████████──────────█████─────
          any loud sound (high false positive)

Recording:+++++++++++++++++++++++++++++++++++++
          always positive (useless alone)

FlyingSquid fusion:
          ─0.9─────0.95─────0.03────0.88──0.04─
          ↑                 ↑       ↑
          bambird+teachers  silence  bambird+teachers
          agree: confident  all LFs  agree: confident
          positive          agree:   positive
                           confident
                           negative
```

The power is in the **intersection**: bambird says "something is vocalizing at 5-10s", ConvNeXT says "this 5s window contains species X" → high confidence. bambird detects energy at 20s but ConvNeXT says "no species X here" → resolved as negative (probably a different sound). Silence regions where all LFs agree → confident negative.

### Confidence-weighted training loss

Not all soft labels are equally trustworthy. A token with P=0.91 (all LFs agree) should contribute more to training than P=0.55 (LFs disagree):

```python
def confidence_weighted_bce(logits, soft_labels, padding_mask):
    """BCE weighted by label confidence."""
    loss = F.binary_cross_entropy_with_logits(
        logits, soft_labels, reduction='none'
    )

    # Confidence = how far from 0.5 (uncertain)
    confidence = (soft_labels - 0.5).abs() * 2  # 0 at P=0.5, 1 at P=0/1

    # Weight loss by confidence and mask
    weighted = loss * confidence * padding_mask.unsqueeze(-1)
    return weighted.sum() / (padding_mask.sum() * num_classes + 1e-8)
```

This means:
- P=0.95 → confidence=0.90, strong positive gradient
- P=0.05 → confidence=0.90, strong negative gradient
- P=0.55 → confidence=0.10, almost ignored (LFs disagree, don't trust this token)

## Recommended Approach: Progressive Complexity

### Level 0: Weighted Average (zero dependencies)

Start here for exp 007. Combine LF outputs as a weighted soft label per token:

```python
soft_label = (w_bambird * bambird_score
            + w_energy * energy_score
            + w_convnext * convnext_score
            + w_recording * recording_prior)
soft_label = soft_label.clamp(0, 1)
loss = F.binary_cross_entropy_with_logits(logits, soft_label)
```

**When to use**: first iteration, validating that token-level SED works at all.

### Level 1: FlyingSquid Triplet Method (principled, ~30 lines numpy)

Upgrade when you have 3+ LFs and want principled accuracy estimation. No dependencies beyond numpy, no hyperparameters, instant computation.

**When to use**: the default for multi-teacher fusion. Simple, correct, fast.

### Level 2: Cleanlab Refinement (post-training)

After training a model with Level 0 or 1 labels, use Cleanlab to identify and fix label errors via the model's own predictions. Iterate train → clean → retrain.

**When to use**: after initial training, to squeeze out extra performance.

## Integration with Training Pipeline

FlyingSquid is **completely offline** — it runs once as a pre-computation step, caches soft labels, and the training loop loads them like any other label. There is zero runtime cost during GPU training.

### Job structure (4 sbatch jobs, sequential)

```
Job 1: precompute_mels.sbatch        (GPU, ~30 min)
  ├── Load audio from HF dataset
  ├── Compute mel spectrograms per 30s chunk
  └── Save: {chunk_id}.pt = {mel, labels, real_frames}

Job 2: run_teachers.sbatch           (GPU+CPU, ~1 hour)
  ├── ConvNeXT inference: 5s sliding windows over all recordings → species confidence
  ├── BirdNET inference: 3s sliding windows → species confidence (optional)
  ├── bambird: extract detected_events per recording (CPU, from BirdSet or rerun)
  ├── Energy detection: per-frame RMS energy (CPU, trivial)
  └── Save: {recording_id}_teachers.npz = {convnext_5s, birdnet_3s, bambird_events, energy}

Job 3: fuse_labels.sbatch            (CPU only, ~5 min)
  ├── For each species in the dataset:
  │   ├── Build label matrix L: (n_total_tokens, m_lfs) from teacher outputs
  │   ├── Run flying_squid(L) → soft_labels, accuracies
  │   └── Log: "species {name}: bambird={acc:.2f}, convnext={acc:.2f}, ..."
  ├── Reshape soft_labels back to per-chunk, per-species arrays
  └── Save: {chunk_id}_fused.pt = {soft_labels: (3000, C), confidence: (3000, C)}
      where C = num_species

Job 4: train.sbatch                  (GPU, ~12 hours)
  └── Standard training loop, loading pre-fused labels
```

FlyingSquid itself is ~10 seconds of the 5-minute Job 3. The rest of Job 3 is I/O — loading teacher outputs and saving fused labels.

### Cache format

Each training chunk gets two files:

```python
# From Job 1 (mel cache):
torch.save({
    "mel": mel_spec,           # (1, 128, 3000) float16
    "labels": clip_labels,     # (C,) binary recording-level labels
    "real_frames": 2847,       # int, for padding mask
}, f"cache/{chunk_id}.pt")

# From Job 3 (fused labels):
torch.save({
    "soft_labels": soft_labels,  # (3000, C) float16, per-token per-species
    "confidence": confidence,    # (3000, C) float16, |soft_label - 0.5| * 2
}, f"cache/{chunk_id}_fused.pt")
```

Or merge into a single file to simplify the dataloader:

```python
# Combined cache (preferred):
torch.save({
    "mel": mel_spec,             # (1, 128, 3000)
    "soft_labels": soft_labels,  # (3000, C)
    "confidence": confidence,    # (3000, C)
    "real_frames": 2847,
}, f"cache/{chunk_id}.pt")
```

### Dataset and training loop

The training code change is minimal — the dataset loads soft labels from cache, and BCE works with soft targets out of the box:

```python
class CachedMelDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(Path(cache_dir).glob("*.pt"))

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return {
            "mel": data["mel"],                # (1, 128, 3000)
            "soft_labels": data["soft_labels"], # (3000, C)
            "confidence": data["confidence"],   # (3000, C)
            "real_frames": data["real_frames"],  # int
        }

def training_step(self, batch, batch_idx):
    mel = batch["mel"]                    # (B, 1, 128, 3000)
    soft_labels = batch["soft_labels"]    # (B, 3000, C)
    confidence = batch["confidence"]      # (B, 3000, C)
    real_frames = batch["real_frames"]    # (B,)

    # Build padding mask
    mask = torch.arange(3000, device=mel.device).expand(B, -1)
    mask = (mask < real_frames.unsqueeze(1)).float()  # (B, 3000)

    logits = self.model(mel, padding_mask=mask)       # (B, 3000, C)

    # BCE with soft targets — works identically to hard labels
    loss = F.binary_cross_entropy_with_logits(
        logits, soft_labels, reduction='none'
    )

    # Weight by confidence and mask
    loss = (loss * confidence * mask.unsqueeze(-1)).sum()
    loss = loss / (mask.sum() * C + 1e-8)
    return loss
```

### Key point: no runtime overhead

The entire weak supervision pipeline (teachers + FlyingSquid + caching) runs **before** training starts. During training, the model sees pre-computed `(mel, soft_labels, confidence)` tuples — exactly the same data loading path as hard labels. Switching from hard bambird labels to fused soft labels requires:

1. Running Jobs 2-3 once (~1 hour)
2. Changing `token_labels[t] = 1.0 if event_overlaps(t) else 0.0` to `token_labels[t] = cached_soft_label[t]`

That's it. No model architecture changes, no new dependencies at training time, no additional GPU cost.

## Open Questions

- **Weight tuning**: For Level 0, how to set weights without a labeled validation set? Could use LF agreement as a proxy — weights proportional to how often an LF agrees with the majority.
- **Temporal resolution mismatch**: ConvNeXT gives 5s resolution, bambird gives event-level, energy gives frame-level. How to handle the granularity mismatch when fusing? Linear interpolation? Step function?
- **Abstention**: Should LFs abstain on tokens where they have no signal (e.g., ConvNeXT at a boundary between two 5s windows)? Snorkel/FlyingSquid handle abstention natively; weighted average needs explicit handling.
- **Confidence weighting in loss**: Should uncertain tokens (soft label near 0.5) contribute less to gradient? Could multiply loss by `|soft_label - 0.5| * 2` to down-weight ambiguous tokens.
- **Iterative refinement**: How many train → clean → retrain cycles are useful before diminishing returns?

## Testable Hypotheses

1. **Soft labels outperform hard labels**: Token-level SED trained with fused soft labels achieves higher val/cmAP than the same model trained with bambird-only hard labels, because the soft labels are more accurate and better calibrated.

2. **Principled fusion outperforms weighted average**: Dawid-Skene or FlyingSquid-derived labels outperform hand-weighted averages, especially when LFs are correlated (bambird and energy detection both trigger on loud sounds).

3. **Cleanlab refinement improves trained models**: A retrain cycle with Cleanlab-identified label corrections improves val/cmAP by at least 0.02 over the initial training with noisy labels.

4. **Uncertainty weighting helps**: Down-weighting ambiguous tokens (soft label near 0.5) in the loss improves calibration and val/cmAP compared to uniform token weighting.

## Connections

- **Based on**: [Unsupervised event detection](unsupervised-event-detection-labeling.md) (bambird as a labeling function), [bambird + scikit-maad](../vendor/bambird-scikit-maad.md)
- **Related to**: [Prompted SED](prompted-sound-event-detection.md) (label fusion provides training targets for any SED approach), [Soundscape domain shift](soundscape-domain-shift.md) (label quality affects domain transfer)
- **Could lead to**: Integration into experiments 007/008/009 as a label quality improvement

## Notes

- The progressive complexity approach (Level 0 → 1 → 2 → 3) means we never invest more engineering than the current bottleneck justifies. If Level 0 weighted average works well enough, there's no need for Dawid-Skene.
- All levels produce the same output format (soft labels in [0, 1] per token), so the training code doesn't change between levels — only the pre-computation step.
- Cleanlab (Level 2) is the only approach that requires a trained model. Levels 0-1 and 3 are pure pre-computation before any training.
- For the BirdCLEF competition, the LF pre-computation is a one-time cost per dataset. Once cached, switching between label fusion methods only requires re-running the fusion step, not re-computing the individual LFs.

## Lessons Learned (exp 013, 014, 015)

Exp 013 implemented the triplet method and produced fused soft labels. Exp 014/015 tested them as training targets. Key outcomes:

### What worked
- **Triplet method is sound**: pure numpy, no dependencies, estimated LF accuracies match intuition (BirdNET 0.785, Perch 0.784, recording label 0.932)
- **Focal soft labels are well-calibrated**: present species ~0.575, absent ~0.001, strongly bimodal
- **v2 raw logit LFs**: adding BirdNET/Perch raw logits as extra LFs for bird species improved estimated accuracy to 0.920/0.912

### What failed
- **Soft labels as direct training targets hurt linear probes**: exp 014 (0.238 cmAP) vs exp 012 hard-label baseline (0.336 cmAP). Adding soundscape data (+65%) had zero effect. Don't replace ground-truth hard labels with teacher soft labels.
- **SED head on frozen backbone didn't help**: exp 015 (0.237 cmAP) ≈ exp 014. Bottleneck is representation quality, not head architecture.
- **Energy-attenuated probe labels rejected**: BI normalization produces weights clustered at ~0.5, not enough contrast.

### Soundscape fusion is weak
- LF accuracy drops: BirdNET 0.735 (vs 0.788 focal), Perch 0.695 (vs 0.779)
- **High LF correlation on soundscapes: ~0.6** (vs ~0.2 focal) — violates conditional independence, inflates accuracy estimates
- Only 142/206 species have learnable soundscape signal; 64 stuck at coin-flip
- **No recording label LF** for soundscapes (biggest loss — it's the strongest LF at 0.932 on focal)

### Implication for the pipeline
Soft labels are useful for **pseudo-labeling** (generating labels for unlabeled data) but should **not replace** hard labels where they exist. The right role is:
1. Use hard labels for the 35,549 focal recordings (ground truth)
2. Use soft labels to pseudo-label the 176.5h unlabeled soundscapes
3. Train on hard + pseudo-labeled data together, with confidence weighting on pseudo-labels

## Soundscape Ground-Truth Labels: Integration Strategy

Exp 100 EDA revealed that **66 soundscapes have ground-truth labels** (739 unique windows, 75 species, 5s resolution). These are sparse but high-quality — and currently unused by the weak supervision pipeline.

### How they can help

**1. Evaluation oracle (highest priority, zero engineering)**

The triplet method estimates LF accuracies without ground truth, but high LF correlation on soundscapes (~0.6) means those estimates are likely inflated. The 739 labeled windows provide actual ground truth to compute real precision/recall/AUROC on soundscape soft labels.

This directly answers: **are the current soundscape pseudo-labels any good?** If they score poorly, the pipeline needs fixing before scaling to 176.5h unlabeled.

**2. Probe calibration / domain-adapted probes**

BirdNET/Perch probes were trained on focal embeddings and degrade on soundscapes. The 739 labeled windows could:
- **Recalibrate thresholds**: the top-5%/bottom-50% percentile thresholds tuned on focal may be wrong for soundscape embedding distributions
- **Train soundscape-specific probes**: small linear heads on BirdNET/Perch embeddings supervised on the 739 windows. Even 739 samples across 75 species is thin, but per-species binary classification only needs ~10+ positive examples per species, and 42 species appear in all 12 available windows.

**3. Soundscape recording label LF (limited scope)**

For the 66 labeled soundscapes, add a per-window label LF analogous to focal LF3: +1 for species present in that window, -1 for absent. Accuracy ~1.0 (ground truth). But this only helps fuse labels on those 66 files — the 10,592 unlabeled soundscapes still lack this LF.

**4. Soundscape-specific prevalence prior**

Current prevalence prior is focal-dominated. The labeled soundscapes reveal the real test-domain distribution: **76.6% amphibian, 28.7% bird**. A soundscape-specific prior could shift soft labels toward amphibians where the evidence is ambiguous — matching the actual test domain.

### Falsifiable validation ladder

**Step 1**: Evaluate current soundscape soft labels against the 739 ground-truth windows. Compute per-species AUROC and calibration. This tells us the current quality baseline.

**Step 2**: Recalibrate probe thresholds using ground-truth windows. Re-run fusion. Compare AUROC against Step 1.

**Step 3**: Train soundscape-domain probes on the labeled windows. Add as new LFs or replace focal probes for soundscape fusion. Compare AUROC against Step 2.

**Step 4**: Use the best fusion variant to pseudo-label all 10,592 unlabeled soundscapes. Train downstream classifier on hard labels + pseudo-labeled soundscapes. Compare cmAP against hard-label-only baseline.

Each step gates the next. If Step 1 shows current soft labels already have >0.8 AUROC on soundscapes, the calibration work may not be necessary. If Step 1 shows <0.6 AUROC, the probes need domain adaptation before pseudo-labeling is viable.

### Connection to source separation

Source separation (biodenoising) could improve pseudo-labels via max-channel classification — see [source-separation-training.md](source-separation-training.md) validation ladder. This is orthogonal to probe calibration: separation improves the *input* to the classifiers, while calibration improves the *thresholds*. Both could help independently.

## References

- Ratner, A. et al. (2017). "Snorkel: Rapid Training Data Creation with Weak Supervision." VLDB. [Paper](https://arxiv.org/abs/1711.10160)
- Fu, D. et al. (2020). "Fast and Three-rious: Speeding Up Weak Supervision with Triplet Methods." ICML. [Paper](https://arxiv.org/abs/2002.11955)
- Northcutt, C. et al. (2021). "Confident Learning: Estimating Uncertainty in Dataset Labels." JAIR. [Paper](https://arxiv.org/abs/1911.00068)
- Dawid, A.P. & Skene, A.M. (1979). "Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm." Applied Statistics.
- [Cleanlab library](https://github.com/cleanlab/cleanlab) (v2.9.0, Jan 2026)
- [Crowd-Kit library](https://github.com/Toloka/crowd-kit) (v1.4.2, Oct 2025)
