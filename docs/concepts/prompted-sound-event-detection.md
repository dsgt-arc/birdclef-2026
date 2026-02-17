# Concept: Prompted Sound Event Detection

**Status**: developing
**Created**: 2026-02-08
**Updated**: 2026-02-17

## Idea

Use a reference audio clip as a "prompt" to condition a Mamba SED model for species-specific detection. Instead of a fixed classification head with one output per species, the model takes a few seconds of a target species' call as prefix tokens, then scans a soundscape and outputs per-token activation scores indicating where that species is vocalizing.

This is analogous to LLM text completion: the prompt establishes what to look for, and the model "completes" by identifying matching patterns in the input sequence. New species require only a reference clip, not retraining — enabling few-shot and zero-shot detection.

With 128x1 patches (10ms per token), each token corresponds to one mel frame. Per-token SED at 10ms resolution is natural — `Linear(D, 1)` on each token embedding. The model learns temporal structure finer than its supervision signal: coarse 3-5s teacher labels provide the training signal, but the acoustic match between reference and target drives the model to localize at token-level precision.

## Motivation

BirdCLEF has three compounding challenges:

1. **Domain gap**: Training data is clean XenoCanto focal recordings; test data is noisy overlapping soundscapes. A fixed classification head learns decision boundaries that may not transfer.

2. **Low-resource species**: Some species have only 1-5 training recordings. A per-species classifier weight learned from 3 examples is fragile.

3. **Zero-shot species**: Test soundscapes may contain species absent from training entirely. A fixed-head classifier cannot detect what it has no head for.

A prompted architecture sidesteps all three: the model learns a general skill — "find this sound pattern in that audio" — rather than memorizing per-species classifiers. Species identity is specified at inference time via the reference clip, not baked into the weights.

### Competition Evidence: SED is Table Stakes at the Top

Analysis of BirdCLEF 2025 Kaggle notebooks (see `docs/references/kaggle/birdclef-2025/README.md`) reveals that **every solution above 0.85 uses SED with attention pooling**:

| Place/Score | SED? | Key Detail |
|-------------|------|------------|
| 1st | Yes | 7-model ensemble, GeM frequency pooling, framewise predictions |
| 9th | Yes | AttBlockV2, framewise max at inference |
| 0.878 | Yes | NFNet + SEResNeXt blend, temporal TTA |
| 0.857 | Yes | Single SED model — best non-ensemble score |
| 0.834 | No | EfficientNetV2-S image classification (ceiling) |

SED + AttBlockV2 appears in 5/35 notebooks (all top-5) with +0.03-0.05 gain over clip classification. The train/inference divergence is key: train with attention-weighted pooling, infer with framewise max (more sensitive to brief calls).

The 1st place solution's largest contributor was **iterative pseudo-labeling** (5/7 models), not architecture — but those pseudo-labels trained SED models, not clip classifiers.

## Background

### ProtoCLR (clip-level prototypes)

[ProtoCLR](https://arxiv.org/abs/2409.08589) learns domain-invariant bird audio embeddings via prototypical contrastive learning. Each species is represented by a prototype (mean embedding of support examples). Classification is by nearest prototype in embedding space. This works at the clip level — one embedding per 5s clip.

Limitations:
- Clip-level: no temporal localization of when the species vocalizes
- Requires a separate embedding + nearest-neighbor step at inference
- Prototype is a single vector — loses temporal structure of the call

### Keyword spotting

Keyword spotting (e.g., "Hey Siri") detects a specific sound pattern in continuous audio. Our proposed approach generalizes this: instead of a fixed keyword, the "keyword" is provided dynamically as a prefix.

### LLM prompting

In language models, the prefix (prompt) conditions the hidden state, and the model generates/predicts based on that conditioning. The same principle applies here: the reference call conditions Mamba's hidden state to "listen for" a specific acoustic pattern.

## Architecture

### Inference flow

```
Input: [ref_call: ~500 tokens (5s)] [SEP] [soundscape: 3000 tokens (30s)]
                                       ↓
                         SSAMBA backbone (24 layers)
                                       ↓
                   token embeddings (B, ~3501, D)
                                       ↓
                  discard ref portion, keep soundscape tokens
                                       ↓
                       SED head: Linear(D, 1) per token
                                       ↓
                   activations (B, 3000, 1) → sigmoid
                                       ↑
                   "is the prompted species vocalizing at this frame?"
```

The SED head outputs a **single scalar per token** (binary: active or not), because the species identity comes from the prompt, not the head. This is fundamentally different from exp 007's `Linear(D, C)` which has one output per species.

### Training Pipeline (3 stages)

Training progresses from unsupervised to weakly supervised, maximizing what each data source contributes.

#### Stage 1: SSL Pre-Training (unsupervised)

Masked token prediction on all available audio — XenoCanto, soundscapes, anything. No labels needed. The backbone learns general audio representations: harmonic structure, noise characteristics, temporal patterns.

```
Input: mel spectrogram with ~40% of tokens masked
Task: predict masked tokens from context
Loss: MSE reconstruction
Data: all audio (unlabeled)
```

This gives the backbone a head start before it ever sees a species label. It also learns representations that are invariant to recording conditions, which helps bridge the domain gap. Exp 009 pretrain-full/pretrain-warm are running this stage.

#### Stage 2: Episode-Based Prompted SED (weakly supervised)

The core training stage. Each training sample is an **episode**: a (reference, target, per-token labels) tuple. Teacher pseudo-labels from BirdNET/Perch (exp 010) provide temporal supervision at 3-5s resolution.

**2a. Within-recording episodes (near self-supervised):**

Takes two non-overlapping segments from the same recording. No species labels needed — same recording = same species implicitly.

```
Recording of species X (60s):
  Segment A [0-5s]:  reference (teacher says high confidence here)
  Segment B [5-35s]: target (teacher labels: active at 8-10s, 22-25s)

Input: [A tokens] [SEP] [B tokens]
Labels: [ignore] [0,0,...,1,1,...,0,0,...,1,1,1,...,0,0]
Loss: BCE on B portion only
```

**2b. Cross-recording episodes (needs species labels):**

Same species, different recordings. Reference from recording 1, target from recording 2. Teaches the model that species X can sound different across recordings (individual variation, recording quality, microphone distance).

```
Recording 1 of species X: reference segment with high teacher confidence
Recording 2 of species X: 30s target chunk with teacher event labels

Input: [rec1 ref tokens] [SEP] [rec2 target tokens]
Labels: [ignore] [teacher labels from rec2]
Loss: BCE on target portion
```

**2c. Negative episodes:**

Mismatched species or silent chunks. Teaches the model to not hallucinate.

- **Cross-species negative**: ref = species X, target = recording of species Y → all-zeros target labels
- **Silent negative**: ref = species X, target = chunk from X's recording where teacher found no activity → all-zeros target labels
- **Hard negatives**: ref = species X, target = recording of a similar-sounding species Y → all-zeros but the model must learn fine-grained discrimination

Episode sampling per batch: mix of 2a, 2b, 2c. Ratio is a hyperparameter (e.g., 40% within-recording, 30% cross-recording, 30% negative).

#### Stage 3: Soundscape Adaptation

Fine-tune with prompted episodes where the target is soundscape audio instead of XenoCanto. **This is where soundscapes become the most valuable training signal** — every episode directly trains on the target domain.

```
Reference: XenoCanto segment of species X (clean)
Target: soundscape 30s chunk (noisy, multi-species)
Labels: BirdNET/Perch pseudo-labels on soundscape (from exp 010)

Input: [focal ref tokens] [SEP] [soundscape target tokens]
Labels: [ignore] [teacher pseudo-labels for species X]
Loss: BCE on target portion
```

This stage transforms soundscapes from "unlabeled data" into "the most valuable training signal" because:
- The model learns to match clean references to noisy targets
- This is directly training on the domain gap
- The combinatorial space of (focal reference, soundscape target) pairs is enormous

#### Pipeline summary

```
Stage 1: SSL            → backbone learns audio structure
  Data: all audio           (unsupervised)
  Loss: masked reconstruction

Stage 2: Prompted SED   → model learns "find this sound in that audio"
  Data: XenoCanto +         (weakly supervised: species tags + teacher labels)
        teacher labels
  Loss: per-token BCE

Stage 3: Adaptation     → model handles noisy soundscapes
  Data: focal refs +        (pseudo-labeled soundscape targets)
        soundscape targets
  Loss: per-token BCE
```

The heaviest human annotation in the entire pipeline is the species tag per XenoCanto recording — which already exists in the competition data.

## Episode Mixup: Combinatorial Data Augmentation

Episode-based training naturally produces a combinatorial explosion of training data. Mixup amplifies this further, making the dataset effectively infinite.

### Episode Combinatorics (without mixup)

```
28,564 focal recordings × ~6 segments each = ~170K reference candidates
9,726 soundscapes × 6 segments (30s chunks) = ~58K target candidates
206 species for cross-species negatives

Positive pairs (same species):    ~170K refs × ~6 targets/species = ~1M episodes
Negative pairs (cross-species):   ~170K refs × ~58K targets = ~10B episodes
Soundscape episodes (Stage 3):    ~170K refs × ~58K soundscape targets = ~10B episodes
```

Every epoch samples a different subset — the model never sees the same episode twice.

### Mixup Strategies

**Target mixup** — blend two target spectrograms, blend per-token labels proportionally:
```python
target = α * target_A + (1-α) * target_B
labels = α * labels_A + (1-α) * labels_B
# Soft per-token labels, natural regularization
```

**Reference mixup** — blend two reference clips of the same species:
```python
ref = α * ref_clip_1 + (1-α) * ref_clip_2  # same species, different recordings
# Forces model to learn species-invariant template, not memorize one call
```

**Cross-domain mixup** — blend focal and soundscape targets:
```python
target = α * focal_target + (1-α) * soundscape_target
# Curriculum from clean to noisy, smooth domain adaptation
```

**Multi-species target mixup** — blend targets containing different species:
```python
target = α * soundscape_with_X + (1-α) * soundscape_with_Y
labels_for_X = α * labels_X + (1-α) * zeros  # species X fades proportionally
# Teaches the model to handle overlapping species at varying intensities
```

### Why Mixup is Especially Powerful Here

In standard classification, mixup blends two images and interpolates label vectors. Here, mixup operates at three levels simultaneously:
1. **Acoustic level**: blended spectrograms create realistic multi-source audio
2. **Temporal level**: per-token labels preserve timing information through the blend
3. **Episode level**: the reference stays fixed while the target varies, so the model must learn robust matching regardless of background

This naturally simulates the test conditions: real soundscapes contain multiple overlapping species at varying volumes.

## Metric Learning Connection

The prompted SED formulation is structurally a **metric learning** problem, richer than standard triplets:

| Standard Triplet Loss | Prompted SED |
|----------------------|--------------|
| One embedding per sample | One embedding per token (3000/sample) |
| Scalar distance | Per-token activation map |
| Anchor/positive/negative clips | Reference/target with per-token pos/neg |
| Learns embedding space | Learns temporal matching function |
| Static comparison | Sequential — state accumulates evidence |

The Mamba hidden state after processing the reference acts as the "query." Each target token's embedding is a "key." The SED head computes a learned similarity. Training pushes similarity high for matching tokens, low for non-matching.

Key differences from pure triplet loss:
1. **Per-token granularity** — localization, not just "present in clip"
2. **Implicit hard negative mining** — a soundscape with species X and similar species Y naturally contains hard negatives at the token level
3. **Sequential evidence accumulation** — unlike a static embedding, the Mamba state evolves as it scans. Token 2000 has more context than token 100. Repeated calls reinforce the initial state conditioning.

### Teacher Labels as Coarse Supervision for Fine Predictions

BirdNET (3s windows) and Perch (5s windows) provide coarse temporal labels. But the model predicts at 10ms resolution. This works because:

- A 3s BirdNET-positive window contains maybe 0.5s of actual call and 2.5s of silence
- The model sees many such windows across training
- The acoustic match between reference and target drives the model to localize at token-level, even though supervision is coarse
- This is exactly how winning SED models work: train with coarse labels, produce framewise predictions, take max within the evaluation window

The fine temporal structure is **learned, not labeled**.

## Why Mamba is Uniquely Suited

| Architecture | Query-soundscape cost | Mechanism |
|---|---|---|
| Transformer + cross-attention | O(n * m) | Explicit cross-attention between query and soundscape |
| Mamba (prompted) | O(n + m) | Query compressed into hidden state, soundscape scanned linearly |

Mamba compresses the reference clip into its recurrent hidden state — essentially a **learned matched filter**. As it scans the soundscape, the state resonates when the acoustic pattern matches. This is O(n+m) instead of O(n*m), and the state acts as a fixed-size summary regardless of reference clip length.

### Multi-species detection

At inference, to detect K species in a soundscape:
- Run K forward passes, each with a different species reference as prefix
- Or batch them: `(K, 1, 128, 3500)` — K copies of the soundscape, each with a different prefix
- Output: K activation tracks, one per species

For 206 species on Kaggle CPU: 206 × 12 segments × ~13ms ≈ 32s per soundscape. Feasible within the ~2 hour limit for ~200 soundscapes.

### Comparison with fixed-head approaches

| | Fixed head (exp 007) | Prompted (this concept) |
|---|---|---|
| **Output** | (B, T, C) — all species at once | (B, T, 1) — one species per pass |
| **New species** | Requires retraining | Just provide reference clip |
| **Low-resource** | Weak classifier weight from few examples | Same architecture, reference clip is sufficient |
| **Inference cost** | 1 forward pass for all species | K passes for K species (batchable) |
| **Training** | Standard supervised | Episode-based (sample species pairs) |
| **Species capacity** | Fixed at training time | Unlimited at inference |
| **Training data** | Bounded by labeled recordings | Combinatorial episodes, effectively infinite |

## Open Questions

- **Reference clip selection**: How to choose the best reference clip for a species at inference? Highest teacher confidence segment? Mean of multiple clips? Random? Most typical?
- **Reference clip length**: How many tokens of reference are needed? 500 (5s)? 200 (2s)? Does more help?
- **Multi-species per pass**: Can we concatenate multiple species references in one prefix, separated by [SEP] tokens, and output multi-channel activations? Or does this confuse the hidden state?
- **Negative strategy**: How to construct hard negatives? Similar-sounding species as reference with target containing the confusable species?
- **Domain gap in references**: If reference clips are clean XenoCanto and target is noisy soundscape, does the hidden state still transfer? SSL pre-training and cross-domain mixup may help here.
- **Comparison with cross-attention**: Is Mamba's compressed state sufficient, or would explicit cross-attention between reference and target embeddings perform better? Could test with a small cross-attention layer on top of Mamba features.
- **State capacity**: Can Mamba's fixed-size hidden state capture enough detail about a species' call? Complex songs with multiple syllable types may exceed what the state can represent.
- **Mixup ratio scheduling**: Should the mix between clean focal targets (Stage 2) and noisy soundscape targets (Stage 3) be a curriculum, or a fixed ratio?
- **Episode batch balance**: What's the optimal ratio of positive/negative/hard-negative episodes per batch? Does this need to change as training progresses?

## Testable Hypotheses

1. **Mamba state encodes species identity**: Mamba's hidden state after processing a reference clip is more similar (cosine) to states from same-species references than different-species references — the state is a functional species embedding. *This is the cheapest test and should be run first as a feasibility gate.*

2. **Prompted SED outperforms fixed-head SED on low-resource species**: For species with < 5 training recordings, the prompted model achieves higher per-species AP than the fixed-head model (exp 007), because it leverages the reference clip directly rather than learning a weak classifier weight.

3. **Zero-shot transfer**: The prompted model can detect species not seen during training, achieving > 0 AP on held-out species given only a reference clip at inference time.

4. **Episode mixup outperforms no-mixup**: Cross-domain mixup (focal + soundscape targets) improves soundscape evaluation metrics over training with pure focal or pure soundscape targets.

5. **Longer context improves prompted detection**: 30s target windows outperform 10s windows in prompted mode, because repeated calls provide cumulative evidence that reinforces the initial state conditioning.

6. **Soundscape adaptation closes the domain gap**: Stage 3 (soundscape target episodes) improves soundscape cmAP by at least 0.03 over Stage 2 alone (focal-only episodes).

## Connections

- **Based on**: [ProtoCLR](../references/literature.md) (prototypical learning for bird audio), [SSAMBA](ssamba-audio-mamba.md) (Mamba architecture), [Weak supervision label fusion](weak-supervision-label-fusion.md) (teacher labels for supervision)
- **Related to**: [Soundscape domain shift](soundscape-domain-shift.md) (the domain gap this approach bridges), [Context recovery](context-recovery.md) (using temporal context for better predictions)
- **Depends on**: Experiment 010 (teacher labels — BirdNET/Perch predictions on all audio), Experiment 009 (SSL pretrain for Stage 1 backbone)
- **Could lead to**: Experiment 011 — prompted SED with episode-based training

## Notes

- The prompted approach makes the model a general **acoustic pattern matcher** rather than a species classifier. This is a philosophical shift from "learn 206 species" to "learn to match sounds."
- The BirdCLEF competition provides reference recordings for target species — these could be used directly as prompts at inference time, which aligns perfectly with this architecture.
- Episode-based training is well-studied in few-shot learning literature. The main engineering challenge is efficient episode sampling with teacher label alignment.
- The original concept depended on BamBird for temporal event labels. BamBird failed in exp 010 (0 ROIs across all files). **BirdNET/Perch teacher labels are the replacement** — coarser (3-5s vs 10ms) but species-specific and functional. The model can learn finer resolution from the acoustic signal.
- A hybrid approach could use both: fixed heads for common species (efficient, one pass) and prompted detection for rare/unseen species (flexible, multiple passes).
- Competition evidence strongly supports SED over clip classification (+0.03-0.05 gain, used by all top-5 BirdCLEF 2025 solutions). The question is not "SED or not" but "which flavor of SED."
- The mixup/episode formulation creates a form of **implicit metric learning**: the model learns an embedding space where the reference conditions a "query" and each target token is a "key," with training maximizing similarity for matching tokens and minimizing it for non-matching tokens.
