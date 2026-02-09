# Concept: Prompted Sound Event Detection

**Status**: nascent
**Created**: 2026-02-08
**Updated**: 2026-02-08

## Idea

Use a reference audio clip as a "prompt" to condition a Mamba SED model for species-specific detection. Instead of a fixed classification head with one output per species, the model takes a few seconds of a target species' call as prefix tokens, then scans a soundscape and outputs per-token activation scores indicating where that species is vocalizing.

This is analogous to LLM text completion: the prompt establishes what to look for, and the model "completes" by identifying matching patterns in the input sequence. New species require only a reference clip, not retraining — enabling few-shot and zero-shot detection.

## Motivation

BirdCLEF has three compounding challenges:

1. **Domain gap**: Training data is clean XenoCanto focal recordings; test data is noisy overlapping soundscapes. A fixed classification head learns decision boundaries that may not transfer.

2. **Low-resource species**: Some species have only 1-5 training recordings. A per-species classifier weight learned from 3 examples is fragile.

3. **Zero-shot species**: Test soundscapes may contain species absent from training entirely. A fixed-head classifier cannot detect what it has no head for.

A prompted architecture sidesteps all three: the model learns a general skill — "find this sound pattern in that audio" — rather than memorizing per-species classifiers. Species identity is specified at inference time via the reference clip, not baked into the weights.

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

This gives the backbone a head start before it ever sees a species label. It also learns representations that are invariant to recording conditions, which helps bridge the domain gap.

#### Stage 2: Episode-Based Prompted SED (weakly supervised)

The core training stage. Uses species metadata + bambird events (both cheap/free) for supervision.

**2a. Within-recording episodes (near self-supervised):**

Takes two non-overlapping segments from the same recording. No species labels needed — same recording = same species implicitly. bambird provides per-token temporal labels.

```
Recording of species X (60s):
  Segment A [0-5s]:  reference (bambird found call at 1-3s)
  Segment B [5-35s]: target (bambird labels: events at 8-10s, 22-25s)

Input: [A tokens] [SEP] [B tokens]
Labels: [ignore] [0,0,...,1,1,...,0,0,...,1,1,1,...,0,0]
Loss: BCE on B portion only
```

The only external input is bambird (unsupervised signal processing). A recording with 10 vocalization events yields many within-recording episode pairs.

**2b. Cross-recording episodes (needs species labels):**

Same species, different recordings. Reference from recording 1, target from recording 2. Teaches the model that species X can sound different across recordings (individual variation, recording quality, microphone distance).

```
Recording 1 of species X: reference segment with detected call
Recording 2 of species X: 30s target chunk with bambird event labels

Input: [rec1 ref tokens] [SEP] [rec2 target tokens]
Labels: [ignore] [bambird events from rec2]
Loss: BCE on target portion
```

This is where species labels earn their keep — they pair recordings that the model couldn't pair by itself.

**2c. Negative episodes:**

Mismatched species or silent chunks. Teaches the model to not hallucinate.

- **Cross-species negative**: ref = species X, target = recording of species Y → all-zeros target labels
- **Silent negative**: ref = species X, target = chunk from X's recording where bambird found no events → all-zeros target labels
- **Hard negatives**: ref = species X, target = recording of a similar-sounding species Y → all-zeros but the model must learn fine-grained discrimination

Episode sampling per batch: mix of 2a, 2b, 2c. Ratio is a hyperparameter (e.g., 40% within-recording, 30% cross-recording, 30% negative).

**Low-resource species benefit especially:**
- A species with 3 recordings contributes many within-recording episodes (2a)
- 3 recordings = 6 ordered cross-recording pairs (2b)
- Negative episodes against all other species (2c)
- The model never needs to learn a per-species weight — it just needs enough reference clips to condition on

#### Stage 3: Soundscape Adaptation (optional)

Fine-tune with prompted episodes where the target is soundscape audio instead of XenoCanto. Bridges the domain gap between clean references and noisy targets.

Data sources:
- BirdSet validation sets (clip-level labels, convert to token-level via bambird on soundscapes)
- ConvNeXT pseudo-labels on competition soundscapes (species + temporal)
- Any labeled soundscape data available

```
Reference: XenoCanto segment of species X (clean)
Target: soundscape 30s chunk containing species X (noisy, overlapping)
Labels: bambird events from soundscape (or ConvNeXT pseudo-labels)
```

This stage is optional but addresses the core domain shift problem: the model learns that the same species sounds different through a distant microphone with background noise.

#### Pipeline summary

```
Stage 1: SSL            → backbone learns audio structure
  Data: all audio           (unsupervised)
  Loss: masked reconstruction

Stage 2: Prompted SED   → model learns "find this sound in that audio"
  Data: XenoCanto +         (weakly supervised: species tags + bambird)
        bambird events
  Loss: per-token BCE

Stage 3: Adaptation     → model handles noisy soundscapes
  Data: labeled             (supervised or pseudo-labeled)
        soundscapes
  Loss: per-token BCE
```

The heaviest human annotation in the entire pipeline is the species tag per XenoCanto recording — which already exists in the competition data.

### Why Mamba is uniquely suited

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

This scales linearly with species count per forward pass, but the passes are independent and batchable.

### Comparison with fixed-head approaches

| | Fixed head (exp 007) | Prompted (this concept) |
|---|---|---|
| **Output** | (B, T, C) — all species at once | (B, T, 1) — one species per pass |
| **New species** | Requires retraining | Just provide reference clip |
| **Low-resource** | Weak classifier weight from few examples | Same architecture, reference clip is sufficient |
| **Inference cost** | 1 forward pass for all species | K passes for K species (batchable) |
| **Training** | Standard supervised | Episode-based (sample species pairs) |
| **Species capacity** | Fixed at training time | Unlimited at inference |

## Open Questions

- **Reference clip selection**: How to choose the best reference clip for a species at inference? Mean of multiple clips? Random? Most typical?
- **Reference clip length**: How many tokens of reference are needed? 500 (5s)? 200 (2s)? Does more help?
- **Multi-species per pass**: Can we concatenate multiple species references in one prefix, separated by [SEP] tokens, and output multi-channel activations? Or does this confuse the hidden state?
- **Negative strategy**: How to construct hard negatives? Similar-sounding species as reference with target containing the confusable species?
- **Domain gap in references**: If reference clips are clean XenoCanto and target is noisy soundscape, does the hidden state still transfer? SSL pre-training may help here.
- **Comparison with cross-attention**: Is Mamba's compressed state sufficient, or would explicit cross-attention between reference and target embeddings perform better? Could test with a small cross-attention layer on top of Mamba features.
- **State capacity**: Can Mamba's fixed-size hidden state capture enough detail about a species' call? Complex songs with multiple syllable types may exceed what the state can represent.

## Testable Hypotheses

1. **Prompted SED outperforms fixed-head SED on low-resource species**: For species with < 5 training recordings, the prompted model achieves higher per-species AP than the fixed-head model (exp 007), because it leverages the reference clip directly rather than learning a weak classifier weight.

2. **Zero-shot transfer**: The prompted model can detect species not seen during training, achieving > 0 AP on held-out species given only a reference clip at inference time.

3. **Mamba state encodes species identity**: Mamba's hidden state after processing a reference clip is more similar (cosine) to states from same-species references than different-species references — the state is a functional species embedding.

4. **Longer context improves prompted detection**: 30s target windows outperform 10s windows in prompted mode, because repeated calls provide cumulative evidence that reinforces the initial state conditioning.

## Connections

- **Based on**: [ProtoCLR](../references/literature.md) (prototypical learning for bird audio), [SSAMBA](ssamba-audio-mamba.md) (Mamba architecture), [Unsupervised event detection](unsupervised-event-detection-labeling.md) (bambird for training labels)
- **Related to**: [Soundscape domain shift](soundscape-domain-shift.md) (the domain gap this approach could help bridge), [Context recovery](context-recovery.md) (using temporal context for better predictions)
- **Could lead to**: Experiment 009 — prompted SED with episode-based training

## Notes

- The prompted approach makes the model a general **acoustic pattern matcher** rather than a species classifier. This is a philosophical shift from "learn 21 species" to "learn to match sounds."
- The BirdCLEF competition provides reference recordings for target species — these could be used directly as prompts at inference time, which aligns perfectly with this architecture.
- Episode-based training is well-studied in few-shot learning literature. The main engineering challenge is efficient episode sampling with bambird event alignment.
- Could combine with exp 008's multi-signal approach: SSL pre-trains the backbone to understand audio structure, then episode-based prompted training teaches the matching skill.
- A hybrid approach could use both: fixed heads for common species (efficient, one pass) and prompted detection for rare/unseen species (flexible, multiple passes).
