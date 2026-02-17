# Proposal: Prompted Sound Event Detection

**Status**: proposed
**Created**: 2026-02-17
**Author**: Anthony

## Hypothesis

A Mamba-based SED model conditioned by a reference audio prompt can learn to match acoustic patterns across domains (clean focal → noisy soundscape), achieving competitive soundscape cmAP while enabling zero-shot detection of unseen species. The episode-based training with combinatorial mixup produces an effectively infinite training set, and the model learns finer temporal localization than its coarse (3-5s) supervision signal.

## Background

BirdCLEF's core challenge is the domain gap between clean XenoCanto focal recordings (training) and noisy multi-species soundscapes (evaluation). Current approaches use fixed classification heads (`Linear(D, 206)`) that bake species identity into learned weights — fragile for low-resource species and impossible for unseen ones.

The prompted SED concept (see `docs/concepts/prompted-sound-event-detection.md`) proposes using a reference clip as a prefix to condition Mamba's hidden state, turning the model into a general acoustic pattern matcher. This experiment tests whether the approach is feasible and competitive.

**Key findings informing this experiment:**
- SED is used by all top-5 BirdCLEF 2025 solutions (+0.03-0.05 over clip classification)
- 1st place: iterative pseudo-labeling was the dominant factor (5/7 models)
- Exp 010: BirdNET/Perch teacher labels available on all 38K recordings
- Exp 009: SSL pretrained SSAMBA backbone available (pretrain-full, pretrain-warm)
- SSAMBA Tiny fixed-head: 0.173 cmAP on BirdSet HSN (exp 007)

## Architectural Considerations: SSAMBA Bidirectionality

SSAMBA has **two levels of bidirectionality** that fundamentally affect the prompted SED design:

1. **Inner (per-block)**: each `BidirectionalMamba` module has forward + backward SSMs (bimamba_type="v2")
2. **Outer (layer pairs)**: layers processed in pairs — even layers see normal input, odd layers see `.flip([1])` (reversed sequence), results summed

Additionally, SSAMBA uses **learned positional embeddings** and a **CLS token inserted at the middle** of the sequence (`num_patches // 2`).

### Implications for Prompted SED

Because every token's output embedding reflects the **entire sequence** from both directions:

- **No single "state compression" token** — unlike unidirectional Mamba where the last token compresses the full forward state, in SSAMBA every token has a different positional perspective on the same global information. The last token emphasizes the forward scan's recency, the first token the backward scan's recency, but both see everything.

- **Bidirectionality helps prompting** — the backward scan from target tokens reads *back into the reference*. Both scan directions carry reference information into the target. This is strictly better than unidirectional prompting. However, the reference is only ~14% of input (500 of 3500 tokens), so dilution is a concern.

- **CLS token as global summary** — SSAMBA's middle-inserted CLS token is designed as a global representation. It should be included in the Phase 0 clustering test.

### Positional Embedding Handling

The pretrained model has `pos_embed` of shape `(1, num_patches + 1, embed_dim)` for a fixed sequence length. For prompted SED with `[ref: 500] [SEP] [target: 3000]` = 3501 tokens, we need to handle the length mismatch. Options:

1. **Single interpolated pos_embed** — treat the full `[ref][target]` as one sequence, interpolate pos_embed to length 3501. Simple, but the ref tokens get positional IDs that meant "early target audio" during pretraining.
2. **Separate pos_embed for ref and target** — ref gets positions 0-499, target gets 0-2999, with a learned segment embedding. More principled but adds parameters.
3. **No pos_embed on ref** — let the reference content speak for itself, only add pos_embed to target tokens. The ref conditions the state purely through content, not position.

Option 1 is the simplest starting point. Options 2-3 are refinements to try if position confusion becomes a problem.

## Method

### Approach

Three-phase experiment: (1) feasibility gate via representation analysis, (2) episode-based prompted SED training, (3) soundscape adaptation with teacher pseudo-labels.

### Prerequisites

Before starting the main experiment:

1. **Map BirdNET/Perch species IDs → BirdCLEF 206 species** (exp 010 TODO)
2. **Verify exp 009 pretrain checkpoint quality** (pretrain-full or pretrain-warm, whichever is better)

### Setup

| Component | Details |
|-----------|---------|
| Data | 28,564 focal recordings + 9,726 soundscapes (BirdCLEF 2025) |
| Teacher labels | BirdNET (3s/1s, 939K focal + 564K soundscape clips) + Perch (5s/5s, 188K + 117K clips) from exp 010 |
| Backbone | SSAMBA Tiny (7M params), 128x1 patches, 10ms/token, bidirectional |
| Pretrained weights | Exp 009 pretrain-full or pretrain-warm checkpoint (Stage 1 done) |
| SED head | Linear(D, 1) per token — single binary output |
| Compute | 1x RTX 6000 (24GB), PACE gpu-rtx6000 partition |
| Dependencies | ssamba_hf (exp 004), mamba-ssm, birdclef_009 (data loading) |

### Procedure

#### Phase 0: Feasibility Gate — Representation Clustering

**Goal**: Test whether the pretrained SSAMBA backbone encodes species-discriminative information, and where that information lives in the token sequence.

Because SSAMBA is bidirectional, every token sees the full sequence from both directions. There is no single "state compression" token — each position has a different perspective on the global information. Phase 0 tests multiple pooling strategies in a single forward pass to understand where species identity is encoded.

1. Load pretrained SSAMBA backbone (exp 009 checkpoint)
2. Feed 5s clips from ~50 species (10 clips each = 500 total), `return_all_tokens=True`
3. Extract four representations per clip:

```python
embeddings = backbone(mel_5s, return_all_tokens=True)  # (1, T+1, D)

# a) Mean pool — clip-level representation (ProtoCLR-style)
clip_repr = embeddings[:, non_cls_mask, :].mean(dim=1)

# b) CLS token — SSAMBA's own global summary (middle-inserted)
cls_repr = embeddings[:, cls_position, :]

# c) Last token — forward scan's accumulated state
last_repr = embeddings[:, -1, :]

# d) First token — backward scan's accumulated state
first_repr = embeddings[:, 0, :]
```

4. For each representation type, compute:
   - Pairwise cosine similarity matrix (500 × 500)
   - Silhouette score (50 species as clusters)
   - Linear probe accuracy (logistic regression, 5-fold CV)
   - t-SNE visualization colored by species

**Interpreting results:**

| Mean pool | CLS | Last | First | Interpretation |
|-----------|-----|------|-------|---------------|
| Good | Good | Good | Good | Backbone encodes species everywhere — any pooling works for SED |
| Good | Good | Poor | Poor | Species info is distributed, not compressed — mean pool SED is viable, but prompting via state compression may not work |
| Good | Good | Good | Poor | Forward state compresses species identity — promising for prompting |
| Poor | Poor | Poor | Poor | Backbone doesn't encode species — needs more pretraining or the model is too small |

**Decision point**: If mean-pool linear probe < 30% accuracy, the backbone doesn't carry enough species information for any downstream task. Abort or retrain. If mean-pool works but last-token doesn't, standard SED (mean/attention pooling) is viable but the prompted mechanism needs redesign (e.g., add cross-attention instead of relying on state compression).

**Compute**: ~10 minutes on 1 GPU, no training needed.

#### Phase 1: Episode Training on Focal Recordings

**Goal**: Train the prompted SED model to match reference clips to target segments.

1. Build episode sampler:
   - Index all recordings by species
   - For each recording, identify high-confidence teacher windows (BirdNET > 0.5)
   - Sample episodes: 40% within-recording, 30% cross-recording, 30% negative
2. Construct input: `[ref: 500 tokens] [SEP] [target: 3000 tokens]`
3. Construct per-token labels from teacher predictions:
   - BirdNET window with confidence > 0.5 for target species → tokens in that window = 1.0
   - BirdNET window with confidence < 0.1 for target species → tokens = 0.0
   - Intermediate confidence → soft label (sigmoid-scaled)
4. Train with per-token BCE loss on target portion only
5. Apply episode mixup: reference mixup (same species, different recordings), target mixup (α-blend two targets with proportional labels)

**Training config**:
- Batch size: 8 episodes (each is ~3500 tokens)
- LR: 1e-4 with cosine schedule
- Epochs: 30 (but each "epoch" is a random sample of episodes, not a fixed dataset)
- Mixup: α ~ Beta(0.4, 0.4) for both reference and target

**Compute**: ~4 hours on 1 GPU.

#### Phase 2: Soundscape Adaptation

**Goal**: Bridge the domain gap by training on (focal reference, soundscape target) episodes.

1. Extend episode sampler to include soundscape targets
2. Use BirdNET/Perch pseudo-labels on soundscapes as target labels
3. Mix Phase 1 and Phase 2 episodes: start with 80/20 focal/soundscape ratio, shift to 50/50 over training
4. Apply cross-domain mixup: blend focal and soundscape targets
5. Continue training from Phase 1 checkpoint

**Challenge**: Soundscapes are mostly quiet (only 5-11% of segments exceed 0.50 teacher confidence). Need careful episode sampling to avoid overwhelming the model with negative episodes. Strategy: oversample soundscape segments where at least one teacher reports activity.

**Compute**: ~4 hours on 1 GPU.

#### Phase 3: Evaluation

1. **BirdSet HSN cmAP**: Compare prompted SED vs fixed-head baseline (exp 007: 0.173)
   - For prompted: run 206 passes per soundscape, one per species
   - Aggregate per-token activations into 5s windows (max within window)
2. **Zero-shot evaluation**: Hold out 20 species from training, provide only reference clips at inference
3. **Low-resource evaluation**: Compare per-species AP for species with < 5 training recordings
4. **Temporal localization**: Qualitative visualization of per-token activations vs teacher labels
5. **Inference timing**: Verify 206-species inference fits within Kaggle constraints

### Variables

- **Independent**: Training formulation (prompted vs fixed-head), episode mixup (on/off), soundscape adaptation (on/off), reference clip length (2s/5s/10s)
- **Dependent**: cmAP on BirdSet HSN, zero-shot AP on held-out species, per-species AP for low-resource species
- **Controlled**: Backbone architecture (SSAMBA Tiny), pretrained weights (same exp 009 checkpoint), mel spectrogram params (128 bins, 128x1 patches)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| cmAP (BirdSet HSN) | Primary competition-relevant metric |
| Zero-shot AP | AP on 20 held-out species (never seen in training) |
| Low-resource AP | Mean AP for species with < 5 training recordings |
| Silhouette score | Hidden state clustering quality (Phase 0) |
| Inference time | Seconds per 60s soundscape for 206 species |

### Baseline

| Baseline | cmAP | Source |
|----------|------|--------|
| SSAMBA Tiny fixed-head (pretrained) | 0.173 | Exp 007 |
| SSAMBA Tiny from scratch | 0.086 | Exp 006 |
| ConvNeXt (ImageNet pretrained) | 0.48 | Murilo exp 001 |

### Success Criteria

- **Phase 0 gate**: Silhouette score > 0.1 AND linear probe > 30% accuracy on species classification from hidden states
- **Confirm prompted SED if**: cmAP >= 0.17 (matches fixed-head baseline) AND zero-shot AP > 0 on held-out species
- **Strong confirmation if**: cmAP > 0.20 (exceeds fixed-head) OR zero-shot AP > 0.05
- **Reject if**: cmAP < 0.10 (worse than scratch baseline) AND Phase 0 gate fails — the hidden state can't encode species identity

## Limitations

- **Bidirectional state compression is unproven**: The prompted SED concept was originally conceived for unidirectional Mamba, where the last token naturally compresses the forward state. SSAMBA's bidirectionality means every token sees the full sequence — there's no single "compressed query" token. The prompting mechanism may work through distributed conditioning rather than state compression, or it may need architectural changes (e.g., cross-attention between ref and target embeddings, or a forward-only scan mode for the reference).
- **Positional embedding mismatch**: Concatenating `[ref: 500] [target: 3000]` creates a 3501-token sequence longer than the pretrained pos_embed. Interpolation is straightforward but ref tokens receive positional encodings that originally corresponded to "early audio," not "reference prompt." This position confusion may need to be addressed with separate segment embeddings or ref-only position ablation.
- **Reference dilution**: The reference (500 tokens) is ~14% of the total input (3501 tokens). In a bidirectional model, the target tokens dominate both forward and backward context. The reference signal may get washed out, especially in deeper layers.
- **Backbone capacity**: SSAMBA Tiny (7M params) is small. Poor results may reflect model capacity, not the prompted approach. Larger SSAMBA or hybrid CNN+Mamba could be needed.
- **Teacher label quality on soundscapes**: BirdNET/Perch are weakest on soundscapes (median max-prob 0.07-0.09). Pseudo-labels may be noisy, especially for rare species.
- **Inference cost**: 206 forward passes per soundscape is 206x more expensive than a fixed-head model. Feasible on Kaggle but tight (~32s per soundscape).
- **No ground-truth temporal labels**: We evaluate at 5s clip level (competition format), not at the 10ms token level. The model may learn excellent temporal localization that we can't directly measure.
- **Single backbone**: Results are specific to SSAMBA Tiny. The approach may work better or worse with other architectures.
- **BamBird unavailable**: The original concept relied on BamBird for precise temporal labels. We substitute with coarser BirdNET/Perch labels (3-5s). This is adequate for training but the model's temporal precision is limited by the supervision signal quality.
