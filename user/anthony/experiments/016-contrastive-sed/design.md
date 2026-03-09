# Design: Prototypical SED — Architecture and Inference

## Competition Task

For each 5-second soundscape window, predict probability of each of 206 species.

## Key Insight: Prototypes Are Classifier Weights

The prototypical approach produces a `prototypes (206, D)` matrix that serves the
same role as `Linear.weight (206, 1280)` in a standard classifier. The difference
is how they're derived:

| | Linear Probe (exp 012) | Prototypical SED (exp 016) |
|---|---|---|
| Weights | `Linear(1280, 206).weight` | `prototypes (206, D)` |
| Derived from | End-to-end gradient descent | Mean of projected training embeddings |
| Optimized for | Classification loss directly | Indirectly (projection is optimized, prototypes are derived) |
| Dimensions | 1280 (full backbone dim) | D (projected, may be bottlenecked) |
| Jointly optimized | Yes (all 206 vectors trained together) | No (each prototype computed independently) |
| Inference cost | 1 matmul | 1 ONNX forward + 1 matmul |
| Few-shot capable | No (fixed head) | Yes (swap prototypes) |
| SED capable | No (clip-level only) | Yes (per-frame similarity) |

At inference time, both reduce to: features @ weights.T → logits. The prototypical
approach trades classifier expressiveness for SED capability and few-shot flexibility.

## Design Evolution

### v0 (original): Pairwise contrastive episodes

The initial design used (reference, query) pairs with binary BCE loss. Critical
review identified fatal flaws:

1. **186x sparser supervision**: 1 species per episode vs 206 per sample.
2. **Uniform frame labels**: Labeling all 20 frames as 1.0 for positive pairs is
   actively wrong for silent frames — worse than the soft labels that failed in
   exp 014/015.
3. **BCE + tau=0.07**: Cosine similarity / 0.07 saturates sigmoid → dead gradients.
4. **Clip attention `Linear(1,1)`**: A 1-input linear layer is just scale+bias,
   cannot learn frame importance.

### v1: Prototypical classification (cross-entropy)

Switched to full 206-class prototypical classification with cross-entropy/softmax.
Fixed the supervision sparsity but introduced a new problem: softmax trains mutually
exclusive predictions, but soundscape inference is multi-label.

### v2 (current): Prototypical classification (BCE)

Same as v1 but with BCE (focal, gamma=2) loss instead of cross-entropy:
- BCE treats each species independently → consistent with multi-label inference.
- Matches exp 012's loss function (focal BCE).
- Same supervision density as the linear probe.
- No episode sampling, no reference/query split during training.
- Framewise max for SED at inference.
- Fixed tau=1.0.
- Prototypes re-computed every epoch (not every 5 — cheap with cached features).
- Added cosine gap analysis (step 1) and ablation baselines (step 2) before training.

### Deferred enhancements (if v1 shows promise)

- Reference attention: learn to weight frames by informativeness for prototype computation.
- Learnable temperature: log-parameterized with clamping, as in CLIP.
- Energy-weighted frame loss: weight BCE by mel RMS energy to suppress silent frames.
- Few-shot holdout: 20 species excluded from training, evaluated with K=5 references.
- Prototype momentum: EMA update instead of full re-computation.

## Relationship to Prototypical Networks and ProtoCLR

This experiment is **prototypical networks extended to frame-level SED**. The table
below clarifies the lineage:

### Prototypical Networks (Snell et al., 2017)

Standard few-shot classification: embed support examples, compute class prototypes
as means, classify queries by softmax over distances. Episodic training with N-way
K-shot tasks.

### ProtoCLR (Medina et al., 2020)

Combines prototypical networks with contrastive learning (InfoNCE loss). Still uses
episodic training. Learns representations where same-class embeddings cluster.

### This experiment

Uses the prototypical classification idea (similarity to prototype embeddings) but:
- **Full 206-class training** instead of N-way episodes → no supervision sparsity.
- **Frame-level similarity** instead of clip-level → enables SED.
- **Frozen backbone + projection** instead of end-to-end training → constrains the
  problem but limits expressiveness.
- **Re-computed prototypes** every epoch from training data → not per-episode.
- **BCE loss** → multi-label compatible, not InfoNCE or softmax.

This is **not a novelty claim** — it is a standard application of prototypical
classification extended to per-frame SED. The experiment tests two separable
questions: (1) does framewise max improve over mean pooling on frozen features,
and (2) can prototypical classification match a linear probe. These are tested
via ablation baselines (see proposal procedure).

## Zero-Training Baseline

The most important experiment: test whether the frozen backbone's features already
have metric structure for species discrimination **before writing any training code**.

```python
# 1. Extract features for all clips
features = {}  # {clip_id: (1280, 20) L2-normed frame features}
for clip_id, mel in all_clips:
    feat = backbone(mel)             # (1, 1280, 8, 20)
    feat = feat.mean(dim=2)          # (1, 1280, 20)
    feat = F.normalize(feat, dim=1)  # L2-norm per frame
    features[clip_id] = feat[0]      # (1280, 20)

# 2. Compute prototypes from training clips
prototypes = {}  # {species: (1280,)}
for species in species_list:
    all_frames = [features[c] for c in train_clips[species]]  # list of (1280, 20)
    stacked = torch.cat(all_frames, dim=1)                     # (1280, N*20)
    proto = stacked.mean(dim=1)                                # (1280,)
    prototypes[species] = F.normalize(proto, dim=0)

proto_matrix = torch.stack(list(prototypes.values()))  # (206, 1280)

# 3. Evaluate two aggregation strategies
for clip_id, label in val_clips:
    frames = features[clip_id]   # (1280, 20)

    # Clip-level: mean pool frames, then cosine sim
    clip_embed = F.normalize(frames.mean(dim=1), dim=0)   # (1280,)
    clip_sim = clip_embed @ proto_matrix.T                  # (206,)

    # Frame-level: cosine sim per frame, then max
    frame_sim = frames.T @ proto_matrix.T                   # (20, 206)
    frame_max_sim = frame_sim.max(dim=0).values             # (206,)

# 4. Compute val/cmap for both
# If frame_max_sim cmAP > clip_sim cmAP → SED has value
# If both < 0.10 → abort
```

### Decision matrix

| Clip-level cmAP | Frame-level cmAP | Interpretation | Action |
|-----------------|-------------------|---------------|--------|
| < 0.10 | < 0.10 | No metric structure | **Abort** experiment |
| 0.10 - 0.25 | similar to clip | No SED benefit | Reconsider — may just be a worse linear probe |
| 0.10 - 0.25 | > clip by 0.03+ | SED adds value | **Proceed** with prototypical training |
| > 0.25 | > clip by 0.03+ | Strong features + SED value | **Proceed** — projection head should exceed baseline |
| > 0.30 | similar to clip | Features already great, no SED need | Consider if projection adds anything over linear probe |

## Inference Pipeline

### Step 1: Pre-compute Prototypes (once, before inference)

```python
prototypes = np.zeros((206, D), dtype=np.float32)

for i, species in enumerate(species_list):
    ref_clips = select_references(species, K=5)
    ref_mels = np.stack([compute_mel(clip) for clip in ref_clips])

    # ONNX forward: backbone + projection + L2-norm
    frame_embeds = session.run(["frame_embeddings"], {"mel": ref_mels})[0]  # (K, 20, D)

    # Mean pool all frames across all K clips
    prototypes[i] = frame_embeds.reshape(-1, D).mean(axis=0)  # (D,)
    prototypes[i] /= np.linalg.norm(prototypes[i])

np.save("prototypes.npy", prototypes)
```

On Kaggle, pre-computed `prototypes.npy` ships with the weights dataset.

### Step 2: Per-Window Inference

```python
audio = load_audio_chunk(path, sr=32000, start_s=t, duration_s=5.0)
mel = compute_mel(audio, mel_cfg)                       # (1, 1, 256, 625)

frame_embeds = session.run(
    ["frame_embeddings"], {"mel": mel}
)[0]                                                    # (1, 20, D)

similarity = frame_embeds[0] @ prototypes.T             # (20, 206)
clip_logits = similarity.max(axis=0) / tau              # (206,)
clip_probs = sigmoid(clip_logits)                       # (206,)
```

### Step 3: Write Submission

Same format as exp 012 — one row per 5s window with 206 species probability columns.

## ONNX Export Strategy

Single ONNX model exports the backbone + projection head:

- Input: `mel (B, 1, 256, 625)`
- Output: `frame_embeddings (B, 20, D)` — L2-normalized projected frame features

Prototypes are stored separately as `prototypes.npy (206, D)`. This separates the
model graph from the class definitions, enabling prototype swapping for few-shot.

## Kaggle Submission Artifacts

```
birdclef-2025-weights/
├── prototypical_sed.onnx      # Backbone + projection head
├── prototypes.npy             # (206, D) pre-computed species prototypes
├── species_list.json          # Ordered species names
├── mel_config.json            # Mel spectrogram parameters
└── kaggle_inference.py        # Self-contained inference script
```

## Timing Estimate (Kaggle CPU)

| Step | Time |
|------|------|
| Load ONNX model | ~2s |
| Load prototypes.npy | <0.1s |
| Per 5s window: mel computation | ~50ms |
| Per 5s window: ONNX forward (backbone + projection) | ~200ms |
| Per 5s window: similarity matmul (20×D) @ (D×206) | <1ms |
| **Per 60s soundscape (12 windows)** | **~3s** |
| **Total for ~100 soundscapes** | **~5 min** |

Well within the Kaggle 9-hour time limit. Comparable to exp 012 inference speed.

## Implementation Notes

### Gradient safety

- L2-normalization: clamp norm to >= 1e-8 before dividing to prevent NaN gradients
  on zero-norm vectors (silence frames).
- Temperature: use fixed tau=1.0. If making learnable later, use log-parameterization
  `tau = exp(log_tau)` with clamping `log_tau.clamp(-2, 2)` to prevent collapse.
- BCE: operates on cosine similarities in [-1/tau, 1/tau]. With tau=1.0,
  this is [-1, 1] → sigmoid maps to [0.27, 0.73]. Well-behaved gradient range,
  no saturation. If tau is later reduced, monitor for gradient collapse.

### Memory budget

- Batch of 256 clips: backbone forward produces 256 × 1280 × 8 × 20 × 4 bytes = ~200MB
  feature maps. With `torch.no_grad()` on the frozen backbone, no activation storage.
- Projection head gradients: 256 × 20 × D × 4 bytes ≈ 5-40MB (D=256-1280). Negligible.
- Prototype matrix: 206 × D × 4 bytes ≈ 0.2-1MB. Negligible.

### Validation loop

Pre-compute all val clip frame embeddings (through projection head) at the start
of each validation epoch. Then compute similarity against current prototypes in one
batched matrix multiply: `(N_val × 20, D) @ (D, 206)`. This takes ~1 second, not
the 3 minutes per epoch that per-clip ONNX inference would require.

## References

### Concepts

- `docs/concepts/prompted-sound-event-detection.md` — Original prompted SED concept
  that motivated this line of work. Key ideas: reference conditioning, per-token
  detection, domain-gap bridging (focal → soundscape).
- `docs/concepts/unsupervised-event-detection-labeling.md` — Unsupervised event
  detection approaches; relevant for temporal localization without ground truth.

### Prior Experiments

| Experiment | Relevance |
|-----------|-----------|
| `011-prompted-sed/` | Direct predecessor. Phase 0 showed SSAMBA backbone too weak. Prototypical similarity identified as more promising. Exp 016 implements this with EfficientNet. |
| `012-efficientnet-linear-probe/` | Primary baseline (0.336 val/cmap, 0.649 Kaggle). Provides the frozen backbone, data pipeline, and val split that exp 016 reuses. |
| `013-weak-supervision-flyingsquid/` | Produced soft labels that exp 014/015 used. Their failure motivated avoiding teacher labels entirely. |
| `014-efficientnet-soft-labels/` | Negative result (0.238). Soft labels degrade linear probe. Key lesson: don't replace ground truth. |
| `015-efficientnet-sed/` | Negative result (0.237). SED + soft labels adds nothing. Confirmed bottleneck is label quality. |

### Kaggle References

- `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` — 1st place: SED + attention pooling, framewise max at inference.
- `docs/references/kaggle/birdclef-2025/bird2025-single-sed-model-inference-lb-0-857.md` — Single SED model, 0.857 LB.
- `docs/references/kaggle/birdclef-2025/pseudo-label-creator.md` — Pseudo-labeling; prototypes could generate pseudo-labels as follow-up.

### Deep Research

- `docs/references/deep-research/2026-02-14-frame-level-ssl-pretraining.md` — Frame-level SSL context.
