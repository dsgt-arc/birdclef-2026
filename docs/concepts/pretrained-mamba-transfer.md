# Concept: Pre-trained Mamba Transfer with Re-initialized Tokenizer

**Status**: developing
**Created**: 2026-02-09
**Updated**: 2026-02-17

## Idea

Load pre-trained SSAMBA weights (AudioSet, 16×16 patches) but re-initialize only the patch embedding layer to accept a different input tokenization (e.g., 128×1 frequency frames at 10ms resolution). The Mamba backbone — 24 layers of bidirectional SSM blocks, ~99% of the parameters — transfers directly. This gives data-efficient training for downstream tasks like prompted SED, because the backbone already understands audio temporal dynamics, and Mamba's recurrence naturally extends to longer sequences without architectural changes.

## Motivation

SSAMBA from scratch achieves 0.08 cmAP on BirdSet HSN. Pre-trained ConvNeXT achieves 0.48. The gap is almost entirely explained by pre-training: AudioSet provides 2M clips of diverse audio, giving the model a rich prior over spectral and temporal patterns before it ever sees bird data.

The problem: SSAMBA was pre-trained with 16×16 patches on a mel spectrogram (treating audio as a 2D image). This produces ~1500 tokens for a 10s clip. For token-level SED and prompted detection, we want frame-level tokenization — one token per 10ms (3000 tokens for 30s) — which requires a different patch embedding. Naively, this means discarding the pre-trained patch embedding and losing its learned features.

But the patch embedding is tiny: `Conv2d(1, D, 16, 16)` has ~50K params for D=192. The 24 Mamba layers have ~7M params (SSAMBA-tiny). Re-initializing <1% of the model while keeping >99% pre-trained is a well-established transfer learning pattern. The question is whether the Mamba layers, trained on 16×16-patch tokens, can adapt to a new tokenizer's output distribution — and why Mamba's architecture makes this adaptation particularly natural.

## Background

### Why Mamba transfers to new token types

The Mamba backbone's pre-trained knowledge lives in three places:

**1. State dynamics (A_log, dt_proj):** How to accumulate and decay information over time. The discretized recurrence is:

```
h[t] = exp(Δ·A) · h[t-1] + Δ·B · x[t]
y[t] = C · h[t] + D · x[t]
```

where `A_log` controls decay rates and `dt_proj` controls the input-dependent time step Δ. These learn temporal patterns: "information from 200ms ago is still relevant" or "this event boundary should reset the state." These dynamics are properties of audio, not of the tokenizer. A bird call's temporal structure — onset, sustain, decay, repetition — is the same whether the tokens came from 16×16 patches or 128×1 frames.

**2. Gating (in_proj, out_proj, SiLU):** Which features to pass through and which to suppress. The gated output `y * SiLU(z)` learns feature selection patterns: "when the state indicates a vocalization, amplify these channels." Again, this is about what audio features mean, not how they were tokenized.

**3. Inter-layer feature refinement (24 layers of residual + norm):** Each layer refines the representation. Lower layers extract local patterns; higher layers compose them into global structure. The layer-to-layer feature hierarchy transfers across tokenization changes because it operates on abstract feature vectors, not on raw spectrogram patches.

What does NOT transfer: the patch embedding weights themselves. These learn to extract features from a specific spatial layout (16×16 pixel patches). A 128×1 frame has a fundamentally different shape — the weights can't be reused. But at ~50K params out of ~7M total, this is a small price.

### Why Mamba extends to longer sequences naturally

This is the key architectural advantage over transformers for prompted SED.

**Transformer positional encoding is length-bound.** A ViT/AST pre-trained on 1500 tokens has 1500 learned positional embeddings. To process 3500 tokens (5s ref + 30s soundscape), you must either:
- Interpolate positional embeddings (lossy, especially for large stretches)
- Use relative positional encodings (requires architecture change, retraining)
- Truncate (lose information)

**Mamba has no positional encoding to break.** The recurrence `h[t] = f(h[t-1], x[t])` processes one token at a time, maintaining a fixed-size hidden state `h ∈ ℝ^{d_inner × d_state}`. Position information is implicit in the state evolution — the model "knows" where it is because of what it has accumulated, not because of a position index. Processing 3500 tokens instead of 1500 just means more recurrence steps. The state dynamics are identical.

Concretely, for SSAMBA-small (embed_dim=384, expand=2, d_state=16):
- Hidden state per direction: `(768, 16)` = 12,288 scalar values
- This state is the same size whether processing 500 tokens or 50,000 tokens
- The state equation `h[t] = exp(Δ·A) · h[t-1] + Δ·B · x[t]` is applied identically at every step

**The selective scan makes this input-dependent.** Unlike a fixed linear RNN where A and B are constant matrices, Mamba's B, C, and Δ are functions of the input x[t] at each step. This means:
- The model can choose to "remember" (large Δ, slow decay) or "forget" (small Δ, fast decay) at each token
- When processing a reference clip prefix, it can selectively compress the call's signature into the state
- When scanning the soundscape suffix, it can "resonate" when the input matches the state pattern

This selective compression is why prompted SED is natural for Mamba: the reference clip gets compressed into the hidden state, and the soundscape processing uses that state as a learned matched filter. No cross-attention mechanism needed — the recurrence IS the attention mechanism, just amortized over time.

### Train short, evaluate long

Because the recurrence is length-agnostic, the training sequence length can be much shorter than the inference length. The optimal training length is driven by what the model needs to learn per example, not by the target deployment length.

For prompted SED, the model needs to learn one skill per episode: "compress reference into state, detect matches in query." This requires:
- A reference segment containing at least one full call (~0.1-2s for most birds, so 3-5s of reference is sufficient)
- A query segment with some positive tokens (call present) and some negative tokens (silence/other sounds) for contrast (~7-10s)

This gives a natural training length of **10-15s total** (3s ref + 7s query, or 5s ref + 10s query). At inference, you evaluate on 30-60s where the model benefits from more temporal context — more chances to observe a match — but the per-token matching skill was learned on short windows.

**Why 10s is a sweet spot:**

| Consideration | Implication |
|---|---|
| AudioSet pre-training length | 10s — backbone state dynamics calibrated for this timescale |
| Bird call duration | 0.1-2s — 3s reference almost certainly contains a full call |
| Training efficiency | 10s = 1000 tokens at 10ms. Batches are cheap, iteration is fast |
| Gradient signal density | 7-10s query contains 1-3 call events — enough positive/negative contrast per example |
| GPU memory | Shorter sequences = larger batches = better gradient estimates |

**What short training can't teach:** Long-range repetition patterns (e.g., a species that calls every 15s). But this is an inference bonus, not a training requirement. The state dynamics generalize — a decay rate learned on 10s windows works identically on 60s. The model accumulates evidence from repeated calls at inference even if it never saw multi-call sequences during training.

**What matters more than length: episode diversity.** A batch of diverse 10s episodes — within-recording pairs, cross-recording pairs, negative pairs, hard negatives — teaches more than a single 60s example, and you get far more episodes per GPU hour.

**Comparison with transformers:**

| | Mamba | Transformer |
|---|---|---|
| Train 10s, evaluate 60s | Works natively — same recurrence, more steps | Positional embeddings break — need interpolation or retraining |
| Compute scaling | 6x longer = 6x compute (linear) | 6x longer = 36x compute (quadratic) |
| What transfers across lengths | State dynamics, gating, decay rates | Nothing without positional encoding surgery |

**What about the pre-trained positional embeddings?** SSAMBA does use learned absolute positional embeddings (`pos_embed`), added after the patch embedding. These are tied to the original number of patches. When re-initializing the patch embedding for a different token count, you also re-initialize `pos_embed`. For Mamba, this is less damaging than for a transformer because:
- Mamba's temporal modeling is primarily in the recurrence, not the positional embeddings
- The positional embeddings mainly help the first few layers orient to the sequence; deeper layers rely on accumulated state
- Empirically, Mamba models are less sensitive to positional encoding perturbations than transformers

### Bidirectional structure and prefix conditioning

SSAMBA uses two levels of bidirectionality:

**Inner level:** Each Mamba block has separate forward and backward SSMs with independent A, B, C, D, conv1d parameters. The forward SSM scans left-to-right; the backward SSM scans right-to-left.

**Outer level:** Layers are processed in pairs. Even layers process the sequence forward; odd layers process it flipped. Outputs are summed:

```
for i in range(num_layers // 2):
    h_fwd = layer[2i](tokens)
    h_bwd = layer[2i+1](tokens.flip())
    tokens = h_fwd + h_bwd.flip()
```

#### Two conditioning regimes

The original framing of prompted SED assumed unidirectional Mamba: the forward SSM compresses the reference into a state, then scans the soundscape with that state as a "learned matched filter." But SSAMBA's bidirectionality changes this picture fundamentally:

**Regime 1: State compression** — the forward scan's accumulated state after the reference carries species identity. The last reference token has compressed the reference into a fixed-size state that conditions all subsequent target processing. The backward scan provides complementary information (target context flowing back into reference positions). This is the "learned matched filter" model.

**Regime 2: Distributed conditioning** — species identity is encoded across the **spatial pattern** of token embeddings, not compressed into any single position. Because both forward and backward scans see the entire sequence, every token's output reflects global information. The reference conditions target tokens through deep feature interaction across 24 layers of bidirectional mixing, not through a single compressed state vector.

The key insight: in bidirectional SSAMBA, **every token sees the full sequence from both directions**. There is no position that uniquely compresses the sequence — each token has a different *perspective* on the same global information (different positional emphasis of forward vs backward recency).

#### Implications for prompted SED

**If Regime 1 holds** (state compression works): the original prompted SED design works as described. Feed `[ref][SEP][target]`, the forward state carries species identity into target processing.

**If Regime 2 holds** (distributed, not compressed): the prompting still works, just through a different mechanism. The reference modulates the entire target representation through deep bidirectional interaction. Each target token's embedding becomes a function of (local audio content, full reference context mediated by 24 layers).

Distributed conditioning is actually **less lossy** than state compression — a fixed-size state is an information bottleneck, while distributed conditioning preserves all reference information across the sequence. But the concern is **dilution**: the reference (500 tokens) is ~14% of the total input (3501 tokens). In a bidirectional model, target tokens dominate both forward and backward context, and the reference signal may get washed out in deeper layers.

#### The critical diagnostic: conditioning delta

The Phase 0 test for prompted SED (exp 011) checks whether the backbone encodes species at all (clustering). But the **prompted-SED-specific** diagnostic is whether the reference actually influences target processing:

```python
# Same target, different references
emb_A = backbone(concat(ref_species_X, target))[:, 500:, :]  # target portion
emb_B = backbone(concat(ref_species_Y, target))[:, 500:, :]  # same target, different ref

delta = (emb_A - emb_B).norm(dim=-1).mean()
```

If `delta ≈ 0`: the model ignores the reference entirely. Target embeddings are the same regardless of what reference is prepended. Prompted SED cannot work through concatenation — need explicit comparison mechanism.

If `delta > 0`: the reference IS influencing target embeddings through the bidirectional layers. The SED head has something to work with. The magnitude of delta relative to the overall embedding norm indicates conditioning strength.

#### Fallback strategies if conditioning is weak

If the conditioning delta is small (reference influence is diluted), three approaches in order of increasing complexity:

**1. Prototypical similarity head** — process ref and target separately, compare explicitly:
```python
ref_emb = backbone(mel_ref, return_all_tokens=True).mean(dim=1)   # (B, D)
tgt_emb = backbone(mel_tgt, return_all_tokens=True)               # (B, T, D)
similarity = (tgt_emb * ref_emb.unsqueeze(1)).sum(-1)             # (B, T)
```
No concatenation needed. The backbone just needs to produce good per-token features (which the mean-pool clustering test confirms). This is ProtoCLR at token level — simple, robust, no architectural risk.

**2. Lightweight cross-attention** — add one cross-attention layer between ref and target embeddings:
```python
attn_out = cross_attention(
    query=target_embeddings,   # (B, 3000, D)
    key=ref_embeddings,        # (B, 500, D)
    value=ref_embeddings
)
logits = sed_head(attn_out)   # (B, 3000, 1)
```
Makes the comparison explicit. Cost is O(3000 × 500) — cheap. Adds one layer of parameters but doesn't modify the backbone.

**3. Standard fixed-head SED** — abandon prompting, use `Linear(D, 206)` per token. This is what the BirdCLEF 2025 winners use. Lose zero-shot capability but keep the SED gains (+0.03-0.05 over clip classification). The distributed features are good for classification, just not for the specific prompting mechanism.

The prototypical similarity head (option 1) is the strongest fallback because it preserves zero-shot capability (just provide a new species' reference clip at inference) while avoiding the conditioning question entirely.

### Pre-trained weight structure

The weights that transfer (for SSAMBA-tiny, embed_dim=192):

| Component | Params | Transfers? |
|---|---|---|
| `patch_embed.proj` (Conv2d 16×16) | 50K | **No** — shape mismatch |
| `pos_embed` (1, N, 192) | ~290K | **No** — length mismatch |
| `cls_token` (1, 1, 192) | 192 | Yes (if used) |
| `layers.{0-23}.mixer.*` (Mamba blocks) | ~6.5M | **Yes** |
| `layers.{0-23}.norm.*` (LayerNorm) | ~9K | **Yes** |
| `norm_f.*` (final norm) | 384 | **Yes** |
| **Total transferable** | **~6.5M / 7M** | **~93%** |

For SSAMBA-small (384) and base (768), the ratio is even higher because the Mamba layers scale quadratically with embed_dim while the patch embedding scales linearly.

## Practical Transfer Recipe

### Step 1: Load pre-trained weights, skip mismatched layers

```python
pretrained = load_checkpoint("ssamba_tiny_audioset.pth")
model = SSAMBAModel(config)  # config with new patch_size=(128, 1)

# Load only matching keys
model_dict = model.state_dict()
transfer = {k: v for k, v in pretrained.items()
            if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(transfer)
model.load_state_dict(model_dict)

# patch_embed.proj, pos_embed: randomly initialized
# layers.*.mixer.*, layers.*.norm.*, norm_f.*: pre-trained
```

### Step 2: Freeze backbone initially (optional)

```python
# Freeze Mamba layers, train only patch_embed + head
for name, param in model.named_parameters():
    if "patch_embed" not in name and "head" not in name:
        param.requires_grad = False

# Train for N epochs to adapt patch embedding to pre-trained backbone
# Then unfreeze everything for end-to-end fine-tuning
```

### Step 3: Extend to prompted SED

```python
# For prompted SED: concatenate reference + soundscape tokens
ref_tokens = model.patch_embed(ref_mel)       # (B, ~500, D)
snd_tokens = model.patch_embed(soundscape_mel) # (B, 3000, D)
sep = model.sep_token.expand(B, 1, -1)         # new learned token

tokens = torch.cat([ref_tokens, sep, snd_tokens], dim=1)  # (B, ~3501, D)
# pos_embed: new, covers full length
tokens = tokens + model.pos_embed[:, :tokens.size(1)]

# Mamba backbone processes the full sequence
# Pre-trained layers handle temporal dynamics
# Fine-tuning teaches: "prefix = query, suffix = search space"
output = model.backbone(tokens)

# SED head on soundscape portion only
logits = model.sed_head(output[:, 501:, :])  # (B, 3000, 1)
```

## Open Questions

- **How much does re-initializing pos_embed hurt?** Mamba relies less on positional embeddings than transformers, but the first few layers may use them heavily. Could compare: (a) random init, (b) interpolated from pre-trained, (c) no positional embeddings at all.
- **Freeze-then-unfreeze vs end-to-end from the start?** Freezing the backbone while the patch embedding adapts prevents gradient noise from corrupting pre-trained features. But it also prevents the backbone from adapting to the new token distribution. What's the right schedule?
- **Which pre-trained model size?** SSAMBA-tiny (7M) is fast but less expressive. SSAMBA-small (26M) has richer representations but needs more GPU memory. For BirdCLEF's inference budget, tiny may be the right trade-off.
- **Does the double-norm quirk matter?** The HF port preserves a bug from the original SSAMBA where `norm_f` is applied twice. Pre-trained weights were trained with this behavior, so removing it would break transfer. Worth testing whether fixing it after transfer improves fine-tuning.
- **CLS token positioning:** Pre-trained SSAMBA inserts the CLS token at the middle of the sequence, not the beginning. For prompted SED, middle insertion splits the reference-soundscape boundary oddly. Should we move it to the end, or drop it entirely and use mean pooling?

## Testable Hypotheses

1. **Pre-trained backbone + random patch embed >> random everything**: SSAMBA-tiny with pre-trained Mamba layers and a randomly initialized 128×1 patch embedding achieves >0.15 cmAP on BirdSet HSN (vs 0.08 from scratch), confirming that the backbone transfers despite the tokenizer change.

2. **Mamba extends to 3500 tokens without degradation**: A model pre-trained on ~1500 tokens (10s, 16×16 patches) processes 3500 tokens (5s ref + 30s soundscape, 128×1 frames) with no loss in temporal modeling quality, as measured by comparable or better cmAP vs a model trained from scratch on 3500 tokens.

3. **Prompted SED works with pre-trained weights**: After fine-tuning with episodic training, the pre-trained model achieves >0 AP on held-out species (zero-shot), confirming that Mamba's recurrence can learn to use the reference prefix as a species-specific matched filter.

4. **Freeze-then-unfreeze outperforms end-to-end**: Freezing the backbone for the first N epochs while the patch embedding adapts, then unfreezing for end-to-end fine-tuning, achieves higher final cmAP than training everything from the start, because it prevents early gradient noise from corrupting pre-trained features.

5. **Positional embeddings are less critical for Mamba than transformers**: Removing positional embeddings entirely from a pre-trained SSAMBA degrades cmAP by <5% relative, compared to >20% degradation for a comparable transformer (AST), because Mamba's temporal modeling is in the recurrence, not the positions.

6. **Train on 10s, evaluate on 60s with no degradation**: A prompted SED model trained on 10s episodes (3s ref + 7s query) achieves comparable or better per-species AP when evaluated on 60s soundscapes vs 10s soundscapes, because longer windows provide more opportunities for the learned per-token matching skill to fire, and the state dynamics generalize across lengths.

## Connections

- **Based on**: [SSAMBA architecture](ssamba-audio-mamba.md) (pre-trained weights and model structure), [Learned frequency tokenizer](cnn-mamba-hybrid-frontend.md) (alternative tokenizers that this concept enables)
- **Related to**: [Prompted SED](prompted-sound-event-detection.md) (primary downstream application), [Weak supervision label fusion](weak-supervision-label-fusion.md) (provides training labels for fine-tuning)
- **Could lead to**: Experiment combining pre-trained transfer + prompted SED + weak supervision labels — the full stack

## Notes

- The concept of re-initializing the tokenizer while keeping the backbone is standard in NLP (swap tokenizer, keep transformer) and vision (swap patch size, keep ViT layers). The difference with Mamba is that the sequence length extension comes for free — no positional encoding surgery needed.
- This concept is orthogonal to the frequency attention tokenizer from [cnn-mamba-hybrid-frontend](cnn-mamba-hybrid-frontend.md). You could combine both: pre-trained Mamba backbone + LEAF + freq attention as the new tokenizer. The backbone provides temporal modeling; the freq attention provides richer spectral features. They compose naturally because the tokenizer output interface is the same: `(B, T, D)`.
- For the BirdCLEF competition, this is arguably the highest-leverage single change: going from 0.08 (from scratch) to potentially >0.15 (pre-trained backbone) with minimal engineering — just a weight loading script and a new patch embedding layer.
- The prompted SED application is particularly exciting because it aligns with how BirdCLEF provides data: reference recordings of target species are available at inference time. A model that can condition on a reference clip is using the competition's own data format as an architectural advantage.
- The state size for SSAMBA-tiny is `(384, 16)` per direction = 12,288 values. This is the bottleneck for how much information from the reference clip can influence soundscape processing. For complex songs with multiple syllable types, this may be insufficient. SSAMBA-small doubles the state to `(768, 16)` = 24,576 values, which may be necessary for prompted detection of complex vocalizations.
