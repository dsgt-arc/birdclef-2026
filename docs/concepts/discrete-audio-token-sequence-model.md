# Concept: Discrete Audio Token Sequence Model for Bioacoustics

**Status**: shelved/partially falsified
**Created**: 2026-01-30
**Updated**: 2026-01-30

## Idea

Instead of treating audio classification as a spectrogram patch problem (the current BEATs/AST paradigm), treat it as a **sequence modeling problem over discrete audio tokens**. Neural audio codecs like WavTokenizer can compress audio into 40-75 discrete tokens per second. A Mamba or transformer sequence model over these tokens could process full soundscapes (minutes of audio) as a single sequence, support few-shot "prompting" by prepending reference call tokens, and potentially reconstruct audio from its token representation. This reframes bioacoustic identification as something closer to language modeling over an audio vocabulary.

This is a fundamentally different approach from spectrogram-patch models. Rather than patchifying a 2D spectrogram into visual tokens, we tokenize the raw waveform into a 1D sequence of discrete codes that already encode acoustic structure. The sequence model then learns temporal patterns over these codes -- analogous to how an LLM learns linguistic patterns over text tokens.

## Motivation

Several limitations of the current spectrogram-patch approach motivate this:

1. **Spectrogram resolution is baked into every model.** BirdNET (48kHz, 3s chunks), Perch (32kHz, 5s chunks), BEATs/AST (16kHz, 128 mel bins, 10ms hop) all have fixed spectrogram frontends. The CNN or transformer architecture is designed around a specific time-frequency resolution. Changing the resolution means retraining. Even BirdNET and Perch, which output single embeddings per chunk, have internal architectures shaped by their spectrogram frame size -- the number of convolutional layers, kernel sizes, and stride patterns are all calibrated to a specific mel resolution.

2. **Chunking loses temporal context.** Current models process 3-10s windows independently. Species identification cues that span longer timescales (repeated call patterns, dawn chorus structure, response calls between individuals) are invisible to chunk-level models.

3. **No reconstruction path.** Spectrogram patch models are lossy in one direction -- you can't listen to what the model "heard." Discrete acoustic tokens from codecs like WavTokenizer can be decoded back to audio, enabling interpretability (listen to what the model attends to) and generation (synthesize reference calls).

4. **Few-shot prompting is natural with sequences.** If audio is a token sequence, you can prepend reference examples to a query sequence -- exactly like few-shot prompting in LLMs. Systems like UniAudio 1.5 already demonstrate this. For BirdCLEF, this could mean: "here are 3 examples of Species X calls [tokens], does this soundscape contain Species X? [tokens]."

## Background

### Spectrogram frame sizes across existing models

All models start from a spectrogram, but the resolution varies and shapes the architecture:

| Model | Sample Rate | Spectrogram | Frame Size | Chunk | Output |
|-------|-------------|-------------|------------|-------|--------|
| BirdNET | 48kHz | mel (CNN frontend) | 3s | 3s | 1024-dim embedding |
| Perch | 32kHz | PCEN mel | 5s | 5s | 1280-dim embedding |
| BEATs | 16kHz | 128 mel, 10ms hop | 10s (1024 frames) | 10s | 768-dim per token |
| AST | 16kHz | 128 mel, 10ms hop | 10s (1024 frames) | 10s | 768-dim per token |
| SSAMBA | 16kHz | 128 mel, 10ms hop | 10s (1024 frames) | 10s | 768-dim per token |

For BirdNET and Perch, the spectrogram frame size determines the internal feature map dimensions. BirdNET at 48kHz produces a denser spectrogram than BEATs at 16kHz -- more frequency resolution, more temporal detail per second. The CNN's receptive field, number of pooling layers, and output stride are all designed for this specific resolution. You can't trivially change the sample rate or mel configuration without retraining the entire network.

For transformer models, the spectrogram is explicitly patchified (16×16 patches), making the resolution/token-count relationship explicit. But the patch embedding weights are still tied to a specific spectrogram configuration.

### Neural audio codecs (discrete tokenization)

| Codec | Tokens/sec | Codebooks | Bitrate | Reconstruction Quality |
|-------|-----------|-----------|---------|----------------------|
| SoundStream | ~600 | 8-12 (RVQ) | ~6 kbps | Good speech |
| EnCodec | ~900 | 8 (RVQ) | ~6-24 kbps | Good speech + music |
| DAC | ~900 | 9 (RVQ) | ~9 kbps | High quality |
| **WavTokenizer** | **40-75** | **1** | **0.9 kbps** | **Outperforms DAC at 9kbps** |

WavTokenizer (ICLR 2025) is a breakthrough: single-codebook, 40-75 tokens/second, better quality than multi-codebook models at 10× the bitrate. This makes audio token sequences comparable in length to text.

Token counts for bioacoustic recordings at 75 tokens/sec:

| Duration | Tokens | Comparable to... |
|----------|--------|-----------------|
| 5s | 375 | A long paragraph |
| 10s | 750 | A short document |
| 30s | 2,250 | A page of text |
| 60s | 4,500 | ~2 pages of text |
| 5min | 22,500 | A long article |

At 75 tokens/sec, a 60s soundscape is 4,500 tokens. For Mamba (linear complexity), this is very tractable. For attention, this is on the edge -- feasible with flash attention but expensive.

### Token types: semantic vs acoustic

- **Acoustic tokens** (EnCodec, WavTokenizer): Encode waveform fidelity. Can reconstruct audio. Don't necessarily encode "meaning."
- **Semantic tokens** (HuBERT k-means, BEATs codebook): Encode high-level content. Good for classification. Cannot reconstruct audio.
- **Hybrid** (X-Codec, BiCodec): Separate streams for content and acoustics.

For bioacoustics, we likely want **both**: semantic tokens for species ID, acoustic tokens for reconstruction/verification. Or a hybrid codec trained on bird audio.

### Few-shot prompting with audio tokens

- **UniAudio 1.5** (2024): Concatenates audio tokens from reference examples with query audio as a single LM sequence. Few-shot audio tasks without retraining.
- **PALM** (EMNLP 2024): Few-shot prompt learning for audio-language models. Optimizes learnable context embeddings prepended to audio features.
- **Prototypical networks**: Compute class prototypes from support examples, classify by nearest prototype. Standard few-shot approach for audio.

The token-sequence approach enables a richer form of matching than embedding similarity: the model can attend to temporal patterns in the reference and look for them in the query.

### Positional encoding for long audio sequences

Current audio transformers use **learned positional embeddings** (from ViT), which don't extrapolate beyond training length. Alternatives:

- **RoPE (Rotary Position Embedding)**: Encodes relative position via rotation matrices. Enables length generalization -- train on 10s, inference on 60s. Widely adopted in LLMs (LLaMA, Mistral, Qwen). Applied to audio in BS-RoFormer and Mel-RoFormer for music source separation.
- **2D axial RoPE**: Extends to two dimensions (time + frequency). Directly applicable to spectrograms.
- **ALiBi**: Linear attention bias, another length-generalization approach.

**RoPE needs further investigation** for this application. Key questions: does 1D RoPE suffice for a 1D token sequence (from a codec), or do we need 2D RoPE if we keep the spectrogram-patch approach? How well does RoPE extrapolate for audio specifically (most evidence is from text)?

References:
- [RoFormer (Su et al.)](https://arxiv.org/abs/2104.09864) - original RoPE paper
- [BS-RoFormer](https://arxiv.org/abs/2309.02612) - RoPE for audio source separation, won SDX23
- [ComRoPE (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_ComRoPE_Scalable_and_Robust_Rotary_Position_Embedding_Parameterized_by_Trainable_CVPR_2025_paper.pdf) - trainable rotation matrices

## Open Questions

- **Do acoustic tokens from WavTokenizer preserve the fine spectral structure of bird calls?** WavTokenizer is trained on speech/music. Bird vocalizations have different characteristics (higher frequencies, rapid FM sweeps, harmonic stacks). May need domain-specific codec training or evaluation.
- **What's the right vocabulary size for bioacoustic tokens?** WavTokenizer uses a large codebook (4096+). Bird audio might need a smaller, specialized vocabulary. Or the general vocabulary might already cover it.
- **Can a Mamba model over discrete tokens match BEATs-quality representations?** The discrete bottleneck loses information. Is the token sequence expressive enough for fine-grained species classification?
- **How does RoPE behave for audio length extrapolation?** Train on 10s sequences, test on 60s. Does it degrade gracefully or catastrophically? (Needs empirical testing -- flagged for deeper research.)
- **Is few-shot prompting actually better than embedding retrieval for species ID?** Prepending reference tokens is more expensive than cosine similarity over embeddings. The question is whether the additional context helps for difficult cases (similar species, noisy recordings).
- **What sample rate should the codec operate at?** WavTokenizer uses 24kHz. Bird audio is often recorded at 32-48kHz with relevant content up to 10-12kHz. Is 24kHz sufficient, or do we need a higher-rate codec?

## Testable Hypotheses

1. **WavTokenizer tokens preserve bird call structure**: WavTokenizer round-trip (encode → decode) on BirdCLEF recordings will produce audio where BirdNET classification confidence drops by <10%, indicating the codec preserves species-discriminative features.

2. **Discrete token sequences enable longer-context classification**: A Mamba model processing 60s of WavTokenizer tokens (4,500 tokens) will outperform the same model processing 12 × 5s chunks independently on multi-species soundscape classification, because it captures cross-call temporal patterns.

3. **Few-shot token prompting outperforms embedding retrieval on rare species**: For species with <10 training examples, prepending 3 reference call token sequences to the query and running through a sequence model will achieve higher F1 than cosine similarity on BEATs embeddings.

4. **A codec-based model can fit BirdCLEF's CPU inference budget**: Processing 60s of audio as 4,500 WavTokenizer tokens through a small Mamba model will complete in <2s on CPU, compared to BEATs' ~6s for the same duration (chunked).

## Connections

- **Based on**: [temporal-resolution-sequence-length](temporal-resolution-sequence-length.md) (token count analysis, patch shape tradeoffs), [pretraining-compute-feasibility](pretraining-compute-feasibility.md) (training budget constraints), [context-recovery](context-recovery.md) (why local context isn't enough)
- **Related to**: [distilled-sequence-model](distilled-sequence-model.md), [ssamba-audio-mamba](ssamba-audio-mamba.md)
- **Could lead to**: Experiment evaluating WavTokenizer reconstruction quality on bird audio; Mamba-over-tokens pilot on ESC-50; RoPE integration into SSAMBA-HF

## Notes

### Bidirectional Mamba for classification and SED

For BirdCLEF, we have the full recording available for post-hoc analysis -- there's no streaming requirement. This means we can use **bidirectional Mamba** (as SSAMBA already does), giving every token context from the entire sequence.

Standard (causal) Mamba is sequential: token N only sees tokens 1..N. Bidirectional Mamba runs forward + backward scans and merges them, so every token gets full-sequence context. SSAMBA-HF (Exp 004) already implements this with 24 layers processed as 12 forward-backward pairs.

#### Architecture: shared encoder, multiple heads

```
                        ┌─────────────────────────────────┐
                        │     Mel Spectrogram (32kHz)      │
                        │     128 bins × 6000 frames       │
                        └──────────────┬──────────────────┘
                                       ↓
                        ┌──────────────────────────────────┐
                        │   Patch Embed Conv2d(128, 2)     │
                        │   → 3000 tokens × D dims         │
                        └──────────────┬───────────────────┘
                                       ↓
                        ┌──────────────────────────────────┐
                        │   Bidirectional Mamba (L layers)  │
                        │   Forward scan + Backward scan    │
                        │   Each token sees full 60s        │
                        └──────────────┬───────────────────┘
                                       ↓
                    ┌──────────────────┼──────────────────┐
                    ↓                  ↓                  ↓
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │  Attn Pool   │  │  Per-Token   │  │  Per-Token   │
            │  → clip label│  │  → SED       │  │  → reconstruct│
            │  (species)   │  │  (when)      │  │  (listen)    │
            └──────────────┘  └──────────────┘  └──────────────┘
```

Three heads from the same encoder:

1. **Classification head**: Attention-weighted pooling over all 3000 tokens → multi-label sigmoid → "which species are in this soundscape?" This is the BirdCLEF submission output.

2. **SED head**: Per-token linear → sigmoid per species → "when does each species call?" At 128×2 patches (50 tokens/sec), this gives **20ms resolution** sound event detection:

```
Time:    0s──────10s──────20s──────30s──────40s──────50s──────60s
Sp. A:   ....░░░░░░......░░░░░░.............................░░░░░░....
Sp. B:   ....................░░░░........░░░░........░░░░........░░░░..
Sp. C:   ................................................................
```

3. **Reconstruction head** (optional): Decode tokens back to audio via codec decoder for interpretability -- listen to what the model processed.

#### Training with weak labels

BirdCLEF data is **weakly labeled** -- recordings have species tags but no timestamps for when each species calls. Three approaches for training SED from weak labels:

**Multiple Instance Learning (MIL) pooling** (most common):
- Compute per-token logits for each species
- Attention-weighted pool to clip-level prediction
- Train with clip-level BCE loss
- SED emerges from the per-token activations: tokens contributing to a positive clip prediction are implicitly the ones where the species is calling
- No per-frame annotations needed

**Clip-level train, token-level infer**:
- Train: pool all token embeddings → clip-level species prediction with weak labels
- Infer: read the per-token activations before pooling → SED output
- Simpler than MIL, and per-token predictions naturally peak at call locations

**Teacher-guided pseudo-labels**:
- Run BirdNET (3s chunks) or Perch (5s chunks) on training data
- Use their per-chunk predictions as approximate timestamps
- Train the Mamba SED head with these noisy frame-level labels
- Bootstrapping: use production models to generate training signal for the new model

#### Few-shot prompting with reference calls

With the full sequence available, few-shot conditioning becomes a natural extension:

```
[ref call A tokens] [SEP] [ref call B tokens] [SEP] [60s soundscape tokens]
       (2s = 100 tokens)    (2s = 100 tokens)        (60s = 3000 tokens)
                                                            ↓
                                              Bidirectional Mamba
                                                            ↓
                            Per-token: "does this match ref A or ref B?"
```

The model processes reference calls first, building up a hidden state representation of "what to listen for," then scans the soundscape with that context. For rare species with few training examples, this in-context conditioning could outperform a fixed classifier head that has barely seen the species during training.

#### Positional encoding: Mamba doesn't need it

A key advantage of Mamba over transformers for this use case: **no positional encoding required**. Position is implicit in the recurrence -- token 1500 has been processed through 1500 state updates, so its position is inherently encoded. This means:

- Train on 10s (500 tokens) → inference on 60s (3000 tokens) with **zero architectural changes**
- No learned positional embeddings to extrapolate (which break beyond training length)
- No RoPE scaling tricks needed
- The recurrence just keeps running

The tradeoff: Mamba's long-range memory decays with distance (finite hidden state capacity), while attention gives direct access to any token. For audio, where the most relevant context is usually nearby (within the same call or call sequence), this is likely acceptable. Bidirectional processing also helps: the backward scan gives "future" context to early tokens, so no token is more than 30s from both boundaries in a 60s clip.

#### Comparison: position encoding by architecture

| Architecture | Position Encoding | Train 10s → Infer 60s | Long-range access | CPU cost (60s) |
|---|---|---|---|---|
| Transformer + learned pos | Learned (ViT) | **Breaks** | Perfect | O(n²) |
| Transformer + RoPE | Rotary | **Works** (extrapolates) | Perfect | O(n²) |
| **Bidirectional Mamba** | **Implicit (recurrence)** | **Works** (just keeps running) | Decays with distance | **O(n)** |

### Two paradigms compared

| Aspect | Spectrogram Patches (current) | Discrete Audio Tokens (proposed) |
|--------|------------------------------|--------------------------------|
| Input | 2D mel spectrogram | 1D waveform |
| Tokenization | Conv2d patchify (16×16) | Neural codec (WavTokenizer) |
| Token rate | ~50/sec (16×16 patches) | 40-75/sec (WavTokenizer) |
| Resolution | Fixed by mel config + patch size | Fixed by codec design |
| Reconstruction | No (lossy, one-way) | Yes (decode tokens → audio) |
| Positional encoding | Learned (ViT), fixed length | RoPE possible, length-flexible |
| Few-shot prompting | Not natural | Concatenate reference + query |
| Pretrained weights | Abundant (ViT → AST → BEATs) | Limited (codec + LM needed) |
| Maturity | Well-studied for audio | Emerging (mostly speech/music) |

### The spectrogram still matters

Even in the discrete token approach, the spectrogram frame size is relevant. Models like BirdNET and Perch are CNN architectures designed around specific spectrogram resolutions:
- BirdNET at 48kHz has more spectral detail per frame than BEATs at 16kHz
- The number of conv layers, pooling operations, and kernel sizes in BirdNET/Perch are calibrated to their mel configuration
- Changing the frontend spectrogram breaks the architecture assumptions

Neural codecs sidestep this by learning the frontend end-to-end, but they introduce their own resolution assumptions (WavTokenizer at 24kHz, 40-75 tokens/sec). The resolution isn't gone -- it's just moved into the codec's learned representation.

### RoPE: needs deeper research

RoPE is well-established for text (LLaMA, Mistral, etc.) and has been applied to audio separation (BS-RoFormer). But several questions remain open for bioacoustic token sequences:

- **1D vs 2D**: If we use waveform-based tokens (codec), 1D RoPE is natural. If we keep spectrogram patches, 2D axial RoPE would encode both time and frequency position.
- **Extrapolation behavior**: NTK-aware scaling, YaRN, and other RoPE extensions exist for text. Do they transfer to audio sequences with very different distributional properties?
- **Interaction with Mamba**: Mamba doesn't use positional embeddings (position is implicit in the recurrence). RoPE would only apply to attention-based models. If we go the Mamba route, positional encoding is a non-issue.

This needs a dedicated research spike before we can make architectural decisions.

### Reconstruction as interpretability

One underappreciated benefit of acoustic tokens: you can **listen to what the model is working with**. If species classification fails, you can decode the tokens back to audio and hear whether the codec lost the discriminative features. With spectrogram patches, you can visualize but not hear. For bioacoustics, where domain experts identify species by ear, this could be a powerful debugging tool.
