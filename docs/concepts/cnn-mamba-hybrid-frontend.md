# Concept: Learned Frequency Tokenizer for SSAMBA

**Status**: nascent
**Created**: 2026-02-08
**Updated**: 2026-02-08

## Idea

Replace SSAMBA's single-layer linear patch embedding (`Linear(128, 192)`) with a learned frequency encoder that produces richer per-frame tokens. Mamba still processes 3000 tokens at 10ms resolution — the only change is that each token better captures spectral content. The frequency encoder can be pre-trained with SSL independently of Mamba, using a frequency band masking objective.

## Motivation

SSAMBA from scratch achieves 0.08 cmAP on BirdSet HSN. ConvNeXT pre-trained achieves 0.48. ViT from scratch achieves 0.164. SSAMBA is worse than ViT from scratch, suggesting the architecture struggles with spectrogram feature extraction.

The current tokenizer is `Linear(128, 192)` — a single affine transform per time frame. This maps 128 mel bins to 192 dimensions with no nonlinearity, no hierarchy, and no locality bias. It can only learn linear combinations of frequency bins. Nonlinear spectral features — harmonic series detection, spectral envelope shape, bandpass energy — require at minimum a nonlinear function, which a single linear layer cannot represent.

The question is: **what is the right learned frequency encoder, and what inductive biases should it have?**

## Architecture

### Recommended: LEAF + frequency attention + Mamba

Three stages, each with a clear role and appropriate inductive bias:

```
Raw audio (B, 960000)
         ↓
┌────────────────────────────────────────────┐
│ LEAF (learnable Gabor filterbank + PCEN)    │
│ "What frequency bands have energy?"         │
│ Inductive bias: bandpass filter structure    │
│ ~30K params                                 │
└────────────────────────────────────────────┘
         ↓
(B, 128, 3000)  — 128 learned filter energies per 10ms frame
         ↓
┌────────────────────────────────────────────┐
│ Frequency attention (per time frame)        │
│ "Which frequency bands relate to each       │
│  other?" (harmonics, spectral shape)        │
│ Inductive bias: pairwise interaction,       │
│  position-aware                             │
│ 1-2 transformer layers over 128 freq tokens │
│ ~200K params                                │
└────────────────────────────────────────────┘
         ↓
(B, 3000, D)  — same token interface as current patch embedding
         ↓
┌────────────────────────────────────────────┐
│ Bidirectional Mamba (24 layers)             │
│ "What temporal patterns do these features   │
│  form?" (call repetition, context)          │
│ Inductive bias: sequential recurrence       │
│ ~7M params                                  │
└────────────────────────────────────────────┘
         ↓
(B, 3000, D) → task head
```

#### LEAF frontend

LEAF replaces the fixed mel spectrogram with learnable components:

```
Raw waveform
    ↓
GaborConv1d(n_filters=128)    learnable bandpass filters
    ↓                          each filter = (center_freq, bandwidth)
Squared modulus                energy in each band
    ↓
GaussianLowpass(stride=320)   smooth + downsample to 10ms
    ↓
PCEN                           adaptive per-channel normalization
    ↓
(B, 128, 3000)                same shape as mel spectrogram
```

Why LEAF over mel:
- **PCEN** handles dynamic range better than log compression for field recordings with variable SNR
- **Learnable center frequencies** can specialize to bird vocalization range (1-8 kHz)
- **Learnable bandwidths** can be narrow for tonal calls, wide for noisy calls
- After training, inspect learned filters to verify they're sensible

#### Frequency attention

Treat the 128 LEAF filter outputs at each time frame as frequency tokens. Run a small transformer over the frequency axis:

```python
class FrequencyAttention(nn.Module):
    def __init__(self, n_filters=128, d_freq=32, d_model=192, n_layers=2):
        self.value_proj = nn.Linear(1, d_freq)        # embed scalar → vector
        self.freq_pos = nn.Embedding(n_filters, d_freq)  # learned frequency position
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_freq, nhead=4,
                dim_feedforward=d_freq * 4,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.pool = nn.Sequential(
            nn.Flatten(),                            # (128 * d_freq,)
            nn.Linear(n_filters * d_freq, d_model),  # → D
        )

    def forward(self, leaf_out):
        # leaf_out: (B, 128, T) → (B, T, 128)
        B, F, T = leaf_out.shape
        x = leaf_out.permute(0, 2, 1).reshape(B * T, F, 1)

        # Embed scalars + add frequency position
        x = self.value_proj(x) + self.freq_pos.weight.unsqueeze(0)
        x = self.attn(x)               # (B*T, 128, d_freq)
        x = x.reshape(B * T, -1)       # (B*T, 128*d_freq)
        x = self.pool(x)               # (B*T, D)
        return x.reshape(B, T, -1)     # (B, T, D)
```

What frequency attention learns:
- "Filter 50 (2kHz) and filter 70 (4kHz) both active → harmonic pair" — explicit pairwise interaction
- "Energy concentrated in filters 40-60, nothing above → specific spectral shape" — cross-frequency patterns
- Attention weights are interpretable: which frequency bands interact for which sounds

Cost: 128 tokens × d_freq=32 × 2 layers ≈ 6 GFLOPs for 3000 frames. All parallel (per-frame independent), no sequential overhead.

#### Frequency position encoding from LEAF

The freq attention's positional encoding can be initialized from LEAF's actual learned center frequencies, giving the model explicit knowledge of which physical frequency each filter represents:

```python
# After LEAF is trained, extract center frequencies
center_freqs = leaf.gabor_conv.center_freqs.detach()  # (128,) in Hz
# Initialize positional encoding with sinusoidal encoding of actual frequencies
freq_pos.weight.data = sinusoidal_encoding(center_freqs / nyquist)
```

### Alternative frequency encoders

#### MLP (simplest, no inductive bias beyond nonlinearity)

```python
freq_encoder = nn.Sequential(
    nn.Linear(128, 256), nn.GELU(),
    nn.Linear(256, 256), nn.GELU(),
    nn.Linear(256, D),
)
# Works on mel spectrogram input — no LEAF needed
```

- 130K params, ~780M FLOPs (negligible)
- No locality assumption — sees all 128 bins at once
- **Best for quick diagnostic**: does tokenizer depth matter at all?

#### Conv1d over frequency axis

```python
freq_encoder = nn.Sequential(
    nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),   # 128 → 64
    nn.GELU(),
    nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # 64 → 32
    nn.GELU(),
    nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # 32 → 16
    nn.GELU(),
    nn.Conv1d(128, D, kernel_size=16),                      # 16 → 1
)
# Works on mel spectrogram input
```

- ~50K params, ~7 GFLOPs
- **Concern**: locality in mel-bin space may be the wrong prior — harmonics of a 2kHz call sit at mel bins 50, 70, 82, 90 (gaps of 20, 12, 8 bins). Conv1d(kernel=5) can't see these in early layers. The mel filterbank already applies local smoothing, making a second local filterbank redundant.

### Comparison

| Approach | Prior | Params | Added FLOPs | Cross-freq? | SSL-able? |
|---|---|---|---|---|---|
| Linear(128, D) | None | 24K | baseline | Linear only | No benefit |
| MLP (3-layer) | Nonlinearity | 130K | 0.78G (2.5%) | Yes (implicit) | Marginal |
| Conv1d stack | Freq locality | 50K | 7G (23%) | Only via depth | Freq masking |
| **LEAF + freq attn** | **Bandpass + pairwise** | **230K** | **6.5G (21%)** | **Yes (explicit)** | **Yes (natural)** |

## Modular SSL pre-training

The key insight: the frequency encoder (LEAF + freq attention) and Mamba can be pre-trained **separately**, each with an objective that matches its role. This avoids the SSL gaming problem where an encoder learns to produce temporally smooth features rather than spectrally rich ones.

### Stage 1: Pre-train frequency encoder (frequency band masking)

```
One time frame: LEAF gives 128 filter energies
         ↓
Mask 40% of filters (set to 0)
         ↓
Freq attention sees only unmasked filters
         ↓
Predict energy at masked filter positions
         ↓
Loss: MSE on masked positions
```

This is a masked autoencoder over the **frequency axis**. It teaches:
- LEAF: which bandpass filters best capture the structure of bird audio
- Freq attention: cross-frequency relationships (harmonics, spectral shape, energy distribution)

**Data abundance**: each 30s recording at 10ms = 3,000 independent frames. With 10K recordings = **30 million training frames** for a ~230K parameter model. Converges in hours on a single GPU, feasible even on CPU.

**No temporal objective**: the encoder learns pure spectral structure. It can't game a temporal prediction target because there is none.

**Verification after Stage 1**:
- Inspect LEAF's learned center frequencies — are they concentrated in 1-8 kHz (bird range)?
- Visualize freq attention weights — do harmonically related filters attend to each other?
- Measure reconstruction quality on held-out frames

### Stage 2: Pre-train Mamba on frozen encoder tokens (temporal masking)

```
Audio → frozen freq encoder → frame tokens (3000, D)
                                      ↓
                              mask 40% of frames
                                      ↓
                  Bidirectional Mamba (sees unmasked only)
                                      ↓
                  predict encoder features at masked positions
                                      ↓
                  Loss: MSE(predicted, actual) at masked frames
```

Mamba receives high-quality tokens from day one (the freq encoder is already pre-trained). It learns temporal dynamics: call patterns, repetition structure, silence-vocalization transitions.

**Data**: same audio, but now as 30s sequences. Thousands of sequences, 1-2 days on 1 GPU.

### Stage 3: Fine-tune end-to-end

Unfreeze everything, add task head (classification or SED), fine-tune on labeled data (BirdSet HSN).

### Why separate is better than joint

| | Joint SSL (encoder + Mamba) | Separate SSL (this approach) |
|---|---|---|
| **Encoder objective** | Contaminated by temporal prediction — incentivized to produce smooth features | Pure frequency reconstruction — incentivized to capture spectral structure |
| **Gaming risk** | High — encoder minimizes Mamba's prediction difficulty | None — objectives are independent |
| **Iteration speed** | Slow — retrain everything | Fast — iterate on encoder alone (hours) |
| **Verification** | Hard to disentangle encoder vs Mamba quality | Each component verified independently |
| **Data efficiency** | Sequences (thousands) | Stage 1: frames (millions), Stage 2: sequences |

### Full pipeline summary

```
Stage 1: Pre-train LEAF + freq attention               (hours, 1 GPU)
  ├── Objective: frequency band masking (spectral SSL)
  ├── Data: 30M+ individual frames from all audio
  ├── Model: ~230K params
  └── Result: frozen frequency encoder

Stage 2: Pre-train Mamba on frozen encoder tokens       (1-2 days, 1 GPU)
  ├── Objective: masked frame prediction (temporal SSL)
  ├── Data: 30s sequences from all audio
  ├── Model: ~7M params (Mamba backbone)
  └── Result: pre-trained temporal model

Stage 3: Fine-tune end-to-end                           (~12 hours, 1 GPU)
  ├── Objective: classification / SED
  ├── Data: labeled BirdSet HSN
  ├── Model: unfreeze all, add task head
  └── Result: final model for evaluation
```

## Why didn't SSAMBA go this route?

### The ViT paradigm

The lineage ViT → AST → BEATs/EAT → Vision Mamba → SSAMBA inherits ViT's assumption: **a linear patch embedding is sufficient, and the sequence model handles everything else.** This works for transformers because:

1. **Global attention recovers spatial structure.** Every patch attends to every other — the O(n²) attention map implicitly learns 2D relationships. A frequency patch at 2kHz directly attends to its harmonic at 4kHz.

2. **Enough data compensates for a weak tokenizer.** AudioSet has 2M clips. At that scale, even a linear projection converges to useful features.

3. **Paper incentives favor simplicity.** A paper proposing a new sequence model wants to isolate its contribution, not conflate it with a custom tokenizer.

### Why Mamba can't get away with it

Transformers have O(n²) global interaction; Mamba has O(n) local recurrence. A pattern at token 50 must propagate through the hidden state to influence token 100. The state compresses information — it's a bottleneck for spatial relationships that attention handles directly.

With massive pre-training, Mamba learns state dynamics that propagate frequency information well enough. Without pre-training, on small data, the single-layer tokenizer + Mamba combination fails because:
- The tokenizer produces weak features (linear-only frequency representation)
- Mamba can't compensate because it lacks the global interaction of attention

A deeper tokenizer with frequency attention gives Mamba richer input without requiring it to learn frequency features through its recurrent dynamics.

## Open Questions

- **Is the tokenizer actually the bottleneck?** Quick diagnostic: swap Linear for MLP, same data, compare cmAP. If MLP doesn't help, the problem is elsewhere.
- **LEAF implementation**: use existing `leaf-pytorch` package or reimplement (~100 lines)? Reimplementing avoids a dependency.
- **Freq attention scale**: d_freq=32 with 2 layers, or d_freq=64 with 1 layer? Trade-off between attention capacity and parameter count.
- **Pooling strategy**: flatten + linear, or learned [CLS]-style aggregation token, or mean pooling over frequency tokens?
- **Does SSAMBA with pre-trained weights (16×16 patches) beat this approach?** If yes, the problem is data scale, not architecture.
- **Stage 1 masking ratio**: 40%? Higher masking (60%) forces more cross-frequency reasoning but may be too hard early in training.

## Testable Hypotheses

1. **MLP tokenizer > linear tokenizer (from scratch)**: Quick diagnostic. If MLP(128→256→256→D) doesn't improve over Linear(128, D), the tokenizer isn't the bottleneck.

2. **LEAF + freq attn + Mamba > mel + Linear + Mamba**: The full stack outperforms the current architecture on BirdSet HSN, validating the modular design.

3. **Separate SSL > no SSL**: Pre-training each component independently improves downstream cmAP by ≥50% relative, confirming that the modular SSL pipeline works.

4. **Separate SSL > joint SSL**: Separately pre-trained components outperform jointly pre-trained ones, confirming that the gaming concern is real and separation helps.

5. **Mamba adds temporal value**: LEAF + freq attn + Mamba outperforms LEAF + freq attn + mean pooling on 30s windows, demonstrating that Mamba captures useful temporal structure.

6. **LEAF specializes to bird frequencies**: After Stage 1 SSL on bird audio, LEAF's learned center frequencies are concentrated in 1-8 kHz, confirming domain adaptation.

## Connections

- **Based on**: [SSAMBA architecture](ssamba-audio-mamba.md), exp 006 results (0.08 cmAP from scratch)
- **Related to**: [Prompted SED](prompted-sound-event-detection.md) (this stack as backbone for prompted detection); multi-signal SED was a placeholder experiment direction, not a scaffolded experiment.
- **Could lead to**: Experiment with the full LEAF + freq attn + Mamba pipeline; or quick MLP diagnostic first

## Notes

- **Quick diagnostic first**: before building the full LEAF + freq attn stack, swap `Linear(128, 192)` for `MLP(128, 256, 256, 192)` and retrain. One line change, same data, high signal. If it doesn't help, the tokenizer isn't the problem.
- The frequency encoder is applied per-frame independently — it adds zero sequential computation. All additional cost is parallel. Mamba's sequential processing (the CPU bottleneck) is unchanged.
- This concept is compatible with all other proposed improvements (SED heads, prompted detection, weak supervision). It's a drop-in replacement for the patch embedding layer.
- The LEAF + freq attention module is independently useful — it could be paired with any temporal model (transformer, LSTM, temporal convolution), not just Mamba. If Mamba proves to not add value over simpler temporal models, the frequency encoder still works.
- For the BirdCLEF competition, the practical priority order is: (1) MLP diagnostic, (2) LEAF + freq attn with SSL, (3) temporal SSL for Mamba, (4) fine-tune on task, (5) then pursue SED/prompted concepts on the pre-trained backbone.
