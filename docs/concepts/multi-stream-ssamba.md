# Concept: Multi-Stream SSAMBA for Implicit Source Demixing

**Status**: developing
**Created**: 2026-03-30
**Updated**: 2026-03-30

## Idea

Replace SSAMBA's single patch embedding with K slot embeddings that each apply a learnable
soft frequency gate over the full mel spectrogram, producing K parallel token streams. The
gate is a (K, n_mels) parameter matrix with softmax applied over K — every mel bin's "mass"
sums to 1 across slots, so each bin is routed primarily to one slot but boundaries are learned
and can overlap. The K streams are interleaved into a single sequence and processed by a shared
bidirectional Mamba backbone. A mixture-consistency loss plus optional MixIT and mixup signals
drives slot specialization without any source labels.

The key distinction from pipeline separation (e.g. MixIT → SSAMBA) is that separation and
representation learning are jointly optimized. Mamba's bidirectional context informs the gate:
a slot can claim content at time t based on what it has been tracking across the entire
sequence, not just local frequency content. The gate in turn learns source-specific frequency
routing rather than arbitrary partitioning.

## Motivation

In BirdCLEF soundscapes, every 10ms mel frame is a superposition of multiple species.
Amphibians dominate (76.6% of labeled windows), birds overlay them (28.7%), insects add
high-frequency energy (22.7%), and all three taxa are often present simultaneously. Standard
SSL trains models to predict these mixtures from mixture context — the representations encode
"what acoustic scene is happening here" rather than "what is species X doing." This is a
fundamental bottleneck for:

1. **Prompted SED**: the reference clip conditions Mamba to detect a specific species, but if
   every token mixes multiple species, the state conditioning is noisy from the start
2. **Low-resource species**: representations averaged over co-occurring species dilute the
   signal for rare species that appear in only a few training recordings
3. **Zero-focal species**: the 25 insect sonotypes (`47158son*`) have no focal data — their
   signal must be extracted entirely from mixture soundscapes, which requires source
   disentanglement to be learnable at all

Multi-stream addresses this by making source disentanglement an explicit part of the SSL
objective, with the backbone providing the contextual understanding needed to do it well.

## Background

### The collapse problem with naive K embeddings

The obvious approach — K separate `Linear(n_mels, D)` embeddings all receiving the same full
mel column — has no structural constraint preventing slot collapse. All K slots receive
identical input; the only differentiation comes from random init and output regularization,
which is fragile. The learnable gate is the fix: it structurally restricts what each slot
sees before the linear projection.

**No gate (K identical inputs — collapses):**
```
mel column (224 values) → E_0 → slot 0 ┐
mel column (224 values) → E_1 → slot 1 │  all see the same thing
mel column (224 values) → E_2 → slot 2 │  nothing forces differentiation
mel column (224 values) → E_3 → slot 3 ┘
```

**With learnable gate (different views — stable):**
```
gates = softmax(W, dim=0)    # W: (K, n_mels)
                              # gates[k, j] = slot k's ownership of bin j
                              # Σ_k gates[k, j] = 1  for every bin j

mel column (224) ──×── gates[0] ──→ E_0 ──→ slot 0   (soft low-freq view)
mel column (224) ──×── gates[1] ──→ E_1 ──→ slot 1   (soft mid-freq view)
mel column (224) ──×── gates[2] ──→ E_2 ──→ slot 2   (soft high-freq view)
mel column (224) ──×── gates[3] ──→ E_3 ──→ slot 3   (soft residual view)
```

After training, gates are expected to look like:

```
mel bin:  0────────56────────112────────168────────224
slot 0:   ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   frogs    (low)
slot 1:   ░░░░░░░░░░░░████████████████░░░░░░░░░░░░░░   birds    (mid)
slot 2:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████░░   insects  (high)
slot 3:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████   residual
```

But boundaries are learned and can overlap — a bird harmonic at bin 170 can be co-owned by
slot 1 and slot 2 if the data supports it. Hard frequency splits (K×n_mels/K patches) cannot
do this.

### Full architecture

```
Input: mixture mel (B, n_mels, T)
              │
    gates = softmax(W, dim=0)       W: (K, n_mels), learned
              │
    ┌─────────┼──────────┐
    ↓         ↓          ↓
 g[0]*x    g[1]*x  ...g[K]*x        soft-masked views
    │         │          │
  E_0(·)   E_1(·)  ...E_K(·)        Linear(n_mels, D) per slot
    │         │          │
  s0_t      s1_t    ...sK_t          K slot tokens at time t

Interleave across time:
  [s0_t0, s1_t0, ..., sK_t0,  s0_t1, s1_t1, ..., sK_t1,  ...]
   └─── K adjacent same-time ──┘  └─── next timestep ───┘

Adjacent same-time tokens → cross-slot interaction (sources relating now)
Same-slot across time     → within-slot temporal dynamics (source evolving)

Shared bidirectional Mamba backbone (24 layers)

K reconstruction heads: Linear(D, n_mels) each → K spectrograms (B, K, n_mels, T)
```

### Training losses

Three signals compose additively — each teaches the gate something different:

**1. Mixture consistency (base SSL):**
```
loss_recon = MSE(Σ_k recon_k,  original_mixture)   ← slots must partition the mixture
           + λ * Σ_k L1(recon_k)                   ← sparsity prevents 1/K trivial solution
```
Weakest signal — teaches partitioning but not source identity.

**2. MixIT-style (strongest signal):**
```
Take two soundscapes A and B, create super-mixture M = A + B
Feed M → gated multi-stream → K slot reconstructions

Find optimal partition {S_A, S_B} of K slots minimizing:
  MSE(Σ_{k∈S_A} recon_k,  A)  +  MSE(Σ_{k∈S_B} recon_k,  B)

Gradient flows through the reconstruction heads and the gate.
```
The gate and MixIT are complementary: the gate routes by frequency structure, MixIT
supervises which frequency clusters belong to which source. Together the gate learns
source-specific frequency routing.

We already have pre-separated MixIT 4-channel audio (exp 013, 10,372 soundscapes at
`shared/mixit/pretrained/soundscapes/`). These can serve as the "known clean sources" for
MixIT training: take two MixIT channels, re-mix them, ask the model to reconstruct each.

**3. Mixup with known labels (softest signal):**
```
M = α * focal_A  +  (1-α) * focal_B    ← linear blend, α known
Feed M → gated multi-stream

Supervision: slot activations for species A's frequency band should scale by α,
             slot activations for species B's frequency band should scale by (1-α)
Loss: e.g. KL between gate-weighted activation and α-scaled target distribution
```
Weaker than MixIT (no partition assignment) but free given focal recordings with species
labels. Useful as a curriculum because α is a continuous knob.

### How the three signals compose

| Signal | What it teaches the gate | Supervision required |
|--------|--------------------------|---------------------|
| Mixture consistency + L1 | Partition mixture into non-overlapping parts | None |
| MixIT loss | Which frequency slots belong to which source type | Known mixed pairs |
| Mixup + α | Scale slot activations proportionally to source energy | Species labels + α |

A natural curriculum:
```
Stage 1: pure focal recordings (α=1.0)
         gate learns clean species-specific frequency routing

Stage 2: focal mixup (α ∈ [0.5, 1.0]) + MixIT-channel pairs
         gate learns to proportion activations and handle overlap

Stage 3: raw soundscape mixtures + masked patch SSL
         gate generalizes to natural multi-species overlap
         Mamba learns long-range temporal dynamics on separated streams
```

### Masked multi-stream SSL

Standard masked patch prediction applies naturally across K streams:
```
Mask the same T positions across all K streams simultaneously.
Each slot reconstructs its masked portion.
Loss: MSE on masked positions, with mixture consistency constraint on reconstructions.
```
This combines all objectives: temporal structure learning (masked prediction) + frequency
specialization (gate) + source disentanglement (MixIT/mixup on unmasked positions).

### Sequence length

| K | Resolution | Tokens/60s | Regime (SSAMBA-tiny) |
|---|-----------|-----------|----------------------|
| 1 | 10ms | 6,000 | Linear |
| 2 | 10ms | 12,000 | Linear |
| 4 | 10ms | 24,000 | Linear |
| 4 | 20ms | 12,000 | Linear |
| 8 | 10ms | 48,000 | Superlinear boundary |

K=4 at 10ms (24,000 tokens) is within the confirmed linear regime from exp 005. Needs
verification for small/base models.

## Open Questions

- **Does the softmax gate actually prevent collapse?** The gate gives each slot a different
  view of the input, but if all gates converge to uniform (1/K per bin), slots still collapse.
  Need to verify that mixture consistency + L1 is strong enough to maintain differentiation,
  or add a diversity loss directly on the gate weights (e.g. maximize pairwise distance
  between gate vectors).
- **What is the right value of K?** K=4 mirrors MixIT channels and the rough taxa count
  (birds, frogs, insects, residual). The optimal K may differ — too few and sources remain
  mixed per slot, too many and slots become redundant.
- **Does the interleaved sequence order matter?** Grouping by slot ([s0_t0, s0_t1, ...,
  s0_tT, s1_t0, ...]) makes within-slot temporal modeling easier for Mamba but loses
  cross-slot co-temporal interaction. Interleaved by time is the natural choice but worth
  ablating.
- **Should gate initialization be frequency-band biased?** Random init will symmetry-break
  but slowly. Initializing with a hard frequency partition (low/mid/high/residual) as a
  prior should accelerate specialization and reduce the risk of arbitrary assignments.
- **Is MixIT permutation invariant training stable?** With K=4 slots and 2 partitions, there
  are C(4,2)=6 ways to assign slots to sources. The min-cost assignment is differentiable
  via the Sinkhorn operator or a straight-through estimator, but training can be unstable
  early when all assignments have similar cost.

## Testable Hypotheses

1. **Gate specializes to frequency bands**: after training, `argmax(gates, dim=0)` produces
   a near-monotonic assignment — low bins → slot 0, high bins → slot K-1. Test by plotting
   the learned gate matrix and measuring cross-entropy vs a monotonic reference assignment.

2. **Learnable gate > hard split > no gate**: linear probe cmAP ranks as
   `learnable gate ≥ hard (n_mels/K)×1 patches > naive K identical embeddings`,
   because learned overlap is at least as expressive as hard boundaries, both better than
   no structural differentiation.

3. **Multi-stream > single-stream on low-resource species**: linear probe AP for species with
   <10 training recordings is higher for multi-stream SSAMBA than single-stream SSAMBA on
   identical data.

4. **MixIT loss > mixture consistency**: permutation-invariant training on known source pairs
   produces better downstream representations than mixture consistency alone, because it
   provides explicit source identity supervision vs implicit partitioning pressure.

5. **Curriculum (focal→mixup→soundscape) > direct soundscape training**: staged training
   improves final linear probe cmAP vs training on soundscapes from the start, because the
   curriculum gives the gate a clean initialization before exposing it to complex mixtures.

6. **Zero-focal insect improvement**: the 25 `47158son*` insect sonotypes achieve higher
   soundscape AUROC with multi-stream representations than single-stream, because the
   high-frequency slot provides cleaner signal for species with no focal data.

## Connections

- **Based on**: [SSAMBA architecture](ssamba-audio-mamba.md), MixIT (exp 013 — 4-channel
  separations at `shared/mixit/pretrained/soundscapes/`), slot attention (Locatello et al.
  2020)
- **Related to**: [Prompted SED](prompted-sound-event-detection.md) (cleaner slot-based
  reference conditioning), [CNN-Mamba hybrid](cnn-mamba-hybrid-frontend.md) (frequency-axis
  learned processing), [Temporal resolution](temporal-resolution-sequence-length.md)
- **Could lead to**: Exp 103-C (multi-stream SSL ablation within exp 103)

## Notes

- **Structurally similar to slot attention** but the slots here are token streams processed
  by a shared sequential model (Mamba), not a set-based attention mechanism. The gate plays
  the role of the slot attention routing but is simpler: a single matrix multiply + softmax
  rather than iterative attention.
- **Alternative to interleaving**: a lightweight cross-attention layer between K streams at
  each time step gives explicit cross-slot interaction without relying on the Mamba sequence
  ordering. More expensive but more expressive for source interaction modeling.
- **Gate diversity loss**: if L1 + mixture consistency isn't enough to prevent collapse, add
  `loss_div = -Σ_{i≠j} ||gates[i] - gates[j]||_2` to explicitly push gate vectors apart.
- **Compute**: K=4 at 10ms is 24,000 tokens — 4× single-stream. Same as extending
  single-stream context from 60s to 240s. Feasible for small, tight for base.
- Gate the multi-stream concept behind 103-A and 103-B: if single-stream SSL on MixIT
  channels (103-B) doesn't beat raw-mixture SSL (103-A), the demixing hypothesis is weak
  and the gate architecture adds complexity without payoff.
