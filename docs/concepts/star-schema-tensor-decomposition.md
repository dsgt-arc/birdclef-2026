---
title: "Star Schema as Tensor Decomposition"
status: "exploratory/future-work seed"
date: 2026-04-09
experiment: 106
---

# Star Schema as Tensor Decomposition

## Motivation

Experiment 106's star schema pipeline is sequential: compute means → SVD
whitening → Leiden clustering → prototypes → domain transfer → scoring. Each
step involves hand-designed choices (number of SVD components, Leiden resolution,
cohesion threshold, transfer weight formula). The tensor decomposition framing
unifies these into a single optimization problem where the structure falls out
from the data.

## The 3-way tensor

Focal recordings form a sparse 3-way tensor:

```
X[s, i, p]  =  logit value of window i of species s on Perch dimension p
```

- **s** ∈ {1, ..., 234} — species
- **i** ∈ {1, ..., N_s} — windows (ragged: different N_s per species)
- **p** ∈ {1, ..., 14795} — Perch logit dimensions

Soundscapes form a parallel tensor:

```
Y[l, j, p]  =  logit value of window j at site l on dimension p
```

The star schema compositions are contractions of these tensors through the
shared p dimension. Prototyping is finding a compressed representation along
the window axis.

## CP decomposition: prototypes as factors

CP (CANDECOMP/PARAFAC) decomposes the tensor into a sum of rank-1 components:

```
X[s, i, p] ≈ Σ_r  a_r[s] * b_r[i] * c_r[p]
```

Where:
- **c_r** ∈ ℝ^P — the r-th prototype vector in Perch logit space
- **a_r** ∈ ℝ^S — species loadings: which species "use" prototype r
- **b_r** ∈ ℝ^W — window loadings: how strongly each window activates prototype r

The prototype vectors **c_r** are exactly what Leiden finds — acoustic patterns
in P-space — but now they're learned jointly across all species rather than
independently per species. Species that share call types (e.g., similar alarm
calls across related birds) would share the same component c_r with different
species weights a_r.

### Connection to current pipeline

| Sequential pipeline | Tensor decomposition |
|---------------------|---------------------|
| Species mean μ_s | First component: a_r ≈ 1 for species s, c_r ≈ μ_s |
| Within-species residuals | Higher-order components: deviations from mean |
| SVD whitening | Implicit in the P-space factor structure |
| Leiden clusters → K prototypes | K components per species via sparse a_r |
| Cohesion filtering | Regularization on component magnitude |

## Constrained decomposition

The raw CP decomposition doesn't enforce the structure we want. Key constraints:

### Non-negativity

Species shouldn't have negative loadings on prototypes. A species either uses
a call type or it doesn't.

```
a_r[s] ≥ 0  for all r, s
```

This is Non-negative CP (NTF) — well-studied, interpretable.

### Sparsity

Each species should use only a few prototypes (K=1-8), not all R. L1
regularization on a_r or explicit cardinality constraints:

```
||a_r||_0 ≤ K  per species  (or L1 relaxation)
```

This replaces the MAX_PROTOTYPES cap and MIN_CLUSTER_SIZE from Leiden.

### Domain transfer supervision

The decomposition should find prototypes that predict species presence in
soundscapes, not just reconstruct focal recordings. Add a supervised loss:

```
L = L_recon + λ * L_supervised

L_recon = ||X - Σ_r a_r ⊗ b_r ⊗ c_r||²  (focal reconstruction)

L_supervised = BCE(y_sw, σ(Σ_r a_r[s] * c_r @ w_soundscape))
               for labeled soundscape windows (w, y)
```

The supervised term steers prototypes toward features that transfer. Prototypes
that reconstruct focal data well but don't predict soundscape labels get
penalized. This is domain transfer built into the optimization, not bolted on
afterward.

### Site-state structure

The soundscape tensor Y[l, j, p] gets its own decomposition:

```
Y[l, j, p] ≈ Σ_r  d_r[l] * e_r[j] * f_r[p]
```

Site-state prototypes **f_r** are the analog of species prototypes **c_r**.
The species × site-state affinity is the inner product of the two factor
sets in P-space:

```
affinity[s, l_state] = Σ_r,r'  a_r[s] * d_r'[l_state] * (c_r @ f_r')
```

This is a coupled tensor decomposition — the P-dimension is shared, and the
coupling through P is the star schema composition.

## Parallel spaces as coupled decompositions

The same tensor exists in embedding space:

```
X_E[s, i, e]  =  embedding value of window i of species s on dimension e
```

A coupled decomposition ties the species and window factors across spaces:

```
X_P[s, i, p] ≈ Σ_r  a_r[s] * b_r[i] * c_r^P[p]
X_E[s, i, e] ≈ Σ_r  a_r[s] * b_r[i] * c_r^E[e]
```

Same species loadings a_r, same window loadings b_r, but different prototype
vectors in each space (c_r^P in logit space, c_r^E in embedding space). This
enforces that the "call types" are consistent across representations — a
prototype that represents alarm calls should explain the same windows in both
spaces.

The Bayesian model averaging across spaces becomes implicit: the shared a_r
factors weight each prototype's contribution, and the per-space prototypes
c_r^P and c_r^E contribute complementary information.

## Implementation sketch

```python
import jax
import jax.numpy as jnp
from jax import grad, jit

def loss(params, X_focal, Y_sc_labeled, labels):
    a, b, c = params["species"], params["windows"], params["prototypes"]
    
    # Reconstruction loss (focal)
    X_hat = jnp.einsum("sr,ir,pr->sip", a, b, c)
    L_recon = jnp.mean((X_focal - X_hat) ** 2)
    
    # Supervised loss (soundscape labels)
    # Score each labeled window against all prototypes, weighted by species
    scores = jnp.einsum("sr,pr,wp->sw", a, c, Y_sc_labeled)
    L_super = binary_cross_entropy(labels, jax.nn.sigmoid(scores))
    
    # Sparsity on species loadings
    L_sparse = jnp.mean(jnp.abs(a))
    
    return L_recon + λ_super * L_super + λ_sparse * L_sparse

# Optimize with projected gradient descent (non-negativity)
# or use multiplicative updates (standard for NTF)
```

### Scale considerations

- Focal tensor: 234 × ~1300 avg windows × 14795 — too large to materialize.
  Use stochastic optimization: sample species and windows per batch.
- R (number of components): start with R=50-100, tune via held-out supervised loss.
- GPU useful here: the einsum contractions and gradient computation benefit
  from parallelism. JAX on A100/L40S would make this tractable.

## Advantages over sequential pipeline

1. **End-to-end**: no hand-tuned SVD components, Leiden resolution, cohesion
   thresholds. The decomposition rank R and regularization weights are the only
   hyperparameters.

2. **Shared prototypes across species**: related species that share call types
   get linked through common components. The sequential pipeline clusters each
   species independently — it can't discover shared structure.

3. **Domain transfer is in the loss**: prototypes are optimized to predict
   soundscape labels, not just reconstruct focal recordings. The domain gap
   is addressed during training, not corrected post-hoc.

4. **Parallel spaces are coupled**: logit and embedding spaces share the species
   and window factors. No separate pipelines + Bayesian averaging — the coupling
   happens inside the decomposition.

5. **Differentiable**: can be fine-tuned end-to-end with any downstream loss
   (competition metric, calibration, etc.).

## Risks and limitations

1. **Nonconvex optimization**: CP decomposition is NP-hard in general. Multiple
   local minima, sensitive to initialization. The current Leiden prototypes are
   a natural warm start.

2. **Ragged tensor**: different species have different numbers of windows.
   Standard tensor libraries assume rectangular tensors. Need masking or
   batch-by-species approach.

3. **Scale**: 234 × 266K × 14795 is large. Stochastic methods are necessary
   but add variance. Convergence may be slow.

4. **Diminishing returns**: if the sequential pipeline already captures most
   of the structure, the tensor decomposition may not improve much. The
   supervision signal (739 labeled windows for 234 species) is thin.

5. **Engineering cost**: this is a research project — JAX setup, custom
   optimization loop, GPU allocation. Days of work vs. the sequential pipeline
   which is running now.

## TRMF: temporal structure in the factorization

### Temporal Regularized Matrix Factorization

The soundscape tensor has temporal ordering — windows within a site form a
trajectory. Standard CP treats windows as exchangeable. TRMF adds autoregressive
structure to the temporal factor:

```
Y[l, t, p] ≈ Σ_r  d_r[l] * b_r[t] * f_r[p]

subject to: b_r[t] = Σ_lag A_r,lag * b_r[t-lag] + ε
```

- **f_r** = acoustic state prototypes (what Leiden finds as clusters)
- **b_r[t]** = activation over time (what diel distributions characterize)
- **A_r,lag** = state transition dynamics (what the Koopman operator captures)

The AR dynamics on the temporal factor IS a linear Koopman operator in the
factored low-rank space. The failed DMD analysis from phase 2 was trying to
learn this on raw 14,795-d vectors — TRMF does it in the compact factor space.

### Two temporal scales

- **Within-recording** (focal): short-range AR captures call duration, song
  patterns, silence gaps. A bird mid-song at t is likely singing at t+1.
- **Within-site** (soundscape): long-range AR captures diel cycles, weather
  events, seasonal shifts. Dawn chorus transitions to daytime activity.

### Three equivalent views

TRMF connects to established frameworks through different lenses:

#### 1. Bayesian state-space / HMM equivalence

The temporal factor b_r[t] with AR dynamics is a linear-Gaussian state-space
model. The prototypes f_r are the emission matrix. Discretizing the temporal
factor (hard cluster assignments instead of soft activations) recovers an HMM
where hidden states are acoustic regimes.

```
Leiden clusters + transition matrix = discrete HMM
TRMF temporal factor = continuous relaxation of HMM
Full Bayesian (GP-BTTF) = uncertainty on everything
```

The Bayesian version puts priors on all factors:
- Gaussian process on temporal factor (replaces AR with flexible dynamics)
- Spike-and-slab on species loadings (principled sparsity)
- Inverse-Wishart on noise (heteroscedastic across dimensions)

The posterior gives uncertainty on prototypes and predictions — critical for
knowing which species predictions to trust, especially for rare species with
few focal windows.

#### 2. Graph signal processing equivalence

Three relational graphs exist in the star schema:
- **Site graph**: (L, L) from site similarity in P-space (already computed)
- **Species graph**: (S, S) from co-occurrence in P-space (already computed)
- **Temporal graph**: chain graph (t → t+1) within each site

TRMF with graph regularization:

```
L = L_recon + L_supervised
    + λ_site * Σ_(l,l') w_{ll'} ||d_r[l] - d_r[l']||²    (site smoothness)
    + λ_species * Σ_(s,s') w_{ss'} ||a_r[s] - a_r[s']||²  (species smoothness)
    + λ_temporal * Σ_t ||b_r[t] - Σ_lag A * b_r[t-lag]||²  (temporal AR)
```

Similar sites should have similar prototype activations. Similar species should
have similar loadings. The graph Laplacian penalties enforce this smoothness.

This connects to graph neural networks: replacing the Laplacian smoothness
penalty with graph convolution on the factor matrices gives a GNN that operates
on the star schema graphs. The tensor decomposition is message passing through
P-space across the species, site, and temporal relational structures.

#### 3. Koopman operator equivalence

The AR coefficient matrices A_r,lag define a linear dynamical system in the
factor space. This is the finite-dimensional Koopman approximation — the
factors b_r are Koopman eigenfunctions and the AR dynamics are the Koopman
operator restricted to these observables.

```
DMD on raw vectors:  high-dimensional (14795-d), expensive, noisy
TRMF temporal AR:    low-rank factor space, tractable, regularized
Leiden + transitions: discrete Koopman (Markov chain on cluster labels)
```

All three compute the same object at different resolutions.

### Unified view

```
TRMF
 ├── Bayesian state-space
 │   ├── HMM when discretized (≈ Leiden clusters + transition matrix)
 │   └── GP-BTTF when fully Bayesian (uncertainty on everything)
 │
 ├── Graph signal processing
 │   ├── Graph Laplacian smoothness on species/site factor matrices
 │   └── GNN: message passing through P-space on relational graphs
 │
 └── Koopman operator theory
     ├── AR dynamics on temporal factor = linear Koopman in factor space
     └── DMD on projected vectors ≈ TRMF with identity spatial factors
```

The sequential pipeline we built (Leiden → prototypes → transition matrix →
domain transfer → scoring) is computing fragments of this unified
decomposition. Each piece is a special case or approximation:

| Pipeline step | TRMF equivalent |
|---------------|-----------------|
| Species means | Rank-1 approximation per species |
| Leiden prototypes | Discrete factor modes |
| Site-state clusters | Discretized temporal factor |
| Diel characterization | Empirical AR dynamics |
| Transition matrix | Koopman operator (discrete) |
| Species graph regularization | Graph Laplacian on species factor |
| Site similarity | Graph Laplacian on site factor |
| Domain transfer weights | Supervised loss on soundscape labels |
| Bayesian ensemble across spaces | Coupled decomposition |

## Hopfield layers and attention as prototype retrieval

### The connection

Modern Hopfield networks (Ramsauer et al., 2021) and transformer attention are
the same operation: soft prototype matching. The star schema prototypes are
stored patterns, and scoring a window is a retrieval:

```
Our pipeline (hard):   score(s) = max_k cos(w, prototype_{s,k})
Modern Hopfield (soft): output = softmax(w @ C^T / τ) @ V
Transformer attention:  output = softmax(Q @ K^T / √d) @ V
```

Where Q = window, K = prototype matrix C, V = species loading matrix A.

### Prototypes as key-value memory

The CP factors map directly to attention:

```python
attention = softmax(w @ C^T / τ)     # (R,) — which prototypes match this window?
scores = attention @ A                # (S,) — species predictions from matched prototypes
```

Temperature τ interpolates between hard-max (our current pipeline, τ → 0) and
uniform averaging (τ → ∞). The CP decomposition is the τ → ∞ limit (linear
combination); hard-max prototype matching is the τ → 0 limit.

### Multi-head = multi-space

Each representation space becomes an attention head:

```python
head_P = softmax(w_P @ C_P^T / τ) @ A    # logit-space head
head_E = softmax(w_E @ C_E^T / τ) @ A    # embedding-space head
output = W_out @ concat(head_P, head_E)   # fuse heads
```

The parallel spaces with Bayesian averaging become multi-head attention with
shared species loadings A but different key spaces per head.

### Site context as cross-attention

```python
# Layer 1: identify acoustic state (cross-attention to site prototypes)
site_state = softmax(w @ C_site^T / τ) @ A_site

# Layer 2: species scoring (attention to species prototypes)
species_scores = softmax(w @ C_species^T / τ) @ A_species

# Layer 3: modulate by site context
final = species_scores * site_state
```

### Domain transfer as learned gating

Per-dimension transfer weights become a learned gate:

```python
# Manual: score = (w * transfer_weights) @ prototype^T
# Learned: score = (w * σ(W_gate)) @ prototype^T
```

W_gate is trained end-to-end to down-weight dimensions that don't transfer.

### Full architecture

```
Input: window w (P-space + E-space), site l

Layer 1 — Site state (cross-attention):
  Q = w,  K = V = site prototypes for site l
  → site_context

Layer 2 — Species scoring (multi-head attention):
  Q = w,  K = species prototypes (separate heads for P, E)
  V = species loadings (shared)
  → raw_scores per head

Layer 3 — Fusion:
  Combine heads + modulate by site context
  → final_scores (S,)
```

A 3-layer transformer on the star schema. The "sequence" is the prototype set,
not tokens. Attention patterns are interpretable: which prototypes activated
for each prediction.

### Hopfield vs transformer: trade-offs

**Hopfield layers (energy-based, associative memory)**

Pros:
- Theoretical convergence guarantees — energy decreases monotonically during
  retrieval, converges to a fixed point (stored pattern or metastable state)
- Exponential storage capacity (2^(d/2) patterns in d dimensions) vs. linear
  for classical Hopfield — our ~200-500 prototypes are well within capacity
- Natural for one-shot retrieval: input a noisy/partial pattern, retrieve the
  closest stored pattern. This IS our scoring problem — a noisy soundscape
  window retrieves the closest species prototype
- No positional encoding needed — prototypes are a set, not a sequence
- Energy function provides a principled confidence measure: depth of the energy
  well at the retrieved pattern indicates retrieval certainty

Cons:
- Single-step retrieval may not capture interactions between prototypes (species
  that co-occur should jointly modulate each other's scores)
- Fixed stored patterns — modern Hopfield can learn patterns, but the energy
  landscape is more rigid than attention's learned Q/K/V projections
- Less mature tooling and optimization compared to transformer ecosystems

**Transformer attention (learned projections)**

Pros:
- Q/K/V projections add a learnable transformation before matching — can adapt
  the comparison metric (e.g., learn to ignore shifted dimensions without
  explicit transfer weights)
- Multi-head naturally handles parallel spaces with learned fusion
- Stacking layers allows prototype-prototype interactions: attention between
  species prototypes can capture co-occurrence and competition
- Mature ecosystem: flash attention, quantization, efficient inference
- Layer norm, residual connections, MLPs between layers — standard toolkit for
  regularization and expressivity

Cons:
- No convergence guarantees — output depends on learned parameters, not an
  energy landscape
- Requires positional encoding for temporal structure (TRMF connection is less
  natural than in Hopfield where temporal patterns can be stored directly)
- Heavier parameterization: Q/K/V projection matrices add parameters that need
  training data to fit — risky with only 739 labeled samples
- Attention patterns are "soft" by default — may spread mass across many
  prototypes when sharp retrieval (one prototype matches) is what we want

**For our problem specifically:**

The Hopfield framing is more natural because:
1. We want **retrieval**, not generation — find which stored pattern matches
2. The energy function gives **confidence** — useful for Bayesian averaging
3. We have **few labeled samples** (739) — fewer learnable parameters is better
4. Prototypes are a **set**, not a sequence — no positional encoding needed
5. Sharp retrieval (hard-max) is a reasonable starting point — most windows
   should match one dominant prototype per species

The transformer framing is better if:
1. We need **prototype-prototype interactions** (co-occurring species)
2. We want to **learn the comparison metric** end-to-end
3. We have enough data to train Q/K/V projections without overfitting
4. We want **multi-layer refinement** (iterative scoring)

**Hybrid approach**: use Hopfield for the retrieval step (stored prototypes,
energy-based confidence) but add a learned gating layer (transformer-style)
for domain transfer and multi-space fusion. This keeps the retrieval principled
while allowing the fusion to be learned.

### Multi-label: sigmoid attention, not softmax

Soundscape windows contain 5-20 co-occurring species. Standard softmax attention
forces weights to sum to 1 — activating more prototypes dilutes each one. This
is wrong for multi-label: a window with 10 species should activate 10 prototype
groups at full strength.

**Sigmoid attention** (independently gated prototypes):

```python
gates = σ(w @ C^T / τ)              # (R,) — each prototype independently on/off
scores = gates @ A                    # (S,) — multiple species get high scores
```

Each prototype's activation is independent — sigmoid, not softmax. A dense
soundscape window can activate many prototypes simultaneously without dilution.
This is equivalent to running R independent binary classifiers (is this
prototype present?) and aggregating via the species loading matrix.

**Species interaction via the loading matrix**: the matrix A (R, S) encodes
which prototypes co-activate for which species. If species A and B share
background prototypes but have distinct call prototypes, the loading matrix
captures this. Co-occurring species that are acoustically compatible (different
frequency bands, different call patterns) will activate non-overlapping
prototype subsets — no interference. Confusable species (similar calls) will
compete for the same prototypes — the loading matrix must disambiguate.

**Reconstruction with multi-source windows**: a soundscape window w is a
mixture of multiple species' acoustic patterns plus background:

```
w ≈ Σ_r gates[r] * C[r]     (reconstruction as mixture of active prototypes)
```

The reconstruction loss decomposes the window into its constituent prototypes.
This is a form of **source separation in P-space** — the prototypes are the
basis functions and the gates are the mixing coefficients. The species loading
matrix A then maps from "which acoustic patterns are present" to "which species
are present."

This connects to the BirdCLEF problem structure: the competition uses
**row-wise sigmoid** (independent per-species binary predictions), not
softmax (single-label). The sigmoid attention formulation matches this directly.

### Graphical model: star schema as Bayesian network

The star schema IS the graphical model. Each dimension (S, P, E, L, T, W) is a
random variable, and compositions through P are conditional dependencies:

```
                    Prototypes C
                    (R, P)
                   ╱       ╲
                  ╱         ╲
    species_s ──→ focal_w    soundscape_w ←── site_state
       │         (observed    (observed        (L×K, P)
       │          in focal     in sc domain)      ↑
       │          domain)          ↑           site_l ── time_t
       │              ↑            │
       │         P-dimension    gate_d
       │         (shared)      (domain transfer:
       │                        which P-dims
       │                        are reliable)
       │
    primary_label              ground_truth
    (always known              (known for 739,
     in focal)                  unknown for 127K)
```

Three data regimes with different observed/latent variables:

| Data | Species | Context | Domain | Count |
|------|---------|---------|--------|-------|
| Focal | Observed (primary_label) | — | Focal | 266K |
| Soundscape labeled | Observed (multi-label) | Observed (site, time) | Soundscape | 739 |
| Soundscape unlabeled | **Latent** | Observed (site, time) | Soundscape | 127K |

### Bayesian inference with partial information

Each data source contributes a different likelihood term:

```
P(all data | θ) = P(focal | species, prototypes)           [266K windows]
                × P(labels | sc_windows, prototypes, gate)  [739 windows]
                × P(sc_windows | site_states, prototypes)   [127K windows]
```

Where θ = {prototypes C, loadings A, gate, temperature τ, site_states}.

**Focal likelihood** (266K, species known): contrastive — pull species s's
windows toward its prototypes, push away from others. This is metric learning
with free supervision from track-level annotation.

```
L_focal = -Σ_{w,s} log σ(Σ_r A[r,s] * gate(w @ C[r]))
```

**Soundscape supervised likelihood** (739, multi-label known): the bridge
between domains. These windows tell you what species s looks like in a
soundscape, not just in a focal recording.

```
L_super = BCE(y_multi, σ(gates @ A))
```

**Soundscape unsupervised likelihood** (127K, species unknown): reconstruction
in the target domain. Species labels are marginalized out — the model explains
the window as a mixture of prototypes without knowing which species are present.

```
L_recon = ||w - Σ_r gates[r] * C[r]||²
```

The missing species labels for 127K windows are latent variables. In the
Bayesian framework, you marginalize over them. In practice, this is EM:

- **E-step**: given current prototypes, infer which species are likely present
  in each unlabeled window (soft multi-label predictions)
- **M-step**: given inferred labels, update prototypes and gate to maximize
  likelihood

The star schema structure constrains this inference: species predictions must
be compositions through P-space. The site context provides a prior (species
expected at this site-state). The gate modulates which P-dimensions to trust.
The prototypes are shared across both domains — the focal data locks them to
species identity, the soundscape data adapts them to the target domain.

### What each data source teaches

| Source | Count | Teaches | Mechanism |
|--------|-------|---------|-----------|
| Focal (species known) | 266K | Prototype positions — what each species sounds like in P-space | Contrastive loss |
| Soundscape unlabeled | 127K | Target domain manifold + site-state structure | Reconstruction loss |
| Soundscape labeled | 739 | Domain bridge — how focal prototypes map to soundscape | Multi-label BCE |
| Species graph (S,S) | — | Which species share acoustic structure | Graph regularization on A |
| Site graph (L,L) | — | Which sites have similar ecology | Graph regularization on site factors |

The 266K focal labels are the foundation — they define what each species IS in
P-space. The 127K unlabeled soundscapes define what the target domain looks
like. The 739 labeled soundscapes are the Rosetta Stone that connects the two.
This is fundamentally a domain adaptation problem with rich source labels,
abundant unlabeled target data, and sparse target labels.

### Connection to CP decomposition

```
CP:         score(s,w) = Σ_r a_r[s] * (c_r @ w)              [linear]
Attention:  score(s,w) = Σ_r softmax(c_r @ w / τ)[r] * a_r[s] [softmax-weighted]
Hopfield:   score(s,w) = a_{r*}[s]  where r* = argmin E(w, C)  [energy-minimizing]
```

Three points on the same spectrum: linear combination → soft retrieval →
hard retrieval. The tensor decomposition framework accommodates all three —
the choice of combination function determines the expressivity/data-efficiency
trade-off.

## What NMF captures in each space

An important distinction: NMF on Perch logits does NOT find acoustic call types
(song, alarm, contact). It finds **detector co-activation patterns** — which of
the 14,795 Perch species detectors fire together. A component might be "200 bird
logits + 50 insect logits" — a recording condition pattern, not an acoustic
feature.

| Space | NMF/clustering captures | Interpretation |
|-------|------------------------|----------------|
| P (logits, 14795) | Detector activation patterns | Taxonomic/semantic structure |
| E (embeddings, 1536) | Acoustic similarity clusters | Spectral/temporal grouping |
| Spectrograms (raw) | Spectral bases + temporal activations | Actual source separation |

The two-space approach is complementary:
- **NMF on logits**: which Perch taxa patterns characterize this species
- **K-means on embeddings**: which acoustic conditions the species appears in

Neither gives call-type prototypes directly. For that, you'd need NMF on
spectrograms — which is classic audio source separation (V ≈ W @ H, with
W = spectral bases, H = time activations). Spectrogram NMF allows audio
reconstruction: keep selected components, inverse STFT → separated audio.

This connects to MixIT/source separation work: NMF on soundscape spectrograms
could separate the multi-species mixture into components, then run Perch on
each component independently for cleaner per-source logits/embeddings.

### Implicit covariance handling

NMF on logits implicitly resolves the covariance problem. If 50 cicada logits
always co-activate, NMF puts them in one component. The prototype similarity
for that component is one number, not 50 correlated features. The explicit
14795×14795 precision matrix is replaced by the NMF projection from 14795-d
into K-d prototype similarity space.

Similarly, ridge regression on K prototype features doesn't need a covariance
prior — the redundancy is resolved by the decomposition. This is why the
practical MVP (ridge on prototype features) can skip the heavy covariance
computation entirely.

## When to pursue this

- **After** multi-prototype results are in and we know the sequential pipeline
  works (or doesn't)
- **If** shared prototypes across species seem valuable (e.g., many species
  have similar call types that Leiden finds independently)
- **If** domain transfer is the bottleneck (prototypes are good on focal but
  don't transfer) — the supervised loss directly addresses this
- **If** we have GPU budget for optimization (A100/L40S, ~2-4 hours)

## References

- Kolda & Bader (2009). "Tensor Decompositions and Applications." SIAM Review.
- Kim & Park (2012). "Nonnegative Matrix and Tensor Factorizations." Wiley.
- Acar et al. (2011). "All-at-once optimization for coupled matrix and tensor
  factorizations." MLG Workshop.
- Yu et al. (2016). "Temporal Regularized Matrix Factorization for High-Dimensional
  Time Series Prediction." NeurIPS.
- Zhao et al. (2019). "Bayesian CP Factorization of Incomplete Tensors with
  Automatic Rank Determination." IEEE TPAMI.
- Brunton & Kutz (2019). "Data-Driven Science and Engineering: Machine Learning,
  Dynamical Systems, and Control." Cambridge University Press. (Koopman/DMD)
- Sandryhaila & Moura (2013). "Discrete Signal Processing on Graphs." IEEE TSP.
  (Graph signal processing foundations)
- Ramsauer et al. (2021). "Hopfield Networks is All You Need." ICLR.
  (Modern Hopfield = continuous attention with exponential storage capacity)
- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
  (Transformer attention as soft key-value retrieval)
- Smaragdis & Brown (2003). "Non-negative Matrix Factorization for Polyphonic
  Music Transcription." IEEE WASPAA. (NMF on spectrograms for source separation)
- Févotte et al. (2009). "Nonnegative Matrix Factorization with the
  Itakura-Saito Divergence." Neural Computation. (NMF for audio with
  multiplicative updates and reconstruction)
