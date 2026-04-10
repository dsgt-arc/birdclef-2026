# Star Schema: Dimensional Analysis and Bayesian Interpretation

## The problem

Given a 5-second soundscape window from site l at time t, predict which of S=234
species are present. Available data:

| Source | Count | Domain | Labels |
|--------|-------|--------|--------|
| Focal recordings | 265K windows | Clean, single-species | Species known |
| Unlabeled soundscapes | 127K windows | Target (Pantanal) | None |
| Labeled soundscapes | 739 windows | Target (Pantanal) | Species known |

All windows are encoded by Perch v2 into embeddings (ℝ^E, E=1536) and logits
(ℝ^P, P=14795). The challenge: focal and soundscape domains are severely shifted
(96.8% of logit dimensions shift significantly via KS test).

## Entities and dimensions

| Symbol | Name | Size | Description |
|--------|------|------|-------------|
| S | Species | 234 | BirdCLEF targets |
| L | Sites | 21 | Pantanal monitoring locations |
| K_s | Species prototypes | 4 | Acoustic modes per species |
| K_l | Site prototypes | 6 | Acoustic states per site |
| D | Shared space | 256 | Learned projection (exp 107) |
| P | Perch logits | 14,795 | Pretrained space (exp 106) |
| E | Perch embeddings | 1,536 | Pretrained space |
| W | Windows | 127K+ | Individual observations |

## The star schema structure

Every entity is represented as a set of vectors in a shared D-dimensional space.
Relationships between entities are dot products through D.

### Entity representations

**Species memory**: M_s ∈ ℝ^{S × K_s × D}

Each species has K_s=4 prototype vectors in the shared space. These represent
distinct acoustic modes — e.g., dawn song, alarm call, flight call, juvenile.
The multi-prototype structure gives a piecewise-linear decision boundary:
the acceptance region for species s is the union of K_s cones in D-space.

**Site memory**: M_l ∈ ℝ^{L × K_l × D}

Each site has K_l=6 prototype vectors capturing acoustic states — e.g., dawn
chorus, midday quiet, nocturnal insect, rain event. These are data-driven
(discovered by NMF/k-means in exp 106, learned end-to-end in exp 107).

**Window query**: q ∈ ℝ^D

A test window projected into the shared space: q = proj(emb_w) where
proj: ℝ^E → ℝ^D is a learned linear projection.

### Compositions

All compositions are bilinear forms through the shared D dimension:

**Species scoring** (what it sounds like):

    score_species(w, s) = max_{k=1..K_s} σ(β_s · q_w · M_s[s, k])

Dimensions: (D,) · (D,) → scalar, over K_s prototypes → scalar per species.
Full matrix: (W, D) @ (S·K_s, D)^T → (W, S·K_s) → reshape + max → (W, S).

The σ (sigmoid) gives multi-label probabilities. The max over K selects the
best-matching prototype — the mode of species s most similar to window w.

**Site scoring** (where it is):

    score_site(w, l) = max_{k=1..K_l} σ(β_l · q_w · M_l[l, k])

Dimensions: same structure, (D,) · (D,) → scalar.

This scores how well the window matches its site's acoustic state prototypes.
High score = the window is typical for this site. Low score = unusual
(rare event, equipment anomaly, etc.).

**Species × Site affinity** (what's expected where):

    affinity(s, l) = max_{k_s, k_l} M_s[s, k_s] · M_l[l, k_l]

Dimensions: (D,) · (D,) → scalar, over K_s × K_l pairs → scalar.
Full matrix: (S·K_s, D) @ (L·K_l, D)^T → (S·K_s, L·K_l) → reshape + max → (S, L).

This is the star schema's core composition: species and sites interact through
the shared D-space. No cross-domain features needed — both representations
are in the same learned space.

**Composed score** (posterior):

    score(w, s, l) = score_species(w, s) × score_site(w, l)

    or with gating:

    score(w, s, l) = g · score_species + (1-g) · score_species × score_site

The gate g ∈ [0,1] (learned) controls how much site context influences
the prediction. If sites aren't informative, g → 1 and the model ignores them.

### Dimensional summary

| Operation | Input shapes | Shared dim | Output shape |
|-----------|-------------|------------|--------------|
| Species scoring | (W, D) × (S, K_s, D) | D | (W, S) |
| Site scoring | (W, D) × (L, K_l, D) | D | (W,) |
| Species×Site affinity | (S, K_s, D) × (L, K_l, D) | D | (S, L) |
| Composed score | (W, S) × (W,) | — | (W, S) |
| Projection | (W, E) | — | (W, D) |

Total learnable parameters:

| Component | Shape | Parameters |
|-----------|-------|-----------|
| Projection | (E, D) = (1536, 256) | 393K |
| Species memory | (S, K_s, D) = (234, 4, 256) | 240K |
| Site memory | (L, K_l, D) = (21, 6, 256) | 32K |
| Temperatures | β_s, β_l | 2 |
| Gate | (D, 1) = (256, 1) | 257 |
| **Total** | | **~665K** |

## Bayesian interpretation

The star schema implements approximate Bayesian inference over a latent
variable model.

### Generative model

For window w at site l, time t:

    1. Site l is in acoustic state k_l ~ Cat(π_l)         prior over site states
    2. Species s is present with prob P(s | l, k_l)        species-site prior
    3. If present, s vocalizes in mode k_s ~ Cat(π_s)      species mode prior
    4. Perch encodes the audio: x_w ~ P(x | s, k_s)        likelihood

### What each component computes

| Component | Bayesian role | Distribution |
|-----------|--------------|-------------|
| q_w · M_s[s,k] | Log-likelihood log P(x_w \| s, k_s) | Gaussian with shared Σ |
| max_k(...) | ≈ log max_k P(x_w \| s, k_s) | Hard EM on mixture |
| σ(β · score) | Calibrated P(s present \| x_w) | Sigmoid link |
| affinity(s,l) | Log-prior log P(s \| l) | Composition through D |
| score × site | ∝ P(s \| x_w, l) | Posterior ∝ likelihood × prior |

### The cosine-Gaussian connection

If per-species-mode representations are approximately Gaussian in D-space:

    P(x | s, k) ∝ exp(-½ (x - μ_{s,k})^T Σ^{-1} (x - μ_{s,k}))

With shared covariance Σ and after whitening (Σ → I):

    log P(x | s, k) ∝ -½ ||x - μ_{s,k}||²
                     = x · μ_{s,k} - ½||x||² - ½||μ_{s,k}||²

After L2 normalization (||x|| = ||μ|| = 1), this is proportional to cosine
similarity. So cosine matching = Gaussian log-likelihood with shared covariance,
evaluated on the unit sphere.

The sigmoid σ(β · cos) then maps this to a probability. The temperature β
controls the sharpness — high β means the model is confident, low β means
uncertain. Learning β from data is equivalent to estimating the precision
(inverse variance) of the Gaussian.

### Why max over K, not sum

The full Bayesian marginal is:

    P(x | s) = Σ_k π_k P(x | s, k)

We approximate with:

    P(x | s) ≈ max_k P(x | s, k)

This is a hard EM approximation. It's exact when one mode dominates (high β),
which is typical for bird vocalizations (a window contains a dawn song OR an
alarm call, rarely both from the same species). The max is also more robust
to "dead" prototypes (modes with no matching windows) than sum, which would
dilute the signal.

### The semi-supervised regularizer

The 127K unlabeled windows provide two signals:

**Entropy minimization**: P(s | x_w) should be confident (low entropy) even
on unlabeled windows. This encourages the model to use the full capacity of
its prototypes — each window should clearly match or not match each species.

    L_ent = -Σ_w Σ_s [p log p + (1-p) log (1-p)]     where p = score(w, s)

**Prototype coherence**: Each species prototype should be close to the centroid
of the windows it claims (high sigmoid score). This is the M-step of EM, made
differentiable:

    L_coh = Σ_s Σ_k || M_s[s,k] - E_{w: σ(q_w · M_s[s,k]) > τ}[q_w] ||²

These regularizers use unlabeled data to shape the shared space without
requiring labels. The labeled 739 windows provide the supervised BCE signal
that anchors species identity.

## Why exp 106 failed and exp 107 should work

### Exp 106: fixed space, broken compositions

Exp 106 used Perch's pretrained logit space (P=14795) as the shared dimension D.
The assumption: Perch logits are compositional — each species' contribution to
the logit vector is independent and additive.

This assumption fails because:

1. **Perch is a neural network.** Its representations are entangled — a species'
   logit activation depends on what else is in the audio (other species, noise,
   recording conditions). The logit vector is not a superposition of per-species
   signals.

2. **Domain shift corrupts the shared space.** 96.8% of logit dimensions shift
   between focal and soundscape. The species prototypes (from focal) and the
   site prototypes (from soundscape) live in different manifolds of P-space.
   Their dot product is dominated by the domain gap, not the species-site
   relationship.

3. **Domain shift and site structure are entangled.** The dimensions that shift
   most (cicada and frog logits) are the same dimensions that discriminate
   between sites. Transfer weights that suppress shift also suppress site signal.
   You can't fix one without breaking the other in a fixed space.

Results: Species cosine similarity works (0.816 AUROC on focal species) because
it only requires within-domain scoring (soundscape window × focal prototype).
But site affinity has zero effect because it requires cross-domain composition
(focal prototype × soundscape site prototype through shifted P-space).

### Exp 107: learned space, valid compositions

Exp 107 learns the shared space D (256-d) via a projection from Perch embeddings.
The projection is trained to make compositions valid:

1. **The space is task-specific.** The projection learns which directions in
   Perch embedding space are compositional for BirdCLEF species × Pantanal sites.
   Entangled directions are disentangled by the linear projection.

2. **Everything is in one domain.** Both species and site memories are updated
   by backprop on soundscape-domain data (739 labeled + 127K unlabeled). There
   is no cross-domain composition — the shared space is defined by how entities
   relate in the target domain.

3. **The gate handles uncertainty.** If site information doesn't help (some species
   are ubiquitous), the learned gate g → 1 and the model ignores the site bank.
   No need for transfer weights — the model learns what to ignore.

4. **Semi-supervision exploits unlabeled data.** The 127K unlabeled soundscapes
   shape the shared space via entropy minimization and prototype coherence.
   This is the untapped resource that exp 106's unsupervised compositions
   couldn't leverage.

### The key shift

Exp 106 asks: "Does Perch's pretrained space support compositional inference?"
Answer: no, the domain shift breaks it.

Exp 107 asks: "Can we learn a small space where compositional inference works?"
The star schema structure (factored species × site) remains the same. Only the
instantiation changes — from a pretrained fixed space to a learned task-specific
space.

## Relationship to existing approaches

### Murilo's ProtoSSM (0.918 LB)

ProtoSSM uses BiSSM + cross-attention + gated fusion on Perch features. This is
architecturally similar to what exp 107 proposes — temporal modeling (BiSSM) +
cross-attention (Hopfield retrieval) + gated fusion (species × context). The
difference: ProtoSSM is a larger model trained end-to-end on more data.
Exp 107 is a minimal version testing whether the factored structure alone
(without temporal modeling) captures meaningful signal.

### Recommender systems

The star schema is a factored recommendation model. Species are "items," sites
are "users," and the score is a bilinear form through a shared embedding space.
The semi-supervised regularizer corresponds to implicit feedback (unlabeled
windows = "impressions" without "clicks"). Matrix factorization, neural
collaborative filtering, and two-tower retrieval models all share this structure.

### Modern Hopfield networks (Ramsauer et al. 2020)

The species and site memory banks with sigmoid attention are exactly the modern
Hopfield framework. The energy function:

    E(q, M) = -log Σ_k exp(β · q · M[k])     (log-sum-exp)

has a unique global minimum (for β < critical temperature) and the dynamics
(iterating the softmax/sigmoid attention) provably converge. In our case,
we don't iterate at inference time (one-shot attention is sufficient), but
the energy provides a principled confidence measure: low energy = the query
clearly matches a stored pattern; high energy = ambiguous.
