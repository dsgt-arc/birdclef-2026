# Perch Logit Star Schema: Dimensional Analysis

## Core idea

The Perch logit space (P = 14,795 dimensions) is a shared semantic axis that
connects all entities in the BirdCLEF problem. Every entity — species, site,
time period, individual recording — can be represented as a vector in P-space.
Any pairwise relationship is computed by composing through this shared dimension.

## Dimensions

| Symbol | Name | Size | Description |
|--------|------|------|-------------|
| **S** | Species | 234 | BirdCLEF 2026 target species |
| **P** | Perch logits | 14,795 | Perch v2 taxonomy (shared dimension) |
| **E** | Embeddings | 1,536 | Perch v2 embedding features (parallel shared dim) |
| **L** | Locations | 21 | Pantanal monitoring sites (S01–S22, some missing) |
| **T** | Time periods | 4 | Diel periods: dawn, day, dusk, night |
| **W** | Windows | 127K+ | Individual 5-second soundscape observations |

## Data layout

Partitioned parquets on scratch (created by `PartitionFeatures` task):

```
~/scratch/birdclef-2026/features/exp106/
├── focal/by_species/{species}.parquet    # 206 files, one per species
├── soundscape_unlabeled/by_site/{site}.parquet  # 21 files, one per site
└── soundscape_labeled.parquet            # 739 labeled windows
```

All downstream analysis loads per-entity parquets directly — no shard scanning.

## Profile matrices (projections into shared space)

Each entity type has a profile matrix — its representation in Perch logit space.
All profiles should be **centered** (global mean subtracted) so that dot products
compute Pearson correlation, not raw cosine.

| Matrix | Shape | Source | Artifact |
|--------|-------|--------|----------|
| Species profiles (single-mean) | (S, P) | Mean logit per species from focal | `species_graph.npz` |
| Species prototypes (multi) | (S×K_s, P) | Leiden cluster centroids from focal | `multi_prototypes.npz` |
| Location profiles (single-mean) | (L, P) | Mean logit per site, James-Stein shrunk | `site_residuals.npz` |
| Location prototypes (multi) | (L×K_l, P) | Leiden cluster centroids from soundscapes | `site_prototypes.npz` |
| Window observations | (W, P) | Raw per-window Perch logit vectors | `by_site/{site}.parquet` |
| Species embeddings | (S, E) | Mean embedding per species from focal | `species_graph.npz` |

## Compositions (products through shared dimension)

Any two profile matrices compose by multiplying through P (or E):

### Species × Location: (S, P) @ (L, P)ᵀ = (S, L)

Which species are expected at which sites? This is a **site-level prior**.
With multi-prototypes: `(S×K_s, P) @ (L×K_l, P)^T → group_max → (S, L)`.

### Species × Window: (S, P) @ (W, P)ᵀ = (S, W)

Per-window species predictions. With multi-prototypes, each window is scored
against all prototypes and the max per species is taken. Piecewise-linear:
the decision boundary is a union of cones in P-space.

### Species × Species: (S, P) @ (S, P)ᵀ = (S, S)

Species co-occurrence / confusion matrix. Done — `species_graph.npz`.
Asymmetric when using window-level scoring.

### Location × Location: (L, P) @ (L, P)ᵀ = (L, L)

Site similarity. Done — `site_residuals.npz`.

## Centering

**Critical**: all compositions must use centered profiles (global mean subtracted).
Without centering, raw cosine is dominated by shared background — all species
pairs show 0.95+ similarity from ubiquitous insect detector activations.

After centering, cosine = Pearson correlation: measures how two entities
**co-deviate** from the average.

- Species profiles centered by global focal mean
- Location profiles centered by global soundscape mean
- Windows centered by the same mean as the entity they're compared to

## Within-entity metric tensors

Raw dot products treat all P dimensions equally. But Perch logits are correlated
— 50 cicada-related logits that fire together shouldn't count as 50 features.

For clustering within entities, we use **domain-specific residual covariance**:

### Within-species residual covariance (for species clustering)

1. Per species s: residuals `r_{s,i} = w_{s,i} - μ_s`
2. Pool residuals across all species → within-species covariance Σ_within
3. SVD → top 200 components + whitening weights

This captures "what varies around a species centroid" — the right metric for
asking "within a species, which windows are similar?" Uses **focal recording**
statistics, not soundscape.

### Within-site residual covariance (for site clustering)

Same approach but with soundscape windows:

1. Per site l: residuals `r_{l,i} = w_{l,i} - μ_l`
2. Pool across all sites → within-site covariance
3. SVD → top 200 components + whitening

Uses **soundscape** statistics — the right domain for clustering soundscape
windows. Pooling assumes the within-site variation structure is shared across
sites (rain sounds similar everywhere, insect choruses have similar structure).

### Whitening = Mahalanobis

Projecting residuals into whitened PCA space and computing cosine similarity
is equivalent to computing the Mahalanobis inner product in the original
P-space. The dense Gram matrix in whitened space IS the Mahalanobis-weighted
similarity graph.

## Multi-prototype species profiles

### Pipeline

For each species (loaded from `focal/by_species/{species}.parquet`):

1. Compute residuals (logits - species mean)
2. Project through within-species whitening transform → (N_s, 200)
3. Build dense Gram matrix: `projected_normed @ projected_normed.T`
4. Run Leiden community detection (variable number of clusters)
5. Filter: reject clusters below MIN_CLUSTER_SIZE or MIN_COHESION
6. Each accepted cluster centroid (in original P-space) = one prototype

### Cohesion filtering

Leiden assigns every node to a community — it doesn't label noise. Noise
filtering happens through:

- **Small communities** (< MIN_CLUSTER_SIZE): rejected
- **Low-cohesion communities**: mean pairwise cosine within the cluster must
  exceed MIN_COHESION (0.3). Rejects incoherent grab-bag clusters that are
  large but not acoustically coherent.
- **Cap** (MAX_PROTOTYPES): prevents fragmentation

### Updated species profile matrix

`(S, P)` → `(S×K, P)` with variable K per species. Scoring:

```python
scores = prototypes @ window.T          # (total_prototypes,)
per_species = scatter_max(scores, K)    # (S,) — max over each species' prototypes
```

## Site-state prototypes

Same pipeline as species, but on soundscape windows per site. Each prototype
is characterized by:

- **Temporal distribution**: fraction of dawn/day/dusk/night windows — reveals
  whether the state is diel-locked or weather/event-driven
- **Top Perch taxa**: which detectors activate most strongly — interpretable
  label (e.g., "night insect cluster" vs "dawn bird chorus")

Replaces fixed diel bins with data-driven acoustic states.

### Species × Site-state affinity

```
(S, P) @ (L×K_l, P)^T = (S, L×K_l)
```

For a test window from site l, identify which site-state prototype it's closest
to, then use species affinities for that state as a context-dependent prior.

## Domain transfer precision

96.8% of Perch logit dimensions shift significantly between focal and soundscape
domains (KS test, p < 0.05). Raw focal prototypes applied to soundscape windows
are degraded by this shift. The domain transfer precision matrix corrects for
this by down-weighting unreliable dimensions.

### Three components

**1. Mahalanobis decorrelation** (within-species Σ⁻¹): correlated detector
banks (50 cicada logits) collapse to one effective dimension. Already applied
via the SVD whitening in the prototype pipeline.

**2. Domain shift reliability per dimension**:

```
reliability_d = σ²_within(d) / (σ²_within(d) + σ²_shift(d))
```

σ²_within is the within-species (or within-site) variance — "real" variation.
σ²_shift is the between-domain variance — focal vs soundscape divergence.
Dimensions where shift dominates get weight → 0. Distribution shift stats
already computed in `distribution_shift.json`.

**3. Species specificity from profile matrix**:

```
specificity_d = 1 / (number of species activating dimension d)
```

A logit dimension that fires for 200/206 species is background — high domain
shift, low discriminative value. A dimension that fires for 3 species is
discriminative and likely stable across domains. Column sparsity of the species
profile matrix `(S, P)` provides this directly.

### Combined transfer weight

```
transfer_weight_d = decorrelation_d × reliability_d × specificity_d
```

Applied as a diagonal weighting when scoring soundscape windows against focal
prototypes:

```
score = (prototype * transfer_weights) @ (window * transfer_weights).T
```

This is the diagonal approximation to the full domain transfer precision matrix.

## Parallel spaces

The star schema is space-agnostic. Any model that produces per-window feature
vectors can serve as the shared mediating dimension:

| Space | Dim | Source | Strengths |
|-------|-----|--------|-----------|
| **P** (logits) | 14,795 | Pretrained Perch v2 | Interpretable, mediating taxa visible |
| **E** (embeddings) | 1,536 | Pretrained Perch v2 | Compact, better conditioned |
| **F_P** (finetuned logits) | N_classes | Finetuned model (e.g., exp 102) | Task-specific discriminative |
| **F_E** (finetuned embeddings) | hidden_dim | Finetuned model | Task-specific acoustic similarity |

Each space produces the full pipeline independently: prototypes, site states,
affinity matrices, domain transfer weights. After composing through the
mediating dimension, all collapse to the same shape: **(S, W)** species scores
per window.

### Bayesian model averaging across spaces

Each space m gives a per-species, per-window score. Combine via species-dependent
mixing weights:

```
P(species s | window w) = Σ_m  π_m(s) * P(s | w, space_m)
```

Mixing weights `π_m(s)` estimated from per-species AUROC on labeled soundscapes:

```python
# Per species, softmax over per-space AUROC
π_P(s) = exp(auroc_P(s) / τ) / Σ_m exp(auroc_m(s) / τ)
```

Temperature τ controls concentration. Species-dependent: unmapped insects get
π_P ≈ 0 (no logit signal), π_E ≈ 1. Well-mapped birds might get π_P = 0.6,
π_E = 0.4.

## Scoring a soundscape window

Full pipeline for scoring a test window w from site l at time t:

### Step 1: Identify acoustic state

```
w_proj = whiten_site(w - μ_l)
k* = argmax_k cos(w_proj, site_prototype_{l,k})
```

Now we know: site l is in state k* (e.g., "dawn chorus", "night insects").

### Step 2: Context-dependent species prior

```
prior(s) = affinity[s, (l, k*)]
```

Which species are expected in this site-state, from the composition
`(S×K_s, P) @ (L×K_l, P)^T`.

### Step 3: Multi-prototype species scoring (per space)

```
w_corrected = (w - global_focal_mean) * transfer_weights
raw_score(s) = max_k cos(w_corrected, prototype_{s,k})
```

Piecewise-linear: union of cones per species in the transfer-corrected space.

### Step 4: Combine across spaces + prior

```
score_m(s) = raw_score_m(s) * prior(s)     # per space
final(s) = Σ_m π_m(s) * score_m(s)         # Bayesian average
```

Perch logits mediate every step: prototypes, site states, affinity, scoring,
transfer weights — all expressed in and composed through the shared dimension.

## Computed artifacts

| Composition | Status | Artifact |
|-------------|--------|----------|
| (S, S) logit | Done | `species_graph.npz: logit_cooc` |
| (S, S) embedding | Done | `species_graph.npz: emb_cooc` |
| (L, L) | Done | `site_residuals.npz` |
| (S×K_s, P) | Running | `multi_prototypes.npz` |
| (L×K_l, P) | Running | `site_prototypes.npz` |
| (S, L×K_l) | Running | `site_prototypes.npz: species_site_affinity` |
| (S, W) | Not yet | Per-window predictions via multi-prototype scoring |

## Next steps

### Phase 2 (in progress)
1. Multi-prototype + site-state prototypes via Leiden (running)
2. Evaluate multi-prototype vs single-mean AUROC on labeled soundscapes

### Phase 3 (domain transfer)
3. Implement domain transfer precision weights (reliability × specificity)
4. Domain-corrected multi-prototype scoring
5. Per-species domain gap analysis (focal prototype vs soundscape labeled windows)

### Phase 4 (parallel spaces + ensemble)
6. Run E-space (embedding) pipeline — same code, swap feature column
7. Per-species Bayesian mixing weights across P and E spaces
8. F-space pipeline if finetuned model available (exp 102 EfficientNet)
9. Calibrate local↔LB mapping from initial submissions
10. Bayesian ensemble with submission-informed weights
