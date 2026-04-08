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
| **L** | Locations | 23 | Pantanal monitoring sites (S01–S23) |
| **T** | Time periods | 4 | Diel periods: dawn, day, dusk, night |
| **W** | Windows | 127K+ | Individual 5-second soundscape observations |

## Profile matrices (projections into shared space)

Each entity type has a profile matrix — its representation in Perch logit space.
All profiles should be **centered** (global mean subtracted) so that dot products
compute Pearson correlation, not raw cosine. Centering removes the shared
background signal that otherwise dominates.

| Matrix | Shape | Source | Artifact |
|--------|-------|--------|----------|
| Species profiles | (S, P) | Mean logit activation per species from focal recordings | `species_graph.npz: logit_means_normed` |
| Location profiles | (L, P) | Mean logit activation per site from unlabeled soundscapes | `site_residuals.npz: shrunk_residuals` |
| Location-time profiles | (L×T, P) | Per-site per-diel-period signatures | `soundscape_diel_signatures.parquet` |
| Window observations | (W, P) | Raw per-window Perch logit vectors | `soundscape_unlabeled/shard-*.parquet` |
| Species embeddings | (S, E) | Mean embedding per species from focal | `species_graph.npz: emb_means_normed` |
| Location embeddings | (L, E) | Mean embedding per site | (not yet computed) |

## Compositions (products through shared dimension)

Any two profile matrices compose by multiplying through P (or E):

### Species × Location: (S, P) @ (L, P)ᵀ = (S, L)

Which species are expected at which sites? High entry (s, l) means species s's
focal signature aligns with location l's acoustic profile. This is a **site-level
prior** — before seeing any window, you know which species are likely where.

### Species × Location-Time: (S, P) @ (L×T, P)ᵀ = (S, L×T)

Species-site affinity resolved by time of day. Amphibians should show high
affinity for nighttime site profiles, birds for dawn. This gives a
**spatiotemporal prior** for each species.

### Species × Window: (S, P) @ (W, P)ᵀ = (S, W)

Per-window species predictions. Each window is scored against every species'
focal signature. This is an interpretable alternative to the ridge model — the
prediction is a direct dot product in Perch species space. Can serve as a
complement or sanity check against the learned ridge weights.

### Species × Species: (S, P) @ (S, P)ᵀ = (S, S)

Species co-occurrence / confusion matrix. Already computed in the species graph.
Asymmetric when using window-level scoring. Mediating Perch taxa are readable
from the element-wise product of the two species' P-vectors.

### Location × Location: (L, P) @ (L, P)ᵀ = (L, L)

Site similarity. Already computed in site residuals (cosine on mean profiles).
After centering + James-Stein shrinkage, this reflects genuine ecological
differences between sites, not just shared Pantanal background.

### Species × Species (embedding): (S, E) @ (S, E)ᵀ = (S, S)

Acoustic similarity in embedding space. Less interpretable than logit-space
co-occurrence but captures broader acoustic similarity (timbre, background,
call structure) beyond taxonomic detector overlap.

## Metric tensor: Mahalanobis weighting

Raw dot products treat all P dimensions equally. But Perch logits are correlated
— 50 cicada-related logits that all fire together shouldn't count as 50
independent features.

The logit covariance matrix **(P, P)** from 127K unlabeled soundscape windows
provides the metric tensor. Mahalanobis-weighted composition:

```
(S, P) @ (P, P)⁻¹ @ (L, P)ᵀ = (S, L)  [decorrelated]
```

This is equivalent to matched filtering: the precision matrix down-weights
correlated logit clusters so each independent acoustic feature contributes once.

We use PCA-reduced precision (top 500 eigenvalues, condition number ~653) to
make the inversion numerically stable.

## Centering

**Critical**: all compositions must use centered profiles (global mean subtracted).
Without centering, raw cosine similarity is dominated by the shared background —
e.g., all species pairs show 0.95+ similarity because they share ubiquitous
insect detector activations (Sceliphron curvatum / mud dauber wasp).

After centering, cosine = Pearson correlation: measures how two entities
**co-deviate** from the average, not how similar they are to the average.

Centering is done per entity type against its own global mean:
- Species profiles centered by global focal mean
- Location profiles centered by global soundscape mean
- Windows centered by the same mean as whatever they're being compared to

## James-Stein shrinkage for low-sample entities

Locations with few windows (S21: 36, S23: 36) have noisy profile estimates.
James-Stein shrinkage pulls them toward the global mean proportional to their
estimation uncertainty:

```
shrunk[l] = α[l] * residual[l]
α[l] = max(0, 1 - (P * σ²_within) / (n[l] * ||residual[l]||²))
```

Sites with many windows (α ≈ 1.0) keep their raw profiles. Sites with few
windows (α ≈ 0.92) are regularized. This prevents noisy small-sample profiles
from producing spurious correlations in the compositions.

## Parallel schema: embedding space

The same star schema works with E = 1,536 embedding dimensions as the shared
axis. Embedding-space compositions capture acoustic similarity without
taxonomic interpretability. The two schemas are complementary:

- **Logit schema (P)**: interpretable, mediating taxa visible, but noisier
  (14,795-d, many near-zero logits)
- **Embedding schema (E)**: better conditioned (1,536-d), captures broader
  acoustic similarity, but opaque

Both should be centered. The embedding covariance (E, E) from unlabeled data
provides the Mahalanobis metric for the embedding schema.

## Computed artifacts

| Composition | Status | Artifact |
|-------------|--------|----------|
| (S, S) logit | Done | `species_graph.npz: logit_cooc` |
| (S, S) embedding | Done | `species_graph.npz: emb_cooc` |
| (L, L) | Done | `site_residuals.npz` (cosine on residuals) |
| (S×K_s, P) | Pending | `multi_prototypes.npz` — multi-prototype species profiles |
| (L×K_l, P) | Pending | `site_prototypes.npz` — site-state prototypes |
| (S, L×K_l) | Pending | `site_prototypes.npz: species_site_affinity` |
| (S, L×T) | Not yet | Spatiotemporal species prior (superseded by site prototypes) |
| (S, W) | Not yet | Per-window species predictions via multi-prototype graph |

## Multi-prototype species profiles via Leiden clustering

### Motivation

The single-mean species profile averages over silent windows, background noise,
and multiple call types. This dilutes the signal — a species with distinct song
and alarm calls gets a centroid between the two, matching neither well.

### Within-species residual covariance

To cluster windows within a species, we need a similarity metric. Raw cosine in
P-space has the same correlated-dimension problem the star schema identifies
elsewhere (50 cicada detectors counting as 50 features). The fix: use the
**within-species residual covariance** as the metric tensor.

1. For each species s, compute residuals: `r_{s,i} = w_{s,i} - μ_s`
2. Pool residuals across all species → within-species covariance `Σ_within`
3. SVD of pooled residuals → top-k components + eigenvalues
4. Whitening transform: project into PCA space, scale by 1/√eigenvalue

This estimates "what varies around a species' centroid" — the right metric for
asking "within a species, which windows are similar?" Crucially, this uses focal
recording statistics, not soundscape statistics (the unlabeled covariance would
describe the wrong domain).

### Leiden community detection

After whitening, the clustering pipeline:

1. Per species: project residuals through whitening transform → (N_s, k) matrix
2. Build k-NN graph on whitened vectors (cosine = Mahalanobis in P-space)
3. Remove low-degree nodes (pre-filter for isolated/noise windows)
4. Run Leiden algorithm → communities of variable number and size
5. Each non-trivial community centroid (mapped back to P-space) = one prototype

Leiden is preferred over HDBSCAN because:
- Works natively with graph structure (cosine k-NN)
- No need to specify number of clusters
- Handles high-dimensional spaces well (via the k-NN graph indirection)
- Naturally identifies isolated nodes for removal

### Updated species profile matrix

Instead of `(S, P)` with one row per species, the result is `(S×K, P)` with
variable K prototypes per species. Window scoring becomes:

```
scores = prototypes @ window.T        # (total_prototypes,)
per_species = group_max(scores, K)    # (S,) — max over each species' prototypes
```

This is piecewise-linear: the decision boundary is the union of K cones in
P-space rather than a single halfspace. It captures multi-modal call types
without requiring nonlinear models.

### Validation

- Primary cluster should activate the species' mapped Perch logit index
- Species with single stereotyped call → K=1 (no change from mean)
- Species with song + alarm + contact call → K=3-4
- Compare multi-prototype (S,W) scoring against single-mean scoring on the
  739 labeled soundscape windows

## Site-state prototypes via Leiden clustering

### Motivation

A single mean per site flattens distinct ecological states: dawn chorus, night
insect regime, rain events, quiet midday. The fixed diel bins (dawn/day/dusk/
night) are coarse and may not align with the acoustic reality at each site.

### Within-site residual covariance

Same approach as species prototypes, but using soundscape windows:

1. Per site l, subtract site mean: `r_{l,i} = w_{l,i} - μ_l`
2. Pool residuals across all sites → within-site covariance `Σ_within_site`
3. SVD → whitening transform

This is computed from soundscape statistics (127K windows), not focal — the
right domain for clustering soundscape windows. Pooling across sites assumes
the within-site variation structure is shared (rain sounds similar everywhere,
insect choruses have similar spectral structure), which is reasonable.

### Pipeline

Same as species prototypes: whiten residuals → k-NN cosine graph → Leiden →
prototype centroids. Each prototype is characterized by:

- **Temporal distribution**: fraction of dawn/day/dusk/night windows in the
  cluster — reveals whether the state is diel-locked or weather-driven
- **Top Perch taxa**: which detectors activate most strongly — interpretable
  label for the state (e.g., "night insect cluster" vs "dawn bird chorus")

### Updated location dimension

Instead of `(L, P)` with one row per site, the result is `(L×K_l, P)` with
variable K_l prototypes per site. The site-state prototypes replace both the
mean site profiles and the fixed diel period binning.

### Species × Site-state composition

```
(S×K_s, P) @ (L×K_l, P)^T = (S×K_s, L×K_l)
group_max on both sides → (S, L)
```

Or keep the full resolution: for a test window from site l, identify which
site-state prototype it's closest to, then use species affinities for that
state as a context-dependent prior. This gives each window a tailored species
prior based on what ecological state the site is currently in.

## Next steps

1. **Multi-prototype profiles** (this section) — within-species Leiden clustering
2. Compute (S, L) species × site affinity from prototypes
3. Compute (S, L×T) spatiotemporal prior using diel signatures
4. Compare multi-prototype (S, W) predictions against ridge OOF predictions
5. Mahalanobis-weighted compositions using within-species precision
6. Use (S, L×T) as a prior for the ridge model on soundscapes
7. Focal-to-soundscape domain gap per species: correlate each species' focal
   profile with its soundscape labeled windows
