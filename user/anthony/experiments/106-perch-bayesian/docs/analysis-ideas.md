# Analysis Ideas: Perch v2 Feature Exploration

## Data available

- **739 labeled windows** (66 files, 9 sites) — ground truth for 75 species
- **127,104 unlabeled windows** (10,592 files, 23 sites) — no labels
- **~245K focal windows** (35,549 files, 206 species) — ground truth per recording
- Per window: embedding (1536-d), perch_logits (14,795-d), metadata (site, time, filename)
- Mapping: 203 BirdCLEF species → Perch logit index, 6 genus proxies, 25 unmapped

## 1. Zero-shot baseline: direct logit readout

For the 203 directly-mapped species, the Perch logit at the corresponding index
is a free predictor. No training needed.

- Evaluate on labeled windows: per-species AUROC, cmAP
- Compare against exp 105 linear probe (0.684 focal cmAP, 0.052 sc_cmap)
- Tells us the ceiling for non-learned approaches

## 2. Ridge regression (logits → BirdCLEF targets)

Multi-output ridge: 14,795 Perch logits → 234 BirdCLEF species.

- Closed-form: `W = (X'X + λI)^{-1} X'Y`, factorize once, solve all 234 targets
- Cross-validate λ on the 739 labeled windows (LOSO by site, reuse exp 101 splits)
- Compare features: raw logits vs sigmoid(logits) vs embeddings (1536-d)
- Embeddings might work better — 1536 features is a much friendlier ratio with 739 samples

## 3. Distribution shift: focal vs soundscape

Compare feature distributions between the three data populations:
- Focal recordings (206 species, controlled, mostly diurnal birds)
- Labeled soundscapes (75 species, Pantanal wetland, amphibian-dominated)
- Unlabeled soundscapes (same sites, unknown species composition)

Methods:
- Per-logit KL divergence or Kolmogorov-Smirnov test (which Perch outputs shift most?)
- UMAP/t-SNE of embeddings, colored by source (focal/soundscape) and taxa
- MMD between focal and soundscape embedding distributions
- Which Perch logits are "always on" in soundscapes but not focal? (background species)

## 4. Temporal activation patterns

Soundscapes are 60s (12 windows). Analyze how Perch activations change over time:
- Per-file activation timelines: which logits fire in which 5s windows?
- Species-specific temporal signatures (frogs call continuously, birds call briefly)
- Optimal aggregation width per species: does averaging over 2, 3, or all 12 windows help?
- Autocorrelation of logit activations within a file — do species come in bursts?

## 5. Site-level and diel patterns

The soundscape filenames encode site and timestamp. Analyze spatial/temporal structure:
- Per-site activation profiles: which Perch logits characterize each site?
- Diel patterns: which logits are active at night vs dawn vs day?
  (amphibians = nocturnal, birds = dawn chorus — this is a strong prior)
- Site × time interaction: do species assemblages shift predictably?
- Use unlabeled soundscapes (127K windows, 23 sites) — no labels needed, just activations
- Could inform a site-aware or time-aware prior for the ridge model

## 6. Non-bird species via indirect correlations

Amphibians, mammals, and insects have no direct Perch mapping. But their calls
may consistently co-activate certain bird logits (acoustic similarity, shared
frequency bands, co-occurrence).

- On labeled windows: correlate each of the 14,795 Perch logits with each non-bird
  species' presence. Which bird logits predict frog/mammal presence?
- Sparse patterns would be interpretable (e.g., "Perch logit for species X always
  fires when frog Y is calling")
- Ridge regression subsumes this — but the raw correlations are diagnostic

## 7. Insect sonotype discrimination (hardest subproblem)

25 `47158sonXX` targets with zero focal data, zero Perch mapping, only 3-12 labeled
windows each. All defined by soundscape annotation only.

- Cluster their Perch embeddings: do the 25 sonotypes separate in embedding space?
- If yes: nearest-centroid or prototype classifier could work
- If no: Perch doesn't encode the relevant acoustic differences (frequency, rhythm)
- Compare embeddings vs logits for this task — embeddings might preserve more low-level
  acoustic information than the taxonomic logit space

## 8. State space model / Extended Kalman Filter

Species presence is a latent state that evolves over time. Perch logits are
noisy observations. A state space model makes this explicit.

### Within-clip dynamics (12 steps per 60s file)

- **State**: 234-d species presence (logit scale, so approximately Gaussian)
- **Observation model**: ridge weights from section 2 (H matrix)
- **Transition model**: species-specific persistence — frogs call continuously
  (high autocorrelation, low process noise), birds call in brief bouts (low
  autocorrelation, high process noise)
- **Benefit**: smooths noisy per-window predictions using temporal continuity.
  A frog detected in windows 1-4 and 6-12 was probably present in window 5 too.

### Across-clip dynamics (site-level, hours/days)

- Longer timescale: species assemblages at a site evolve over hours (diel cycle)
  and days (seasonal migration, weather)
- State propagates across consecutive clips at the same site
- Unlabeled clips participate as observations — they update the state even without
  ground truth labels

### What the EKF transition matrix tells us

The learned transition matrix A encodes **species dynamics**:
- Diagonal entries = persistence (how sticky is each species' presence?)
- Off-diagonal entries = co-occurrence dynamics (does species X arriving predict
  species Y departing? Does a dawn chorus species predict the end of nocturnal callers?)
- Eigenvalues = timescales of the system (fast vs slow-changing species groups)

### Clustering from the transition function

Once we learn A, its structure reveals ecological groupings:

1. **Spectral clustering on A**: eigendecomposition of the transition matrix groups
   species by dynamical similarity. Species in the same cluster have similar
   persistence and co-occurrence patterns. This could discover:
   - Nocturnal assemblage (frogs + nightjars that appear/disappear together)
   - Dawn chorus group (birds that activate together at sunrise)
   - Background species (always present, diagonal ≈ 1)
   - Transient visitors (low persistence, high process noise)

2. **Community detection**: treat A as a weighted directed graph, run modularity-based
   clustering. Edges = dynamical coupling between species.

3. **Regime identification**: if A is time-varying (different transition dynamics at
   night vs day), fit separate A matrices per diel period. The difference between
   A_night and A_day reveals which species swap in/out at transitions.

4. **Anomaly detection**: clips where the Kalman innovation (prediction error) is
   large indicate unusual species activity — rare species, unusual timing, or
   recording artifacts. Could flag interesting unlabeled clips for manual review.

### Role of each dataset

The three datasets serve distinct roles in the state space framework:

**Focal recordings (~245K windows, labeled)** — parameter estimation (supervised):
- Learn the **observation model H**: what does each species look like in Perch
  logit space? Thousands of clean, labeled examples per species.
- Estimate **process noise Q** per species: how variable are Perch logits across
  consecutive windows within a single focal recording of that species?
- Measure **call persistence** per species: average consecutive active windows.
  Directly parameterizes diagonal of A.
- Focal data has NO temporal continuity between recordings — each file is an
  independent snapshot. The EKF does not run on focal data.

**Labeled soundscapes (739 windows, 66 files)** — calibration + validation:
- Calibrate H for domain shift: focal-learned observation profiles may not match
  soundscape conditions (noise, multi-species overlap, different mics).
- Validate EKF predictions against ground truth.
- Tune λ (ridge) and Q (process noise) via cross-validation.

**Unlabeled soundscapes (127K windows, 10,592 files)** — dynamics + inference:
- Learn **transition matrix A** from temporal autocorrelation of Perch logits
  across consecutive clips at each site. No labels needed.
- Learn A at two timescales: within-clip (12 steps) and across-clip at a site
  (thousands of chronologically ordered files).
- 12 steps within a clip is enough for *applying* A (inference/smoothing) but
  not for *learning* A from a single clip. The power comes from 10,592 files
  providing ~116K within-clip transitions plus thousands of cross-clip transitions.
- Run EKF state estimation to produce smoothed species predictions on all
  unlabeled soundscapes.

### Practical considerations

- EKF needed (not linear KF) because species presence is binary but we model on
  logit scale — the observation model (sigmoid ∘ ridge) is nonlinear
- Alternative: particle filter if the Gaussian approximation is too poor
- Start simple: diagonal A (independent species persistence), then add off-diagonal
  terms if cross-species dynamics improve predictions
- The 127K unlabeled windows provide massive data for learning A — transition
  statistics can be estimated purely from temporal autocorrelation of Perch logits
  without any species labels

## 9. Dynamic Mode Decomposition on Perch time series

The learned transition matrix A from the EKF is analyzable via DMD. This
connects the Bayesian state space model to a modal decomposition of the
ecosystem dynamics.

### DMD = eigendecomposition of A

Given consecutive Perch logit observations z_t, z_{t+1} from unlabeled
soundscapes, DMD finds the best-fit linear operator A ≈ z_{t+1} z_t^+
(where z_t^+ is the pseudoinverse). Then A = VΛV^{-1}:

- **Eigenvectors (DMD modes)**: spatial activation patterns. Each mode is a
  coherent pattern of Perch logits that evolve together. Interpretable as
  ecological communities — "these bird logits + these amphibian-correlated
  logits activate and decay as a unit."

- **Eigenvalues (DMD spectrum)**: temporal behavior of each mode.
  - |λ| < 1 → decaying (transient events: brief bird calls, alarm calls)
  - |λ| ≈ 1 → persistent (background species: always-present frogs, insects)
  - Complex λ → oscillatory. Period = 2π / arg(λ). A complex pair with
    period ≈ 24h would be the diel cycle (dawn chorus ↔ nocturnal assemblage).
  - |λ| > 1 → growing (unstable — shouldn't appear in a stationary ecosystem,
    would indicate model misfit or non-stationarity)

- **Mode amplitudes**: how much each mode contributes to the overall dynamics.
  Dominant modes explain most of the variance in species turnover.

### What DMD reveals

1. **Ecosystem timescales**: the eigenvalue spectrum shows the characteristic
   frequencies of species turnover. Expect clusters at:
   - Very fast (|λ| ≈ 0): individual call events
   - Slow (|λ| ≈ 0.9-0.99): gradual assemblage shifts
   - Oscillatory (~24h period): diel cycle
   - Near-unit (|λ| ≈ 1): persistent background

2. **Ecological communities**: DMD modes group species by dynamical similarity.
   Species that load on the same mode appear/disappear together. This is a
   data-driven discovery of species assemblages — no taxonomy needed.

3. **Diel regime structure**: if dominant modes are oscillatory with ~24h period,
   the real/imaginary parts of those eigenvectors describe the dawn→day→dusk→night
   species rotation. Projecting a specific time-of-day onto these modes gives
   the expected species composition.

4. **Low-rank dynamics**: if a few DMD modes explain most variance, the effective
   dimensionality of species dynamics is low. This means the 234-species state
   space can be compressed to a handful of community modes — massive simplification
   for the EKF (run in mode space, not species space).

### Computational approach

DMD is cheap — just an SVD:
1. Stack consecutive Perch logit vectors into matrices X = [z_1, ..., z_{T-1}]
   and Y = [z_2, ..., z_T]
2. SVD of X: X = UΣV*
3. Project: Ã = U* Y V Σ^{-1} (reduced-rank A in the SVD basis)
4. Eigendecompose Ã → modes and eigenvalues

For 127K windows with 14,795-d logits, work in the reduced basis (truncated SVD
to rank r ≈ 50-200). Total cost: one truncated SVD of a 14795 × 127K matrix,
which is minutes on a CPU with randomized SVD.

Can also run DMD separately per site or per diel period to discover site-specific
or time-varying dynamics.

### Pipeline integration

1. Ridge regression → static observation model (H)
2. DMD on unlabeled Perch logit time series → dynamical modes and A
3. Interpret modes as ecological communities, eigenvalues as timescales
4. Use DMD modes to inform/initialize EKF structure
5. EKF with informed A → temporally smoothed species predictions
6. Analyze EKF innovations (prediction errors) for anomaly detection

DMD can be done *before* the EKF as a standalone analysis — it only needs the
raw Perch logit time series from unlabeled soundscapes.

## 10. Feature space selection: logits vs embeddings

The two Perch outputs encode different information and suit different analyses.

### Logits (14,795-d)

- Each dimension = a specific species in Perch's pretrained taxonomy
- Sparse, mostly near-zero, interpretable (you can name each dimension)
- Bird-biased: non-bird sounds produce indirect activations ("which birds does
  Perch think this frog sounds like?")
- Best for: DMD (modes are interpretable species patterns), direct logit readout
  (zero-shot), correlation analysis with BirdCLEF targets

### Embeddings (1536-d)

- Penultimate-layer acoustic representation, before taxonomic projection
- Dense, encodes spectral/temporal features more generally
- Better for non-bird species: captures acoustic similarity even for sounds
  Perch was never trained to classify
- Best for: ridge regression (1536 features vs 739 samples is much friendlier
  than 14795 vs 739), prototype classification, UMAP visualization

### Which to use where

| Analysis | Logits | Embeddings | Both |
|----------|--------|------------|------|
| Zero-shot baseline | ✓ (direct readout) | | |
| Ridge regression | try | **prefer** (better p/n ratio) | concatenate |
| DMD | **prefer** (interpretable modes) | try (acoustic modes) | |
| EKF observation model | | **prefer** (generalization) | concatenate |
| Distribution shift | ✓ (per-logit KS test) | ✓ (UMAP, MMD) | both |
| Site/temporal patterns | ✓ (which species active?) | ✓ (acoustic clusters) | both |
| Insect sonotypes | | **prefer** (prototype/centroid) | |

### Prototype learning connection

Prototype networks classify by distance to class centroids in embedding space.
The EKF approach extends this with temporal dynamics:

- **Static prototype**: each species has a centroid in embedding space. Classify
  by nearest prototype (cosine similarity). No temporal information.
- **EKF observation model**: each species has an expected activation profile
  (effectively a prototype), but beliefs are updated probabilistically and
  temporal priors propagate information across windows.
- **DMD modes**: dynamic prototypes — not a single point but a trajectory
  pattern. A "nocturnal frog community mode" describes how a species group
  activates and persists together over time.

For **data-starved species** (insect sonotypes with 3-12 windows), static
prototype classification in embedding space is the right starting point.
The EKF can then smooth those prototype-based predictions temporally.

The approaches layer naturally:
1. Prototypes (embedding space) → baseline for rare species
2. Ridge (embedding or logit space) → static observation model for common species
3. EKF → temporal smoothing on top of either
4. DMD (logit space) → interpret the dynamics

## Priority order

1. **Zero-shot baseline** — minutes of work, establishes floor
2. **Ridge regression** — hours, try embeddings first, then logits, then concat
3. **Distribution shift** — diagnostic, both feature spaces
4. **Site/temporal patterns** — uses unlabeled data, informs priors
5. **DMD on unlabeled logit time series** — dynamical modes, timescales, communities
6. **State space / EKF** — temporal smoothing using DMD-informed dynamics
7. **Prototype classification** — for insect sonotypes and other data-starved species
8. **Transition matrix clustering** — ecological groupings from DMD modes
9. **Non-bird correlations** — diagnostic for understanding ridge weights
