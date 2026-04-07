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

## Priority order

1. **Zero-shot baseline** — minutes of work, establishes floor
2. **Ridge regression** — hours, main result
3. **Distribution shift** — diagnostic, informs whether ridge generalizes
4. **Site/temporal patterns** — uses unlabeled data, informs priors
5. **Non-bird correlations** — diagnostic for understanding ridge weights
6. **Temporal aggregation** — extension to ridge
7. **Insect sonotypes** — hardest, do last
