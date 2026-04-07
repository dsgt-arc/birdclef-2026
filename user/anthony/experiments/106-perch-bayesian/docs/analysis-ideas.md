# Analysis Ideas: Perch v2 Feature Exploration

## Data available

- **739 labeled windows** (66 files, 9 sites) — ground truth for 75 species
- **127,104 unlabeled windows** (10,592 files, 23 sites) — no labels
- **~245K focal windows** (35,549 files, 206 species) — ground truth per recording
- Per window: embedding (1536-d), perch_logits (14,795-d), metadata (site, time, filename)
- Mapping: 203 BirdCLEF species → Perch logit index, 6 genus proxies, 25 unmapped

---

## Static methods (no temporal structure)

### 1. Zero-shot baseline: direct logit readout

For the 203 directly-mapped species, the Perch logit at the corresponding index
is a free predictor. No training needed.

- Evaluate on labeled windows: per-species AUROC, cmAP
- Compare against exp 105 linear probe (0.684 focal cmAP, 0.052 sc_cmap)
- Tells us the ceiling for non-learned approaches

### 2. Empirical Bayes ridge / logistic regression

Use the 127K unlabeled embeddings to estimate the feature covariance, then use
that as the prior for Bayesian regression on the 739 labeled windows.

**Empirical Bayes approach** (Munch et al. 2022):
- Estimate Σ_unlabeled from 127K unlabeled embedding vectors (1536×1536 covariance)
- Use Σ_unlabeled^{-1} as the precision prior for ridge/logistic regression
- Fit on 739 labeled windows → posterior weights for 234 species
- Closed-form for ridge: `W = (X'X + λΣ^{-1})^{-1} X'Y`
- The unlabeled covariance tells the model which feature directions have high
  variance (important) vs low variance (noise) — better regularization than λI

**Why this is the right starting point:**
- Semi-supervised: uses all 127K unlabeled points through the prior
- Closed-form, runs in seconds on CPU
- Ridge on embeddings (p/n ≈ 2:1) is well-validated for frozen pretrained features
  (Tang et al. 2020, CLIP linear probe literature)
- For logits (p/n ≈ 20:1), apply PCA to 200-500 dims first, or use elastic net
  (L1+L2) to exploit sparsity — most of 14,795 logits are irrelevant

**Compare features:**
- Embeddings (1536-d) — better p/n ratio, generalizes better for non-birds
- Logits (14,795-d) — interpretable, but needs dimensionality reduction
- Sigmoid(logits) — bounded [0,1], may be better conditioned
- PCA-reduced logits (200-500d) — best of both worlds?
- Concatenated embedding + top-k logits

Cross-validate λ via LOSO by site (reuse exp 101 splits).

### 3. GP classification on embeddings

Gaussian process classification on 1536-d Perch embeddings.

**Why GP:** (Patacchiola et al. NeurIPS 2020 — Deep Kernel Transfer)
- GP with RBF kernel on frozen embeddings is competitive with fine-tuning for few-shot
- Exact GP is feasible at n=739 (cost ≈ 739³ ≈ 400M ops)
- Provides calibrated uncertainty per prediction — useful for downstream thresholding
- Multi-output GP (Ingram et al. 2020) models species correlations via coregionalization
  matrix B = WW', capturing inter-species structure

**Semi-supervised extension:**
- Scalable SVGP (Hensman et al. AISTATS 2015) handles 127K points via inducing points
- GPyTorch implements this with GPU acceleration
- 500-1000 inducing points on 1536-d features is computationally feasible

### 4. Distribution shift: focal vs soundscape

Compare feature distributions between the three data populations. **This is the
go/no-go diagnostic** — if features don't transfer, no downstream model will work.

Methods:
- Per-logit KL divergence or Kolmogorov-Smirnov test (which Perch outputs shift most?)
- UMAP/t-SNE of embeddings, colored by source (focal/soundscape) and taxa
- MMD between focal and soundscape embedding distributions
- Which Perch logits are "always on" in soundscapes but not focal? (background species)

### 5. Prototype classification for rare species

For insect sonotypes (25 targets, 3-12 windows each) and other data-starved species:
- Compute centroid of each species' embeddings from labeled windows
- Classify by cosine similarity to centroids (Snell et al. 2017 — prototypical networks)
- Embeddings preferred over logits (preserve acoustic similarity for non-bird sounds)
- Compare: cosine vs Mahalanobis distance (using unlabeled covariance)

### 6. Non-bird species via indirect correlations

Amphibians, mammals, and insects have no direct Perch mapping. But their calls
may consistently co-activate certain bird logits.

- On labeled windows: correlate each of the 14,795 Perch logits with each non-bird
  species' presence. Which bird logits predict frog/mammal presence?
- Sparse patterns would be interpretable
- Ridge regression subsumes this — but the raw correlations are diagnostic

---

## Temporal methods (exploit time structure)

### 7. Simple temporal baselines

Before any state-space model, establish what simple pooling achieves:
- Max-pool ridge predictions across 12 windows of a 60s clip
- Mean-pool ridge predictions across 12 windows
- Exponential moving average of predictions within a clip
- **These are the baselines any temporal model must beat**

### 8. HMM temporal regimes

Hidden Markov Models are the established approach for PAM temporal structure
(D'Angelo et al. 2026 — general HMM framework for PAM time series).

**Per-site Gaussian HMM on embedding sequences:**
- Fit a 3-5 state HMM per site on unlabeled embedding time series
- States correspond to diel regimes: dawn chorus, daytime, dusk, night, transition
- No labels needed — learns regimes from temporal structure alone
- Then condition species predictions on the inferred regime:
  P(species | embedding, regime) vs P(species | embedding)

**Why HMM before EKF:**
- Simpler, well-validated in bioacoustics (D'Angelo et al. 2026)
- Discrete states (dawn/day/dusk/night) match ecological reality better than
  continuous Gaussian state evolution
- Diel patterns are strong in this data (Perez-Granados & Schuchmann 2025):
  amphibians = nocturnal, birds = dawn chorus
- Viterbi-decoded state labels are directly useful as features for ridge/GP

### 9. State space model / UKF

For continuous-state temporal smoothing. **Prefer UKF over EKF** — Newman et al.
(2023) showed UKF consistently outperforms EKF for ecological SSMs with
moderate nonlinearity, at similar computational cost.

**Key clarification from the literature:**
- Ecology's canonical SSMs (Royle & Kery 2007) use binary latent states with
  Bernoulli observations → Kalman filters don't apply, need MCMC
- BUT our setting is different: we model continuous Perch embeddings evolving
  over time, then threshold for species detection. The Gaussian assumption is
  reasonable for high-dimensional embeddings (central limit effects).
- Run UKF on continuous embeddings → threshold for detection after

**Architecture:**
- **State**: embedding-space representation of species community (1536-d, or
  reduced to DMD mode space, ~50-100d)
- **Observation model H**: empirical Bayes ridge weights (section 2)
- **Transition model A**: learned from DMD (section 10) or temporal autocorrelation
- **Process noise Q**: species-specific, estimated from focal within-recording
  variability

**Role of each dataset:**

| Dataset | Role | Why |
|---------|------|-----|
| Focal (~245K) | Learn H (observation model), Q (noise), call persistence | Lots of labeled data, clean signal, no temporal continuity between files |
| Labeled soundscapes (739) | Calibrate H for domain shift, validate, tune hyperparams | Small but in-domain ground truth |
| Unlabeled soundscapes (127K) | Learn A (dynamics), run UKF inference | Massive temporal/spatial structure, no labels needed |

**Within-clip (12 steps):** enough for applying A (smoothing), not for learning A.
**Across-clip (thousands per site):** where A is learned from temporal autocorrelation.

### 10. Dynamic Mode Decomposition

DMD extracts dominant spatiotemporal modes from time series — connecting the
Bayesian state space model to a modal decomposition of ecosystem dynamics.

**Standard DMD** — eigendecomposition of the best-fit linear operator:
- Eigenvectors = spatial patterns (species/logit activation patterns that evolve together)
- Eigenvalues = temporal dynamics (|λ|<1 decaying, |λ|≈1 persistent, complex λ oscillatory)
- Modes interpretable as ecological communities

**DMD variants worth trying:**

| Variant | Reference | Why relevant |
|---------|-----------|--------------|
| **Bayesian DMD** | Takeishi & Kawahara, IJCAI 2017 | Posterior uncertainty over modes — critical with short 12-step clips |
| **BOP-DMD** | Sashidhar & Kutz, Phil Trans A 2022 | Bagged optimized DMD, uncertainty via bootstrap. Implemented in PyDMD |
| **Multi-resolution DMD** | Kutz et al. 2015 | Separates slow (community shifts) from fast (vocalization events). Two timescales |
| **DMDc** | Proctor et al. SIAM 2016 | DMD with control inputs — site ID and time-of-day as exogenous forcing |
| **Online DMD** | Zhang et al. SIAM 2019 | Streaming updates, O(n²) memory. Feasible for 1536-d embeddings |
| **Kernel DMD** | Williams et al. 2015 | Nonlinear Koopman in kernel space. Cost depends on snapshots, not dim |
| **Non-Stationary DMD** | Baddoo et al. 2023 | Handles drifting dynamics (seasonal shifts). Fits modulated modes |
| **Sparse DMD** | Jovanovic et al. 2014 | L1 penalty selects minimal set of important modes |

**Run DMD on logits** for interpretable species-level modes.
**Run DMD on embeddings** for acoustic-level modes (and better numerics at 1536-d).

**What DMD reveals:**
1. Ecosystem timescales (eigenvalue spectrum)
2. Ecological communities (species grouped by dynamical similarity)
3. Diel regime structure (oscillatory modes with ~24h period)
4. Low-rank dynamics → compress 234-species state space to a few community modes

**Practical note:** run DMD per site separately first. Cross-site DMD assumes
shared dynamics, which may not hold across the 23 sites.

### 11. Classification-occupancy model

Ogawa et al. (Ecology, 2025) — directly our setting. Jointly models false-positive
and false-negative rates from AI classifiers (BirdNET) within an occupancy framework.

**How it applies:**
- Perch predictions have both false positives (bird logit fires for a frog) and
  false negatives (species present but Perch doesn't detect it)
- The 739 labeled windows serve as the validation subset that calibrates FP/FN rates
- Chambert et al. (2018): foundational framework — a subset of human-validated
  detections dramatically improves occupancy estimates from automated classifiers
- Wright et al. (2020): multi-species extension with confusion matrix modeling

**Integration:** run ridge/GP to get initial predictions → feed into occupancy model
with site/time covariates → get calibrated occupancy estimates accounting for
imperfect detection.

### 12. HMSC (Hierarchical Modelling of Species Communities)

Ovaskainen/Tikhonov et al. (2020) — joint model for hundreds of species with
shared environmental responses and species-to-species associations.

- GPU-accelerated via Hmsc-HPC (Tikhonov et al. 2024)
- Models species correlations as latent factors
- Could be downstream of Perch features: feed site-level detection summaries
  into HMSC with environmental covariates
- Produces community-level occupancy estimates with residual correlations

---

## Feature space selection: logits vs embeddings

| Analysis | Logits | Embeddings | Both |
|----------|--------|------------|------|
| Zero-shot baseline | ✓ (direct readout) | | |
| Ridge / empirical Bayes | try (PCA first) | **prefer** (p/n ≈ 2:1) | concatenate |
| GP classification | | **prefer** | |
| DMD | **prefer** (interpretable modes) | try (acoustic modes) | |
| HMM regimes | | **prefer** (dense features) | |
| UKF observation model | | **prefer** (generalization) | concatenate |
| Distribution shift | ✓ (per-logit KS) | ✓ (UMAP, MMD) | both |
| Prototypes | | **prefer** (acoustic similarity) | |
| Occupancy model | | **prefer** (fewer params) | |

---

## Priority order

### Phase 1: Static baselines (days 1-3)
1. **Zero-shot logit readout** — minutes, establishes floor
2. **Empirical Bayes ridge on embeddings** — hours, main static result
3. **Distribution shift diagnostics** — go/no-go gate for everything downstream
4. **Prototype classifier for insect sonotypes** — hours, low-hanging fruit

### Phase 2: Temporal structure (days 4-7)
5. **Simple temporal pooling** — hours, establishes temporal baseline
6. **HMM diel regimes on unlabeled data** — 1-2 days, practical temporal model
7. **DMD on per-site time series** — 1 day, discover dynamical modes

### Phase 3: Full pipeline (days 7-10)
8. **UKF temporal smoothing** — using DMD-informed dynamics + ridge observation model
9. **Classification-occupancy model** — calibrate FP/FN rates
10. **GP classification** — if ridge saturates, try GP for uncertainty
11. **Transition matrix / DMD mode analysis** — ecological interpretation

### Research exploration (if time permits)
12. **HMSC community modeling** — ecological gold standard
13. **Multi-resolution DMD** — separate fast/slow dynamics
14. **Bayesian DMD** — uncertainty over modes from short clips

---

## References

### Bayesian transfer learning
- Tang et al. 2020 — Deep Transfer Learning with Ridge Regression
- Munch et al. 2022 — Semi-supervised empirical Bayes factor regression
- Patacchiola et al. NeurIPS 2020 — Deep Kernel Transfer (GP on frozen features)
- Hensman et al. AISTATS 2015 — Scalable Variational GP Classification
- Ingram et al. 2020 — Multi-output GPs for species distribution modelling
- Kim et al. 2023 — Bayesian continual learning on frozen features

### State space models in ecology
- Royle & Kery 2007 — Dynamic occupancy models as Bayesian SSMs
- Newman et al. 2023 — Practical SSM fitting (EKF vs UKF vs particle filter)
- Auger-Methe et al. 2021 — Guide to state-space modeling of ecological time series
- D'Angelo et al. 2026 — HMM framework for PAM time series
- Perez-Granados & Schuchmann 2025 — Diel vocal activity patterns via PAM

### Occupancy-detection models
- Ogawa et al. 2025 — Classification-occupancy model for automated species detection
- Chambert et al. 2018 — Framework for acoustic monitoring with imperfect classifiers
- Wright et al. 2020 — Multi-species misclassification modeling
- Ovaskainen/Tikhonov et al. 2020 — HMSC 3.0 (hierarchical community modeling)

### DMD variants
- Takeishi & Kawahara, IJCAI 2017 — Bayesian DMD
- Sashidhar & Kutz, Phil Trans A 2022 — BOP-DMD (bagged, uncertainty)
- Kutz et al. 2015 — Multi-resolution DMD
- Proctor et al. SIAM 2016 — DMDc (with control inputs)
- Zhang et al. SIAM 2019 — Online DMD
- Williams et al. 2015 — Kernel DMD
- Baddoo et al. 2023 — Non-Stationary DMD
- Frelat et al. 2017 — Tensor decomposition for ecological communities

### Bioacoustic uncertainty
- Bett et al. 2025 — Uncertainty calibration of bird sound classifiers
- Iqbal et al. 2024 — Variational uncertainty for bioacoustic label smoothing
- Denton 2024 — Bayesian call density estimation on Perch confidence scores
- Metcalf et al. 2023 — Contextual information (temporal post-processing) for classifiers
- Kershenbaum et al. 2025 — Practical guide to automatic detection in bioacoustics
- Snell et al. 2017 — Prototypical networks for few-shot classification
