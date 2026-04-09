---
title: "Star Schema as Tensor Decomposition"
status: concept
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
