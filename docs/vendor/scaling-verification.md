---
name: "Weaver (scaling-verification)"
source: "https://github.com/HazyResearch/scaling-verification"
license: "MIT"
category: "utility"
pipeline_stage:
  - "data-processing"
relevance: "Reference for latent variable label fusion -- ideas transfer to weak supervision in exp 013"
status: "reference"
---

# Weaver (scaling-verification)

## Overview

Weaver is a framework from Hazy Research (Stanford) for combining multiple weak, imperfect verifiers into a strong unified verifier without requiring ground-truth labels. It was developed for LLM response selection (choosing the best reasoning chain from multiple candidates), but the core statistical machinery -- latent variable models for estimating annotator accuracy, diversity-aware verifier selection, and calibrated score fusion -- transfers directly to our multi-source species label fusion problem in BirdCLEF (exp 013). The paper is "Shrinking the Generation-Verification Gap with Weak Verifiers" (Saad-Falcon et al., 2025, arXiv:2506.18203).

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `WeakSupervised` | Core label model (extends `metal-ama` LabelModel). Estimates per-verifier TPR/TNR without ground truth. Supports both binary (triplet method) and continuous (covariance-based) modes. | `weaver/models.py:1014` |
| `tensor_decomp` | Tensor decomposition for unsupervised class balance estimation. Decomposes 3rd-order moments of verifier outputs to recover mixture weights. | `weaver/tensor_decomp.py` |
| `_drop_deps` | Diversity-aware verifier selection. Finds the maximally independent subset of k verifiers by minimizing off-diagonal magnitude of the inverse covariance matrix. | `weaver/models.py:1070` |
| `_get_balanced_idxs` | Adaptive imbalanced verifier filtering. Drops verifiers whose marginal vote rate is too extreme (>0.9 or <0.1) based on class balance. | `weaver/models.py:1187` |
| `NaiveBayes` | Supervised Naive Bayes baseline. Computes per-verifier TPR/TNR from labeled data, applies Bayes rule with conditional independence. | `weaver/models.py:1498` |
| `Unsupervised` / `Unsupervised2` | EM-based latent variable models. Cluster problems by difficulty, learn per-cluster logistic regression (v1) or per-cluster Gaussian emission (v2) models. | `weaver/models.py:315,540` |
| `NaiveEnsemble` | Simple mean-score baseline (no weighting, no accuracy estimation). | `weaver/models.py:1813` |
| `VerificationDataset` | Data loading/preprocessing: score normalization (min-max, quantile, winsorize), difficulty-stratified splitting, verifier binarization. | `weaver/dataset.py` |
| `selection/run.py` | Main train/eval harness. Hydra configs, per-problem vs per-dataset vs cluster training strategies, W&B logging. | `selection/run.py` |
| `distillation/` | Knowledge distillation from Weaver ensemble into a 400M cross-encoder (ModernBERT), capturing 98.7% accuracy at 0.03% compute. | `distillation/` |

## Core Algorithms in Detail

### 1. Weak Supervision Label Model (WeakSupervised)

The primary model extends `metal-ama`'s `LabelModel` (a reimplementation of the FlyingSquid/MeTaL triplet method). Key mechanisms:

**Binary mode** (default):
- Binarizes continuous verifier scores at a threshold (default 0.5)
- Learns per-verifier class-conditional accuracies: P(LF_i = y | y=1) and P(LF_i = y | y=0)
- Uses observable agreement statistics (no ground truth needed) via the triplet method
- Estimates class balance either from labels, a fixed prior, or tensor decomposition

**Continuous mode** (`use_continuous=True`):
- Works with raw continuous scores (no binarization)
- Estimates verifier-label covariance Sigma_hat[i, m] via triplet identities:
  ```
  acc_i = sqrt(O[i,j] * O[i,k] * var_y / O[j,k])
  ```
  where O is the verifier-verifier covariance matrix
- Prediction via linear MMSE: `p(y|X) = Sigma[y,X] * Sigma[X,X]^{-1} * X`

**Tensor decomposition for class balance** (`use_tensor_decomp=True`):
- When class balance is unknown, decomposes 3rd-order moment tensors of verifier outputs
- Samples 30 random triples of verifiers, runs `mixture_tensor_decomp_full` on each
- Averages recovered mixture weights across triples for robust estimation

### 2. Diversity-Aware Verifier Selection (_drop_deps)

Critical insight: correlated verifiers violate the conditional independence assumption and degrade label model quality. Weaver selects the most independent subset:

1. For all C(n, k) subsets of k verifiers (default k=3):
   - Compute inverse covariance matrix of [selected_verifiers, ground_truth]
   - Remove the ground-truth row/column and the diagonal
   - Record the max absolute off-diagonal element (sparsity metric)
2. Select the subset with the smallest sparsity (most conditionally independent)
3. Warn if any selected verifier has extreme marginals (>0.9 or <0.1)

This directly parallels our exp 013 design: with 10 LFs across 4 error modes, we need to identify which subsets are truly independent for FlyingSquid.

### 3. Adaptive Imbalanced Verifier Filtering

Verifiers that vote nearly unanimously (>90% positive or >90% negative) provide little discriminative signal. Weaver's adaptive strategy:
- If class balance < 0.2 (hard problems): drop verifiers that vote mostly "yes" (rule="large")
- If class balance > 0.8 (easy problems): drop verifiers that vote mostly "no" (rule="small")
- Otherwise: drop any verifier with marginal outside [0.1, 0.9] (rule="all")

### 4. Unsupervised EM Model (Unsupervised2)

A full generative model per difficulty level:
- Latent variable: y_ij in {0,1} (correct/incorrect)
- Emission: X_ij | y_ij, d_i ~ N(mu(y_ij, d_i), Sigma(y_ij, d_i))
- Prior: P(y_ij=1 | d_i) = sigmoid(gamma(d_i))
- EM alternates between computing posterior P(y|X,d) and updating (mu, Sigma, gamma)

This is analogous to learning per-species emission distributions for BirdNET/Perch embeddings in our pipeline.

## Relevance to Pipeline

### Direct Transfer to Exp 013 (Weak Supervision Label Fusion)

Weaver's architecture maps closely to our BirdCLEF label fusion problem:

| Weaver Concept | BirdCLEF Analog |
|---|---|
| Verifier (reward model / LM judge) | Labeling function (BirdNET probe, Perch probe, NDSI, etc.) |
| Response correctness (binary) | Species presence (binary per species per time step) |
| Problem difficulty clustering | Species/habitat difficulty (taxa, recording quality) |
| Verifier score normalization | LF output calibration (sigmoid on logits, min-max on indices) |
| Diversity-aware subset selection | Independent error mode verification for FlyingSquid |
| Class balance estimation | Positive rate estimation (most species absent most of the time) |
| Continuous label model | Continuous embedding-based LF scoring |

### Key Ideas to Borrow

1. **Inverse covariance for independence testing** (`_drop_deps`): Before running FlyingSquid on our 10 LFs, use the inverse covariance method to verify that our 4 error modes (neural, metadata, acoustic events, spectral indices) are actually conditionally independent. If two LFs within the same tier are highly correlated, either drop one or model the dependency.

2. **Adaptive verifier filtering** (`_get_balanced_idxs`): Our acoustic index LFs (NDSI, spectral contrast) may be nearly constant for certain species or habitats. Apply the same adaptive filtering: if a species is rare (low positive rate), drop LFs that always vote positive.

3. **Tensor decomposition for class balance** (`tensor_decomp.py`): When we lack ground truth for species presence rate, we can estimate it from verifier agreement patterns using the triplet method -- identical to what FlyingSquid does internally.

4. **Continuous covariance-based fusion**: For embedding-based LFs (BirdNET 1024-d, Perch 1280-d), the continuous label model's covariance approach may outperform binarized triplets. The formula `acc_i = sqrt(O[i,j] * O[i,k] * var_y / O[j,k])` works on any real-valued LF outputs.

5. **Per-difficulty clustering**: Weaver trains separate models per difficulty cluster. We could analogously train per-taxa label models (Aves vs Amphibia vs Insecta), since the LF error profiles differ systematically across taxonomic groups.

## Usage

This is a **reference-only** repository. We will not install or run Weaver directly. The transferable algorithms are:

```python
# 1. Independence testing (from _drop_deps)
import numpy as np
from itertools import combinations

def find_most_independent_subset(lf_votes, k=3):
    """Select k most conditionally independent LFs."""
    n_lfs = lf_votes.shape[1]
    best_sparsity = np.inf
    best_subset = None

    for subset in combinations(range(n_lfs), k):
        subset = list(subset)
        cov = np.cov(lf_votes[:, subset].T)
        cov += 1e-6 * np.eye(len(subset))
        inv_cov = np.linalg.inv(cov)
        np.fill_diagonal(inv_cov, 0)
        sparsity = np.abs(inv_cov).max()
        if sparsity < best_sparsity:
            best_sparsity = sparsity
            best_subset = subset

    return best_subset, best_sparsity


# 2. Continuous accuracy estimation (from WeakSupervised.fit continuous mode)
def estimate_lf_accuracies(lf_scores, var_y_estimate):
    """Estimate LF accuracies from pairwise covariances (no labels needed)."""
    n, m = lf_scores.shape
    O = lf_scores.T @ lf_scores / n  # verifier covariance

    accuracies = np.zeros(m)
    for i in range(m):
        acc_estimates = []
        for j in range(m):
            for k in range(j + 1, m):
                if j == i or k == i:
                    continue
                if O[j, k] > 1e-8:
                    acc = np.sqrt(
                        np.abs(O[i, j] * O[i, k] * var_y_estimate / O[j, k])
                    )
                    acc_estimates.append(acc)
        if acc_estimates:
            accuracies[i] = np.mean(acc_estimates)

    return accuracies


# 3. Adaptive filtering (from _get_balanced_idxs)
def filter_extreme_lfs(lf_votes, class_balance, threshold=(0.1, 0.9)):
    """Drop LFs with extreme marginal vote rates."""
    marginals = lf_votes.mean(axis=0)
    if class_balance < 0.2:
        keep = marginals < threshold[1]  # drop always-yes voters
    elif class_balance > 0.8:
        keep = marginals > threshold[0]  # drop always-no voters
    else:
        keep = (marginals > threshold[0]) & (marginals < threshold[1])
    return np.where(keep)[0]
```

## Dependencies

Weaver itself requires (we do NOT install these):
- `torch >= 2.0`, `transformers`, `vllm` (for generation/scoring -- not relevant to us)
- `metal-ama` (fork of MeTaL label model -- the triplet method core)
- `scikit-learn`, `scipy`, `numpy` (for the statistical models we care about)
- `hydra-core`, `wandb` (experiment management)

For our exp 013, the relevant algorithms are pure numpy/scipy and need no additional dependencies beyond what we already have.

## Notes

- **metal-ama** is a fork of the MeTaL/FlyingSquid label model by Mayee Chen (co-author). It provides the `LabelModel.train_model()` and `LabelModel.predict_proba()` methods that `WeakSupervised` inherits. This is functionally equivalent to FlyingSquid's triplet method, which we plan to use directly in exp 013.

- **Continuous vs binary mode matters**: Weaver found that continuous mode (covariance-based) can outperform binary mode when verifier scores are well-calibrated. For our embedding-based probes (BirdNET logits, Perch logits), continuous mode is worth testing alongside the binarized FlyingSquid approach.

- **Diversity > quantity**: Weaver consistently found that 3 diverse, independent verifiers outperform 15+ correlated ones. This validates our exp 013 design of 4 independent error modes rather than stacking many similar neural classifiers.

- **Logit adjustment for class imbalance** (line 226): When using logistic regression on imbalanced data, Weaver subtracts `log(pi_1 / pi_0)` from the intercept post-training. This corrects for the prior shift when train and test class distributions differ -- directly applicable to our rare-species problem where positive rates can be <1%.

- **The distillation stage** (training a 400M cross-encoder to mimic the ensemble) is conceptually interesting for deployment but not immediately relevant -- our inference pipeline already runs on GPU, and the bottleneck is teacher label quality, not inference speed.

- **Key paper result**: On MATH-500 with Llama 3.3 70B, Weaver achieves 93.4% selection accuracy (vs 83.0% majority vote, 73.0% best single verifier), demonstrating that principled weak supervision fusion substantially outperforms naive averaging even with imperfect components.
