---
name: "FlyingSquid"
source: "https://github.com/HazyResearch/flyingsquid"
license: "Apache 2.0"
category: "utility"
pipeline_stage:
  - "data-processing"
  - "label-fusion"
relevance: "Fuses noisy labeling functions into soft labels for weak supervision in BirdCLEF species classification"
status: "active"
---

# FlyingSquid

## Overview

FlyingSquid is a weak supervision framework from Stanford HazyResearch that learns a
label model from the agreements and disagreements between multiple noisy labeling
functions (LFs), without requiring any ground-truth labels. It uses a triplet method
(ICML 2020) to estimate each LF's accuracy via observable second- and third-order
moments, then combines LF votes into calibrated probabilistic labels. The key
advantage over Snorkel's matrix completion approach is speed: FlyingSquid runs in
closed form with no iterative optimization, making it practical for large-scale or
streaming settings.

## Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| `LabelModel` | Core class: fits a graphical model over LFs, predicts soft labels | `flyingsquid/label_model.py` |
| `_triplets.Mixin` | Triplet method: forms LF triplets, recovers mean parameters from moments | `flyingsquid/_triplets.py` |
| `_graphs.Mixin` | Graph structure checks: verifies Y separates LF pairs (triplet identifiability) | `flyingsquid/_graphs.py` |
| `_observables.Mixin` | Computes class balance and Y marginals from dev set or uniform prior | `flyingsquid/_observables.py` |
| `_lm_parameters.Mixin` | Generates B-matrix and r-vector for solving joint LF-Y marginals | `flyingsquid/_lm_parameters.py` |
| `FSLoss` | PyTorch loss wrapper: online label model retraining + BCE loss for end models | `flyingsquid/pytorch_loss.py` |
| `helpers.py` | Utility: `dict_product` for Cartesian product over dictionaries | `flyingsquid/helpers.py` |

## The Triplet Method

The core insight is that for conditionally independent LFs given Y:

```
E[lambda_i * Y] * E[lambda_k * Y] = E[lambda_i * lambda_k]
```

Since `E[lambda_i * lambda_k]` (the second moment) is directly observable from LF
outputs, and we can form triplets `(i, j, k)` of mutually separated LFs, we recover
individual LF accuracies via:

```
E[lambda_i * Y] = sqrt(|E[lambda_i * lambda_j] * E[lambda_i * lambda_k] / E[lambda_j * lambda_k]|)
```

Three solve methods are available:
- **`triplet`**: uses a single triplet per LF (fast, less stable)
- **`triplet_mean`** (default): averages over all valid triplets per LF (recommended)
- **`triplet_median`**: takes the median over all valid triplets (most robust to outliers)

## Label Matrix Convention

The label matrix `L_train` is an `(n_samples, m_lfs)` numpy array with values:
- **+1**: LF votes positive (species present)
- **-1**: LF votes negative (species absent)
- **0**: LF abstains (no opinion)

This is the standard weak supervision convention. When `allow_abstentions=True`
(the default), FlyingSquid correctly handles sparse LFs that only vote on a subset
of data points.

## Relevance to Pipeline

FlyingSquid is the label fusion engine for **experiment 013 (weak supervision)**:

- **Phase 2 of exp 013**: For each of the 206 BirdCLEF species, construct a label
  matrix from 10 labeling functions across 4 independent error modes (neural probes,
  metadata, acoustic events, spectral indices), then run FlyingSquid per species to
  produce calibrated soft labels at 1-10 Hz resolution.
- **Downstream consumers**: The fused soft labels feed into exp 009 (SSL pretraining)
  and exp 012 (EfficientNet finetuning) as training targets, replacing noisy
  single-teacher labels with denoised probabilistic labels.
- **LF accuracy diagnostics**: `estimated_accuracies()` reveals which LFs are
  informative per species, guiding LF development and debugging.

## Usage

### Basic: Fit and Predict

```python
from flyingsquid.label_model import LabelModel
import numpy as np

# L_train: (n_samples, m_lfs) array of {-1, 0, +1}
# Minimum 3 LFs required for the triplet method
L_train = np.load("label_matrix.npy")

m = L_train.shape[1]  # number of labeling functions
label_model = LabelModel(m)

# Fit: learns LF accuracies from LF agreement patterns (no ground truth needed)
label_model.fit(L_train, solve_method="triplet_mean")

# Predict soft labels: P(Y=+1) for each sample
soft_labels = label_model.predict_proba(L_train)  # (n, 2) array: [P(Y=-1), P(Y=+1)]
proba_positive = soft_labels[:, 1]                # just P(Y=+1)

# Or get marginalized probabilities directly (single task):
proba = label_model.predict_proba_marginalized(L_train)  # (n,) array of P(Y=+1)

# Hard predictions: {-1, +1}
hard_preds = label_model.predict(L_train)  # (n, 1) array

# Inspect estimated LF accuracies: P(lambda_i == Y)
accuracies = label_model.estimated_accuracies()  # list of m floats
```

### Per-Species Binary Fusion (exp 013 pattern)

```python
from flyingsquid.label_model import LabelModel
import numpy as np

def fuse_species_labels(label_matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run FlyingSquid independently per species.

    Args:
        label_matrices: {species_code: (n_windows, m_lfs) array of {-1, 0, +1}}

    Returns:
        {species_code: (n_windows,) array of P(species_present)}
    """
    soft_labels = {}
    for species, L in label_matrices.items():
        m = L.shape[1]
        if m < 3:
            raise ValueError(f"{species}: need >= 3 LFs, got {m}")

        lm = LabelModel(m)
        lm.fit(L, solve_method="triplet_mean")
        soft_labels[species] = lm.predict_proba_marginalized(L)

    return soft_labels
```

### With Class Balance Prior

```python
# If you have a dev set with ground-truth labels:
label_model.fit(L_train, Y_dev=Y_dev)  # Y_dev: (n_dev,) array of {-1, +1}

# Or specify class balance directly (for rare species):
# [P(Y=-1), P(Y=+1)] -- e.g., 95% negative, 5% positive
label_model.fit(L_train, class_balance=np.array([0.95, 0.05]))
```

## Dependencies

Core (required):
- **pgmpy**: Probabilistic graphical models -- Markov model, junction tree, discrete factors
- **numpy**: Array operations, moment computation
- **tqdm**: Progress bars for verbose mode
- **networkx** (>=2.2): Graph operations (transitive dependency via pgmpy)

Optional:
- **PyTorch** (>=1.0): Only for `FSLoss` / `MajorityVoteLoss` end-model integration
- **scipy**, **scikit-learn**: Used in examples/tutorials, not in core

Note: The package targets Python 3.7 in its environment.yml, but the core code uses
no version-specific features and works on Python 3.11.

## Notes

- **Minimum 3 LFs**: The triplet method requires at least 3 labeling functions.
  `LabelModel(m)` raises `NotImplementedError` if `m < 3`.

- **Conditional independence assumption**: LFs must be conditionally independent given
  Y for the triplet method to produce correct accuracy estimates. In exp 013, we
  ensure this by drawing LFs from 4 independent error modes (neural, metadata,
  acoustic events, spectral indices). LFs within the same error mode (e.g., BirdNET
  probe and Perch probe) may be correlated, which the `lambda_edges` parameter can
  model if needed.

- **LF dependencies**: For correlated LFs, pass `lambda_edges=[(i, j)]` to
  `LabelModel()` to add edges in the graphical model. This changes the junction tree
  structure and the triplets that can be formed, but may reduce the number of valid
  triplets available.

- **`triplet_mean` vs `triplet_median`**: The default `triplet_mean` averages all
  valid triplet estimates for each LF accuracy. Use `triplet_median` when LFs have
  very different coverage rates (high abstention variance), as outlier triplets can
  skew the mean.

- **Negative probabilities**: The linear system solve can produce negative marginal
  probabilities when LFs are noisy. By default, `flip_negative=True` flips the sign
  and renormalizes. Set `clamp=True, flip_negative=False` to clamp to zero instead.

- **FSLoss bug**: The `pytorch_loss.py` file has `super(WSLoss, self).__init__()`
  but the class is named `FSLoss` -- this will raise `NameError` at runtime. This
  module is not used in our pipeline (we run FlyingSquid offline, not as a training
  loss), so this is not a blocker.

- **Scaling**: For the fully independent case (no `lambda_edges`), the moment
  computation uses vectorized `np.einsum` for means and covariances, making it fast
  even for large `n_samples`. The triplet enumeration is O(m^3) but with m=10 LFs
  this is negligible.

- **Paper**: Fu et al., "Fast and Three-rious: Speeding Up Weak Supervision with
  Triplet Methods", ICML 2020. [arXiv:2002.11955](https://arxiv.org/abs/2002.11955)
