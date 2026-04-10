# Technical Design: Exp 107 Implementation

## Scope

This document specifies the module-level design for exp 107: file layout,
module responsibilities, dependencies, and data flow. No code snippets —
see `design.md` for the model API and algorithm details.

## Finalized architecture decisions

Before reading the details, here are the decisions that are locked in:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input features | **Embeddings only** (1536-d) | Simpler baseline; logits deferred as future ablation |
| Memory bank count | **Two** (species + site) | Matches star schema factored structure |
| Species head normalization | **Sigmoid** (max-over-K) | Multi-label: independent per-class probabilities |
| Site head normalization | **Sigmoid** (max-over-K) | Consistency with species head; single-scalar gate |
| Multi-head attention | **No** | One projection is sufficient; multi-head adds complexity for a 2-source problem |
| Iterative Hopfield refinement | **No** | One-shot retrieval; iteration collapses multi-label signal |
| Pattern separation regularizer | **No** | Start simple; add only if prototypes collapse in training |
| Library dependency | **None** (plain PyTorch) | hflayers doesn't support sigmoid; einsum is 5 lines |
| Hopfield framing | **Yes** in docs, not in code | Intellectual lineage preserved; implementation stays simple |

Deferred to future work (explicitly out of scope for exp 107):

- Dual-space (logits + embeddings) with parallel banks
- Per-species site affinity (vs the current scalar site gate)
- Iterative Hopfield refinement
- Transformer-style multi-head attention
- Pattern separation or coherence regularizers
- Calibration (Platt scaling) on output scores

## Directory structure

```
user/anthony/experiments/107-hopfield-prototypes/
├── proposal.md                          # hypothesis, method, success criteria
├── design.md                            # model API, training procedure
├── docs/
│   ├── star-schema-dimensional-analysis.md
│   └── technical-design.md              ← this file
├── pyproject.toml                       # declares dependency on birdclef_106
├── src/birdclef_107/
│   ├── __init__.py
│   ├── cli.py                           # entry point: train, evaluate, ablate
│   ├── data.py                          # datasets + multi-domain batch sampler
│   ├── model.py                         # HopfieldPrototypeModel
│   ├── losses.py                        # three-source loss composition
│   ├── init.py                          # prototype + projection initialization
│   ├── train.py                         # training loop with phased schedule
│   └── evaluate.py                      # LOSO cross-validation
├── configs/
│   ├── base.yaml                        # default hyperparameters
│   ├── ablation_a.yaml                  # species only (no focal, no site, no unlabeled)
│   ├── ablation_b.yaml                  # + focal
│   ├── ablation_c.yaml                  # + site bank
│   └── ablation_d.yaml                  # + unlabeled (full model)
├── sbatch/
│   ├── _setup-venv.sh                   # reuses exp 106 venv, adds torch
│   └── train.sbatch                     # single training + LOSO run
└── artifacts/                           # outputs (gitignored)
    ├── checkpoints/
    ├── metrics/
    └── loso_results.json
```

## Dependencies

### External

| Package | Used for |
|---------|----------|
| `torch` (CPU) | Model, training loop, gradient computation |
| `numpy`, `polars` | Data loading (inherited from exp 106) |
| `scikit-learn` | Metrics (roc_auc_score), PCA init |
| `pyyaml` | Config loading |

### Internal

Exp 107 imports from the exp 106 package (declared in `pyproject.toml`):

- `birdclef_106.data_loading` — cached parquet loaders, label matrix builder,
  site extraction, species list construction
- Artifacts consumed (not imported):
  - `multi_prototypes.npz` — for memory bank initialization
  - `site_prototypes.npz` — for site memory initialization
  - `perch_species_mapping.json` — for species ordering

No code is copied from exp 106. The exp 106 infrastructure is reused as-is.

## Module responsibilities

### `data.py`

- `PrototypeDataset`: wraps a single data source (focal, labeled_sc, or
  unlabeled_sc) as a PyTorch `Dataset`. Handles:
  - Loading embeddings from exp 106 parquets
  - Converting labels (one-hot for focal, multi-hot for labeled_sc, None for unlabeled)
  - Extracting site IDs (or sentinel value for focal)
- `MultiDomainBatchSampler`: yields training batches that combine fixed
  counts from each source per step. Handles:
  - Per-species sampling for focal (to balance class imbalance)
  - Per-site sampling for unlabeled (I/O efficient across 21 parquets)
  - With-replacement sampling for labeled (only 739 windows)

### `model.py`

- `HopfieldPrototypeModel`: the full model. Architecturally a **modern Hopfield
  network** (Ramsauer et al. 2020) with two memory banks (species, site) and
  a shared query projection. The species head uses **sigmoid attention** for
  multi-label retrieval; the site head uses **softmax attention** for
  single-winner state selection. Implemented with plain PyTorch primitives
  (einsum + sigmoid/softmax) — no library dependency, but the computation is
  the same as the hflayers package would perform, just with the
  sigmoid/softmax choice made per head.

  Exposes:
  - `forward(emb, site_idx)` — main entry point, handles both focal (no site)
    and soundscape (with site) windows
  - `species_head(q)` — multi-label Hopfield retrieval:
    `σ(β_s · q · M_s)` max-pooled over K prototypes per species
  - `site_head(q, site_idx)` — single-winner Hopfield retrieval:
    `softmax(β_l · q · M_l[site])` aggregated to a scalar match score
  - Learnable components: projection, species memory, site memory, log-betas, gate

  The Hopfield connection: each memory bank is a set of stored patterns in a
  D-dimensional space. Retrieval is one-shot attention (no iterative dynamics
  — the multi-label setting prefers independent per-class scoring over
  convergence to a single attractor). β parameters control the sharpness of
  the retrieval (inverse temperature in the Hopfield energy function).

### `losses.py`

- `LossConfig`: dataclass with phase-dependent loss weights and enable flags
- `compute_loss(model, batch, config)`: computes the three-source loss
  (focal BCE + labeled SC BCE + unlabeled entropy) and returns components
  for logging

### `init.py`

- `init_from_exp_106(model, artifacts_path)`: initializes species and site
  memories from exp 106 prototypes, projected through the current `proj`
  layer weights
- `init_proj_from_pca(model, embeddings)`: optional PCA initialization of
  the projection layer for faster convergence

### `train.py`

- `TrainConfig`: dataclass with hyperparameters and phase boundaries
- `train(model, sampler, config, eval_fn)`: training loop with:
  - Phased schedule (phase 1: freeze site bank; phase 2: unfreeze; phase 3: add unlabeled loss)
  - Cosine learning rate decay
  - Gradient clipping
  - Periodic evaluation via `eval_fn` callback
  - Best-checkpoint tracking

### `evaluate.py`

- `loso_evaluate(model_factory, config, datasets)`: leave-one-site-out CV
  - For each site with ≥20 labeled windows: split labeled data, re-init model,
    train, evaluate on held-out site, record per-fold AUROC
  - Returns per-fold and per-species AUROC summaries
- `per_species_auroc(model, dataset)`: helper for per-species breakdown

### `cli.py`

- Subcommands: `train` (single run), `loso` (full LOSO), `ablate` (runs all
  ablation configs sequentially)
- Config loading from YAML, output paths, logging setup

## Configuration

Each ablation is a separate YAML file differing in feature flags:

| Config | species bank | site bank | focal source | unlabeled source |
|--------|-------------|-----------|--------------|------------------|
| ablation_a | ✓ | ✗ | ✗ | ✗ |
| ablation_b | ✓ | ✗ | ✓ | ✗ |
| ablation_c | ✓ | ✓ | ✓ | ✗ |
| ablation_d (full) | ✓ | ✓ | ✓ | ✓ |

`base.yaml` is ablation_d with default hyperparameters. Each ablation
inherits from base and overrides flags.

## Data flow

```
Exp 106 artifacts on scratch
    │
    │  (birdclef_106.data_loading)
    ▼
data.py: PrototypeDataset × 3
    │
    ▼
data.py: MultiDomainBatchSampler
    │
    ▼
train.py: training loop
    ├── model.py: HopfieldPrototypeModel.forward
    └── losses.py: compute_loss
    │
    ▼
evaluate.py: loso_evaluate
    │
    ▼
artifacts/loso_results.json + checkpoints/
```

## Testing plan

Unit tests (run before any training):

1. **Shape tests**: model.forward produces correct output shape for focal
   and soundscape inputs
2. **Gradient tests**: all learnable parameters receive gradients for
   appropriate inputs (site memory only for soundscape, etc.)
3. **Freeze tests**: phase 1 freezing correctly zeros gradients on site bank
4. **Init tests**: loading exp 106 prototypes produces finite, correctly
   shaped memory tensors
5. **Overfit test**: model can drive a single batch's loss to near-zero
   in 100 steps (sanity check that learning works)
6. **Parity test**: species-only ablation (ablation_a) produces LOSO AUROC
   within 0.02 of exp 106's ridge baseline (0.749)

Tests 1–5 are seconds each. Test 6 is the meaningful sanity check — if
a linear model can't match ridge, something is broken.

## SLURM setup

- Partition: `cpu-small` (GPU optional but not required)
- Resources: 4 CPUs × 8 GB RAM, 3h walltime (conservative for LOSO + ablations)
- Venv: reuses `~/scratch/birdclef-2026/.venv-106` with added torch (CPU build)
- Spawn multiprocessing fix from exp 106 not needed — we're not using Luigi

## Hopfield framing

The model is a **modern Hopfield network** (Ramsauer et al. 2020) specialized
for multi-label classification. The lineage:

```
Classical Hopfield (1982)
    └─→ discrete patterns, local update rule, polynomial capacity
        │
        ▼
Modern Hopfield (Ramsauer et al. 2020)
    └─→ continuous patterns, exponential capacity,
        update rule = softmax(β · q · M^T) · M
        = scaled dot-product attention
        │
        ▼
Exp 107: multi-memory-bank Hopfield
    ├─→ species bank: sigmoid attention (multi-label retrieval)
    ├─→ site bank: softmax attention (single-winner state selection)
    └─→ gated fusion: learned combination
```

**Why sigmoid instead of softmax for species:**

The classical Hopfield update rule uses softmax because it's designed to
retrieve ONE stored pattern (the nearest attractor). For multi-label output,
we want independent per-class probabilities — a window can contain species
A AND species B AND species C simultaneously. Sigmoid gives this; softmax
forces the attention weights to sum to 1 (normalized blend).

Mathematically, the species head computes:

    score_s = σ(β_s · q · m_s)   for each species s independently

This is equivalent to the Hopfield energy gradient with sigmoid output
instead of softmax. The energy landscape still has attractors near each
stored pattern, but multiple attractors can be simultaneously "high" without
competing.

**Site head: also sigmoid (for simplicity and consistency):**

Although site state selection is conceptually a single-winner problem (a
window is in one acoustic state at its site), we use the same max-over-K +
sigmoid pattern as the species head for implementation consistency. The
site head produces a scalar per window representing "how well does this
window match any of its site's prototypes" — interpretable as a typicality
score. This scalar gates the species predictions via the learned gate.

Strictly speaking, a softmax-attention site head would be more Hopfield-pure
(retrieving a weighted site state and measuring its similarity to the query).
We pick sigmoid for consistency and to avoid the coupling issues softmax
introduces when combining with the species head. If ablations show the site
head is ineffective, we can revisit with softmax attention.

**No library dependency:**

We use plain PyTorch primitives (`nn.Linear`, `nn.Parameter`, `torch.einsum`,
`torch.sigmoid`, `torch.softmax`) rather than the `hflayers` package. The
computation is identical to what `hflayers.Hopfield` would perform with
`scaling=beta, normalization="softmax"` (for the site head) except:

1. The species head uses sigmoid instead of softmax (not supported by hflayers).
2. We don't iterate the update rule — one-shot retrieval is sufficient for
   classification (iteration is useful for memory recall, not discrimination).
3. We keep the projection layer separate from the attention computation for
   clarity and to share it across both memory banks.

The design is consistent with the Hopfield literature and can be described
as such in publications and documentation without introducing a library
dependency that would force design compromises.

## Open decisions

1. **Shared vs per-domain projection**: Start with one `proj` layer for both
   focal and soundscape. Revisit if L_focal plateaus while L_sup improves.
2. **Memory L2 normalization**: Optional projection step after each gradient
   update to constrain memory vectors to the unit sphere. Makes beta
   interpretable but may slow learning. Start without it.
3. **PCA init**: Couples init to embedding distribution but gives meaningful
   starting directions. Try both random and PCA init in early runs.
4. **Focal loss weight α**: Starts at 0.5. Tune based on relative L_focal
   and L_sup magnitudes in early training.
