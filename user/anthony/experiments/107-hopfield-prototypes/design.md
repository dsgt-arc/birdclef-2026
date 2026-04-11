# Design: Prototype Retrieval for Amphibian-Dominated Soundscapes

## Overview

A PyTorch memory-bank classifier over frozen Perch v2 embeddings.
Species and site prototypes live in a learned 512-d projection space;
species retrieval is trained via BCE on strong + weak focal labels;
site retrieval is trained via **Hopfield energy minimization** on 127K
unlabeled soundscape windows. Both banks compose at inference via
gated fusion on per-species Platt-calibrated probabilities.

See `proposal.md` for the thesis and success criteria, and
`docs/star-schema-dimensional-analysis.md` for the tensor framework,
the soft-EM gradient identity, and the formal derivation of the
unsupervised site clustering loss from the Hopfield retrieval energy.

## Locked architecture decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input features | Perch v2 pooled embeddings (1536-d, frozen) | Cached from exp 106; a strong general bioacoustic feature space |
| Output dim | 234 (no silence class) | Axiomatic simplification — bag labels + K=8 prototype slots + family hierarchy absorb silent focal noise |
| Memory bank count | Two (species + site), star schema factored | Eval-time affinity matrix is a deliverable |
| Species bank parameterization | Hierarchical: `M_species[s] = M_family[f(s)] + Δ_species[s]` | Low-resource species inherit family prior under weight decay; high-resource specialize |
| Shared space dim | D = 512 | Enough capacity to disentangle focal ↔ soundscape manifolds |
| Projection architecture | `Linear(1536, 1024) → GELU → Linear(1024, 512) → LayerNorm` | Nonlinear; shaped by labeled BCE gradient (projection is detached from the unsupervised site loss) |
| Retrieval projections | Explicit `W_Q`, `W_K_species`, `W_K_site` (all `(512, 512)`) | Ramsauer formulation of modern Hopfield networks |
| Prototype slots | K_s = K_l = 8 | Enough for intra-species acoustic modes (song/call, dawn/night, chorus) without overparameterizing |
| Retrieval aggregation | `(1/β)·logsumexp_k(β · q_r · W_K · M_{*,k})`, learnable `log_β` per head | Modern Hopfield energy — all prototypes receive softmax-weighted gradient |
| Per-species calibration | Built-in Platt `sigmoid(a_s · score + b_s)` | Load-bearing for Bayesian gated fusion (`species · site_score` needs calibrated probabilities) |
| Gated fusion | `g · species + (1 - g) · species · site_score`, `g = σ(Linear(q))` | Per-window mixing between species-only and species × site |
| Loss composition | `L_labeled + λ_site · L_site_unsup` | Two terms, one novel mechanism |
| Projection gradient on unsupervised loss | **Detached** | Prevents the projection from encoding site-identifying artifacts |
| Initialization | Random (no k-means) | Soft-EM gradient identity justifies BCE+Hopfield convergence without warm start |
| Tensor operations | einops `einsum`, `reduce`, `rearrange` | Project convention; self-documenting contractions |

## Built-in Platt scaling

The species head applies per-species affine calibration before the
sigmoid:

    score_s = σ(a_s · (1/β_s) · logsumexp_k(β_s · q_r · W_K · M_{s,k}) + b_s)

where `a_s` (scale) and `b_s` (bias) are learnable per-species parameters
initialized to identity. Total: 2 × 234 = 468 params.

Two reasons calibration belongs in the model:

1. **Multi-label composition needs calibrated probabilities.** The
   gated fusion multiplies `species · site_score`. This is only a
   valid Bayesian composition ("probability species s is present and
   compatible with the site") when both factors are proper
   probabilities. Without per-species Platt, `species · site_score` is
   a product of arbitrary-scale sigmoid outputs; systematic per-species
   offsets break the composition.
2. **Cleaner BCE gradients.** Uncalibrated scores produce biased BCE
   gradients; per-class scale + bias absorbs systematic offsets that
   would otherwise be learned into the projection.

The site head doesn't need per-site Platt — it's a single scalar per
window, and the gate handles systematic bias.

## Model

```python
from einops import einsum, rearrange

class PrototypeRetrievalModel(nn.Module):
    """
    Hierarchical species (family + delta) and site memory banks with
    modern-Hopfield logsumexp retrieval over a learned 512-d projection
    of Perch v2 embeddings. Per-species Platt calibration on the species
    head, gated fusion with a site-typicality scalar.
    """

    def __init__(self, n_species=234, n_families=30, n_sites=21,
                 K_species=8, K_site=8,
                 d_input=1536, d_hidden=1024, d_proj=512,
                 family_of=None):
        super().__init__()
        assert family_of is not None and family_of.shape == (n_species,)
        self.register_buffer("family_of", family_of.long())

        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
            nn.LayerNorm(d_proj),
        )

        self.W_Q = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_species = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_site = nn.Linear(d_proj, d_proj, bias=False)

        # Hierarchical species memory
        self.family_memory = nn.Parameter(
            torch.zeros(n_families, K_species, d_proj))
        self.species_delta = nn.Parameter(
            torch.zeros(n_species, K_species, d_proj))

        # Flat site memory (21 sites, no taxonomic hierarchy)
        self.site_memory = nn.Parameter(
            torch.zeros(n_sites, K_site, d_proj))

        # Learnable Hopfield temperatures
        self.log_beta_species = nn.Parameter(torch.zeros(()))
        self.log_beta_site = nn.Parameter(torch.zeros(()))

        # Per-species Platt (identity init)
        self.cal_scale = nn.Parameter(torch.ones(n_species))
        self.cal_bias = nn.Parameter(torch.zeros(n_species))

        self.gate = nn.Linear(d_proj, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.family_memory, std=0.02)
        nn.init.zeros_(self.species_delta)
        nn.init.normal_(self.site_memory, std=0.02)

    def effective_species_memory(self):
        """M_species[s, k] = M_family[family_of(s), k] + Δ_species[s, k]."""
        return self.family_memory[self.family_of] + self.species_delta

    def project(self, emb):
        return self.proj(emb)

    def forward(self, emb, site_idx=None):
        """
        Args:
            emb: (B, 1536) Perch embeddings
            site_idx: (B,) int site indices, or None for focal windows

        Returns:
            scores: (B, 234) calibrated multi-label probabilities
        """
        q = self.proj(emb)
        q_r = self.W_Q(q)

        # Hierarchical species retrieval
        beta_s = self.log_beta_species.exp()
        M_s = self.W_K_species(self.effective_species_memory())           # (s k d)
        sp_sim = einsum(q_r, M_s, 'b d, s k d -> b s k')                  # (b s k)
        sp_agg = torch.logsumexp(beta_s * sp_sim, dim=-1) / beta_s        # (b s)
        sp_score = torch.sigmoid(self.cal_scale * sp_agg + self.cal_bias)

        if site_idx is None:
            return sp_score

        # Site retrieval → per-window typicality scalar
        beta_l = self.log_beta_site.exp()
        M_l = self.W_K_site(self.site_memory[site_idx])                   # (b k d)
        site_sim = einsum(q_r, M_l, 'b d, b k d -> b k')                  # (b k)
        site_agg = torch.logsumexp(beta_l * site_sim, dim=-1) / beta_l    # (b,)
        site_score = torch.sigmoid(site_agg)                              # (b,)

        # Gated fusion
        g = torch.sigmoid(self.gate(q)).squeeze(-1)                       # (b,)
        return (g * sp_score
                + (1 - g) * sp_score * rearrange(site_score, 'b -> b 1'))

    def unsupervised_site_energy(self, emb, site_idx):
        """
        Hopfield energy minimization on unlabeled soundscape windows.

        Projection is DETACHED — this loss trains only site_memory,
        W_K_site, and log_beta_site. It does NOT update the projection,
        which prevents the site bank from pulling the projection toward
        site-identifying artifacts (exp 106's 'site entangled with
        domain shift' failure mode).

        Returns a scalar loss: the mean negative log-sum-exp retrieval
        energy, which pulls each unlabeled SC window toward SOME
        prototype in its known site's memory bank via soft-EM clustering.
        """
        q_det = self.proj(emb).detach()
        q_r = self.W_Q(q_det)
        beta_l = self.log_beta_site.exp()
        M_l = self.W_K_site(self.site_memory[site_idx])                   # (b k d)
        sim = einsum(q_r, M_l, 'b d, b k d -> b k')
        energy = -(torch.logsumexp(beta_l * sim, dim=-1) / beta_l)        # (b,)
        return energy.mean()
```

### Weight decay asymmetry

Weight decay on `species_delta` is the "fall back to family prior"
mechanism. Low-resource species can't generate enough BCE gradient to
counteract the strong shrinkage and stay near `Δ = 0`; high-resource
species overcome it and specialize.

```python
param_groups = [
    {"params": [model.species_delta], "weight_decay": 1e-2},    # strong shrinkage
    {"params": [p for n, p in model.named_parameters()
                if n != "species_delta"], "weight_decay": 1e-5},
]
optimizer = torch.optim.AdamW(param_groups, lr=3e-4)
```

### Eval-time species × site affinity matrix

Computed outside `forward` for interpretation / reporting:

```python
beta = model.log_beta_species.exp()
M_species = model.effective_species_memory()                             # (s ks d)
M_s = model.W_K_species(M_species)                                       # (s ks d)
M_l = model.W_K_site(model.site_memory)                                  # (l kl d)
pair = einsum(M_s, M_l, 's ks d, l kl d -> s l ks kl')
affinity = rearrange(pair, 's l ks kl -> s l (ks kl)')
A = torch.logsumexp(beta * affinity, dim=-1) / beta                      # (s l)
```

## Label preparation

One-time preprocessing pass producing per-window labels for the three
streams. Far simpler than previous iterations — no Perch filter, no
tri-state, no silence detection, no RMS thresholds.

### Inputs (from exp 106)

- `~/scratch/birdclef-2026/features/exp106/focal/by_species/*.parquet` —
  cached Perch v2 embeddings per species
- `~/scratch/birdclef-2026/features/exp106/soundscape_labeled.parquet` —
  739 labeled SC windows
- `~/scratch/birdclef-2026/features/exp106/soundscape_unlabeled/by_site/*.parquet` —
  127K unlabeled SC embeddings per site
- **Species → family mapping** (from BirdCLEF metadata or iNat/eBird
  fallback CSV) — verified before training starts

### Processing

```python
def prepare_labels(focal_pq, labeled_sc_pq, unlabeled_sc_pq,
                   species_to_family):
    # Strong: labeled SC, 234-d window-level ground truth
    strong = labeled_sc_pq.select(
        "window_id", "emb", "site_id", "labels_234"
    )

    # Weak focal: every focal window with its recording's primary-species
    # label. No filter, no silence relabeling. Same rule for all regimes
    # (bird/non-bird, in-Perch/not-in-Perch, common/rare).
    focal_labels = one_hot(focal_pq["species_id"], dim=234)
    weak_focal = focal_pq.with_columns(
        pl.Series("labels_234", focal_labels),
        pl.col("species_id").map_dict(species_to_family).alias("family_id"),
    )

    # Unsupervised: all 127K unlabeled SC windows with only site_idx.
    # No species labels. These flow through L_site_unsup during training.
    unsupervised = unlabeled_sc_pq.select("window_id", "emb", "site_id")

    return {
        "strong": strong,
        "weak_focal": weak_focal,
        "unsupervised": unsupervised,
    }
```

### Preflight statistics computed in the same pass

| Stat | Use |
|------|-----|
| Per-species focal window count | Informs per-species-balanced sampler weights |
| Per-family species count | Sanity check for family hierarchy gradient balance |
| Species without focal + without 739 positives + without family sibling with focal | Unlearnable species list (excluded from aggregate AUROC) |
| Perch top-1 direct AUROC on labeled SC | Reference baseline for the sanity-check section |

## Training

### Batch construction

Three streams, sampled independently per step. The weak focal stream
uses **per-species balanced sampling** (`WeightedRandomSampler`);
strong and unsupervised use uniform sampling.

```python
# Build weak focal sampler once
weak_counts = weak_focal_df.groupby("species_id").size()
weak_weights = [1.0 / weak_counts[row.species_id] for row in weak_focal_df]
weak_sampler = WeightedRandomSampler(
    weights=weak_weights,
    num_samples=len(weak_focal_df),
    replacement=True,
)

batch = {
    "strong":     sample(strong_windows, n=32),              # 739 cycled
    "weak_focal": sample_with(weak_sampler, n=256),          # ~265K, balanced
    "unsup":      sample(unsup_windows, n=256),              # 127K, uniform
}
```

### Loss

```python
def compute_loss(model, batch, lambda_site=0.1):
    # ----- Labeled term: strong + weak focal, single BCE call -----
    # Strong has site_idx (labeled SC); weak focal has site_idx=None.
    # Forward pass routes them to the right branches internally.
    sp_strong = model(batch.strong.emb, batch.strong.site_idx)      # (32, 234)
    sp_weak   = model(batch.weak_focal.emb, site_idx=None)          # (256, 234)

    L_strong = F.binary_cross_entropy(sp_strong, batch.strong.labels)
    L_weak   = F.binary_cross_entropy(sp_weak,   batch.weak_focal.labels)
    L_labeled = L_strong + L_weak

    # ----- Unsupervised term: Hopfield energy on unlabeled SC -----
    L_site_unsup = model.unsupervised_site_energy(
        batch.unsup.emb, batch.unsup.site_idx
    )

    return L_labeled + lambda_site * L_site_unsup
```

Single hyperparameter: `lambda_site`. Default 0.1. No ramp, no warm-up,
no `gamma_weak`.

### Schedule

- AdamW, `lr = 3e-4`, cosine decay
- 30–40 epochs
- Early stopping on LOSO AUROC evaluated every 5 epochs
- Weight decay: `species_delta` → 1e-2, everything else → 1e-5

## LOSO evaluation

7 folds (sites with ≥20 labeled SC windows). Each fold:

1. Split strong stream: train on labeled SC from all sites except the
   held-out site; evaluate on the held-out site's labeled SC
2. Reset full model, re-run random init
3. Train: 30–40 epochs of the two-term loss, early stopping on held-out
   AUROC every 5 epochs
4. Evaluate: macro AUROC on held-out labeled SC, with subset breakdowns

Final metric: mean of per-fold AUROC. Reported alongside per-fold
variance (so go/no-go against the 0.749 baseline is statistically
meaningful given LOSO's high fold-to-fold variability).

Subset breakdowns:
- Overall (primary headline)
- Bird vs non-bird
- With-focal vs without-focal
- Has-family-sibling-with-focal vs orphaned

Unlearnable species (no focal + no 739 positives + no family sibling
with focal) excluded from aggregate, listed separately in the report.

Leakage audit: unlabeled SC is shared across folds (no labels to
leak), focal data is site-less. Standard semi-supervised LOSO.

## Kaggle submission path

1. **LOSO validation** on 7 folds → confirm architecture (overall ≥ 0.749)
2. **Final training on all data** (focal + 739 labeled SC + all
   unlabeled SC, no fold hold-out), LOSO-best hyperparameters, single
   GPU run ~1h
3. **Package**: exp 107 checkpoint (~4.1M params) + Perch v2 ONNX
   (reused from exp 102/105 Kaggle package) + preprocessing scripts
4. **Inference**: Perch v2 → 1536-d embedding per 5-sec window → exp
   107 model → 234-d multi-label probabilities. No silence class to
   strip (we don't have one).
5. **Standalone LB submission**
6. **Ensemble with exp 102** best checkpoint: probability-space
   average, blend weight tuned on LOSO against the 739 labels (not
   on LB — LB overfitting risk). Optional per-species-group weighting:
   higher weight on non-bird species for exp 107, higher on birds for
   exp 102.

### Submission checklist

- [ ] LOSO primary validates architecture (≥ 0.749 overall)
- [ ] Final retraining on all data
- [ ] Model export (PyTorch checkpoint; ONNX only if einops/logsumexp
      convert cleanly, else raw `state_dict`)
- [ ] Kaggle package assembly
- [ ] Standalone LB submission
- [ ] Ensemble LB submission

## Directory structure

```
user/anthony/experiments/107-hopfield-prototypes/
├── proposal.md                          # thesis, method, success criteria
├── design.md                            # ← this file
├── docs/
│   └── star-schema-dimensional-analysis.md  # tensor framework + Hopfield formalism
├── pyproject.toml                       # declares dependency on birdclef_106
├── src/birdclef_107/
│   ├── __init__.py
│   ├── cli.py                           # entry: prepare-labels, train, evaluate, submit
│   ├── label_prep.py                    # one-time preprocessing pass
│   ├── data.py                          # datasets + 3-stream sampler (weak focal balanced)
│   ├── model.py                         # PrototypeRetrievalModel
│   ├── losses.py                        # L_labeled + λ_site · L_site_unsup
│   ├── train.py                         # one-phase training loop
│   ├── evaluate.py                      # LOSO cross-validation + subset metrics
│   └── kaggle_export.py                 # final retraining + submission packaging
├── configs/
│   └── base.yaml                        # default hyperparameters (one config in v1)
├── sbatch/
│   ├── _setup-venv.sh                   # reuses exp 106 venv + torch + einops
│   ├── prepare_labels.sbatch            # label prep (CPU)
│   ├── train_loso.sbatch                # LOSO training (GPU)
│   └── train_final.sbatch               # full-data retrain for Kaggle (GPU)
└── artifacts/                           # gitignored
    ├── labels/                          # per-window label parquets
    ├── checkpoints/
    │   ├── fold_{i}/best.pt
    │   └── final_full.pt                # for Kaggle submission
    ├── metrics/
    │   └── loso_results.json            # overall + subset breakdowns
    └── affinity/
        └── species_site_matrix.npz      # (234, 21) eval-time affinity
```

## Dependencies

**External**

| Package | Used for |
|---------|----------|
| `torch` | Model, training loop, GPU (RTX 6000 recommended) |
| `einops` | `einsum`, `rearrange` — memory-bank contractions |
| `polars` | Cached parquet IO (inherited from exp 106) |
| `numpy`, `scikit-learn` | Metrics (`roc_auc_score`) |
| `pyyaml` | Config loading |

**Cached artifacts consumed from exp 106**

| Artifact | Used for |
|----------|----------|
| Perch v2 embeddings (265K focal + 739 labeled + 127K unlabeled) | Model input |
| Species → family mapping | Hierarchical species bank |

Note: Perch logits are **not** consumed in v1 (no distillation, no
pseudo-labeling, no Perch-based filtering). Only the pooled embeddings
are used.

**Internal**

Exp 107 imports from the exp 106 package (declared in `pyproject.toml`):
- `birdclef_106.data_loading` — cached parquet loaders, LOSO utilities

No prototype transfer from exp 106 — memory banks are randomly
initialized and trained end-to-end.

## Testing plan

Four tests, run before any LOSO job:

1. **Shape test**: `model(emb, site_idx)` returns `(B, 234)` for both
   soundscape (int site_idx) and focal (`site_idx=None`) inputs.
   `model.unsupervised_site_energy(emb, site_idx)` returns a scalar.
2. **Gradient routing test**: during `L_site_unsup`, `model.proj`
   parameters receive **zero** gradient (projection is detached) while
   `site_memory`, `W_K_site`, and `log_beta_site` all receive non-zero
   gradient.
3. **Affinity correctness**: `A(s, l)` computed via einsum matches a
   brute-force quadruple loop over `(s, l, k_s, k_l)` on a small
   synthetic model.
4. **Overfit-on-one-batch**: full model drives a single 32-window
   labeled BCE loss to near-zero in 100 steps. Confirms the labeled
   path is wired correctly end-to-end.

## SLURM setup

- **Label prep**: one-time CPU job, ~30min, `cpu-small`, 4 CPUs × 8 GB
- **LOSO training**: GPU job, `gpu-rtx6000`, 1× RTX 6000, 16 GB RAM,
  12h walltime. One job runs all 7 folds sequentially.
- **Final retraining for Kaggle**: separate GPU job, ~1h walltime.
- Venv: reuses `~/scratch/birdclef-2026/.venv-106` with torch + einops
  added. Per `CLAUDE.md`, never install `.venv` under `$HOME`.

## Connection to exp 106

Reused infrastructure:
- Cached Perch v2 embeddings (739 + 127K + 265K)
- Partitioned parquets (`by_species/`, `by_site/`)
- LOSO fold utilities
- Species mapping

**Not inherited**: exp 106's `multi_prototypes.npz` and
`site_prototypes.npz`. Exp 107's memory banks are initialized from
random Gaussian noise and trained end-to-end in the learned projection
space; exp 106's prototypes live in Perch's fixed logit space and
aren't dimensionally compatible.

Exp 107 adds:
- Label preparation pass (bag-label focal + unsupervised SC preparation)
- PyTorch memory-bank model with hierarchical species parameterization + unsupervised site Hopfield energy loss (~200 LOC)
- One-phase training loop with species-balanced sampling (~120 LOC)
- LOSO cross-validation wrapper with subset metrics (~80 LOC)
- Kaggle packaging + ensemble blend (~100 LOC)

## Open decisions

1. **`lambda_site` default**: 0.1 is a guess. Tune on LOSO if the
   unsupervised site loss either dominates training (too high) or
   produces no visible effect on the site bank's prototype diversity
   (too low).
2. **Weight decay on `species_delta`**: 1e-2 is a guess for the
   "collapse low-resource species to family prior" strength. Can be
   swept if low-resource subset AUROC doesn't behave as expected.
