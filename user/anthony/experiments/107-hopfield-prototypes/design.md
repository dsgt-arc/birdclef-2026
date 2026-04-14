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
| Output dim | 234 (no silence class) | Axiomatic simplification — bag labels + K=8 prototype slots absorb silent focal noise |
| Memory bank count | Two (species + site), star schema factored | Eval-time affinity matrix is a deliverable |
| Species bank parameterization | **Flat** `M_species : (234, K=8, 512)`. Family hierarchy is a training-time shrinkage regularizer (`L_family`), not a parameterized bank. | Partial pooling via stop-grad mean over siblings; zero inference-time cost |
| Shared space dim | D = 512 | Enough capacity to disentangle focal ↔ soundscape manifolds |
| Projection architecture | `Linear(1536, 1024) → GELU → Linear(1024, 512) → LayerNorm` | Nonlinear; shaped by labeled BCE gradient (projection is detached from the unsupervised site loss) |
| Retrieval projections | Explicit `W_Q`, `W_K_species`, `W_K_site` (all `(512, 512)`) | Ramsauer formulation of modern Hopfield networks |
| Prototype slots | K_s = K_l = 8 | Enough for intra-species acoustic modes (song/call, dawn/night, chorus) without overparameterizing |
| Retrieval aggregation | `(1/β)·logsumexp_k(β · q_r · W_K · M_{*,k})`, learnable `log_β` per head | Modern Hopfield energy — all prototypes receive softmax-weighted gradient |
| Per-species calibration | Built-in Platt `sigmoid(a_s · score + b_s)` | Load-bearing for Bayesian gated fusion (`species · site_score` needs calibrated probabilities); also the only per-species differentiator for zero-data orphans |
| Gated fusion | `g · species + (1 - g) · species · site_score`, `g = σ(Linear(q))` | Per-window mixing between species-only and species × site |
| Loss composition | `L_labeled + λ_site · L_site_unsup + λ_fam · L_family` | Three terms; `L_family` is hierarchical-Bayes partial pooling on `M_species` |
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
    Flat species and site memory banks with modern-Hopfield logsumexp
    retrieval over a learned 512-d projection of Perch v2 embeddings.
    Per-species Platt calibration on the species head, gated fusion
    with a site-typicality scalar.

    Family hierarchy is NOT part of the forward pass — it lives
    entirely in the `L_family` training loss as a stop-grad shrinkage
    regularizer on `M_species`. At inference, the model contains no
    family-level parameters.
    """

    def __init__(self, n_species=234, n_sites=23,
                 K_species=8, K_site=8,
                 d_input=1536, d_hidden=1024, d_proj=512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
            nn.LayerNorm(d_proj),
        )

        self.W_Q = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_species = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_site = nn.Linear(d_proj, d_proj, bias=False)

        # Flat species memory (234 species, K=8 slots each). Decisions
        # are made from this bank alone. The family hierarchy is
        # implemented via the `L_family` loss at training time.
        self.M_species = nn.Parameter(
            torch.zeros(n_species, K_species, d_proj))

        # Flat site memory (23 Pantanal sites S01..S23)
        self.site_memory = nn.Parameter(
            torch.zeros(n_sites, K_site, d_proj))

        # Learnable Hopfield temperatures
        self.log_beta_species = nn.Parameter(torch.zeros(()))
        self.log_beta_site = nn.Parameter(torch.zeros(()))

        # Per-species Platt (identity init). Load-bearing: the only
        # per-species differentiator for zero-data orphans, which
        # otherwise collide with their data-rich siblings.
        self.cal_scale = nn.Parameter(torch.ones(n_species))
        self.cal_bias = nn.Parameter(torch.zeros(n_species))

        self.gate = nn.Linear(d_proj, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.M_species, std=0.02)
        nn.init.normal_(self.site_memory, std=0.02)

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

        # Species retrieval (flat — no family-level concatenation)
        beta_s = self.log_beta_species.exp()
        M_s = self.W_K_species(self.M_species)                            # (s k d)
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

    def family_shrinkage_loss(self, family_idx):
        """
        Hierarchical-Bayes partial pooling on M_species. At each step,
        compute the family centroid as a stop-grad uniform mean over
        siblings, and pull each species's prototypes toward its
        centroid.

        Args:
            family_idx: (n_species,) long tensor mapping species → family

        Returns:
            scalar shrinkage loss (squared Euclidean to family mean)
        """
        # Scatter-mean siblings into family centroids: (F, K, D)
        M_species = self.M_species                                        # (s k d)
        n_families = int(family_idx.max().item()) + 1
        sums = torch.zeros(n_families, *M_species.shape[1:],
                           device=M_species.device, dtype=M_species.dtype)
        counts = torch.zeros(n_families, device=M_species.device,
                             dtype=M_species.dtype)
        sums.index_add_(0, family_idx, M_species)
        counts.index_add_(0, family_idx,
                          torch.ones_like(family_idx, dtype=M_species.dtype))
        M_family_mean = sums / counts.clamp_min(1).view(-1, 1, 1)         # (f k d)

        # stop_grad: target is a fixed point of view for M_species
        target = M_family_mean[family_idx].detach()                       # (s k d)
        return ((M_species - target) ** 2).mean()
```

### Eval-time species × site affinity matrix

Computed outside `forward` for interpretation / reporting:

```python
beta = model.log_beta_species.exp()
M_s = model.W_K_species(model.M_species)                                 # (s ks d)
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
- **Species → family mapping** — `docs/families.csv`, built via
  the iNat API (see `scripts/fetch_families.py`). 67 families total,
  including a pseudo-family `Insecta_unresolved` for the 25
  `47158son01..25` morphotypes. Consumed only by the `L_family`
  training loss, not by the model itself.

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
| Per-family species count | Sanity check for partial-pooling centroid stability; singletons (33/67) have a no-op shrinkage term |
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
def compute_loss(model, batch, family_idx, lambda_site=0.1, lambda_fam=1e-2):
    # ----- 1. Labeled term: strong + weak focal, single BCE call -----
    # Strong has site_idx (labeled SC); weak focal has site_idx=None.
    # Forward pass routes them to the right branches internally.
    sp_strong = model(batch.strong.emb, batch.strong.site_idx)      # (32, 234)
    sp_weak   = model(batch.weak_focal.emb, site_idx=None)          # (256, 234)

    L_strong = F.binary_cross_entropy(sp_strong, batch.strong.labels)
    L_weak   = F.binary_cross_entropy(sp_weak,   batch.weak_focal.labels)
    L_labeled = L_strong + L_weak

    # ----- 2. Unsupervised term: Hopfield energy on unlabeled SC -----
    L_site_unsup = model.unsupervised_site_energy(
        batch.unsup.emb, batch.unsup.site_idx
    )

    # ----- 3. Family partial pooling (stop-grad shrinkage) -----
    # family_idx: (n_species,) long tensor, registered once at train start
    L_family = model.family_shrinkage_loss(family_idx)

    return L_labeled + lambda_site * L_site_unsup + lambda_fam * L_family
```

Two hyperparameters: `lambda_site` (default 0.1) and `lambda_fam`
(default 1e-2). Both tuned on the S22 LOSO fold. No ramp, no warm-up,
no `gamma_weak`.

### Schedule

- AdamW, `lr = 3e-4`, cosine decay
- 30–40 epochs
- Early stopping on LOSO AUROC evaluated every 5 epochs
- Weight decay: `1e-5` on all parameter groups. No asymmetric
  shrinkage — `L_family` handles low-resource regularization directly.

## LOSO evaluation

**9 folds** (sites with any labeled SC windows: S03, S08, S09, S13,
S15, S18, S19, S22, S23). Each fold:

1. Split strong stream: train on labeled SC from all sites except the
   held-out site; evaluate on the held-out site's labeled SC.
2. Reset full model, re-run random init.
3. Train: 30–40 epochs of the three-term loss with the **locked**
   `λ_fam` and `λ_site` values (tuned on S22 — see below), early
   stopping on held-out AUROC every 5 epochs.
4. Evaluate: pooled held-out predictions for the primary report;
   per-fold AUROC for diagnostics.

**Every fold uses an identical training recipe** — same loss, same
locked `λ_fam`, same sampler, same schedule. The only variable
across folds is which site's labels are withheld.

### Fold imbalance and reporting

S22 alone = 954/1478 = 65% of all labeled SC windows; the other 8
folds have 30–120 windows each. Naive macro averaging across folds
is dominated by noise from tiny folds. Reporting strategy:

- **Pooled AUROC** (**primary metric**): concat held-out predictions
  across all 9 folds into one prediction set, compute per-species
  AUROC, macro across species. Each window contributes once; fold
  size is irrelevant. Directly comparable to exp 106 ridge 0.749.
- **S22 as decision fold**: large enough (954 windows) for stable
  per-fold AUROC. Used for all hyperparameter tuning and the family
  ablation below. Tuning on pooled would eval-overfit.
- **Other 8 folds**: macro mean + per-fold table as a variance /
  generalization check. Noisy estimates of unseen-site behavior,
  not point estimates.
- **Weighted fold mean**: sanity check that pooled and weighted agree.

Subset breakdowns (all computed on the pooled predictions):
- Overall (primary headline)
- Bird vs non-bird
- Amphibian subset (where partial pooling should pay off)
- With-focal vs without-focal
- Insecta (reported separately; expected near-random)
- Has-family-sibling-with-focal vs orphaned

Unlearnable species (no focal + no 739 positives + no family sibling
with focal) excluded from aggregate, listed separately in the report.

Leakage audit: unlabeled SC is shared across folds (no labels to
leak), focal data is site-less. Standard semi-supervised LOSO.

### Pre-registered `λ_fam` sweep on S22 (doubles as family ablation)

Before running the full 9-fold LOSO, sweep `λ_fam ∈ {0, 1e-3, 1e-2,
1e-1, 1}` on the S22 fold alone. `λ_fam = 0` is the flat ablation
(no family shrinkage — pure per-species prototypes). Track three
subset AUROCs on S22's held-out set:

| Subset | What to watch for |
|--------|-------------------|
| Data-rich birds (Aves with ≥50 focal recordings) | Should be **flat** across `λ_fam`. A drop at high `λ_fam` = over-shrinkage distorting prototypes that have enough data to speak for themselves. |
| Medium-resource amphibians | Should **rise** from `λ_fam = 0`, saturate, possibly drop at very high `λ_fam`. Peak is the target. |
| Overall S22 AUROC | Secondary — used only if the subset curves are ambiguous. |

**Decision rule**: pick the smallest `λ_fam` that saturates the
amphibian curve without measurably reducing the bird curve. If no
setting of `λ_fam > 0` beats `λ_fam = 0` on the amphibian subset
by ≥ 0.5 AUROC points, v2 drops `L_family` entirely.

Lock the chosen `λ_fam` and run the full 9-fold LOSO without
re-tuning.

## Kaggle submission path

1. **Tune `λ_fam` on S22 fold** (pre-registered sweep above). Lock
   `λ_fam` and `λ_site`.
2. **LOSO validation** on 9 folds with locked hyperparameters →
   confirm architecture (pooled AUROC ≥ 0.749, target ≥ 0.76).
3. **Final training on all data** (focal + 739 labeled SC + all
   unlabeled SC, no fold hold-out), locked hyperparameters, single
   GPU run ~1h.
4. **Package**: exp 107 checkpoint (~3.4M params) + Perch v2 ONNX
   (reused from exp 102/105 Kaggle package) + preprocessing scripts.
   Note: `L_family` is a training-time loss — the shipped checkpoint
   contains no family-level parameters, only `M_species` and the
   other inference-time modules.
5. **Inference**: parse `site_idx` from the row_id
   (`BC2026_Test_NNNN_S##_<datetime>_<window>`) → Perch v2 →
   1536-d embedding → exp 107 model → 234-d multi-label probabilities
   per 5-sec window. No silence class to strip (we don't have one).
6. **OOD-site fallback**: if `site_idx` ∉ `S01..S23` (hidden rerun
   uses new sites), set `site_score = 1` to fall back to pure
   species retrieval. Revisit if first submission shows many unknown
   sites.
7. **Sample-submission sanity check**: first submission is a trivial
   forward with exp 102 weights + a minimal exp 107 stub, to confirm
   the row_id format matches the sample. Only commit to a full run
   after that.
8. **Standalone LB submission.**
9. **Ensemble with exp 102** best checkpoint: probability-space
   average, blend weight tuned on LOSO against the 739 labels (not
   on LB — LB overfitting risk). Prior hypothesis: exp 107 helps
   amphibians and low-resource birds; exp 102 dominates on common
   birds. Insect morphotypes are unreachable in v1 for either model.

### Submission checklist

- [ ] `λ_fam` sweep on S22 fold
- [ ] LOSO primary validates architecture (pooled AUROC ≥ 0.749)
- [ ] Final retraining on all data
- [ ] Model export (PyTorch checkpoint; ONNX only if einops/logsumexp
      convert cleanly, else raw `state_dict`)
- [ ] Kaggle package assembly
- [ ] Sample-submission sanity check
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
        └── species_site_matrix.npz      # (234, 23) eval-time affinity
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
| Species → family mapping (`docs/families.csv`) | `L_family` training loss (not consumed by the model itself) |

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
  12h walltime. First a single-fold S22 run for the `λ_fam` sweep,
  then one job running all 9 LOSO folds sequentially with the locked
  hyperparameters.
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
- PyTorch memory-bank model with flat species bank + unsupervised site
  Hopfield energy loss + `L_family` partial pooling (~220 LOC)
- One-phase training loop with species-balanced sampling (~120 LOC)
- LOSO cross-validation wrapper with pooled AUROC + subset metrics (~100 LOC)
- Kaggle packaging + ensemble blend (~100 LOC)

## Open decisions

1. **`lambda_site` default**: 0.1 is a guess. Tune on LOSO if the
   unsupervised site loss either dominates training (too high) or
   produces no visible effect on the site bank's prototype diversity
   (too low).
2. **`lambda_fam` default**: 1e-2 is an order-of-magnitude guess
   based on the effective-sample-size interpretation (shrinkage
   equivalent to ~a handful of virtual samples, matching the amphibian
   median of 6 recordings). The pre-registered S22 sweep
   `{0, 1e-3, 1e-2, 1e-1, 1}` locks the actual value used in LOSO.
