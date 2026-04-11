# Design: Prototype Retrieval for Amphibian-Dominated Soundscapes

## Overview

A PyTorch memory-bank classifier operating on cached Perch v2 embeddings.
Species and site prototypes live in a learned 512-d projection space,
scored by modern-Hopfield logsumexp aggregation over prototype slots,
gated by a per-window mixing factor, and calibrated by per-species Platt
scaling. The projection is shaped by BCE losses across multiple label
sources that differ in quality by species regime (see `proposal.md`).

**The core experimental question**: can a learned projection aligned by
cross-domain bird supervision carry non-bird prototypes across the
focal↔soundscape domain gap?

The implementation uses plain PyTorch + einops — no Hopfield library,
no distillation infrastructure, no multi-phase scheduler. One training
loop, one loss, one optimizer.

See `docs/star-schema-dimensional-analysis.md` for the tensor framework
underpinning the memory-bank forward pass and the eval-time species ×
site affinity matrix.

## Locked architecture decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input features | Perch v2 pooled embeddings (1536-d) | Cached from exp 106; logits used only for the label preparation pass |
| Output dim | 235 (234 species + silence) | Explicit silence class removes "silence = recording label" pathology on focal between-call windows |
| Memory bank count | Two (species + site) | Star schema factored structure; eval-time affinity matrix is a deliverable |
| Species bank parameterization | Hierarchical: `M_species[s] = M_family[family_of(s)] + Δ_species[s]` | Low-resource species inherit family prior under weight decay; high-resource species specialize. Structural fix instead of per-species hacks. |
| Shared space dim | D = 512 | Enough capacity for a nonlinear projection to align focal and soundscape manifolds; ~3× compression from 1536 |
| Projection architecture | `Linear(1536, 1024) → GELU → Linear(1024, 512) → LayerNorm` | Nonlinear alignment of focal and soundscape manifolds; data regime (400K+ BCE-supervised windows across strong + weak streams) supports the extra capacity |
| Retrieval projections | Explicit `W_Q`, `W_K_species`, `W_K_site` (all `(512, 512)`) | Standard Ramsauer et al. 2020 modern Hopfield formulation; separates retrieval space from projection output space |
| Prototypes per bank | K_s = K_l = 8 | Enough to capture intra-species acoustic modes (song/call, dawn/night, solo/chorus) without over-parameterizing against small per-site fine-tune data |
| Retrieval aggregation | Logsumexp over K, then sigmoid | Soft-max Hopfield energy — all prototypes receive gradient weighted by softmax attention; avoids dead-prototype pathology |
| Per-species calibration | Built-in Platt scaling (`a_s · score + b_s` before sigmoid) | BCE assumes calibrated outputs; per-class scale + bias corrects systematic offsets that would bias BCE gradients |
| Gated fusion | `g · species + (1 - g) · species · site` with `g = σ(Linear(q))` | Per-window fallback to species-only when site context doesn't help |
| Pretraining | **None** | Reframed 2026-04-11: the training signal is BCE on weak + strong streams in one phase. Perch-verified label prep replaces distillation. |
| Family memory init | K-means on raw Perch focal embeddings pooled across all species in each family | Families with any focal data get well-populated prototypes; seeds the inheritance path for low-resource species |
| Species delta init | Zero | Starts as pure family fallback; gradient descent + weight decay control whether the delta develops |
| Site memory init | K-means on raw Perch unlabeled SC embeddings per site | Per-site acoustic mode initialization from 127K unlabeled windows |
| Multi-head attention | No | One projection suffices for a 2-bank problem |
| Iterative Hopfield refinement | Deferred (one-shot) | Multi-label scoring wants independent per-species probs; iteration drags the query toward one attractor |
| Stacked cross-attention layers | Deferred to exp 108 | Out of scope for the headline comparison |
| Tensor operations | einops `einsum`, `reduce`, `rearrange` | Matches project convention; makes star-schema contractions self-documenting |

## Built-in Platt scaling

The species head applies per-species affine calibration before the sigmoid:

    score_s = σ(a_s · (1/β) · logsumexp_k(β · q_r · W_K · M_{s,k}) + b_s)

where `a_s` (scale) and `b_s` (bias) are learnable per-species parameters,
initialized to identity (`a_s = 1`, `b_s = 0`). Total extra: 2 × 235 = 470
parameters (one pair per species, including the silence class).

Three reasons calibration belongs in the model, not as a post-hoc fix:

1. **BCE assumes calibrated outputs.** Uncalibrated scores produce biased
   BCE gradients. In-model calibration improves training dynamics.
2. **Multi-label composition needs calibrated probabilities.** The gated
   fusion multiplies species × site scores. Valid as Bayes rule only if
   both sides are proper probabilities.
3. **Entropy minimization assumes probabilities.** The entropy term on
   unconfident unlabeled windows only makes sense if `p` represents
   confidence in a binary event.

The site head produces a single scalar per window and doesn't need
per-site Platt; the gate handles systematic bias.

## Model

```python
from einops import einsum, rearrange

class PrototypeRetrievalModel(nn.Module):
    """
    Hierarchical species (family + delta) and site memory banks with
    modern-Hopfield logsumexp retrieval over a learned 512-d projection
    of Perch v2 embeddings. Per-species Platt calibration on the species
    head, gated fusion with site head. Silence is output class index 234
    (last position, after the 234 competition species at indices 0..233).

    Parameters:
        n_species: 235 (234 competition species + 1 silence class)
        n_families: ~30 (from BirdCLEF taxonomy metadata + 1 silence "family")
        n_sites: 21
        K_species: 8 prototype slots per species
        K_site: 8 prototype slots per site
        d_input: 1536 (Perch embedding)
        d_hidden: 1024 (MLP hidden)
        d_proj: 512 (shared projection)
        family_of: (n_species,) int tensor — species_idx → family_idx
    """

    def __init__(self, n_species=235, n_families=30, n_sites=21,
                 K_species=8, K_site=8,
                 d_input=1536, d_hidden=1024, d_proj=512,
                 family_of=None):
        super().__init__()
        assert family_of is not None and family_of.shape == (n_species,)
        self.register_buffer("family_of", family_of.long())

        # Nonlinear projection into the shared D-space. Extra capacity vs
        # a linear projection is supported by the 400K+ BCE-supervised
        # windows (strong + weak streams combined).
        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
            nn.LayerNorm(d_proj),
        )

        # Explicit W_Q / W_K (Ramsauer et al.): separate retrieval-space
        # projections for query and memory banks.
        self.W_Q = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_species = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_site = nn.Linear(d_proj, d_proj, bias=False)

        # Hierarchical species memory: M_species[s] = M_family[f(s)] + Δ_species[s]
        # Family prototypes: shared across all species in the family, trained
        # from the pooled gradient of every species in that family.
        # Species deltas: zero-initialized, weight-decay-regularized toward 0.
        self.family_memory = nn.Parameter(
            torch.zeros(n_families, K_species, d_proj))
        self.species_delta = nn.Parameter(
            torch.zeros(n_species, K_species, d_proj))

        # Site memory bank (flat; no hierarchy on the site side).
        self.site_memory = nn.Parameter(
            torch.zeros(n_sites, K_site, d_proj))

        # Learnable Hopfield temperatures (stored as log for positivity).
        self.log_beta_species = nn.Parameter(torch.zeros(()))
        self.log_beta_site = nn.Parameter(torch.zeros(()))

        # Per-species Platt calibration, initialized to identity.
        self.cal_scale = nn.Parameter(torch.ones(n_species))
        self.cal_bias = nn.Parameter(torch.zeros(n_species))

        self.gate = nn.Linear(d_proj, 1)

    def effective_species_memory(self):
        """M_species[s] = M_family[family_of(s)] + Δ_species[s]   (s k d)"""
        return self.family_memory[self.family_of] + self.species_delta

    def forward(self, emb, site_idx=None):
        """
        Args:
            emb: (B, 1536) Perch embeddings
            site_idx: (B,) int site indices, or None for focal windows

        Returns:
            scores: (B, 235) calibrated multi-label probabilities
                    (index 234 is the silence class)
        """
        q = self.proj(emb)                                              # (b d)
        q_r = self.W_Q(q)                                               # (b d)

        # Hierarchical species retrieval: compute effective memory, then
        # project into the retrieval space, then logsumexp over K.
        beta_s = self.log_beta_species.exp()
        M_species = self.effective_species_memory()                     # (s k d)
        M_s = self.W_K_species(M_species)                               # (s k d)
        sp_sim = einsum(q_r, M_s, 'b d, s k d -> b s k')                # (b s k)
        sp_agg = torch.logsumexp(beta_s * sp_sim, dim=-1) / beta_s      # (b s)
        sp_score = torch.sigmoid(self.cal_scale * sp_agg + self.cal_bias)

        if site_idx is None:
            # Focal: species-only scoring
            return sp_score

        # Site retrieval: per-window site typicality scalar
        beta_l = self.log_beta_site.exp()
        M_l = self.W_K_site(self.site_memory[site_idx])                 # (b k d)
        site_sim = einsum(q_r, M_l, 'b d, b k d -> b k')                # (b k)
        site_agg = torch.logsumexp(beta_l * site_sim, dim=-1) / beta_l  # (b,)
        site_score = torch.sigmoid(site_agg)                            # (b,)

        # Gated fusion
        g = torch.sigmoid(self.gate(q)).squeeze(-1)                     # (b,)
        score = (g * sp_score
                 + (1 - g) * sp_score * rearrange(site_score, 'b -> b 1'))
        return score
```

### Weight decay on species deltas

Weight decay on `Δ_species` is load-bearing for low-resource species —
it's what keeps them from overfitting their few windows and pushes them
toward the family prototype. Use a separate optimizer parameter group:

```python
param_groups = [
    {"params": model.species_delta, "weight_decay": 1e-2},   # strong shrinkage
    {"params": [p for n, p in model.named_parameters()
                if n != "species_delta"], "weight_decay": 1e-5},
]
optimizer = torch.optim.AdamW(param_groups, lr=3e-4)
```

The high weight decay on species deltas is the "fall back to family
prior when data is scarce" mechanism. Low-resource species can't
generate enough gradient to counteract the 1e-2 shrinkage and stay
near Δ = 0. High-resource species generate plenty of gradient and
push Δ away from zero into a specialized position. Tune the shrinkage
rate via LOSO if needed.

Eval-time species × site affinity matrix (computed outside forward):

```python
beta = self.log_beta_species.exp()
M_species = self.effective_species_memory()                            # (s ks d)
M_s = self.W_K_species(M_species)                                      # (s ks d)
M_l = self.W_K_site(self.site_memory)                                  # (l kl d)
pair = einsum(M_s, M_l, 's ks d, l kl d -> s l ks kl')
affinity = rearrange(pair, 's l ks kl -> s l (ks kl)')
A = torch.logsumexp(beta * affinity, dim=-1) / beta                    # (s l)
```

See `docs/star-schema-dimensional-analysis.md` for the dimensional
framework and recommendation-system interpretation.

## Label preparation pass (one-time, pre-training)

Before training, run a one-time preprocessing step that computes
per-window labels from raw focal recording labels + cached Perch logits.
Output: per-window parquet files alongside the cached embeddings in
`~/scratch/birdclef-2026/features/exp107/labels/`.

### Inputs (from exp 106)

- `~/scratch/birdclef-2026/features/exp106/focal/by_species/*.parquet` —
  per-species focal embeddings + raw Perch logits
- `~/scratch/birdclef-2026/features/exp106/soundscape_labeled.parquet` —
  739 labeled SC windows + raw Perch logits
- `~/scratch/birdclef-2026/features/exp106/soundscape_unlabeled/by_site/*.parquet` —
  per-site unlabeled SC embeddings + raw Perch logits
- `perch_species_mapping.json` — Perch species index → BirdCLEF species id
  (for the 234 competition taxonomy subset)
- Species classification into **bird** vs **non-bird** — from exp 100 EDA
- **Species → family mapping** — from BirdCLEF species list (eBird taxonomy
  for birds, iNaturalist for amphibians/insects). Verified during label
  prep; pulled from an external taxonomy CSV if missing from the comp metadata.
- **Per-window RMS energy** — computed during label prep (or pre-cached
  during embedding extraction in exp 106 if easy to add there). Used to
  assign the silence class.

### Processing

```python
SILENCE_CLASS_IDX = 234   # 235th output, index after the 234 species

def prepare_labels(
    focal_pq, labeled_sc_pq, unlabeled_sc_pq,
    bird_species_ids, non_bird_species_ids,
    perch_taxo_map, conf_threshold=0.3,
    silence_db_threshold=-60.0,
):
    # ---- Silence detection: RMS-based, not Perch-based
    # (Perch has low confidence on amphibians too — don't conflate those
    # with silence.) Compute per-window dB, classify below threshold as silent.
    def silence_mask(df):
        rms = df["rms_linear"]  # linear amplitude, cached in label prep
        db = 20.0 * log10(rms + 1e-10)
        return db < silence_db_threshold

    # ---- Strong stream: labeled SC, window-level GT (235-d)
    strong = labeled_sc_pq.select("window_id", "emb", "site_id", "labels_235")

    # ---- Weak (bird focal): Perch-verified + silence override
    bird_focal = focal_pq.filter(col("species_id").is_in(bird_species_ids))
    bird_focal_topk = topk_in_taxo(bird_focal["perch_logits"], perch_taxo_map, k=5)
    bird_focal_verified = bird_focal.filter(
        verified(bird_focal_topk, bird_focal["species_id"], conf_threshold)
    )
    # Multi-hot over 235 outputs (234 species + silence)
    bird_focal_labels = multi_hot_from_topk(bird_focal_topk, conf_threshold, dim=235)
    # Silence windows get silence-positive, species-negative
    bird_focal_labels = override_silence(
        bird_focal_labels, silence_mask(bird_focal_verified), SILENCE_CLASS_IDX
    )

    # ---- Weak (non-bird focal): bag-level, with silence override
    # This is where the silence class pays off most — silent windows of
    # non-bird focal recordings no longer propagate "silence = species s".
    non_bird_focal = focal_pq.filter(col("species_id").is_in(non_bird_species_ids))
    non_bird_focal_labels = one_hot(non_bird_focal["species_id"], dim=235)
    non_bird_focal_labels = override_silence(
        non_bird_focal_labels, silence_mask(non_bird_focal), SILENCE_CLASS_IDX
    )

    # ---- Weak (unlabeled SC): Perch pseudo-labels, split by confidence + silence
    unlabeled_topk = topk_in_taxo(unlabeled_sc_pq["perch_logits"], perch_taxo_map, k=5)
    silent_sc = unlabeled_sc_pq.filter(silence_mask(unlabeled_sc_pq))
    silent_sc_labels = silence_only_labels(len(silent_sc), dim=235)
    not_silent = unlabeled_sc_pq.filter(~silence_mask(unlabeled_sc_pq))
    not_silent_topk = topk_in_taxo(not_silent["perch_logits"], perch_taxo_map, k=5)
    confident = not_silent.filter(max_conf(not_silent_topk) >= conf_threshold)
    confident_labels = multi_hot_from_topk(
        confident["perch_topk"], conf_threshold, dim=235
    )
    unconfident = not_silent.filter(max_conf(not_silent_topk) < conf_threshold)

    return {
        "strong": strong,
        "weak_bird_focal": (bird_focal_verified, bird_focal_labels),
        "weak_non_bird_focal": (non_bird_focal, non_bird_focal_labels),
        "weak_unlabeled_sc_confident": (confident, confident_labels),
        "weak_silence_sc": (silent_sc, silent_sc_labels),
        "unlabeled_sc_unconfident": unconfident,  # entropy-min only
    }
```

Expected output sizes (rough estimates):
- Strong: 739 windows
- Weak bird focal (verified): ~60–70% of ~150K bird focal windows ≈ 90–105K
- Weak non-bird focal (bag-level): ~115K
- Weak unlabeled SC (confident): ~50% of 127K ≈ 60K
- Unconfident unlabeled SC (entropy only): ~70K

Total windows touched by training: ~260K across strong + weak streams,
plus ~70K for entropy minimization.

### Baselines computed in the same pass

| Baseline | Computation |
|----------|-------------|
| Perch top-1 direct | For each labeled SC window, take Perch's top-1 in taxonomy, compute macro AUROC against GT |
| Perch top-1 in-taxonomy | Same but compute multi-hot over top-k, useful sanity check |
| Coverage statistics | What fraction of each species' focal windows pass the Perch-verified filter? Per-species retention rate. |

## Batch construction

Three streams sampled independently per step:

```python
batch = {
    "strong": sample(strong_windows, n=32),              # 739 total, resample heavily
    "weak":   sample(weak_windows, n=256),               # ~260K, uniform
    "unconfident": sample(unconfident_windows, n=128),   # ~70K, entropy only
}
```

`weak` is sampled uniformly across the combined (bird focal, non-bird
focal, unlabeled SC confident) pool. No per-stream subsampling — the
loss doesn't need to know which weak subsource a window came from, only
whether it has a label and whether it has a site.

`strong` is sampled with replacement (739 will be cycled many times per
epoch).

## Loss

```python
def compute_loss(model, batch, gamma_weak, lam_ent):
    # Strong: labeled SC, window-level GT
    pred_s = model(batch["strong"]["emb"], batch["strong"]["site_idx"])
    L_strong = F.binary_cross_entropy(pred_s, batch["strong"]["labels"])

    # Weak: Perch-verified focal + Perch-pseudo unlabeled SC
    pred_w = model(batch["weak"]["emb"], batch["weak"]["site_idx"])
    L_weak = F.binary_cross_entropy(pred_w, batch["weak"]["labels"])

    # Entropy min on unconfident unlabeled SC
    pred_e = model(batch["unconfident"]["emb"], batch["unconfident"]["site_idx"])
    p = pred_e.clamp(1e-7, 1 - 1e-7)
    L_ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

    return L_strong + gamma_weak * L_weak + lam_ent * L_ent
```

### Ramp schedule

```python
def ramp(epoch, end_epoch=5, target=1.0):
    return target * min(1.0, epoch / end_epoch)

gamma_weak = ramp(epoch, end_epoch=5, target=1.0)
lam_ent    = ramp(epoch, end_epoch=5, target=0.1)
```

Rationale: start from the clean 739 GT, then progressively trust the
weak streams and the entropy regularizer. Same protection the four-phase
schedule gave, without the phase boundaries.

## Initialization

**Projection MLP**: standard Kaiming init.

**Family memory** (K-means on raw Perch focal embeddings pooled per family):

1. Group focal windows by family (using the species → family mapping).
   For each family f, concatenate focal embeddings from all its member
   species. Families with at least one species having focal data will
   typically have ≥100 windows — plenty for k-means.
2. Run k-means with K=8 on the pooled 1536-d raw Perch embeddings.
3. Project the 8 cluster centers through the randomly-initialized `proj`
   to get 512-d vectors, write to `self.family_memory[family_idx]`.
4. Families with zero focal data across all their member species
   (expected rare): initialize with small Gaussian noise; their
   prototypes will only move if a species in the family has labeled SC
   positives in the training strong stream.

Why pooling per family works: species within a family usually share
acoustic signatures (call morphology, frequency ranges, temporal
structure). Pooling gives us more data per clustering, so k-means finds
stable family-level modes that generalize across sibling species.

**Species delta memory**: zero-initialized. Species start as pure family
fallback; weight decay + BCE gradient determine whether each species
develops a specialized delta.

**Site memory**: k-means on unlabeled SC embeddings partitioned by site
(21 × K=8 = 168 prototype slots). Per-site window counts are high (6K+),
so k-means is well-posed.

**W_Q, W_K_species, W_K_site**: default linear init.

**Platt calibration**: identity (`a_s = 1`, `b_s = 0`).

**Hopfield temperatures**: `log β = 0` (β = 1).

**Gate**: default linear init (gate output ~0.5 at the start).

## LOSO evaluation

7 folds (sites with ≥20 labeled SC windows). Each fold:

1. **Split strong stream**: train on labeled SC from all sites except
   the held-out site; test on held-out site's labeled SC.
2. **Init**: k-means memory banks on focal / unlabeled SC as above. Full
   model reset per fold (no carry-over from previous folds).
3. **Train**: 30–40 epochs of the one-phase loop with ramped γ_weak,
   λ_ent. Early stopping on held-out AUROC evaluated every 5 epochs.
4. **Evaluate**: macro AUROC on held-out labeled SC windows, with
   subset breakdowns (bird/non-bird, with/without focal,
   has-family-sibling-with-focal/orphaned, silence class).

Final metric: mean of per-fold AUROC. Report overall + subset splits.
Species flagged as **unlearnable** (no training signal from any source
— no focal, no positive in strong stream, no family sibling with focal)
are excluded from the aggregate AUROC and listed separately.

Leakage audit: unlabeled SC and focal data are shared across folds
(no held-out labels exist for unlabeled SC, and focal has no site_id).
This is standard semi-supervised LOSO practice.

## Ablations

| Config | Tests |
|--------|-------|
| **Primary**: full model, all streams, ramped weights, hierarchical species bank, silence class | Headline LOSO AUROC |
| Architecture: linear classifier on the projection | Does the memory bank beat a linear head on non-bird species? |
| Label cleaning: raw focal one-hot, no Perch-verified filter | Does label cleaning matter? (exp 102 crossover) |
| Weak stream off: `γ_weak = 0` | How much do 260K weak-labeled windows add beyond 739 GT? |
| Site bank off: species-only, no gate | Does site composition work in the learned space? |
| Entropy min off: `λ_ent = 0` | Is entropy min on unconfident unlabeled SC useful? |
| Silence class off: 234 outputs, no RMS labeling, drop silent windows instead | Does the explicit silence class help, or can the model learn "all species → 0" on its own? |
| Flat species bank: replace `M_family + Δ_species` with direct species memory `(235, K, D)`, no weight decay split | Does hierarchical prototyping beat flat prototypes, especially on low-resource and zero-focal species? |
| Site classifier aux: add a small CE loss predicting `site_idx` from the site bank | Does explicit site supervision help or hurt? Prior: hurts (encodes artifacts). |
| Self-training round 2 | Use trained bank to pseudo-label non-bird species in unlabeled SC, retrain. Risk: feedback loops. |

## Directory structure

```
user/anthony/experiments/107-hopfield-prototypes/
├── proposal.md                          # thesis, method, success criteria
├── design.md                            # ← this file
├── docs/
│   └── star-schema-dimensional-analysis.md  # tensor framework + recsys framing
├── pyproject.toml                       # declares dependency on birdclef_106
├── src/birdclef_107/
│   ├── __init__.py
│   ├── cli.py                           # entry: prepare-labels, init-banks, train, evaluate
│   ├── label_prep.py                    # one-time Perch-verified label preparation
│   ├── data.py                          # datasets + multi-stream batch sampler
│   ├── model.py                         # PrototypeRetrievalModel
│   ├── losses.py                        # strong + weak + entropy composition
│   ├── init.py                          # k-means init for species and site banks
│   ├── train.py                         # one-phase training loop with ramp
│   └── evaluate.py                      # LOSO cross-validation + subset metrics
├── configs/
│   ├── base.yaml                        # default hyperparameters
│   ├── ablation_architecture.yaml       # linear classifier variant
│   ├── ablation_label_cleaning.yaml     # raw focal one-hot
│   ├── ablation_no_weak.yaml            # 739 GT only
│   ├── ablation_no_site.yaml            # species-only
│   ├── ablation_no_entropy.yaml         # λ_ent = 0
│   └── ablation_self_training.yaml      # round-2 self-training
├── sbatch/
│   ├── _setup-venv.sh                   # reuses exp 106 venv, adds torch + einops
│   ├── prepare_labels.sbatch            # one-time label prep (CPU)
│   └── train_loso.sbatch                # one training + LOSO run (CPU-tractable)
└── artifacts/                           # gitignored
    ├── labels/                          # per-window label parquets from label prep
    ├── checkpoints/
    │   └── fold_{i}/{epoch}.pt
    ├── metrics/
    │   ├── loso_results.json            # overall + subset breakdowns
    │   └── projection_distances.json    # bird focal↔SC diagnostic
    └── affinity/
        └── species_site_matrix.npz      # (235, 21) eval-time affinity
```

## Dependencies

**External**

| Package | Used for |
|---------|----------|
| `torch` | Model, training loop (GPU optional — CPU-tractable on cached features) |
| `einops` | `einsum`, `reduce`, `rearrange` — memory-bank contractions and affinity matrix |
| `polars` | Cached parquet IO (inherited from exp 106) |
| `numpy`, `scikit-learn` | Metrics, k-means init |
| `pyyaml` | Config loading |

**Cached artifacts consumed from exp 106**

| Artifact | Used for |
|----------|----------|
| Perch v2 embeddings (265K focal + 739 labeled + 127K unlabeled) | Model input |
| Perch v2 logits (same windows) | Label preparation pass only |
| `perch_species_mapping.json` | Restrict Perch's 14,795 outputs to the 234 competition taxonomy |
| Species bird/non-bird classification (from exp 100) | Regime split in label prep |

**Internal**

Exp 107 imports from the exp 106 package (declared in `pyproject.toml`):

- `birdclef_106.data_loading` — cached parquet loaders, label matrix builder,
  site extraction, species list construction

No code copied. Memory banks are k-means-initialized from raw Perch
focal / unlabeled SC embeddings — **not** from exp 106's
`multi_prototypes.npz` or `site_prototypes.npz`, which live in a fixed
space that exp 107's learned projection replaces.

## Testing plan

Run before any training:

1. **Shape tests**: `model.forward` produces correct output shape for
   focal (site_idx=None) and soundscape (site_idx=int) inputs
2. **Gradient tests**: all learnable parameters receive gradients for
   appropriate inputs (site memory and gate only for soundscape inputs)
3. **K-means init tests**: per-family k-means produces finite,
   correctly-shaped `(F, 8, 512)` family memory (F ≈ 30); species
   delta is zero-initialized `(235, 8, 512)`; per-site k-means produces
   `(21, 8, 512)` site memory; families with zero focal data (rare
   edge case) handled via small Gaussian noise init
4. **Affinity test**: `A(s, l)` via einsum matches a brute-force quadruple
   loop over `(s, l, k_s, k_l)` on a small synthetic model
5. **Label prep test**: running label prep on 100 synthetic windows with
   known Perch logits produces the expected filtered output
6. **Overfit test**: full model drives a single 32-window supervised
   batch's BCE to near-zero in 100 steps
7. **Parity test**: ablation "739 GT only, no weak stream" LOSO AUROC
   is within 0.02 of a ridge baseline computed in the same feature
   space. If this fails, the model is broken in a way that prevents
   even matching a linear classifier.
8. **Diagnostic test**: bird focal↔SC projection distance diagnostic
   produces reasonable numbers on untrained model (baseline) and shrinks
   after a few epochs on a toy run

Tests 1–6 are seconds to minutes each. Test 7 is the cheapest
end-to-end correctness check. Test 8 is the mechanism sanity check.

## SLURM setup

- **Label prep**: one-time CPU job, ~1h, `cpu-small` partition. Produces
  the per-window label parquets reused across all folds and ablations.
- **Training + LOSO**: one job runs all 7 folds sequentially. CPU-tractable
  on cached features. Partition `cpu-small`, 8 CPUs × 16 GB RAM, 8h
  walltime. Can also run on GPU if queue is favorable — negligible speedup
  given cached-feature input.
- **Ablations**: each ablation is a separate sbatch with a different
  config yaml. 6 ablations → 6 jobs, same shape as primary.
- Venv: reuses `~/scratch/birdclef-2026/.venv-106` with torch + einops
  added. Per `CLAUDE.md`, never install `.venv` under `$HOME`.

## Connection to exp 106

```
Exp 106 infrastructure (reused):
├── Cached embeddings on scratch (739 + 127K + 265K)
├── Partitioned parquets (by_species/, by_site/)
├── Cached Perch logits (needed by exp 107's label preparation pass)
├── Species mapping (perch_species_mapping.json)
├── Label matrix construction (build_binary_label_matrix)
├── LOSO evaluation utilities
└── data_loading.py (load_focal_species, load_site, load_labeled, etc.)

Not inherited (intentionally):
├── multi_prototypes.npz, site_prototypes.npz — exp 106's prototypes
│   live in Perch's fixed logit space; exp 107's memory banks live in
│   the learned projection's output space.
└── Star schema cosine score (exp 106 baseline only)

Exp 107 adds:
├── Label preparation pass: Perch-verified focal + pseudo-label unlabeled SC
├── PyTorch memory-bank model (~150 LOC)
├── One-phase training loop with ramped weights (~120 LOC)
├── LOSO cross-validation wrapper with subset metrics (~80 LOC)
└── Ablation configs + evaluation harness (~60 LOC)
```

## Open decisions

1. **Perch confidence threshold**: 0.3 is a guess. Sweep 0.2 / 0.3 / 0.5
   during label prep; the right threshold depends on Perch's raw score
   distribution on the 234 taxonomy and on the coverage/quality trade-off.
2. **K_species, K_site**: default K=8. Sweep {4, 8, 16} as a capacity
   ablation if primary results are close to the go/no-go threshold.
3. **Projection dim**: D=512 default. If the model underfits (ablation A
   below ridge), bump to D=768. If it overfits, drop to D=384.
4. **Weak stream weight `γ_weak` target**: 1.0 (equal with strong). May
   need to tune down if weak-stream label noise hurts. Sweep 0.5 / 1.0 / 2.0.
5. **Non-bird focal bag cleaning**: current design leaves non-bird focal
   bag-level. Future work could try self-labeling (train round 1 on
   Perch-verified + bag-level, then use round 1 predictions to clean
   non-bird focal, retrain). Deferred — listed as the "self-training r2"
   ablation.
