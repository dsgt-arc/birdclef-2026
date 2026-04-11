# Design: Hopfield Prototype Retrieval

## Overview

A PyTorch model implementing a **modern Hopfield network** (Ramsauer et
al. 2020) with two memory banks (species, site) over a shared 512-d
space learned by **Perch-logit distillation** on all 400K+ cached
windows. The species bank uses sigmoid attention for multi-label
retrieval; the site bank uses sigmoid attention as a single-scalar
typicality gate. This gives the star schema's factored structure
(species × site) a concrete computational realization: prototype memory
banks with attention-based retrieval over a shared learned projection.

**The core experimental question** is whether a learned shared space
disentangles focal and soundscape manifolds in a way that Perch's fixed
logit space (exp 106) could not. The phase-0 distillation pretraining is
what makes this question testable — without it, the model has to learn
the shared space from 739 labeled windows alone, which is too thin a
supervised signal for the architecture's capacity.

**The Hopfield framing is a theoretical gloss on standard multi-prototype
attention, not a load-bearing mechanism.** The same architecture could
be described as "factored multi-prototype retrieval with learned
projections, explicit Q/K transforms, per-class Platt calibration, and
knowledge distillation pretraining" — and nothing would change. The
Hopfield label gives us a clean lineage back to Ramsauer et al. and
motivates specific choices (logsumexp aggregation, the soft-EM gradient
identity, the one-shot vs iterative decision), but if the experiment
fails it won't be because "Hopfield" was wrong.

Trained on cached Perch v2 embeddings + logits from three domains —
focal recordings (265K, single-label), labeled soundscapes (739,
multi-label), and unlabeled soundscapes (127K, no labels). No audio
processing, no large backbone. Phase 0 needs Perch logits (not just
embeddings) as distillation targets.

The implementation uses plain PyTorch + einops (no `hflayers`
dependency). Both memory-bank heads use sigmoid — species for
multi-label probabilities, site for a single-scalar typicality gate that
modulates species predictions.

See `docs/star-schema-dimensional-analysis.md` for the tensor framework,
recommendation-system framing, and the soft-EM gradient identity that
this design implements.

## Locked architecture decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input features | **Embeddings only** (1536-d) | Simpler baseline; logits deferred as future ablation |
| Memory bank count | **Two** (species + site) | Matches star schema factored structure |
| Shared space dim | **D = 512** | Enough capacity for the projection to disentangle focal/soundscape manifolds |
| Projection architecture | **2-layer MLP** (`Linear(1536, 1024) → GELU → Linear(1024, 512) → LayerNorm`) | Nonlinear alignment of focal and soundscape manifolds; unlocked by phase 0 distillation which provides 400K+ supervised targets for the extra capacity |
| Retrieval projections | **Explicit `W_Q`, `W_K`** (both `(512, 512)`) | Standard Ramsauer et al. 2020 formulation; lets retrieval space differ from embedding space without breaking the Hopfield framing |
| Prototypes per bank | **K_s = K_l = 16** | Abstract capacity; let training discover acoustic modes instead of hard-coding 4 biologically-named ones |
| Species head retrieval | **Logsumexp over K**, then sigmoid | Soft-max Hopfield energy: differentiable, all prototypes receive gradient, avoids dead-prototype pathology at large K |
| Site head retrieval | **Logsumexp over K**, then sigmoid | Consistency with species head; single-scalar gate |
| Per-species calibration | **Built-in Platt scaling** | Per-species scale + bias before sigmoid; learned from BCE |
| Pretraining | **Phase 0: Perch logit distillation** on all 400K+ windows | Main training signal; removes the 739-window bottleneck for projection and species bank learning. Fine-tuning on labels becomes polish, not the primary signal. |
| Species memory init | **K-means on phase-0 projected embeddings** | After phase 0 the projection is meaningful, so k-means on its output is no longer a "stale space" problem; per-species clusters computed from focal windows in the pretrained shared space |
| Site memory init | **K-means on phase-0 projected embeddings** | Same rationale — phase 0 provides a meaningful space to cluster in. Per-site clusters computed from unlabeled soundscape windows |
| Multi-head attention | **No** | One projection suffices; multi-head adds complexity for a 2-source problem |
| Iterative Hopfield refinement | **Deferred** (one-shot) | One-shot retrieval is the right abstraction for multi-label scoring (iteration drags the query toward one attractor); stacked attention layers are the upgrade path if needed |
| Stacked cross-attention layers | **Deferred to exp 108** | Depth beyond one retrieval layer is the natural follow-up *given* a pretrained shared space; out of scope here to keep the headline comparison clean |
| Pattern separation regularizer | **No** | Start simple; add only if prototypes collapse in training |
| Tensor operations | **einops** (`einsum`, `reduce`, `rearrange`) | Matches project convention (used across Murilo's exps, WavTokenizer); makes star-schema contractions self-documenting |

Deferred to future work (out of scope for exp 107):

- Dual-space (logits + embeddings) with parallel banks
- Per-species site affinity (vs the current scalar site gate)
- Stacked cross-attention layers on top of the memory bank (exp 108 hypothesis: does depth help given a pretrained shared space?)
- Pattern separation or coherence regularizers
- Recording-level context (mean-pool, self-attention, triplet SSL, MIL loss) — temporal modeling is a separate hypothesis, exp 108+
- Post-hoc calibration comparison via `netcal` (Platt vs beta vs isotonic vs histogram binning)

## Built-in Platt scaling

The species head applies per-species affine calibration before the sigmoid:

    score_s = σ(a_s · max_k(q · m_{s,k}) + b_s)

where `a_s` (scale) and `b_s` (bias) are learnable per-species parameters.
Total: 2 × 234 = 468 extra parameters. Initialized to identity
(`a_s = 1.0`, `b_s = 0.0`) so the model starts as the uncalibrated baseline
and learns corrections during training.

Three reasons calibration belongs in the model rather than as a post-hoc fix:

1. **BCE assumes calibrated outputs.** Uncalibrated scores produce biased
   BCE gradients. In-model calibration improves training dynamics, not just
   output quality.

2. **Multi-label composition needs calibrated probabilities.** The gated
   fusion multiplies species × site scores. This is only valid as Bayes
   rule if both sides are proper probabilities. Systematic per-species
   offsets break the composition.

3. **Semi-supervised entropy minimization assumes probabilities.** The
   entropy term `-p log p - (1-p) log(1-p)` only makes sense if `p`
   represents confidence in a binary event. Uncalibrated scores make
   entropy min sharpen whatever bias the model has.

The site head doesn't need per-site calibration — it produces a single
scalar per window, and the gate handles any systematic bias.

## Input format

Each training example is a single window:

```
window = {
    "embedding": tensor(1536),     # Perch v2 embedding
    "domain": "focal" | "labeled_sc" | "unlabeled_sc",
    "site_id": int | None,         # None for focal (no site info)
    "labels": tensor(234) | None,  # one-hot (focal), multi-hot (labeled_sc), None (unlabeled)
}
```

All three domains are fed through the same projection and species bank.
The site bank and gate are only active when `site_id` is available (soundscapes).

## Model

```python
from einops import einsum, rearrange

class HopfieldPrototypeModel(nn.Module):
    """
    Species and site memory banks with modern Hopfield (logsumexp) retrieval.
    2-layer MLP projection into a shared D-space, explicit W_Q/W_K on the
    retrieval side, per-species Platt calibration on the species head.

    Parameters:
        n_species: 234
        n_sites: 21
        K_species: 16 (prototype slots per species)
        K_site: 16 (prototype slots per site)
        d_input: 1536 (Perch embedding dim)
        d_hidden: 1024 (MLP hidden dim)
        d_proj: 512 (shared projection dim)
    """

    def __init__(self, n_species=234, n_sites=21,
                 K_species=16, K_site=16,
                 d_input=1536, d_hidden=1024, d_proj=512):
        super().__init__()

        # Nonlinear projection into the shared D-space. The extra capacity vs
        # a single linear layer is what lets the projection *nonlinearly*
        # align focal and soundscape manifolds. Safe to make nonlinear because
        # phase 0 distillation trains it on 400K+ windows, not just 739.
        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
            nn.LayerNorm(d_proj),
        )

        # Explicit Ramsauer et al. W_Q / W_K: query and keys each get their
        # own learned linear projection before the dot product. Lets the
        # retrieval space differ from the embedding space without breaking
        # the Hopfield framing. W_V is identity here — we return the
        # aggregated similarity, not a weighted value vector.
        self.W_Q = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_species = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_site = nn.Linear(d_proj, d_proj, bias=False)

        # Species bank: initialized by k-means on phase-0 projected focal
        # embeddings (see init section). Placeholder zeros here; filled by
        # init_species_memory_kmeans() after phase 0 pretraining completes.
        self.species_memory = nn.Parameter(
            torch.zeros(n_species, K_species, d_proj))

        # Site bank: initialized by k-means on phase-0 projected unlabeled
        # soundscape embeddings. Placeholder zeros here; filled by
        # init_site_memory_kmeans() after phase 0 pretraining completes.
        self.site_memory = nn.Parameter(
            torch.zeros(n_sites, K_site, d_proj))

        # Learnable Hopfield temperatures (stored as log for positivity).
        # β = 1 at init; training picks its own sharpness per head.
        self.log_beta_species = nn.Parameter(torch.zeros(()))
        self.log_beta_site = nn.Parameter(torch.zeros(()))

        # Per-species Platt scaling (built-in calibration).
        # Initialized to identity: score passes through unchanged.
        self.cal_scale = nn.Parameter(torch.ones(n_species))
        self.cal_bias = nn.Parameter(torch.zeros(n_species))

        self.gate = nn.Linear(d_proj, 1)

        # Phase 0 distillation head: projects the D-space query back to
        # Perch's logit space (14795) for the pretraining target. Used only
        # during phase 0; detached and frozen afterwards (or discarded).
        self.distill_head = nn.Linear(d_proj, 14795)

    def project(self, emb):
        """Phase-0 and downstream: shared projection into D-space."""
        return self.proj(emb)                                           # (b d)

    def distill_forward(self, emb):
        """Phase 0: project + predict Perch logits. No memory bank retrieval."""
        q = self.project(emb)
        return self.distill_head(q)                                     # (b 14795)

    def forward(self, emb, site_idx=None):
        """
        Args:
            emb: (B, 1536) Perch embeddings
            site_idx: (B,) int site indices, or None for focal windows

        Returns:
            scores: (B, 234) calibrated multi-label probabilities
        """
        q = self.project(emb)                                           # (b d)
        q_r = self.W_Q(q)                                               # (b d) — retrieval-space query

        # Species retrieval: logsumexp over K (soft-max Hopfield energy).
        # All K prototypes receive gradient weighted by softmax attention;
        # β → ∞ recovers hard max, β → 0 recovers mean.
        beta_s = self.log_beta_species.exp()
        M_s = self.W_K_species(self.species_memory)                     # (s k d)
        sp_sim = einsum(q_r, M_s, 'b d, s k d -> b s k')                # (b s k)
        sp_agg = torch.logsumexp(beta_s * sp_sim, dim=-1) / beta_s      # (b s)
        sp_score = torch.sigmoid(self.cal_scale * sp_agg + self.cal_bias)

        if site_idx is None:
            # Focal: no site info → species-only scoring
            return sp_score

        # Site retrieval: logsumexp over K site prototypes → typicality scalar.
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

The species × site affinity matrix `A(s, l)` from the dimensional analysis
doc is computed at eval time (not in forward) using the W_K-projected
memory banks — both sides live in the retrieval space:

```python
# A(s, l) = (1/β) · logsumexp over (k_s, k_l) pairs
beta = self.log_beta_species.exp()  # or a shared affinity temperature
M_s = self.W_K_species(self.species_memory)                            # (s ks d)
M_l = self.W_K_site(self.site_memory)                                  # (l kl d)
pair = einsum(M_s, M_l, 's ks d, l kl d -> s l ks kl')
affinity = rearrange(pair, 's l ks kl -> s l (ks kl)')
A = torch.logsumexp(beta * affinity, dim=-1) / beta                    # (s l)
```

When `site_idx=None` (focal windows), the model returns species-only scores.
The site bank and gate never see focal data — they are trained exclusively
by soundscape windows.

## Three data sources

| Source | Count | Labels | Site | Domain | Role |
|--------|-------|--------|------|--------|------|
| Focal recordings | 265K | Single-label (one-hot) | None | Clean, single-species | Species identity |
| Labeled soundscapes | 739 | Multi-label (multi-hot) | Yes | Target (Pantanal) | Species identity + site + domain anchor |
| Unlabeled soundscapes | 127K | None | Yes | Target (Pantanal) | Domain structure |

All cached as Perch v2 embeddings on scratch from exp 106:
```
~/scratch/birdclef-2026/features/exp106/
├── focal/by_species/{species}.parquet          # 206 species
├── soundscape_labeled.parquet                   # 739 windows
└── soundscape_unlabeled/by_site/{site}.parquet  # 21 sites
```

## Batch construction

Each training step samples from all three sources independently:

```python
batch = {
    "focal":      sample(focal_windows, n=64),       # single-label, no site
    "labeled_sc": sample(labeled_windows, n=32),      # multi-label, with site
    "unlabeled":  sample(unlabeled_windows, n=256),   # no labels, with site
}
```

Focal sampling: uniform across species (not windows) to handle the 499x class
imbalance. Each species contributes equally to the gradient regardless of how
many focal recordings it has.

Labeled SC sampling: with replacement (739 is small, will be resampled many times).

Unlabeled sampling: uniform across windows (already site-partitioned on disk;
sample from a random site parquet per step for I/O efficiency).

## Loss

```python
def compute_loss(model, batch, lambda_ent=0.1, alpha_focal=0.5):
    # === Focal: species identity, no site ===
    pred_f = model(batch["focal"]["emb"], site_idx=None)   # species bank only
    L_focal = F.binary_cross_entropy(pred_f, batch["focal"]["labels"])

    # === Labeled soundscape: species + site + gate ===
    pred_l = model(batch["labeled_sc"]["emb"], batch["labeled_sc"]["site"])
    L_sup = F.binary_cross_entropy(pred_l, batch["labeled_sc"]["labels"])

    # === Unlabeled soundscape: entropy minimization ===
    pred_u = model(batch["unlabeled"]["emb"], batch["unlabeled"]["site"])
    p = pred_u.clamp(1e-7, 1 - 1e-7)
    L_ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

    return alpha_focal * L_focal + L_sup + lambda_ent * L_ent
```

What each loss teaches:

- **L_focal**: Species memory bank learns what each species sounds like.
  265K clean labels — strongest species identity signal. The projection
  must map focal embeddings into the shared space where species retrieval works.

- **L_sup**: Everything. Species identity in soundscape domain, site bank
  behavior, gate learning. Only 739 windows but RIGHT domain — anchors the
  shared space to soundscape reality.

- **L_ent**: Confidence on unlabeled data. Pushes prototypes apart, pulls
  windows toward nearest prototypes, shapes the site bank with 127K windows.
  No species labels needed.

## Domain bridging

The projection layer sees both focal and soundscape embeddings:

```
proj(focal_emb)     → q_focal     ─┐
proj(sc_emb)        → q_sc        ─┤── same D-space, same species memory
                                    │
species_memory (234, K, D)  ────────┘
```

L_focal forces species memory to work for focal queries.
L_sup forces species memory to work for soundscape queries.
Both losses backprop into the same projection — it must implicitly align
the two domains. This is domain bridging without an explicit domain
adaptation loss.

If the domains are too different for a linear projection to align, the
model will prioritize soundscape (L_sup anchors it) and L_focal will
plateau. That's fine — the focal data becomes less useful but doesn't hurt.

## Initialization

Initialization is no longer a delicate balance between "warm start" and
"bias the optimization" because **phase 0 distillation pretraining does
the heavy lifting** — by the time the memory banks need sensible starting
values, the projection has already been trained on 400K+ windows against
a strong teacher, and k-means in that pretrained space is well-posed for
both species and site banks.

**No transfer from exp 106 prototypes.** `multi_prototypes.npz` and
`site_prototypes.npz` are not loaded. Exp 106's prototypes live in
Perch's entangled logit space; exp 107's memory banks live in the phase-0
projected space. They're not dimensionally compatible.

**Projection — random Kaiming init, then trained in phase 0.** The 2-layer
MLP starts from standard Kaiming init and is trained end-to-end against
Perch logit distillation on all 400K+ windows (see phase 0 below). PCA
warm start is deferred as an ablation — with 20+ epochs of distillation,
the start point matters little.

**Species memory — k-means on phase-0 projected focal embeddings.** After
phase 0 the projection is meaningful (it was trained to preserve Perch's
species-discriminative information in 512-d), so k-means in that space
answers the "cluster species modes" question well. Procedure:

1. Freeze the phase-0 projection.
2. Project all focal recordings through it to get `(265K, 512)` focal
   queries, partitioned by species.
3. For each species s with ≥K_species=16 focal windows, run k-means with
   K=16 on its focal queries. For species with fewer windows, use all
   windows as centers and pad with small Gaussian noise to fill K slots.
4. Write the 16 cluster centers per species into
   `self.species_memory[species_idx]`.

Note the key difference from the earlier "random init, let soft-EM sort
it out" plan: the projection is no longer cold at init time. The soft-EM
gradient identity still applies during fine-tuning (phases 2–3), but the
starting point is a real clustering of focal data in a meaningful space,
not random noise. Small-species edge cases (≤16 focal windows) are
handled by padding with noise around the few available windows.

**Site memory — k-means on phase-0 projected unlabeled soundscapes.** Same
procedure, applied to unlabeled soundscape embeddings partitioned by site:

1. Freeze the phase-0 projection.
2. Project unlabeled soundscapes through it, grouped by site (already
   partitioned on disk as `by_site/{site}.parquet`).
3. Run k-means (K=16) per site on the projected embeddings.
4. Write the 16 cluster centers per site into `self.site_memory[site_idx]`.

This warm-starts the site bank at the actual acoustic states of each site
(dawn chorus, midday quiet, rain, insect wall, …) after the projection
has learned what Perch considers species-relevant. Because both banks are
initialized from the same phase-0 projection, the species × site affinity
matrix `A(s, l)` is meaningful from step 1 of phase 2 — not something
that only becomes valid after many fine-tuning epochs.

**W_Q, W_K_species, W_K_site** start at identity-like random init (standard
linear layer init). They're trained from scratch during fine-tuning, not
phase 0 — phase 0 only touches the projection + distill head.

**Platt calibration** starts at identity (`a_s = 1`, `b_s = 0`).
**Hopfield temperatures** start at `log β = 0` (β = 1).

Deferred alternative (ablation): **skip phase 0 entirely** and fall back
to the earlier "random species init, k-means site init in PCA space"
recipe. This is the most important ablation — it directly measures what
phase 0 distillation is worth.

## Phase 0: Perch logit distillation

Phase 0 is where exp 107 does most of its actual learning. The previous
design treated 739 labeled windows as the anchor and sprinkled focal +
unlabeled in as helpers; that framing put the supervised signal/capacity
ratio in a dangerous regime (2.9M params vs 739 labels). The reframe
inverts the priorities: **distill Perch into a compressed 512-d shared
space using all 400K+ windows, then use the 739 labels only to fine-tune
the site bank and gate.**

### Data

All windows with cached Perch features — no domain split:

| Source | Count | Role in phase 0 |
|--------|-------|-----------------|
| Focal recordings | 265K | Teacher: Perch's logits on focal audio |
| Labeled soundscapes | 739 | Teacher: Perch's logits (labels ignored) |
| Unlabeled soundscapes | 127K | Teacher: Perch's logits |

Labels are intentionally **not used** in phase 0. The supervised signal
is Perch's own output on each window — a soft target over 14,795 species.
The 739 labels are held for later phases.

### Loss

Per window, student predicts the teacher's logits:

```python
def distill_loss(model, emb, perch_logits, T=2.0):
    # Student: projection + distill head
    student_logits = model.distill_forward(emb)                    # (b, 14795)

    # Teacher: cached Perch logits, temperature-softened
    teacher = F.log_softmax(perch_logits / T, dim=-1)
    student = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence (Hinton-style distillation)
    return F.kl_div(student, teacher, reduction='batchmean', log_target=True) * T * T
```

Temperature `T=2.0` softens the teacher's distribution enough to carry
similarity-structure information (not just the argmax). Hinton-style
distillation scaling (`T²`) keeps the gradient magnitude comparable across
temperatures.

### What phase 0 trains

Only the projection MLP and the distill head. The memory banks, W_Q/W_K,
Platt, gate, and log_β are **not touched** in phase 0 — they get their
start from k-means (memory banks) or random init (W_Q/W_K/gate) after
phase 0 completes.

### What phase 0 teaches

The projection learns to compress Perch's 14,795-d logit space into 512-d
in a way that preserves species discrimination. Focal and soundscape
embeddings both flow through the same projection supervised by the same
teacher, so the projection is forced to find a shared representation
that works for both domains. This is the actual domain-alignment step —
the one that in the previous design was hoped to fall out of 739
supervised labels.

Importantly, the student is a *bottleneck* (512-d) vs the teacher's
14,795-d output. It cannot reproduce Perch exactly; it must compress. The
compression is what makes the space useful — irrelevant Perch dimensions
(the 96.8% that shift between domains) get collapsed, and the
species-discriminative axes survive.

## Training schedule

| Phase | Epochs | Sources | Components trained | Loss |
|-------|--------|---------|-------------------|------|
| **0** | **30** | **All 400K+ windows** | **proj, distill_head** | **KL(student, Perch\|T)** |
| 1 | 10 | Focal + labeled SC | proj, species_memory, W_Q, W_K_species, log_β_s, Platt | α·L_focal + L_sup |
| 2 | 10 | Focal + labeled SC | + site_memory, W_K_site, log_β_l, gate | α·L_focal + L_sup |
| 3 | 20 | All three | All (small lr) | α·L_focal + L_sup + λ·L_ent |

Optimizer: AdamW, lr=1e-3 for phase 0, lr=3e-4 for phase 1, lr=1e-4 for
phase 2, lr=3e-5 for phase 3. Cosine decay within each phase.
Early stopping on LOSO AUROC (evaluated every 5 epochs, starting at phase 1).

Phase 0 is the bulk of the compute but runs once per fold (or once across
all folds — see LOSO evaluation below). Phases 1–3 are small fine-tunes.

Phase 1 (run after k-means initialization of memory banks) refines the
species bank in the pretrained shared space. This is the "learned species
retriever" — if it can't beat ridge (0.749) after phase 1, the pretrained
projection isn't adding value and we stop. **This is the go/no-go gate**
— it's the cheapest possible test of whether phase 0 distillation gave
us something better than exp 106 could.

## LOSO evaluation

7 folds (sites with ≥20 labeled windows). **Phase 0 runs once across all
folds** — it uses no labels, so there is no held-out-site leakage risk
from training the projection on unlabeled soundscapes from all sites
simultaneously. Phases 1–3 are re-run per fold with a reset of the
supervised components (species memory, site memory, W_Q/W_K, Platt, gate).

Per fold:

1. **Shared phase 0 checkpoint**: load the once-trained projection +
   distill head.
2. **K-means init**: cluster phase-0 projected focal embeddings into
   species memory, cluster phase-0 projected unlabeled soundscapes (from
   all non-held-out sites) into site memory. Re-initialize W_Q, W_K_*,
   Platt, gate, log_β.
3. **Fold split**: train (labeled SC from all sites except held-out) /
   test (held-out labeled SC).
4. **Train phases 1→2→3** (focal always in train; unlabeled SC from
   non-held-out sites in phase 3).
5. **Evaluate on held-out site**.

Final metric: mean of per-fold macro AUROC (same as exp 106).

Phase 0 is shared across folds to save compute (30 epochs × 400K windows
is the bulk of the cost). Phases 1–3 are small fine-tunes (~40 epochs on
focal + 739 − fold-held-out labeled SC, plus unlabeled entropy min). With
~4.1M params, phase 0 dominates wall time; the per-fold fine-tune is
quick.

Leakage audit: phase 0 sees held-out-site *embeddings* (via unlabeled SC),
but never their *labels* — labels don't exist for unlabeled SC. This is
standard semi-supervised LOSO practice. An optional stricter ablation
re-runs phase 0 per fold with the held-out site's unlabeled windows
excluded; expected to make no measurable difference.

## Ablations

| Config | Phase 0 | Species bank | Site bank | Focal | Unlabeled | Tests |
|--------|---------|-------------|-----------|-------|-----------|-------|
| A0 | ✗ | ✓ | ✗ | ✗ | ✗ | Earlier baseline: random init, labeled SC only (drops to ridge-equivalent) |
| **A** | **✓** | **✓** | **✗** | **✗** | **✗** | **Does phase 0 distillation + k-means init beat ridge on labeled SC alone?** |
| B | ✓ | ✓ | ✗ | ✓ | ✗ | Does focal fine-tuning add lift after phase 0? |
| C | ✓ | ✓ | ✓ | ✓ | ✗ | Does site composition work in the pretrained space? |
| D | ✓ | ✓ | ✓ | ✓ | ✓ | Does entropy-min semi-supervision add value on top? |
| **E** | **✗** | **✓** | **✓** | **✓** | **✓** | **Critical ablation: full model without phase 0 — measures what phase 0 is worth** |

Expected outcome:
- A0 ≈ 0.60–0.70 (random init on tiny supervised data — probably bad)
- A ≈ 0.74–0.78 (matches or beats ridge)
- B > A (focal fine-tune refines species identity)
- C > B (site composition finally works in the pretrained space)
- D ≈ C or slightly higher (entropy min is polish)
- E ≪ D if phase 0 matters; E ≈ D if it doesn't

**E vs D is the headline ablation of this experiment**: it measures
whether distillation pretraining is what makes the learned star schema
work, or whether end-to-end supervised training on 739 windows could
have found the same shared space. If E ≈ D, phase 0 was a complexity
tax with no benefit and the original design was correct. If E ≪ D,
distillation was the unlock — and that result generalizes to other
small-supervised-data problems in this project.

If C ≤ B (even with phase 0), site composition doesn't help even in a
pretrained shared space, and the star schema's factored structure is
rejected for this data.

## Directory structure

```
user/anthony/experiments/107-hopfield-prototypes/
├── proposal.md                          # hypothesis, method, success criteria
├── design.md                            # ← this file
├── docs/
│   └── star-schema-dimensional-analysis.md  # tensor framework + recsys framing
├── pyproject.toml                       # declares dependency on birdclef_106
├── src/birdclef_107/
│   ├── __init__.py
│   ├── cli.py                           # entry point: distill, train, evaluate, ablate
│   ├── data.py                          # datasets + multi-domain batch sampler
│   ├── model.py                         # HopfieldPrototypeModel (incl. distill_forward)
│   ├── losses.py                        # three-source loss composition + KL distillation
│   ├── init.py                          # k-means memory bank init from phase-0 projection
│   ├── distill.py                       # phase 0: Perch logit distillation training loop
│   ├── train.py                         # phases 1-3 training loop
│   └── evaluate.py                      # LOSO cross-validation
├── configs/
│   ├── base.yaml                        # default hyperparameters
│   ├── distill.yaml                     # phase 0 distillation config
│   ├── ablation_a0.yaml                 # no phase 0, species only (old baseline)
│   ├── ablation_a.yaml                  # + phase 0, species only
│   ├── ablation_b.yaml                  # + focal fine-tune
│   ├── ablation_c.yaml                  # + site bank
│   ├── ablation_d.yaml                  # + unlabeled entropy (full model)
│   └── ablation_e.yaml                  # full model without phase 0 (critical ablation)
├── sbatch/
│   ├── _setup-venv.sh                   # reuses exp 106 venv, adds torch + einops
│   ├── distill.sbatch                   # phase 0, GPU, shared across folds
│   └── finetune.sbatch                  # phases 1-3, per-fold LOSO run
└── artifacts/                           # gitignored
    ├── checkpoints/
    │   ├── phase0.pt                    # shared distilled projection
    │   └── fold_{i}/phase{1,2,3}.pt     # per-fold fine-tune checkpoints
    ├── metrics/
    └── loso_results.json
```

## Dependencies

**External**

| Package | Used for |
|---------|----------|
| `torch` | Model, training loop, gradient computation (GPU recommended for phase 0) |
| `einops` | `einsum`, `reduce`, `rearrange` — memory-bank contractions and affinity matrix |
| `numpy`, `polars` | Data loading (inherited from exp 106) |
| `scikit-learn` | Metrics (roc_auc_score), k-means for species and site memory init |
| `pyyaml` | Config loading |

**Cached artifacts consumed from exp 106:**

| Artifact | Size | Used for |
|----------|------|----------|
| Perch v2 embeddings | 265K focal + 739 labeled + 127K unlabeled | Student input for all phases |
| Perch v2 logits (14,795-d) | Same windows | Phase 0 distillation target |
| Species mapping | JSON | Species ordering for fine-tune phases |

Phase 0 distillation requires that Perch *logits* (not just pooled
embeddings) are cached for all windows. Exp 106 stores these in
`~/scratch/birdclef-2026/features/exp106/` alongside the embeddings;
verify both are present before starting phase 0.

Deferred (ablation-only):

| Package | Used for |
|---------|----------|
| `netcal` | Post-hoc calibration comparison — Platt vs beta vs isotonic vs histogram binning against the in-model Platt baseline |

**Internal**

Exp 107 imports from the exp 106 package (declared in `pyproject.toml`):

- `birdclef_106.data_loading` — cached parquet loaders, label matrix builder,
  site extraction, species list construction

Artifacts consumed (not imported):
- `perch_species_mapping.json` — species ordering

No code is copied from exp 106. The species and site memories are
k-means-initialized from phase-0 projected embeddings; exp 106's
`multi_prototypes.npz` and `site_prototypes.npz` are **not** used.

## Testing plan

Run before any training:

1. **Shape tests**: `model.forward` produces correct output shape for focal
   (site_idx=None) and soundscape (site_idx=int) inputs;
   `model.distill_forward` returns `(B, 14795)`
2. **Gradient tests**: all learnable parameters receive gradients for
   appropriate inputs (site memory only for soundscape, W_Q/W_K_* trained
   only in phases 1+ not phase 0, distill_head trained only in phase 0)
3. **Phase-0 freeze test**: during phase 0, only `proj` and `distill_head`
   receive gradients; memory banks, W_Q/W_K_*, Platt, gate, log_β are zero
4. **Phase-1 freeze test**: during phase 1, `site_memory`, `W_K_site`,
   `log_beta_site`, `gate`, `distill_head` receive zero gradients
5. **K-means init tests**: phase-0-projected k-means init produces finite,
   correctly-shaped species `(234, 16, 512)` and site `(21, 16, 512)`
   tensors; per-species/per-site centers are distinct (not collapsed);
   small-species padding case handled (species with <16 focal windows)
6. **Affinity test**: `A(s, l)` computed via einsum using `W_K_species`
   and `W_K_site` matches a brute-force quadruple loop over
   `(s, l, k_s, k_l)` on a small synthetic model
7. **Distillation target test**: phase-0 `distill_forward` can drive KL
   loss on a random 64-window batch to near-zero in 500 steps (sanity
   check that the MLP projection + distill head has capacity to reproduce
   Perch's logits on a tiny sample)
8. **Overfit test**: full fine-tune model can drive a single 32-window
   supervised batch's BCE to near-zero in 100 steps
9. **Phase-0 sanity**: after phase 0, held-out window top-K species
   (argmax of student distill_head output) overlaps with Perch's top-K at
   >80% (student has learned to reproduce the teacher's discriminative
   structure)
10. **Parity test**: ablation A (phase 0 + species-only fine-tune)
    produces LOSO AUROC within 0.02 of exp 106's ridge baseline (0.749).
    If this fails, phase 0 isn't producing a useful shared space and
    something is wrong with the distillation setup.

Tests 1–8 are seconds each. Test 9 runs after phase 0 completes and is
the "did pretraining work" sanity check. Test 10 is the "did the learned
space beat the fixed one" go/no-go gate.

## SLURM setup

Phase 0 wants a GPU for throughput — 400K+ windows × 30 epochs is
non-trivial on CPU even with cached embeddings. Fine-tuning phases are
CPU-tractable.

- **Phase 0**: Partition `gpu-rtx6000`, 1× RTX 6000, 16 GB RAM, 6h walltime.
  Single job, shared across folds.
- **Phases 1–3 + LOSO**: Partition `cpu-small`, 4 CPUs × 8 GB RAM, 3h walltime.
  7 folds × 4 ablation configs.
- Venv: reuses `~/scratch/birdclef-2026/.venv-106` with added torch + einops.
- No Luigi, no spawn multiprocessing — single-process training loop.

## Connection to exp 106

```
Exp 106 infrastructure (reused):
├── Cached embeddings on scratch (739 + 127K + 265K)
├── Partitioned parquets (by_species/, by_site/)
├── Species mapping (perch_species_mapping.json)
├── Label matrix construction (build_binary_label_matrix)
├── LOSO evaluation utilities
└── data_loading.py (load_focal_species, load_site, load_labeled, etc.)

Note: `multi_prototypes.npz` and `site_prototypes.npz` are intentionally
not inherited — memory banks learn from scratch.

Exp 107 adds:
├── PyTorch model (~120 lines)
├── Dataset + batch sampler (~80 lines)
├── Training loop with three-source loss (~100 lines)
├── LOSO cross-validation wrapper (~60 lines)
└── Ablation configs + evaluation (~40 lines)
```

## Open decisions

1. **Shared vs per-domain projection**: Start with one `proj` layer for both
   focal and soundscape. Revisit if L_focal plateaus while L_sup improves.
2. **Memory L2 normalization**: Optional projection step after each gradient
   update to constrain memory vectors to the unit sphere. Makes β
   interpretable but may slow learning. Start without it.
3. **K ablation**: K_s = K_l = 16 is the default. Sweep {4, 8, 16, 32} to
   see where capacity saturates. With logsumexp retrieval, extra slots are
   wasted capacity rather than a hard failure.
4. **Projection init**: PCA on soundscape embeddings vs pure random. PCA is
   expected to converge faster but final LOSO should be similar.
5. **Focal loss weight α**: Starts at 0.5. Tune based on relative L_focal
   and L_sup magnitudes in early training.
