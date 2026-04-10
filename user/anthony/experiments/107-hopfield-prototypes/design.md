# Design: Hopfield Prototype Retrieval

## Overview

A small PyTorch model implementing a **modern Hopfield network** (Ramsauer
et al. 2020) with two memory banks (species, site). The species bank uses
sigmoid attention for multi-label retrieval; the site bank uses softmax
attention for single-winner state selection. This gives the star schema's
factored structure (species × site) a concrete computational realization:
prototype memory banks with attention-based retrieval through a shared
learned projection space.

Trained on cached Perch v2 embeddings from three domains — focal recordings
(265K, single-label), labeled soundscapes (739, multi-label), and unlabeled
soundscapes (127K, no labels). No audio processing, no large backbone.

The implementation uses plain PyTorch primitives (no `hflayers` dependency).
The computation is equivalent to modern Hopfield retrieval with per-head
choice of normalization (sigmoid for multi-label species, softmax for
single-winner site state).

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
class HopfieldPrototypeModel(nn.Module):
    """
    Species and site memory banks with sigmoid cross-attention.

    Parameters:
        n_species: 234
        n_sites: 21
        K_species: 4 (prototypes per species)
        K_site: 6 (prototypes per site)
        d_input: 1536 (Perch embedding dim)
        d_proj: 256 (shared projection dim)
    """

    def __init__(self, n_species, n_sites, K_species, K_site, d_input, d_proj):
        super().__init__()
        self.proj = nn.Linear(d_input, d_proj)

        self.species_memory = nn.Parameter(torch.randn(n_species, K_species, d_proj))
        self.site_memory = nn.Parameter(torch.randn(n_sites, K_site, d_proj))

        self.beta_species = nn.Parameter(torch.tensor(1.0))
        self.beta_site = nn.Parameter(torch.tensor(1.0))

        self.gate = nn.Linear(d_proj, 1)

    def forward(self, emb, site_idx=None):
        """
        Args:
            emb: (B, 1536) Perch embeddings
            site_idx: (B,) int site indices, or None for focal windows

        Returns:
            scores: (B, 234) multi-label probabilities
        """
        q = self.proj(emb)                                           # (B, D)

        # Species attention: sigmoid for multi-label
        sp_sim = torch.einsum('bd,skd->bsk', q, self.species_memory) # (B, S, K)
        sp_score = torch.sigmoid(
            self.beta_species * sp_sim.max(dim=2).values             # (B, S)
        )

        if site_idx is None:
            # Focal: no site info → species-only scoring
            return sp_score

        # Site attention: how well does this window match its site?
        site_mem = self.site_memory[site_idx]                        # (B, K_l, D)
        site_sim = torch.einsum('bd,bkd->bk', q, site_mem)          # (B, K_l)
        site_score = torch.sigmoid(
            self.beta_site * site_sim.max(dim=1).values              # (B,)
        )

        # Gated fusion
        g = torch.sigmoid(self.gate(q)).squeeze(-1)                  # (B,)
        score = (g * sp_score
                 + (1 - g) * sp_score * site_score.unsqueeze(1))     # (B, S)

        return score
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

Species memory from exp 106 focal prototypes:
```python
mp = np.load('multi_prototypes.npz')
proto_emb = mp['proto_emb']  # (234, 4, 1536)
with torch.no_grad():
    model.species_memory.data = model.proj(torch.tensor(proto_emb))
```

Site memory from exp 106 site prototypes (`site_prototypes.npz`).
Projection initialized with PCA of labeled + unlabeled embeddings.

## Training schedule

| Phase | Epochs | Sources | Components trained | Loss |
|-------|--------|---------|-------------------|------|
| 1 | 20 | Focal + labeled SC | proj, species_memory, β_s | α·L_focal + L_sup |
| 2 | 20 | Focal + labeled SC | + site_memory, β_l, gate | α·L_focal + L_sup |
| 3 | 50 | All three | All | α·L_focal + L_sup + λ·L_ent |

Optimizer: AdamW, lr=1e-3, cosine decay.
Early stopping on LOSO AUROC (evaluated every 5 epochs).

Phase 1 is a "learned ridge" warmup. If species-only can't beat ridge (0.749),
the learned projection isn't adding value and we stop.

## LOSO evaluation

7 folds (sites with ≥20 labeled windows). Per fold:

1. Split labeled soundscapes: train (all sites except held-out) / test (held-out)
2. Focal recordings: always included in train (no site overlap concern)
3. Unlabeled soundscapes: always included (no labels to leak)
4. Train phases 1→2→3
5. Evaluate on held-out site

Final metric: mean of per-fold macro AUROC (same as exp 106).

Model is re-initialized per fold. With ~90 epochs and 665K params, each fold
trains in ~2 min on CPU, ~15 sec on GPU.

## Ablations

| Config | Species bank | Site bank | Focal | Unlabeled | Tests |
|--------|-------------|-----------|-------|-----------|-------|
| A | ✓ | ✗ | ✗ | ✗ | Learned projection vs ridge (labeled SC only) |
| B | ✓ | ✗ | ✓ | ✗ | Does focal data help species bank? |
| C | ✓ | ✓ | ✓ | ✗ | Does site composition work in learned space? |
| D | ✓ | ✓ | ✓ | ✓ | Does semi-supervision add value? |

Expected outcome:
- A ≈ 0.749 (ridge-equivalent)
- B > A (focal helps species identity)
- C > B (site composition finally works)
- D > C (127K unlabeled shapes the space)

If C ≤ B, site composition doesn't help even in a learned space, and the
star schema's factored structure is rejected for this data.

## Connection to exp 106

```
Exp 106 infrastructure (reused):
├── Cached embeddings on scratch (739 + 127K + 265K)
├── Partitioned parquets (by_species/, by_site/)
├── Species mapping (perch_species_mapping.json)
├── Label matrix construction (build_binary_label_matrix)
├── LOSO evaluation utilities
├── Prototype initializations (multi_prototypes.npz, site_prototypes.npz)
└── data_loading.py (load_focal_species, load_site, load_labeled, etc.)

Exp 107 adds:
├── PyTorch model (~120 lines)
├── Dataset + batch sampler (~80 lines)
├── Training loop with three-source loss (~100 lines)
├── LOSO cross-validation wrapper (~60 lines)
└── Ablation configs + evaluation (~40 lines)
```
