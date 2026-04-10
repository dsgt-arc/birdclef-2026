# Design: Hopfield Prototype Retrieval

## Overview

A small PyTorch model that implements the star schema's factored structure
(species × site) as learnable Hopfield attention over prototype memory banks.
Trained on cached Perch v2 embeddings — no audio processing, no large backbone.

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
        # Projection: Perch embedding → shared space
        self.proj = nn.Linear(d_input, d_proj)
        
        # Memory banks (learnable)
        self.species_memory = nn.Parameter(torch.randn(n_species, K_species, d_proj))
        self.site_memory = nn.Parameter(torch.randn(n_sites, K_site, d_proj))
        
        # Learnable temperatures
        self.beta_species = nn.Parameter(torch.tensor(1.0))
        self.beta_site = nn.Parameter(torch.tensor(1.0))
        
        # Gated fusion
        self.gate = nn.Linear(d_proj, 1)
    
    def forward(self, emb, site_idx):
        q = self.proj(emb)                              # (B, d_proj)
        
        # Species attention: sigmoid for multi-label
        # q @ memory.T → (B, S, K) → max over K → (B, S)
        sp_sim = torch.einsum('bd,skd->bsk', q, self.species_memory)
        sp_score = torch.sigmoid(self.beta_species * sp_sim.max(dim=2).values)
        
        # Site attention: how well does this window match its site's prototypes?
        # Gather the site memory for each window's site
        site_mem = self.site_memory[site_idx]            # (B, K_site, d_proj)
        site_sim = torch.einsum('bd,bkd->bk', q, site_mem)
        site_score = torch.sigmoid(self.beta_site * site_sim.max(dim=1).values)  # (B,)
        
        # Gated fusion: species_score modulated by site match
        g = torch.sigmoid(self.gate(q)).squeeze(-1)      # (B,)
        score = g * sp_score + (1 - g) * sp_score * site_score.unsqueeze(1)
        
        return score  # (B, 234)
```

## Data pipeline

All data is cached from exp 106. No audio loading needed.

```python
# Labeled (supervised signal)
labeled = np.load('soundscape_labeled.parquet')  # 739 windows
# → embeddings (739, 1536), site_ids (739,), labels (739, 234)

# Unlabeled (semi-supervised regularizer)
for site in list_sites():
    df = load_site(site)  # partitioned parquets from exp 106
    # → embeddings (N, 1536), site_id repeated
```

Batch construction:
- Each batch: 64 labeled + 256 unlabeled windows
- Labeled: random sample with replacement (739 is small)
- Unlabeled: random sample from partitioned site parquets
- Site IDs carried through for site bank indexing

## Initialization

Species memory initialized from exp 106 focal prototypes:
```python
mp = np.load('multi_prototypes.npz')
proto_emb = mp['proto_emb']  # (234, 4, 1536) raw centroids
# Project through initial projection weights, then assign to memory
with torch.no_grad():
    model.species_memory.data = model.proj(torch.tensor(proto_emb))
```

Site memory initialized similarly from `site_prototypes.npz`.

Projection layer initialized with PCA of the labeled embeddings (or identity
if d_proj = d_input).

## Loss

```python
def loss_fn(model, labeled_batch, unlabeled_batch):
    # === Supervised BCE ===
    emb_l, site_l, y_l = labeled_batch
    pred_l = model(emb_l, site_l)
    loss_sup = F.binary_cross_entropy(pred_l, y_l)
    
    # === Semi-supervised: entropy minimization ===
    emb_u, site_u = unlabeled_batch
    pred_u = model(emb_u, site_u)
    # Encourage confident predictions (low entropy)
    entropy = -(pred_u * pred_u.clamp(min=1e-7).log() 
              + (1 - pred_u) * (1 - pred_u).clamp(min=1e-7).log())
    loss_ent = entropy.mean()
    
    # === Prototype coherence ===
    # Memory vectors should be close to the centroid of their high-scoring windows
    # (Implemented as a soft EM constraint — pulls memories toward data)
    # Deferred to phase 3 of training
    
    return loss_sup + 0.1 * loss_ent
```

## Training schedule

| Phase | Epochs | What's trained | Loss |
|-------|--------|---------------|------|
| 1 | 20 | proj + species_memory + beta_species | BCE only (labeled) |
| 2 | 20 | + site_memory + beta_site + gate | BCE only (labeled) |
| 3 | 50 | all | BCE + entropy (labeled + unlabeled) |

Optimizer: AdamW, lr=1e-3 with cosine decay.
Early stopping on LOSO AUROC (evaluated every 5 epochs).

Phase 1 is a "learned ridge" warmup — if it can't beat 0.749, the projection
isn't learning anything useful and we should stop.

## Evaluation

Same LOSO protocol as exp 106: hold out each site, train on remaining,
evaluate on held-out.

For proper LOSO, the model must be retrained per fold (7 folds for sites
with ≥20 windows). With ~90 epochs and tiny model, this is ~10 min total
on CPU, under 1 min on GPU.

Ablations:
1. Species bank only (no site) — does the learned projection beat ridge?
2. Species + site (no unlabeled) — does site bank help with labeled data alone?
3. Species + site + unlabeled — does semi-supervision add value?

## Connection to exp 106

```
Exp 106 infrastructure (reused):
├── Cached embeddings on scratch (739 + 127K + 265K)
├── Partitioned parquets (by_species/, by_site/)
├── Species mapping, evaluation utilities
├── LOSO evaluation code
└── Prototype initializations (multi_prototypes.npz, site_prototypes.npz)

Exp 107 adds:
├── PyTorch model (~100 lines)
├── Training loop (~80 lines)
├── LOSO cross-validation wrapper (~50 lines)
└── Ablation configs (~30 lines)
```

## What this tests

The core question from exp 106: is the star schema's factored structure
(species × site) valuable, or is it an architectural bias that doesn't
match the data?

- If species-only (phase 1) beats ridge: the learned projection finds a
  better space than raw Perch embeddings.
- If species+site beats species-only: site composition DOES help when the
  shared space is learned rather than assumed.
- If unlabeled helps: the 127K soundscapes contain exploitable structure
  beyond what 739 labeled windows capture.

If all three confirm, the star schema is validated as an architecture —
just not in Perch's native space.
