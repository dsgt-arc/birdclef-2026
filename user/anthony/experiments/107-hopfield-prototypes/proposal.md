# Proposal: Hopfield Prototype Retrieval

**Status**: proposed
**Created**: 2026-04-10
**Author**: Anthony

## Hypothesis

A small Hopfield attention model with factored species × site memory banks,
trained on 739 labeled soundscapes and regularized on 127K unlabeled windows,
will learn soundscape-domain prototypes that outperform both the ridge model
(0.749 AUROC) and the star schema cosine matching (0.696 AUROC) from exp 106.

The key insight: exp 106 showed that composing focal prototypes × site prototypes
through Perch logit space fails because domain shift and site structure are
entangled. A learned projection into a shared space — with the star schema's
factored structure as the architecture — can disentangle them.

## Background

Exp 106 established:

1. **Ridge regression**: 0.749 AUROC (LOSO) — best linear model, but doesn't use
   site context or the 127K unlabeled soundscapes.
2. **Star schema cosine**: 0.816 AUROC on 42 focal species, but site affinity has
   zero effect. Root cause: the Perch logit/embedding space isn't compositional
   across domains — focal and soundscape activations live in different manifolds.
3. **Sites ARE different**: species Jaccard 0.00–0.25, residual cosine -0.68 to
   +0.89. The signal exists but can't be accessed through cross-domain dot products.
4. **127K unlabeled soundscapes**: untapped. They provide soundscape-domain coverage
   that could bootstrap domain-appropriate prototypes.

Modern Hopfield networks (Ramsauer et al. 2020) are equivalent to attention with
stored patterns as keys/values. The energy landscape provides convergence guarantees
and natural uncertainty via the Hopfield energy. With separate memory banks for
species and sites, the architecture encodes the star schema's compositional structure
while learning the shared space where compositions work.

## Method

### Architecture

```
Input: Perch v2 embedding (1536-d) for a 5s soundscape window
       Site ID (categorical, 21 sites)

                    ┌─────────────────────────┐
Window emb (1536) ──→  Projection (1536→256)  ──→ query q (256)
                    └─────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌──────────────┐ ┌──────────┐ ┌────────────┐
        │ Species bank │ │Site bank │ │  Optional  │
        │(234, K, 256) │ │(21,K,256)│ │ species×   │
        │  K=4 protos  │ │ K=6     │ │ site cross │
        └──────┬───────┘ └────┬─────┘ └─────┬──────┘
               │              │              │
        sigmoid attn    sigmoid attn   bilinear or
        (multi-label)   (soft select)  cross-attn
               │              │              │
               ▼              ▼              ▼
        species_score   site_score    interaction
          (234,)          (21,)        (234,)
               │              │              │
               └──────────┬───┘──────────────┘
                          ▼
                   Gated fusion → final scores (234,)

Loss: BCE on labeled windows
```

**Species memory bank** (234, K, 256): K=4 prototype vectors per species.
Initialized from exp 106 focal prototypes projected through the learned head.
Each prototype captures one acoustic mode (dawn song, alarm call, etc.).

**Site memory bank** (21, K, 256): K=6 prototype vectors per site.
Initialized from exp 106 site prototypes. Captures site-specific acoustic
environments (cicada-dominated, frog-dominated, etc.).

**Sigmoid attention** (not softmax): allows multi-label retrieval. Multiple
species can be present in one window. The temperature β is learnable per head.

**Gated fusion**: `output = g * species_score + (1-g) * (species_score * site_score)`
where g is a learned gate. When site information doesn't help, g→1 and the model
falls back to species-only scoring.

### Training

**Supervised signal**: 739 labeled soundscape windows with BCE loss.

**Semi-supervised regularizer on 127K unlabeled windows**:
- **Consistency**: augment by adding noise to embeddings, enforce that predictions
  are similar across augmentations (mean-teacher style).
- **Entropy minimization**: encourage confident predictions on unlabeled windows
  (low entropy in the species assignment).
- **Prototype coherence**: prototypes should be close to the centroid of their
  assigned windows (EM-style, differentiable).

**Training schedule**:
- Phase 1: train species bank only on labeled data (ridge-equivalent warmup)
- Phase 2: add site bank, fine-tune jointly
- Phase 3: add unlabeled regularizer

### Setup

| Component | Details |
|-----------|---------|
| Data | Exp 106 cached embeddings: 739 labeled, 127K unlabeled, 265K focal (all on scratch) |
| Compute | CPU sufficient (small model, cached features). GPU optional for speed. |
| Dependencies | PyTorch, exp 106 package (data loading, evaluation) |
| Code | `user/anthony/experiments/107-hopfield-prototypes/` |
| Init | Exp 106 prototypes (`multi_prototypes.npz`, `site_prototypes.npz`) |

### Variables

- **Independent**: architecture (species-only vs species+site vs +cross), K per bank,
  projection dim, β temperature, unlabeled regularizer weight
- **Dependent**: LOSO AUROC on 739 labeled soundscapes, per-species breakdown
- **Controlled**: same Perch v2 features, same LOSO evaluation as exp 106

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| LOSO AUROC | Leave-one-site-out macro AUROC (same as exp 106) |
| Focal vs no-focal split | AUROC on 42 species with focal data vs 28 without |
| Site prior lift | AUROC with vs without site bank (ablation) |
| Kaggle LB | Submission score (if LOSO > 0.75) |

### Baselines

| Model | AUROC (LOSO) | Source |
|-------|-------------|--------|
| Ridge (focal emb + EB) | 0.749 | Exp 106 |
| Star schema cosine | 0.696 | Exp 106 |
| Star schema (focal species only) | 0.816 (42 sp) | Exp 106 |

### Success Criteria

- **Confirm if**: species+site model > 0.76 AUROC (beats ridge + shows site benefit)
- **Bonus**: site bank ablation shows >0.01 lift (site composition finally works)
- **Reject if**: species-only model ≤ 0.749 (can't even match ridge, learned space doesn't help)

## Limitations

- 739 labeled windows is very small for training even a small neural model. Overfitting
  risk is high — must rely on the semi-supervised regularizer.
- LOSO CV with only 7 sites having ≥20 windows means high variance in fold estimates.
- The model learns a shared space for THIS dataset (Pantanal soundscapes). May not
  generalize to different ecosystems without retraining.
- Hopfield convergence guarantees apply to fixed patterns. With learned patterns
  updated by backprop, convergence is not guaranteed (but empirically fine for
  attention-based models).
