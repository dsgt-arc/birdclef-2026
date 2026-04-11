# Proposal: Hopfield Prototype Retrieval

**Status**: proposed
**Created**: 2026-04-10
**Author**: Anthony

## Hypothesis

A factored multi-prototype attention model (species × site memory banks over a
shared 512-d space learned by Perch-logit distillation on all 400K+ cached
windows) will outperform both the ridge model (0.749 AUROC) and the star schema
cosine matching (0.696 AUROC) from exp 106, by disentangling focal and
soundscape manifolds in the learned space.

Two linked insights drive this:

1. **Exp 106 showed that composing focal × site prototypes through Perch's
   fixed logit space fails** because domain shift and site structure are
   entangled. A *learned* projection into a shared space, with the star schema's
   factored structure as the architecture, can in principle disentangle them.

2. **739 labeled windows is too thin a supervised signal** to train a learned
   projection end-to-end. The trick is to use Perch itself as a teacher and
   distill its 14,795-d logits into a 512-d bottleneck over all 400K+ cached
   windows *before* any supervised fine-tuning. This turns 739 labels into a
   fine-tune polish on an already-shaped space, rather than the primary
   training signal for a 5M-parameter model.

Distillation pretraining is the load-bearing mechanism. The "Hopfield" framing
is theoretical pedigree for the logsumexp retrieval — the experiment's
success or failure hinges on whether the distilled shared space lets factored
species × site composition work, not on Hopfield-specific machinery.

## Background

Exp 106 established:

1. **Ridge regression**: 0.749 AUROC (LOSO) — best linear model, but doesn't use
   site context or the 127K unlabeled soundscapes.
2. **Star schema cosine**: 0.816 AUROC on 42 focal species, but site affinity has
   zero effect. Root cause: the Perch logit/embedding space isn't compositional
   across domains — focal and soundscape activations live in different manifolds.
3. **Sites ARE different**: species Jaccard 0.00–0.25, residual cosine -0.68 to
   +0.89. The signal exists but can't be accessed through cross-domain dot products.
4. **127K unlabeled soundscapes + cached Perch logits**: untapped as training
   signal. They provide a dense supervised target (Perch's own output) on
   soundscape-domain windows without needing human labels.

Modern Hopfield networks (Ramsauer et al. 2020) are equivalent to attention with
stored patterns as keys/values — with explicit W_Q, W_K, W_V projections the
formulation matches standard transformer attention. With separate memory banks
for species and sites, the architecture encodes the star schema's compositional
structure. With knowledge distillation from Perch, the shared space where
compositions happen is learned from 400K+ windows, not 739.

## Method

### Architecture

```
Input: Perch v2 embedding (1536-d) + site ID (21 sites)

Window emb (1536)                Phase 0 distill target:
      │                          Perch logits (14795-d)
      ▼                                  ▲
┌─────────────────────┐                  │
│  MLP projection     │                  │
│  1536→1024→512      │──────────────────┘ (distill head, phase 0 only)
│  GELU + LayerNorm   │
└──────────┬──────────┘
           │ q (512)
           ▼
       W_Q (512→512)
           │ q_r (512)
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│ W_K_s   │  │ W_K_l   │
│(512→512)│  │(512→512)│
└────┬────┘  └────┬────┘
     │ keys       │ keys
     ▼            ▼
┌─────────────┐ ┌─────────────┐
│Species bank │ │ Site bank   │
│(234,16,512) │ │(21,16,512)  │
└──────┬──────┘ └──────┬──────┘
  logsumexp/K     logsumexp/K
  + Platt         + sigmoid
  + sigmoid           │
       │              ▼
       │         site_score (1,)
       ▼
  species_score (234,)
       │              │
       └──────┬───────┘
              ▼
    Gated fusion: g·species + (1-g)·species·site → (234,)

Phase 0: KL(student, Perch logits | T=2)  on all 400K+ windows
Phase 1-3: BCE + entropy min on fine-tune data (739 labels)
```

**Phase 0: Perch logit distillation** (the main training mechanism).
The 2-layer MLP projection + a temporary distill head are trained to
reproduce Perch's own logits on all 400K+ cached windows, regardless of
domain. No labels needed. This is where the projection learns to
compress Perch's species-discriminative information into 512-d across
both focal and soundscape domains. The distill head is discarded after
phase 0; only the projection persists into fine-tuning.

**Species memory bank** `(234, 16, 512)`: k-means-initialized on
phase-0 projected focal embeddings after phase 0 completes, then
fine-tuned end-to-end. Slots are soft-EM components — logsumexp
retrieval + SGD performs an implicit E/M update on every step
(see design doc for the gradient identity).

**Site memory bank** `(21, 16, 512)`: k-means-initialized on phase-0
projected unlabeled soundscape embeddings, then fine-tuned.

**Explicit `W_Q`, `W_K_species`, `W_K_site` projections** on the
retrieval side (standard Ramsauer formulation). Lets the retrieval
space differ from the embedding space without breaking the Hopfield
framing.

**Logsumexp retrieval + sigmoid output**: the K slots are aggregated
with the modern Hopfield energy `(1/β) · logsumexp_k(β · q_r · W_K M_{s,k})`,
a differentiable soft-max that routes gradient to all K slots weighted
by softmax attention. The result passes through a per-species Platt
scale + bias and a sigmoid, giving independent multi-label probabilities.
β is learnable per head.

**Gated fusion**: `output = g * species_score + (1-g) * (species_score * site_score)`
where g is a learned gate. When site information doesn't help, g→1 and
the model falls back to species-only scoring.

### Training

Four-phase schedule. Phase 0 is the main training signal; phases 1–3
are fine-tune polish:

| Phase | Data | Signal | What's trained |
|-------|------|--------|----------------|
| 0 | All 400K+ windows | KL distillation from Perch logits | proj, distill head |
| 1 | Focal + labeled SC | BCE (one-hot focal + multi-hot SC) | proj, species bank, W_Q, W_K_species, log_β_s, Platt |
| 2 | Focal + labeled SC | + site bank + gate | + site_memory, W_K_site, log_β_l, gate |
| 3 | All three | + entropy min on unlabeled | all (small lr) |

| Source | Count | Labels | Role |
|--------|-------|--------|------|
| Focal | 265K | One-hot | Phase 0 target + phase 1-3 species supervision |
| Labeled SC | 739 | Multi-hot | Phase 0 target (labels ignored) + phase 1-3 site/gate supervision |
| Unlabeled SC | 127K | None | Phase 0 target + phase 3 entropy min |

**The key reframe vs the earlier design**: the 739 labeled windows are
no longer the primary training signal. Phase 0 distillation is. The
739 labels fine-tune the site bank and gate after the projection and
species bank have already been shaped by 400K+ windows of teacher
signal. This removes the "2.9M params trained on 739 labels" fragility
of the earlier design.

**Domain bridging**: Phase 0 forces the projection to compress Perch's
view of *both* focal and soundscape embeddings into the same 512-d
space, supervised by the same teacher logits on each. This is the
actual domain-alignment mechanism — not a hope that 739 supervised
labels will find it.

### Setup

| Component | Details |
|-----------|---------|
| Data | Exp 106 cached Perch v2 embeddings **and logits**: 265K focal, 739 labeled SC, 127K unlabeled SC |
| Compute | **Phase 0 needs GPU** (RTX 6000, ~6h). Fine-tune phases CPU-tractable. |
| Dependencies | PyTorch, einops, exp 106 package (`birdclef_106.data_loading`) |
| Code | `user/anthony/experiments/107-hopfield-prototypes/` |
| Init | Projection: Kaiming → phase 0. Memory banks: k-means on phase-0 projected embeddings. |
| Params | ~5.0M persistent (~12.6M during phase 0 including temporary distill head) |

### Variables

- **Independent**: phase 0 on/off, architecture (species-only vs +site), K per bank,
  distillation temperature T, unlabeled entropy weight λ, fine-tune lr schedule
- **Dependent**: LOSO AUROC on 739 labeled soundscapes, per-species breakdown,
  student/teacher top-K overlap after phase 0
- **Controlled**: same Perch v2 features, same LOSO evaluation as exp 106

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| LOSO AUROC | Leave-one-site-out macro AUROC (same as exp 106) |
| Phase 0 student/teacher top-K overlap | Did distillation reproduce Perch's discrimination at the bottleneck? |
| Phase 0 ablation lift | AUROC with vs without phase 0 (the critical ablation) |
| Focal vs no-focal split | AUROC on 42 species with focal data vs 28 without |
| Site prior lift | AUROC with vs without site bank |
| Kaggle LB | Submission score (if LOSO > 0.75) |

### Baselines

| Model | AUROC (LOSO) | Source |
|-------|-------------|--------|
| Ridge (focal emb + EB) | 0.749 | Exp 106 |
| Star schema cosine | 0.696 | Exp 106 |
| Star schema (focal species only) | 0.816 (42 sp) | Exp 106 |

### Success Criteria

- **Primary**: full model (with phase 0) > 0.76 AUROC — beats ridge, shows the
  learned shared space adds value.
- **Headline ablation**: phase 0 ablation (E vs D in the design doc) shows
  measurable lift. If E ≈ D, distillation pretraining was a complexity tax
  with no benefit.
- **Bonus**: site bank ablation (C vs B) shows >0.01 lift — site composition
  finally works in the pretrained shared space (which it didn't in exp 106's
  fixed Perch space).
- **Reject if**: ablation A (phase 0 + species only, the minimum configuration)
  fails to match ridge at 0.749 ± 0.02. This would mean phase 0 distillation
  is not producing a useful shared space and the whole construction is
  unsalvageable.

## Limitations

- Phase 0 distillation inherits Perch's discrimination ceiling — the student
  cannot exceed the teacher on tasks where Perch itself is the bottleneck.
  Exp 107 bets that the multi-prototype structure, factored site composition,
  and 739 target-domain fine-tune labels together push the student beyond
  the teacher, but this is not guaranteed.
- 739 labeled windows is small for training the site bank and gate (the
  phase-2 components), even with a pretrained projection. Overfitting risk
  is concentrated there.
- LOSO CV with only 7 sites having ≥20 windows means high variance in fold
  estimates.
- The model learns a shared space for THIS dataset (Pantanal soundscapes).
  May not generalize to different ecosystems without retraining.
- The "Hopfield" framing is theoretical gloss on standard multi-prototype
  attention. If the experiment fails, the cause is almost certainly the
  learned shared space or the supervised signal regime, not anything
  specific to Hopfield retrieval.
- Phase 0 requires Perch *logits* (not just pooled embeddings) cached for all
  400K+ windows. Verify before starting.
