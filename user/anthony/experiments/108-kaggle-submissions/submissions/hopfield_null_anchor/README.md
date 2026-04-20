# Hopfield null-site-anchored residual submission (exp 107)

## Model

Checkpoint: `exp107-ar-softshrink-null-site-anchored-residual-seed42-sbatch/S18/best.pt`

- Variant: `softshrink_null_site_anchored_residual`
- Best single-seed pooled LOSO AUROC: **0.7801** (exp 107, seed 42)
- Fold S18 used as shipping checkpoint: trained on 724/739 = 98% of labeled SC windows
- Architecture: `PrototypeRetrievalModel` with
  - d_hidden=1024, d_proj=512, K_species=K_site=8
  - cross_attention_heads=2, proj/attn dropout 0.1
  - Compact Perch sidecar adapter (mapped_proj + sparse_embed, 14796 slots)
  - Softshrink null-site-anchored residual: `p = σ(shared_logit + scale · softshrink(delta))`
    where `delta = cond_logit − null_logit`

## Pipeline

1. Parse site from filename stem (`BC2026_Test_<file>_<site>_<date>_<time>.ogg`).
2. Perch v2 ONNX → (12, 1536) embedding + (12, 14795) label logits per clip.
3. `PerchSidecarProjector` → compact `mapped_probs (12,234)` + `topk_{idx,val} (12,64)`.
4. OOD sites (outside S01..S23) → `site_idx=None` (shared-species OOD fallback).
5. `PrototypeRetrievalModel.forward` → sigmoid probs (12, 234).

## How to ship

```bash
sbatch sbatch/stage_null_anchor.sbatch       # bundles deps + weights dataset
sbatch sbatch/dry_run_null_anchor.sbatch     # local 3-clip smoke test
# Then: kaggle datasets version + kaggle kernels push + kaggle competitions submit
```

## Result

- Dry-run: pending
- Public LB: pending
