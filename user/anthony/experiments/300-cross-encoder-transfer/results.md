# Experiment 300 Results

**Status**: not yet run — table below is the template to fill after the probe.

## Frozen-embedding ridge LOSO comparison

Macro AUROC / AP over the 9-fold leave-one-site-out split on the labeled
soundscape windows. Source: `artifacts/probe_metrics_frozen.json`.

| Encoder    | Backend      | embed_dim | Macro AUROC | Macro AP | best λ |
|------------|--------------|-----------|-------------|----------|--------|
| beats      | torch        | 768       | —           | —        | —      |
| ast        | torch        | 768       | —           | —        | —      |
| eat        | torch        | 768       | —           | —        | —      |
| ssamba_hf  | torch        | 384       | —           | —        | —      |
| perch      | perch_cache  | 1536      | —           | —        | —      |
| birdnet    | birdnet_bmz  | 1024      | —           | —        | —      |

**Sanity anchor**: Perch ridge here should reproduce ≈ 0.749 AUROC (exp 106
embedding+EB track was 0.749; this probe is identity-prior + standardized, so a
small difference is expected).

## Metric bridge (AUROC ↔ Kaggle LB)

Perch carries both a LOSO AUROC (this table) and a Kaggle LB (Perch V18 = 0.925).
Use the pair to place the other encoders' AUROCs on the leaderboard axis. Record
the calibration here once computed.

## Notes

- Window count: 739 deduped labeled windows (note reconciles "749" pre-dedup).
- Fill `embed_dim` from the actual cache (`embed_dim` column) — values above are expected.
