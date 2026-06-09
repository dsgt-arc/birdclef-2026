# Experiment 301 Results

**Status**: not yet run — tables below are templates to fill.

## Frozen → adapted Δ (ridge LOSO macro AUROC)

Frozen baseline from exp 300 (`probe_metrics_frozen.json`); adapted from the exp
300 probe on the `lora-r8` / `dapt-ep1` variants written by re-extraction.

| Encoder    | Frozen | LoRA r8 | Δ LoRA | DAPT | Δ DAPT |
|------------|--------|---------|--------|------|--------|
| beats      | —      | —       | —      | n/a  | n/a    |
| eat        | —      | —       | —      | —    | —      |
| ssamba_hf  | —      | —       | —      | —    | —      |

(BEATs is LoRA-only by design.)

## KS-shift (frozen vs adapted embeddings)

From `artifacts/ks_<encoder>_<variant>.json`. Higher = adaptation moved the
representation more.

| Encoder | Variant | median KS | % dims shifted (Bonferroni) |
|---------|---------|-----------|------------------------------|
| eat     | lora-r8 | —         | —                            |
| eat     | dapt-ep1| —         | —                            |

## LoRA rank sweep (optional)

Duplicate `configs/lora-<enc>.yaml` with `variant: lora-r{16,32}` and a matching
`lora.rank`, then re-extract + probe each variant.

| Encoder | r8 | r16 | r32 |
|---------|----|-----|-----|
| eat     | —  | —   | —   |

## Notes

- Δ is measured through the SAME ridge probe as the frozen baseline (apples-to-apples).
- DAPT uses a generic latent-bootstrap objective; see proposal for the genericity tradeoff.
