# Paper contribution — methodology section (Overleaf drop-in)

Murilo's 2–3 page contribution to the BirdCLEF 2026 working-note. The paper lives on **Overleaf**;
these files are LaTeX-ready content to paste/merge there. Theme: *the simplest model that gets the
most gains* — a data-driven, Occam's-razor ablation of the supervised baseline.

## Files
| File | What it is | Where it goes in Overleaf |
|---|---|---|
| `methodology-draft.md` | Full prose draft + **source map** (every LB number → tracked result) + appendix tables, for review | — (reference only) |
| `methodology.tex` | LaTeX body prose (subsections) | merge into the `\section{Supervised Baseline System}` source |
| `floats/tab_ablation_ladder.tex` | Centerpiece SED ladder table | `sections/floats/` + `\input` |
| `floats/tab_appendix_curated.tex` | Curated negative-results table (appendix) | `sections/floats/` + `\input` |

Figure 1 (taxa spectrograms) is already committed at `../figures/taxa_spectrograms.{pdf,png}`.

## Before submitting — two reconciliations (see ⚠️ in `methodology-draft.md`)
1. The current draft text says the SED member uses **focal loss** + **time/frequency masking**.
   The deployed `hgnet_b0_sed_v5j.yaml` uses **BCE** + **spectrogram mixup** + **LSEHead** (no
   focal loss, no SpecAugment). Confirm which recipe shipped the 0.917 SED member and fix the
   wording (the prose here follows the committed config).
2. The exp-017 high-res-mel gain (+0.030) was on **EfficientNet-B0** (n_fft=4096/n_mels=224); the
   deployed **HGNet** v5j reverted to n_fft=2048/n_mels=256 (4096 scored −0.002). The shipped
   model is 2048/256 — the ladder labels this honestly.

## Dependencies
- `booktabs` (already used by the existing `40_tab_baseline-ablation`).
- `\cite` keys reused: `hamer2025perch2`, `lin2017focal`.
- New labels introduced: `tab:ablation-ladder`, `tab:curated-negatives`, `sec:negatives`,
  `app:parity`, `app:curated` (add `\section`/`\label` in the appendix for the latter two).
