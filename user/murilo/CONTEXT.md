# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-24 (end of session)

## Read these files first

Read at session start to get full context. Priority matters — skip the "skim"
ones if you're short on time, but read the top three analysis notes fully.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md` | **Headline finding** — ~14/15 roahaw "failures" are silent-crop label noise, 1/15 is a real confusion. Methodology validated. | Read fully (start here) |
| `user/murilo/analysis/notes/2026-04-24-certainty-and-species-composition.md` | `shannon_norm` is redundant with `top1_prob`; `bernoulli_sum` uniquely catches peaked-but-wrong. Species composition identifies roahaw as the highest-signal target. | Read fully |
| `user/murilo/analysis/notes/2026-04-24-triage-v5j-failure-modes.md` | Foundation — triage categorization of 3,654 val samples. Only 1% is `confident_wrong`; 50% of top-20-by-loss is co-occurrence, not failure. | Read fully |
| `user/murilo/analysis/README.md` | Tooling map — the six scripts (triage, certainty, species_composition, export_figures filtered mode, etc.) with usage | Skim |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights (Salman 0.937, Boredom 0.950+) | Skim |
| `user/murilo/FINDINGS.md` | What worked / didn't across exps 006–027 | Skim |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |
| `user/murilo/experiments/028-domain-shift-augmentation/proposal.md` | Still paused — diagnostics first | Skim only if you're resuming exp 028 |
| `user/murilo/notes/sed-architecture-overview.md` | Pipeline reference (13 sections) | Reference material |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Best single SED: 0.906** (v5j, 5s LSEHead, exp 019)
**Target: 0.948** (competition #1). Gap: 0.019.
**Branch**: `mg/model-attention-analysis` — PR #38

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | **Current best — ONNX Perch + SED 4-fold** |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.906** | **Best single SED — our ceiling** |
| v5j-dualval (exp 026) | 0.897 | Soundscape checkpoint selection (worse) |
| v5j-pl-v1 (exp 025 Phase B) | 0.897 | Pseudo-labels as separate rows (regressed) |
| v3-waveform-mix (exp 025 Phase C) | 0.894 | FAILED — Mario-style PL waveform mixing −0.012 |
| v11b simple mel norm (exp 027) | 0.895 | FAILED — mel norm regresses −0.011 |
| v10b addmix+BCE (exp 027) | 0.875 | FAILED — additive MixUp regresses −0.031 |
| v10a addmix+CE (exp 027) | 0.494 | FAILED — CE logits too small for sigmoid |

No LB runs this session — all work was analysis on the existing v5j checkpoint.

## What happened recently

### 2026-04-24 — analysis methodology + roahaw failure-mode headline

Built a reproducible analysis pipeline on top of the 04-23 attention explorer.
All scripts operate on `losses_v5j.parquet` + `losses_v5j_logits.npy` (already
precomputed) — no GPU or retraining needed for any of this work.

**New tooling** (under `user/murilo/analysis/`):
- `triage.py` — label-aware failure classification (6 categories, reads `train.csv` for secondary labels)
- `certainty.py` — four scalars from clipwise logits (`shannon_norm`, `bernoulli_sum`, `top1_top2_gap`, `n_active`), appended to triage parquet
- `species_composition.py` — per-category and pairwise species stats with log2 ratio
- `export_figures.py` filtered mode — triage-aware figure export, `--filter-category`/`--filter-species`/`--filter-saliency-informative`, dedupes on `(filename, offset_sec)`, output at `figures/<model>/<slug>/`
- Triage is now chained after `precompute_losses.py` in the sbatch job

**Three notes (read in this order)**:

1. **[triage note](analysis/notes/2026-04-24-triage-v5j-failure-modes.md)** — classifies all 3,654 val samples. Only 1% (`confident_wrong`, n=38) is unambiguous model failure; 50% of top-20-highest-loss is `cooccurrence_label_ambiguity` (the "wrong" top-1 is actually a known secondary species). `confident_wrong` is 100% focal; soundscape is 76% co-occurrence.

2. **[certainty + species composition note](analysis/notes/2026-04-24-certainty-and-species-composition.md)** — validation showed `shannon_norm` is largely redundant with `top1_prob` (corr −0.86); entropy is a sanity check on triage, not a new signal. `bernoulli_sum` uniquely distinguishes `confident_wrong` from `correct_uncertain` (both are peaked, but confident_wrong has extra low-confidence background classes firing). Species composition: `mixed_uncertainty` is non-uniform but not extreme (top-10 species = 21% of bucket). Identified `roahaw` (n=15, 71% failure rate) as the highest-signal species to investigate.

3. **[roahaw failure-mode analysis (headline)](analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md)** — ran filtered export on all 15 roahaw `mixed_uncertainty` samples, viewed 4 of them. Key findings:
   - **roahaw is NOT rare** (489 focal recordings, 92nd percentile). Falsified the rare-species hypothesis for roahaw, `coffal1`, `socfly1`, `epaori4` — all abundant, all over-represented in failures.
   - **roahaw val is bimodal**: 54% `correct_confident`, 31% `mixed_uncertainty`. Median gt_prob 0.96 but mean 0.62 — heavy near-zero tail.
   - **~14/15 mixed_uncertainty samples look like silent/low-content crops** — mel spectrograms are ambient noise/rain/faint recordings, gradient saliency is scattered, top-1 prob is low (mean ~0.14). The model honestly says "I don't hear roahaw here" and BCE penalizes it because the recording-level label is roahaw.
   - **1/15 is a real acoustic confusion** (rank15, row 2071): clear vocalizations visible in the mel, `saffin` (0.167) vs `roahaw` close behind. This is the ONE sample with `saliency_informative=True`.
   - **Methodology validated**: `saliency_informative` correctly flagged the single real failure out of 15.

**Tentative implication** (needs verification): ~64% of `mixed_uncertainty` val samples (the non-saliency-informative subset) may be silent-crop label noise, not model failures. Would explain the offline-val/LB anti-correlation: recipe changes may be optimizing around a label-noise artifact, not real discrimination.

**Not recommending any training experiments yet.** Sample size is 4 visually inspected out of 15; needs broader verification before proposing a fix.

### 2026-04-23 — model attention explorer + competitor context

- Built `analysis/model_explorer.ipynb` + supporting scripts (`attribution.py`, `model_loader.py`, `precompute_losses.py`, sbatch).
- Updated Kaggle discussion notes:
  - **Salman: 0.922→0.937** via "investigating attention values on examples with highest loss and minimum loss during training." EfficientNetV2-B0, focal-only.
  - **Boredom: 0.950+** (first iteration on unlabeled). Without unlabeled: 0.938–0.939. "Never been able to break 0.94 with a single model."
  - **Tucker Arrants: 0.925 without deep learning** — Perch embeddings + clustering + metadata.
- Paused exp 028 (domain-shift augmentation) — diagnostics first.

## What needs to happen next

### Priority 1: Verify the silent-crop hypothesis beyond roahaw

Before proposing any training experiment, confirm or refute the pattern on more species:

1. Run filtered export for the other abundant over-represented species:
   ```bash
   python export_figures.py --losses data/losses_v5j.parquet --config ... --checkpoint ... \
     --filter-category mixed_uncertainty --filter-species coffal1 --top-k -1 --device cpu
   # repeat for socfly1, epaori4
   ```
2. **Listen to the 15 roahaw `.wav` files** — humans can judge silence faster than reading spectrograms. If they sound silent/noisy, the interpretation is literal.
3. Open 5 roahaw `correct_confident` samples as the "good crop" baseline. Same species, opposite category — the spectrogram difference should be stark if the hypothesis holds.
4. Extend to saliency-informative-only slice across all species (`--filter-category mixed_uncertainty --filter-saliency-informative`, 196 samples) — this should be the "real discrimination failure" set.

### Priority 2: If confirmed, design a single labeled-data-only experiment

Plausible directions (one at a time, in isolation, no stacking):

- **Energy/VAD-based crop filtering during training** — avoid 5s crops without vocalizations. Reduces label-noise penalty without adding data.
- **Energy-weighted BCE loss** — down-weight loss on low-energy crops so the model isn't punished for honesty.
- **Use secondary_labels as positive supervision** — so "wrong" top-1s that are co-occurring species aren't penalized.

Each only worth running if Priority 1 verifies the hypothesis; otherwise we'd be stacking changes on a wrong premise.

### Priority 3: Exp 028 (still paused)

Combined domain-shift augmentation recipe. Fall back if diagnostics dead-end.

## Key patterns (updated 2026-04-24)

- **21+ changes, 0 LB successes** — 19 training recipe changes + 2 pseudo-label approaches. Neither moved LB.
- **NEW: ~64%(?) of `mixed_uncertainty` looks like label-noise from recording-level primary + random crop interaction.** Not yet verified beyond roahaw. If true, explains why training to reduce val loss doesn't help LB — we'd be teaching the model to hallucinate the primary species regardless of crop content.
- **NEW: `saliency_informative` is the important flag.** It restricts to `gt_prob ∈ [0.05, 0.9]` — where sigmoid gradients aren't saturated AND where the model is in genuine discrimination space (not silent-crop dismissal). The roahaw cross-section: 1 of 15 samples was saliency-informative, and that 1 was the real confusion.
- **NEW: rare-species hypothesis falsified** for 4 of 10 top over-represented species in `mixed_uncertainty` (roahaw 92nd pct, coffal1 98th, socfly1 97th, epaori4 65th). Balanced sampling wouldn't help them.
- **Salman's 0.937 is a modeling breakthrough, not a recipe one** — "investigating attention on high/low loss examples." Our findings may be aligned with this: the label-noise issue on focal crops is a modeling framing issue, not a recipe dial.
- **The gap is on labeled data**: Boredom (0.938) and Salman (0.937) beat us by 0.031 using the same focal + labeled soundscape data. No KD, no Perch.
- **Salman trains focal-only** — adding soundscapes drops him to 0.85. He uses PP/TTA but no soundscape training data. Our v5j trains on focal + soundscapes.
- **CE loss is dead** (7 tests). Additive MixUp is dead (3 tests). Pseudo-labels are dead (2 approaches). Mel norm, max trick, multi-context all regressed.
- **Offline metrics remain anti-correlated with LB — DO NOT trust them.**

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/analysis/` | **Active work — analysis tooling and notes** |
| `user/murilo/analysis/triage.py` | Classify val samples into 6 failure-mode categories (label-aware) |
| `user/murilo/analysis/certainty.py` | Add shannon_norm / bernoulli_sum / top1_top2_gap / n_active to triage parquet |
| `user/murilo/analysis/species_composition.py` | Per-category and pairwise species stats |
| `user/murilo/analysis/export_figures.py` | Bulk figure export; filtered mode takes triage categories + species + saliency flags |
| `user/murilo/analysis/model_explorer.ipynb` | Single-sample exploration (superseded by filtered export for bulk, still useful for single deep dives) |
| `user/murilo/analysis/notes/*.md` | All three 2026-04-24 findings notes — read them |
| `user/murilo/analysis/data/triage_v5j.parquet` | Enriched val table (triage + certainty columns) |
| `user/murilo/analysis/data/losses_v5j.parquet` | Precomputed val losses + logits (GPU step done) |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights |
| `user/murilo/FINDINGS.md` | Cross-experiment history |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints |
| `user/murilo/experiments/028-domain-shift-augmentation/` | Paused experiment — diagnostics-first decision stands |
