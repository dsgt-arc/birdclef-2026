# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-24 (end of session)

## Read these files first

Read at session start to get full context. Priority matters — skip the "skim"
ones if you're short on time, but read the top three analysis notes fully.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/analysis/notes/2026-04-24-three-regimes-of-mixed-uncertainty.md` | **Headline finding (supersedes roahaw-only)** — `mixed_uncertainty` splits by median gt_prob into three regimes: bimodal silent-crop (50%), uniformly uncertain (33%), model-blind (17%). 8 abundant-but-model-blind species are the cleanest targets. | Read fully (start here) |
| `user/murilo/analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md` | Earlier finding on roahaw only. Silent-crop hypothesis holds for bimodal species. Read for the tooling-validation story; superseded by three-regimes note for the broader picture. | Read fully |
| `user/murilo/analysis/notes/2026-04-24-certainty-and-species-composition.md` | `shannon_norm` is redundant with `top1_prob`; `bernoulli_sum` uniquely catches peaked-but-wrong. Species composition identifies roahaw as the first investigation target. | Read fully |
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

3. **[roahaw failure-mode analysis](analysis/notes/2026-04-24-roahaw-failure-mode-analysis.md)** — ran filtered export on all 15 roahaw `mixed_uncertainty` samples, viewed 4 of them. Found ~14/15 are silent/low-content crops (BCE penalizing honest dismissal) and 1/15 is a real confusion (the one with `saliency_informative=True`). roahaw is NOT rare (92nd pct in training). Falsified the rare-species hypothesis. **Extended to three species + baseline in the next note.**

4. **[three regimes of mixed_uncertainty (headline)](analysis/notes/2026-04-24-three-regimes-of-mixed-uncertainty.md)** — verified on `coffal1`, `socfly1`, `epaori4` + `roahaw correct_confident` baseline. The silent-crop pattern does NOT generalize uniformly. `mixed_uncertainty` splits by **median gt_prob across all val samples per species** into three distinct regimes:
   - **Bimodal (median gt_prob ≥ 0.8)**: 87 species, 49.8% of mixed_uncertainty. Silent-crop label noise. Training to reduce these wouldn't help.
   - **Uniformly uncertain (0.3–0.8)**: 50 species, 32.8% of mixed_uncertainty. Real acoustic discrimination difficulty.
   - **Model-blind (< 0.3)**: 23 species, 17.4% of mixed_uncertainty. Model hasn't learned the species.

   Within model-blind, 8 species have **150+ focal recordings and still can't be learned** (`compot1` 368 focal, `redjun` 208, `epaori4` 188, `y00678` 188, `plcjay1` 186, `strowl1` 184, `sobcac1` 183, `greela` 224). These are the cleanest targets: no data excuses, genuine acoustic confusability with common species.

**Revised implication**: ~50% of `mixed_uncertainty` is probably label noise (not 64% as the earlier note extrapolated from roahaw alone). The other ~50% is real failure modes that need **regime-specific fixes**, not a single training change.

**Not recommending any training experiments yet.** The three-regime structure means any fix should target one regime and be evaluated against that regime's metric, not aggregate val loss.

### 2026-04-23 — model attention explorer + competitor context

- Built `analysis/model_explorer.ipynb` + supporting scripts (`attribution.py`, `model_loader.py`, `precompute_losses.py`, sbatch).
- Updated Kaggle discussion notes:
  - **Salman: 0.922→0.937** via "investigating attention values on examples with highest loss and minimum loss during training." EfficientNetV2-B0, focal-only.
  - **Boredom: 0.950+** (first iteration on unlabeled). Without unlabeled: 0.938–0.939. "Never been able to break 0.94 with a single model."
  - **Tucker Arrants: 0.925 without deep learning** — Perch embeddings + clustering + metadata.
- Paused exp 028 (domain-shift augmentation) — diagnostics first.

## What needs to happen next

### Priority 1: Broaden visual verification of the three regimes

1 rank01 per regime is thin. Extend to 3–5 species per regime to confirm the median-gt_prob diagnostic holds uniformly. Figures already exported for roahaw, coffal1, socfly1, epaori4 — need more:

1. Pick 2–3 additional species from each regime (e.g., more bimodal species like `houspa` / `yecpar`, more uniformly-uncertain species, more model-blind species like `compot1` / `greela` / `redjun`).
2. Visually confirm rank01/rank02 from each matches the predicted pattern.
3. **Listen to a few WAVs** — particularly from `bimodal` species to confirm the "silent crop" interpretation is literal in audio, not just spectrogram inference.

### Priority 2: Investigate the 8 abundant-but-model-blind species

The cleanest modeling targets — 150+ focal recordings each, median gt_prob <0.3. What's the failure structure?

1. For each of `compot1`, `greela`, `redjun`, `epaori4`, `y00678`, `plcjay1`, `strowl1`, `sobcac1`: what are the top-1 predictions across val samples? Do they cluster (same wrong species repeatedly) or scatter?
2. If they cluster on specific common species, we have **per-species confusion pairs** — actionable for margin loss / hard-negative mining.
3. If they scatter (like the 4 species we already analyzed), the failure is broader — likely needs representation improvement, not pair-specific fixes.

### Priority 3: Regime-specific experiment design (only after P1/P2)

Each regime needs different treatment. Don't stack changes:

| Regime | Plausible lever (labeled data only) |
|---|---|
| Bimodal / silent-crop | VAD or energy-based crop filtering; energy-weighted BCE; secondary_labels as positive supervision |
| Uniformly uncertain | Better mel/backbone features; discrimination-focused losses |
| Model-blind (abundant) | Margin-based or contrastive loss; hard-negative mining on specific pairs (if P2 finds them) |
| Model-blind (rare / no-focal) | Balanced sampling; soundscape pseudo-labels for the 2 no-focal species |

Each tested in isolation. Evaluated against metrics restricted to the targeted regime, not aggregate val loss.

### Priority 4: Exp 028 (still paused)

Combined domain-shift augmentation recipe. Fall back if diagnostics dead-end.

## Key patterns (updated 2026-04-24)

- **21+ changes, 0 LB successes** — 19 training recipe changes + 2 pseudo-label approaches. Neither moved LB.
- **NEW: `mixed_uncertainty` splits into three regimes by median gt_prob per species.** ~50% bimodal (silent-crop label noise), ~33% uniformly uncertain (real discrimination difficulty), ~17% model-blind (real weakness). Any fix must target one regime and be evaluated against that regime's metric — not aggregate val loss. This explains why recipe changes optimizing val loss don't help LB: they may be chasing the 50% label-noise share.
- **NEW: 8 abundant-but-model-blind species are the cleanest modeling targets.** `compot1`, `greela`, `redjun`, `epaori4`, `y00678`, `plcjay1`, `strowl1`, `sobcac1` — all have 150+ focal recordings AND median gt_prob < 0.3. No data excuses. Genuine discrimination failure.
- **NEW: `saliency_informative` flag is essential.** It restricts to `gt_prob ∈ [0.05, 0.9]` — where sigmoid gradients aren't saturated AND where the model is in genuine discrimination space (not silent-crop dismissal). For roahaw mixed_uncertainty: 1 of 15 samples was saliency-informative, and that 1 was the real confusion.
- **NEW: rare-species hypothesis falsified** for 4 of 10 top over-represented species in `mixed_uncertainty` (roahaw 92nd pct, coffal1 98th, socfly1 97th, epaori4 65th). Balanced sampling wouldn't help THESE — but it could help the other ~13 model-blind species that DO have under-represented focal data.
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
