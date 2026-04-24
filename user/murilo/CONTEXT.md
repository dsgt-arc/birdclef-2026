# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-23 (end of session)

## Read these files first

Read ALL of these at session start to get full context.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/analysis/README.md` | **Model attention explorer** — interactive attribution analysis tool | Read fully (start here) |
| `user/murilo/analysis/model_explorer.ipynb` | **Jupyter notebook** — gradient saliency, per-frame logits, LSE attention | Open and run |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | **Competitor insights** — Salman 0.937, Boredom 0.950+, techniques | Read fully |
| `user/murilo/FINDINGS.md` | What worked, what didn't, key patterns (exps 006–027) | Read fully |
| `user/murilo/experiments/028-domain-shift-augmentation/proposal.md` | Exp 028 proposal (PAUSED — diagnostics first) | Skim |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |
| `user/murilo/notes/sed-architecture-overview.md` | Complete pipeline reference (13 sections) | Skim |

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
| v3-waveform-mix (exp 025 Phase C) | 0.894 | FAILED — Mario-style PL waveform mixing -0.012 |
| v11b simple mel norm (exp 027) | 0.895 | FAILED — mel norm regresses -0.011 |
| v10b addmix+BCE (exp 027) | 0.875 | FAILED — additive MixUp regresses -0.031 |
| v10a addmix+CE (exp 027) | 0.494 | FAILED — CE logits too small for sigmoid |

## What happened in this session (2026-04-23)

### Kaggle discussion updated — major new competitor findings

Updated `notes/kaggle-discussion-best-single-model-lb.md` with new comments:

- **Salman: 0.922→0.937 single model** via "investigating attention values on examples
  with highest loss and minimum loss during training." EfficientNetV2-B0, trains on
  focal only (soundscapes drop to 0.85), uses PP/TTA. B0_NS now matches v2B0.
  Key quote: "recently started paying attention to the modeling part... it is one of
  the most important parts."
- **Boredom: 0.950+ single model** (timm+SED, 1st iteration on unlabeled data).
  Without unlabeled: 0.938-0.939. "Never been able to break 0.94 with a single model."
- **Tucker Arrants: 0.925 without deep learning** — clustering on frozen Perch
  embeddings + metadata engineering.

### Built model attention explorer (`user/murilo/analysis/`)

Motivated by Salman's breakthrough. Interactive Jupyter notebook for understanding
WHERE and WHY the model fails on specific samples.

**Tool**: `analysis/model_explorer.ipynb` — Jupyter notebook with:
- Pre-computed per-sample losses for all 3,654 val samples (focal + soundscape)
- Browse by loss rank (highest/lowest), filter by species or source
- 4-panel visualization per sample:
  1. Mel spectrogram — raw input to model
  2. Gradient saliency — which time-frequency regions drive the prediction
  3. Top-10 species bar chart — what the model predicts (with GT/predicted legend)
  4. Per-frame analysis — LSE effective attention + frame probabilities
- GT vs Top-1 saliency comparison — side-by-side when model predicts wrong species
- Batch view of top-5 highest-loss samples
- Audio playback for each sample

**Supporting files**:
- `analysis/model_loader.py` — config-driven model loading (works with any SED model)
- `analysis/attribution.py` — gradient saliency + LSE effective attention computation
- `analysis/precompute_losses.py` — batch GPU script for val loss → parquet
- `analysis/sbatch/run.sh` + `_job.sh` — SLURM job submission
- Pre-computed data in `analysis/data/losses_v5j.parquet` + `losses_v5j_logits.npy`

### Exp 028 paused — diagnostics first

Based on Salman's findings, decided modeling diagnostics should come before domain
shift augmentation. The 0.906→0.937 gap may be about understanding model attention,
not adding more augmentation. Exp 028 remains as-is but is not the immediate priority.

## What needs to happen next

### Priority 1: Analyze attention patterns in the explorer

Use `analysis/model_explorer.ipynb` to identify systematic failure modes:

1. **High-loss samples**: Where does gradient saliency point? Is the model attending
   to silence, noise, or wrong frequency bands?
2. **Low-loss samples**: Confirm saliency highlights the actual bird call — establish
   what "good" attention looks like.
3. **GT vs Top-1 comparison**: When the model confuses species, is it attending to
   the same or different spectrogram regions?
4. **Per-species patterns**: Do certain species (especially rare ones) have
   systematically diffuse attention?
5. **Soundscape vs focal**: Do soundscape samples show different saliency patterns?

### Priority 2: Derive actionable changes from diagnostics

Based on attention analysis findings, determine what to fix:
- If attention is systematically on wrong frequency bands → mel config issue
- If attention is diffuse/unfocused → head architecture or training issue
- If attention is on noise instead of signal → augmentation/domain gap issue
- If certain species are consistently confused → species-specific investigation

### Priority 3: Exp 028 (paused)

If diagnostics don't reveal a clear modeling fix, fall back to exp 028's combined
domain shift augmentation approach.

## Key patterns (updated 2026-04-23)

- **21+ changes, 0 LB successes** — includes 19 training recipe changes AND 2
  pseudo-label approaches. Neither recipe changes nor data additions improved LB.
- **Salman's 0.937 is a modeling breakthrough, not a recipe one** — he investigated
  attention on high/low loss examples and fixed something. We should do the same.
- **The gap is on labeled data**: Boredom (0.938) and Salman (0.937) beat us by
  0.031 using the same focal + labeled soundscape data. No KD, no Perch.
- **Salman trains focal-only** — adding soundscapes drops him to 0.85. He uses
  PP/TTA but no soundscape training data. Our v5j trains on focal + soundscapes.
- **CE loss is dead** (7 tests). Additive MixUp is dead (3 tests). Pseudo-labels
  are dead (2 approaches). Mel norm, max trick, multi-context all regressed.
- **Offline metrics remain anti-correlated with LB** — DO NOT trust them.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/analysis/` | **Active work — model attention explorer** |
| `user/murilo/analysis/model_explorer.ipynb` | Jupyter notebook for attribution analysis |
| `user/murilo/analysis/model_loader.py` | Config-driven model loading (any SED model) |
| `user/murilo/analysis/attribution.py` | Gradient saliency + LSE attention computation |
| `user/murilo/analysis/precompute_losses.py` | Batch GPU script for val losses |
| `user/murilo/analysis/data/losses_v5j.parquet` | Pre-computed val losses + metadata |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights (updated 2026-04-23) |
| `user/murilo/experiments/028-domain-shift-augmentation/` | Exp 028 — paused, diagnostics first |
| `user/murilo/FINDINGS.md` | What worked / didn't across all experiments |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints |
