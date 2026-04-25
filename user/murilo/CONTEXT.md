# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-25 (end of session)

## Read these files first

Read at session start to get full context. Priority matters — read the
top three fully, the rest can be skimmed.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/analysis/notes/2026-04-25-per-frame-attention-analysis.md` | **HEADLINE finding (high confidence)**: v5j is structurally honest. Per-frame attention entropy + GT-class logit std rise monotonically with failure severity; bimodal-species deep dive shows the same species produces qualitatively different head outputs under low-SNR vs clean conditions. Mechanistically explains 21+ failed recipe changes: most "improvements" teach false positives. Captures `silent-class` as exp 030 candidate. | Read fully (start here) |
| `user/murilo/analysis/notes/2026-04-25-audio-validation-low-snr-crops.md` | Audio listening validation. All 15 roahaw `mixed_uncertainty` clips + 10 `correct_confident` baseline + cross-species samples confirmed: failures are dominated by ambient/anthropogenic noise with weak/absent calls. Reframed "silent crop" → "low SNR within 5s window." Includes Salman recipe audit (disclosed vs assumed). | Read fully |
| `user/murilo/experiments/029-2025-pretrain/results.md` | **Active experiment — LB pending.** Stage 1 (2025 pretrain) val/auroc 0.972 ✓. Stage 2 (2026 fine-tune from pretrained init, classifier transfer for 41/234 overlap species) val/auroc 0.850 — within seed noise of v5j-control (0.851). Offline result is **neutral**; LB is the only signal that matters. | Read fully |
| `user/murilo/analysis/notes/2026-04-24-triage-v5j-failure-modes.md` | Foundation — triage categorization of 3,654 val samples. Only 1% is `confident_wrong`; 50% of top-20-by-loss is co-occurrence, not failure. | Skim |
| `user/murilo/analysis/notes/2026-04-24-three-regimes-of-mixed-uncertainty.md` | 04-24 working hypothesis. Largely *superseded* by the 04-25 per-frame finding (the regimes are real but the strategic implication has changed). | Skim for context only |
| `user/murilo/analysis/README.md` | Tooling map — triage, certainty, species_composition, export_figures, **per_frame_analysis (new 04-25)**, model_explorer | Skim |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights (Salman 0.937, Boredom 0.938-0.939 labeled-only / 0.950+ w/ unlabeled, Ali Ozan 0.941 with XC pretrain) | Skim |
| `user/murilo/FINDINGS.md` | Cross-experiment history (006–027) | Skim |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints | Skim |
| `user/murilo/experiments/029-2025-pretrain/proposal.md` | Active experiment — full hypothesis + method + scoping decisions | Reference if working on exp 029 |
| `user/murilo/experiments/028-background-noise-mixing/` | **PAUSED/DEMOTED** by per-frame analysis. Documents preserved as historical reasoning. Don't run. | Skip unless investigating BG-noise specifically |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023)
**Best single SED: 0.906** (v5j, 5s LSEHead, exp 019)
**Target: 0.948** (competition #1). Gap: 0.019.

**Active branch**: `mg/exp029-2025-pretrain` — PR #41 (open)

**Recent merges to main this session (2026-04-25):**
- PR #38 → analysis tooling + audio-validation note
- PR #40 → per-frame analysis pipeline + "v5j is structurally honest" finding
- PR #39 → exp 028 (BG noise mixing) parked as PAUSED/DEMOTED historical artifact

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | Current best — ONNX Perch + SED 4-fold |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.906** | Best single SED — our ceiling |
| **exp 029 stage 2 single fold** | **TBD (in flight 2026-04-25)** | 2025-pretrained → 2026 finetuned. Offline neutral (val/auroc 0.850 ≈ v5j-control 0.851). LB pending. |

## What happened recently

### 2026-04-25 — strategic shift: v5j is structurally honest, exp 029 in flight

**Morning**: audio validation of mixed_uncertainty failures.
- Listened to all 15 `mixed_uncertainty_roahaw` clips + 10
  `correct_confident_roahaw` baseline + samples from coffal1, socfly1,
  epaori4. Pattern across all listened-to mixed_uncertainty: foreground
  is dominated by ambient/anthropogenic noise (chainsaws, white noise,
  wind), bird calls are weak/distant/partial/absent. Baseline correct
  clips have the call as foreground. **Discriminator is foreground/
  background ratio, not call presence.**
- Reframed "silent crop label noise" (04-24 hypothesis) → "low SNR
  within the 5s crop." Same idea, more accurate framing.

**Midday**: per-frame analysis on v5j (LSEHead). Built
`per_frame_analysis.py`. Extended `precompute_losses.py` to dump
per-frame logits. **Headline finding (high confidence)**:

- GT-class attention entropy rises monotonically with failure severity:
  correct_confident 1.05 → mixed_uncertainty 1.53 → confident_wrong
  1.89 → gt_invisible 2.12 nats (uniform = 2.77).
- GT-class logit std *collapses* on failures: 4.31 → 1.97 → 1.30.
- Bimodal-species deep dive: same species (roahaw, coffal1, socfly1,
  compot1, whtdov, barant1) produces *qualitatively different* head
  outputs across categories. Entropy doubles/triples; logit std
  collapses 3-6×.
- Combined with the audio listening: **v5j is honest. Diffuse on hard
  crops because there's nothing to detect.**
- BCE+random-crop training penalizes that honesty (recording-level
  label says species is present whether or not it's in the 5s window).
  This *mechanistically explains* the 21+ failed recipe changes —
  most were trying to teach false positives.

**Strategic implication**:
- Demoted: BG noise mixing (would teach false positives), loss
  reweighting, AttHead substitution as single-variable, most
  augmentation tweaks.
- Promoted: architecture/backbone change, test-time inference, **2025
  pretraining** (Ali Ozan path — adds *information*, not training
  pressure). Captured the silent-class idea as a credible exp 030.

**Afternoon/evening**: exp 028 (BG noise mixing) scaffolded as
single-variable, then paused/demoted by the per-frame finding before
running. Merged as historical artifact (PR #39). exp 029 (2025-
pretrain → 2026-finetune) scaffolded and executed:

| Stage | Outcome |
|---|---|
| Stage 1 (pretrain on 2025 focal, 28K rows, 206 species) | val/auroc 0.972 at ep 15, 30 min walltime, clean convergence |
| Stage 2 (fine-tune on 2026, classifier transfer for 41/234 overlap species, lr 2e-4 warm start) | val/auroc 0.850 at ep 12, 38 min walltime — **within seed noise of v5j-control (0.851)** |
| Stage 3 (LB submission) | **In flight on Kaggle as of end-of-session.** Result pending. |

Triage on the stage-2 checkpoint (compare per-frame structure to v5j)
also queued; user runs the SLURM job when ready.

### 2026-04-24 — analysis methodology + working hypothesis on failure regimes

[Compressed for brevity — see prior CONTEXT.md commit history for
full text. Built triage/certainty/species_composition tooling. Three-
regimes-of-mixed_uncertainty hypothesis. Mostly superseded by the
04-25 per-frame finding which gave the regime story a mechanistic
backing (the model behaves differently on the same species under
different SNR).]

### 2026-04-23 — model attention explorer + competitor context

[See prior CONTEXT.md or `model_explorer.ipynb`. Built attribution.py +
precompute_losses.py + the marimo/jupyter explorer. Captured Salman
0.922→0.937 jump and Boredom 0.950+ insights. Paused exp 028.]

## What needs to happen next

### Priority 1 (in flight): exp 029 LB result

The actual signal we're waiting on. Three branching outcomes:

| LB result | Interpretation | Next |
|---|---|---|
| > 0.906 | Pretrain helped despite flat offline | Plan 4-fold scaleup of exp 029 |
| ≈ 0.906 (±0.005 seed noise) | Pretrain didn't move LB at our scale | Decide: retry with `head_lr: 5e-4` (tests whether reduced LR was wrong given 83% of head was random-init) OR pivot to exp 030 silent-class |
| < 0.906 by ≥0.005 | Pretrain hurt | Diagnose which axis: LR / classifier-transfer / 2025-data quality. Either retry one variable or pivot. |

### Priority 2 (parallel, in flight): triage on exp 029 stage-2 checkpoint

Compare per-frame entropy / logit_std on `mixed_uncertainty` vs v5j
baseline. Tells us whether pretraining changed the head's *internal*
behavior even when val/auroc didn't. SLURM command queued for user to
run; analysis lands after.

### Priority 3 (post-LB): decide next experiment

Three credible candidates depending on exp 029 outcome:

- **4-fold scaleup of exp 029** (if Stage 2 LB wins)
- **exp 030 silent-class** (if exp 029 is flat — engages the structural
  finding directly; supervises a "no signal" class via crop energy.
  Data acquisition for clean silent samples is the open challenge —
  see 04-25 per-frame note for sourcing options reviewed)
- **Salman recipe replication on EffNetV2-B0** (different architecture
  bet; bundle of ~6 simultaneous changes; only viable if narrower paths
  fail. Caveat: Salman's 0.922→0.937 jump is opaque)

### Deferred / superseded

- **04-24 three-regimes validation queue** (histograms, threshold
  sweep, cross-model on v5g, test-set relevance check): low priority
  now. The per-frame analysis already validated the part that mattered
  most for picking a lever, and the strategic frame has shifted from
  "fix mixed_uncertainty" to "the model is doing the right thing,
  add information not training pressure."
- **8 abundant-but-model-blind species investigation**: deferred for
  the same reason. May be revisited if exp 029/030 don't move LB.

## Key patterns (updated 2026-04-25)

- **HEADLINE (new)**: **v5j is structurally honest.** Per-frame
  analysis on 04-25 shows attention diffuses + logit std collapses on
  hard crops *because the call is genuinely weak/absent in those crops*
  (validated by audio listening). The BCE+random-crop training
  objective penalizes this honesty. **Mechanistically explains why 21+
  training-recipe changes have failed to move LB** — most were teaching
  false positives.
- **Direct implication**: stop testing recipe knobs that try to make
  the model more confident on hard crops. The lever is *more
  information* (pretraining, external data) or *better inference*
  (PP/TTA), not loss/augmentation tweaks. Architecture change is the
  remaining recipe-side bet but should be motivated by representation
  quality, not by trying to fix the diffuse-on-hard-crops pattern.
- **The audio finding is high confidence** for the bimodal regime
  (roahaw, coffal1, socfly1, compot1, whtdov, barant1). Generalization
  to all 87 bimodal species is inference-not-direct.
- **21+ recipe failures, 0 LB wins** — pattern continues. Now has a
  mechanistic explanation, not just an empirical curiosity.
- **Salman's 0.937 from "modeling" attention** — reinterpreted in light
  of per-frame finding: probably *not* "make attention less diffuse on
  hard cases" (that's what the head SHOULD do). More likely a
  structural attention/architecture change that doesn't fight the
  honesty pattern.
- **The gap is on labeled data**: Boredom (0.938) and Salman (0.937)
  beat us by 0.031 using the same focal + labeled soundscape data. Ali
  Ozan (0.941) used 2025 pretrain (filtered XC checkpoint) which is
  what exp 029 partially tests at smaller scale.
- **Offline metrics remain anti-correlated with LB — DO NOT trust
  them.** v5j (val/auroc 0.874) → 0.906 LB. v6l (val/auroc 0.874) →
  0.899 LB. Many regressions had higher offline.
- **CE loss is dead** (7 tests). **Additive MixUp is dead** (3 tests).
  **Pseudo-labels are dead** (2 approaches). **BG noise mixing is now
  also demoted** before being run, per the per-frame argument.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/analysis/` | **Active work — analysis tooling and notes** |
| `user/murilo/analysis/per_frame_analysis.py` | **NEW (04-25)** — LSE-implicit attention metrics per sample, aggregated by triage category, bimodal-species deep dive |
| `user/murilo/analysis/precompute_losses.py` | Extended 04-25 to also dump per-frame logits |
| `user/murilo/analysis/triage.py` / `certainty.py` / `species_composition.py` / `export_figures.py` | 04-24 tooling, all still active |
| `user/murilo/analysis/notes/2026-04-25-*.md` | Today's two key findings notes — read both |
| `user/murilo/analysis/data/triage_v5j.parquet` | Enriched val table (triage + certainty cols) |
| `user/murilo/analysis/data/losses_v5j_perframe_logits.npy` | NEW (04-25) — per-frame logits for v5j |
| `user/murilo/experiments/029-2025-pretrain/` | **Active experiment — Stage 2 done, LB pending** |
| `user/murilo/experiments/028-background-noise-mixing/` | PAUSED / DEMOTED historical artifact |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights |
| `user/murilo/FINDINGS.md` | Cross-experiment history |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints (exp 029 stage 1+2 added) |
