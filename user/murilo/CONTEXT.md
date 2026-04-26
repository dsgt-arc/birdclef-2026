# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-25 (end of session, after exp 029 LB result + path A no-op + working notes inventory)

## Read these files first

Read at session start to get full context. Priority matters — read the
top three fully, the rest can be skimmed.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/analysis/notes/2026-04-25-per-frame-attention-analysis.md` | **HEADLINE finding (high confidence)**: v5j is structurally honest. Per-frame attention entropy + GT-class logit std rise monotonically with failure severity; bimodal-species deep dive shows the same species produces qualitatively different head outputs under low-SNR vs clean conditions. Mechanistically explains 21+ failed recipe changes: most "improvements" teach false positives. Captures `silent-class` as exp 030 candidate. | Read fully (start here) |
| `user/murilo/analysis/notes/2026-04-25-audio-validation-low-snr-crops.md` | Audio listening validation. All 15 roahaw `mixed_uncertainty` clips + 10 `correct_confident` baseline + cross-species samples confirmed: failures are dominated by ambient/anthropogenic noise with weak/absent calls. Reframed "silent crop" → "low SNR within 5s window." Includes Salman recipe audit (disclosed vs assumed). | Read fully |
| `user/murilo/experiments/029-2025-pretrain/results.md` | **REJECTED.** Stage 2 LB 0.885 (−0.021 vs v5j). Stage 1 val/auroc 0.972, Stage 2 val/auroc 0.850 (offline-LB anti-correlation continued). Path A retry (split LR) was a silent no-op — `head_lr` is ignored under `scheduler_type: one_cycle` (only used in `cosine` branch). Decision matrix landed on REJECT; pivoting to working-notes review. | Read fully |
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

**Active branch**: `mg/exp029-2025-pretrain` — PR #41 (open). exp 029 is closed (REJECTED) — branch can be merged or kept open as parking lot for the path-A retry that was a no-op. User to decide.

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
| **exp 029 stage 2 single fold** | **0.885 (−0.021, REJECTED)** | 2025-pretrained → 2026 finetuned. Offline neutral, LB regressed. Mechanistic explanation: training-side perturbation of v5j's equilibrium. |

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
| Stage 3 (LB submission) | **0.885 LB (−0.021 vs v5j 0.906) — REJECTED.** Mid-flight slug fix (52→47 chars; original Kaggle slug exceeded the 50-char dataset-slug limit and silently failed to attach to the kernel). |

**Late evening — path A retry**: with offline-LB anti-correlation now
22 in a row, attempted the cheap split-LR diagnostic (`head_lr: 5e-4`,
`lr: 2e-4`) thinking the random-init head was undertrained. **Was a
silent no-op.** Investigation revealed: in `src/birdclef/lightning/sed_finetune.py:783`,
under `scheduler_type: "one_cycle"` the optimizer uses a single param
group with `lr=self.hparams.lr` only — `head_lr` is ignored. (`head_lr`
is wired up only in the `cosine` scheduler branch.) Both runs had
identical effective settings → bit-identical metrics (val/auroc
0.847821… to 16 decimals) → no information gained. Did not submit to
LB.

**Late evening — strategic pivot to working notes review**:
22+ training-side failures + the per-frame mechanistic explanation
make further training-side knob-turning a low-EV bet. Stepped back
to identify what we may have implemented *differently* from
competitors (e.g., waveform mixup details, BG noise mixing recipe,
loss specifics). Searched CEUR-WS for BirdCLEF 2023/2024/2025
working notes and identified gaps. Highest-leverage paper:
**Sydorskyi 2nd place 2025** — explicit domain-shift focus + GitHub
code + same author as 2023 1st place we already have a doc for.

Triage on the stage-2 checkpoint not run (decision: not blocking the
already-decided REJECT, low diagnostic value). Could be revived if
the working-notes review surfaces a specific structural hypothesis
worth testing on that checkpoint.

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

### Priority 1: Working notes deep-dive (start here next session)

After 22+ training-side failures + exp 029 REJECT, the move is to read
the formal CEUR-WS working notes from past BirdCLEF winners — far more
detailed than Kaggle discussion summaries — and identify implementation
differences from our pipeline. The published competition is in
2026; working notes for prior years (2023, 2024, 2025) are the closest
ground-truth recipes available.

**Highest-priority single source: Sydorskyi 2025 2nd place.** Same
author as the 2023 1st place we already have a doc for, explicit
domain-shift focus, has reproducible GitHub code at
`https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place`. Paper:
[CEUR Vol-4038 paper_256](https://ceur-ws.org/Vol-4038/paper_256.pdf).
**This is the recommended starting point** for the next session.

Tier-1 reading priority (likely high-leverage for the 0.03 gap):
1. **Sydorskyi 2nd place 2025** — paper_256 + GitHub repo
2. **Nikita Babych 1st place 2025** — full Kaggle writeup at
   https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
   (we have his inference notebook but the writeup adds: iterative
   noisy-student PL across 4 rounds, SoftAUCLoss, separate insect/
   amphibian pipeline, EffNet-l0/B4/B3/RegNetY ensemble)
3. **Witting et al 2024 — "Domain Shift in Bird Call Classification"** —
   [CEUR Vol-3740 paper-211](https://ceur-ws.org/Vol-3740/) — directly
   targets the focal→soundscape problem we're stuck on

Tier-2 reading priority (verify / complete what we have):
4. **Team Kefir 2024 1st place** — verify our existing doc (2024
   1st place) captures the CE-loss-with-sigmoid-inference details
   (our exp 027 keeps failing on CE; their 1st place used CE >> BCE)
5. **Dmitriev 2024** — [paper-193](https://ceur-ws.org/Vol-3740/) —
   practical CNN training methods for soundscape recordings

Tier-3 reading priority (own prior work — refresh):
6. DS@GT papers Murilo co-authored: BirdCLEF 2023 paper-177, 2024
   paper-202, 2025 "Distilling Spectrograms into Tokens"

Tier-4 (scan for surprises):
7. paper-172 (Hong 2023), paper-175 (Lasseck 2023), paper-204, 206

**Recommended workflow**: produce a methodology-map doc at
`docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md`
(matching our existing `birdclef2023-1st-place-solution.md` pattern)
based on paper_256 + the Sydorskyi GitHub repo. Then map our v5j
implementation against theirs to surface specific differences. ~1-2h
of focused work. After that, the next experiment proposal will have
real grounding rather than yet-another-recipe-tweak.

### Priority 2 (after working-notes review): pick next experiment

Candidates depend on what the lit review surfaces. Pre-review priors:

- **Faithful replication of one specific winning recipe** — most
  likely Sydorskyi 2nd (transfer + SS distillation + PP), since he
  publishes code and explicitly addresses our problem
- **exp 030 silent-class** — engages the structural finding directly;
  may overlap with techniques surfaced by the lit review
- **Salman recipe replication on EffNetV2-B0** — disclosed-parts ceiling
  is ~0.922; the +0.015 jump is opaque, so this is a weaker bet

### Deferred / superseded

- **04-24 three-regimes validation queue**: low priority. Per-frame
  finding already gave the structural backing.
- **8 abundant-but-model-blind species investigation**: deferred.
- **Triage on exp 029 stage-2 ckpt**: decision already made (REJECT);
  diagnostic value low unless lit review surfaces a structural
  hypothesis to test on it.
- **Path A retry (split LR fix)**: would require modifying
  `configure_optimizers` in `sed_finetune.py` to support split LR
  under `one_cycle` (currently only supported under `cosine`). Cheap
  fix (~10 lines). Defer unless lit review suggests split LR is a
  validated technique.

## Working notes / papers — inventory (added 2026-04-25 evening)

What we already have in `docs/references/kaggle/`:

| Year | Have | Source |
|---|---|---|
| 2023 | 1st place (Sydorskyi) summary | Kaggle writeup |
| 2024 | 1st place (Team Kefir) summary | Kaggle writeup |
| 2025 | 1st place (Nikita Babych) inference notebook + Python code; 9th place; ~30 BirdCLEF 2025 notebook references | Kaggle |
| Cross-cutting | `docs/references/kaggle/winning-solutions-analysis.md` (technique comparison matrix across 2023/24/25) | DS@GT internal |

What the formal CEUR working notes have that we don't:

**BirdCLEF 2023 — [Vol-3497](https://ceur-ws.org/Vol-3497/)**
- paper-172 — Hong: CNN training + OpenVINO inference acceleration
- paper-175 — Lasseck: CNN with attention on frequency bands
- paper-177 — Miyaguchi/Zhong/**Gustineli**/Hayduk (DS@GT)
- paper-176, 178 (lower priority)

**BirdCLEF 2024 — [Vol-3740](https://ceur-ws.org/Vol-3740/)**
- paper-193 — Dmitriev: CNN training methods for complex soundscapes
- paper-202 — Miyaguchi/Cheung/**Gustineli**/Kim (DS@GT)
- paper-204 — Porwal: EfficientNet-B0 + B1 ensemble
- paper-206 — Sampathkumar et al (TUC Media Computing): single learning models
- **paper-211 — Witting et al: "Addressing the Challenges of Domain Shift in Bird Call Classification"** ← directly relevant

**BirdCLEF 2025 — [Vol-4038](https://ceur-ws.org/Vol-4038/)**
- paper_232 (overview)
- "Notebook for the LifeCLEF Lab at CLEF 2025" (paper_249)
- **paper_256 — Sydorskyi & Gonçalves 2nd place: "Tackling Domain Shift via Transfer Learning + Semi-Supervised Distillation"** ← highest priority
- "Distilling Spectrograms into Tokens" — Miyaguchi/**Gustineli**/Cheung (DS@GT)
- "One Detector per Bird" — Suthraye Gokulnath et al (binary-classifier-per-species)

Other Tier-1 sources (non-CEUR):

- **Sydorskyi 2025 2nd place GitHub**: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- **Sydorskyi 2023 1st place GitHub**: https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
- **Nikita Babych 2025 1st place writeup**: https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
- **Team Kefir 2024 1st place**: Kaggle /competitions/birdclef-2024/discussion/512197
- **TheoViel 2024 GitHub**: https://github.com/TheoViel/kaggle_birdclef2024
- **JFPuget 2024 GitHub**: https://github.com/jfpuget/birdclef-2024

### Specific implementation questions to answer from the reading

Driven by our 22+ failures and the per-frame finding:

1. **Waveform MixUp specifics** — additive vs interpolative, applied
   before/after BG noise, label combination (max vs sum vs interpolated),
   application probability, alpha distribution
2. **BG noise mixing recipe** — sources, SNR distribution, mixing point
   (waveform vs mel), application probability
3. **Loss function full math** — particularly CE-on-clipwise-and-framewise-max
   (Salman's loss, Team Kefir's CE-vs-BCE win); exact reduction + aggregation
4. **Augmentation cascade order** — full sequence raw audio → input batch
5. **Inference / PP chain** — top winners' PP can be +0.01-0.02; ours has
   TTA + smoothing + TopN + priors but the parameters may be off
6. **Anything resembling silent-class / honest-no-call handling** — would
   inform our exp 030 candidate
7. **Training curriculum** — full schedule: warmup, mixup ramp-up, when
   each augmentation turns on
8. **Domain-shift adaptation** specifically (paper-211 2024, paper_256 2025) —
   what worked for the focal→soundscape transition

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
- **NEW (added end-of-day 2026-04-25)**: **2025 pretrain at our scale
  is rejected** (exp 029 stage 2 → 0.885 LB, −0.021 vs v5j). Training-
  side pretraining is a perturbation of v5j's equilibrium just like
  any other recipe change, and the per-frame finding's prediction
  held cleanly. Also discovered a quirk: `head_lr` is silently ignored
  under `scheduler_type: one_cycle` (only used in the `cosine` branch
  of `configure_optimizers`). Worth fixing if we ever want true split
  LR under one_cycle.
- **NEW**: Working-notes review is the next strategic move. 22+
  recipe-side failures + a mechanistic explanation = the gap is
  unlikely on knob-turning. Reading what winners actually did, in
  full detail, is higher-EV than another isolated tweak.

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
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints (exp 029 stage 1+2 added with REJECTED status) |
| `docs/references/kaggle/winning-solutions-analysis.md` | Cross-cutting comparison matrix of past BirdCLEF winners — useful start point before fetching new papers |
| `docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md` | Sydorskyi 2023 1st place — same author as the highest-priority 2025 paper to read next |
| `docs/references/kaggle/birdclef-2024/birdclef2024-1st-place-solution.md` | Team Kefir 2024 1st place — verify CE+sigmoid details |
| `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` + `birdclef-2025-nikita-1st-place-sed.py` | Nikita Babych 2025 1st place — inference + SED head code |
