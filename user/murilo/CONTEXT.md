# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-26 (end of session, after exp 030 v1 LBs + dual_validation confound diagnosed + v2 retraining launched)

## Read these files first

Read at session start to get full context. Priority matters — read the
top three fully, the rest can be skimmed.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/experiments/030-sydorskyi-mixup/results.md` | **ACTIVE EXPERIMENT — v2 retraining in flight.** v1 LBs all in: 0.890/0.892/0.893/**0.901** (HGNet × EffNetV2-B0 × control × sydorskyi). `dual_validation: true` confound identified (cost ~−0.009 LB via checkpoint selection); v2 dropped it. Best v1 cell **`effnetv2b0_sydorskyi` = 0.901** even at the dual_val handicap — closest single-model to v5j 0.906 in this competition. v2 will tell us if effnetv2b0_sydorskyi crosses 0.906. | Read fully (start here) |
| `user/murilo/analysis/notes/2026-04-26-mixup-implementation-audit.md` | **HEADLINE finding**: v5j and prior wave-mixup attempts (v4j/v6b/v10b) all use Zhang-style INTERPOLATED multi-label targets, not Sydorskyi's `max(y1, y2)`. The audit verifies this with line numbers in `sed_finetune.py`. Refutes that Xu 2018 is the source of Sydorskyi's recipe (Xu actually uses interpolation on log-mel). Reframes "wave-mixup is dead" → "interpolated wave-mixup is dead; the Sydorskyi-faithful recipe was never tested." | Read fully |
| `user/murilo/analysis/notes/2026-04-26-mixup-target-simulation-results.md` | **CPU simulation finding**: 80% of v5j's active training target values are sub-1; 59% of training samples have NO class at full presence. Quantifies the "trained honesty" hypothesis: model rarely sees the multi-label binary test distribution during training. | Read fully |
| `user/murilo/analysis/notes/2026-04-26-d1-train-side-attention-verdict.md` | **D1 verdict**: 84% top-1% failure overlap between v5j (LSEHead) and v6k (AttHead). Failure pattern is data-driven, not head-architecture-driven. v6k's attention is essentially uniform (entropy 2.5/2.77 max) — but this is because v6k's recipe uses framewise-max-only inference (`blend_clip=0.0`), so attention has no gradient pressure. H2 not testable from v6k specifically. | Read fully |
| `user/murilo/analysis/notes/2026-04-26-d1-followup-discussion.md` | Three open questions after the D1 verdict: (1) v6k uses attention only at training, structurally weird; (2) longer windows ruled out (4 prior tests regressed); (3) Sydorskyi-faithful MixUp endorsed as next experiment. Walks through the trade-offs. | Skim |
| `user/murilo/analysis/notes/2026-04-26-sydorskyi-2025-vs-v5j-comparison.md` | Side-by-side mapping of Sydorskyi 2025 2nd place recipe against v5j. Three big levers identified: in-domain pretrain at scale, soft pseudo-distillation, and the augmentation/loss/sampler stack. | Skim |
| `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` | Sydorskyi 2025 2nd place reference doc — full pipeline writeup from paper + GitHub repo. **Source for the MixUp recipe that motivated exp 030.** | Reference |
| `user/murilo/analysis/notes/2026-04-25-per-frame-attention-analysis.md` | **04-25 finding**: v5j is structurally honest. Per-frame attention entropy + GT-class logit std rise monotonically with failure severity. Mechanistically explains 21+ failed recipe changes. **Reproduced on train side in D1 verdict (above).** | Skim (reproduced on train) |
| `user/murilo/analysis/notes/2026-04-25-audio-validation-low-snr-crops.md` | Audio listening validation. mixed_uncertainty crops dominated by ambient/anthropogenic noise; bird calls weak/absent. Reframed "silent crop" → "low SNR within 5s window." | Skim |
| `user/murilo/experiments/029-2025-pretrain/results.md` | **REJECTED.** Stage 2 LB 0.885 (−0.021 vs v5j). 2025 pretrain at our scale doesn't help. | Skim |
| `user/murilo/analysis/notes/2026-04-24-triage-v5j-failure-modes.md` | Foundation — triage categorization of 3,654 val samples. Only 1% is `confident_wrong`. | Skim |
| `user/murilo/analysis/README.md` | Tooling map — triage, certainty, species_composition, export_figures, per_frame_analysis, model_explorer, **train_side_analysis (new 04-26)**, **mixup_target_simulation (new 04-26)** | Skim |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights | Skim |
| `user/murilo/FINDINGS.md` | Cross-experiment history (006–027) | Skim |
| `user/murilo/MODEL-RUNS.md` | All model scores and checkpoints (now includes exp 030 v1 + 2x2 LB matrix) | Skim |
| `user/murilo/experiments/030-sydorskyi-mixup/proposal.md` | Active experiment — full hypothesis + method + decision matrix | Reference if working on exp 030 |
| `user/murilo/experiments/028-background-noise-mixing/` | **PAUSED/DEMOTED** by per-frame analysis. Don't run. | Skip |

## Current state

**Best LB: 0.929** (Perch ONNX + SED v5j-4fold MLP-only ensemble, exp 023) — unchanged.
**Best single SED: 0.906** (v5j, 5s LSEHead, exp 019) — still our ceiling.
**Closest single-model challenger: 0.901** (effnetv2b0_sydorskyi v1, exp 030) — within 0.005 of v5j *with the dual_validation handicap*. v2 retraining in flight.
**Target: 0.948** (competition #1). Gap: 0.019.

**Active branch**: `mg/exp030-sydorskyi-mixup` — PR #43 (OPEN). exp 030 v1 done, v2 retraining running (~40 min remaining).

**Recent merges to main (this session 2026-04-26):**
- PR #42 → D1 diagnosis + Sydorskyi 2025 reference doc + MixUp audit + train-side attention tooling. **Merged at `3a1e18b`.**
- PR #43 → exp 030 scaffolding + Phase A v1 results + Phase B v1 results + v1 LB diagnosis (dual_validation confound) + v2 retrain config. **Open, will continue next session.**

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | Current best — ONNX Perch + SED 4-fold |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.906** | Best single SED — our ceiling |
| **exp 030 effnetv2b0_sydorskyi v1** | **0.901** (dual_val handicap) | EfficientNetV2-B0 + Sydorskyi mixup. Best v1 cell of 2x2. v2 (no dual_val) pending; expected ~0.908-0.912 if dual_val cost recovers. |
| exp 030 effnetv2b0_control v1 | 0.893 | EffNetV2-B0 + spec interp mixup. Mixup on v2-B0 = +0.008 LB. |
| exp 030 v5j_sydorskyi v1 | 0.892 | HGNet + Sydorskyi mixup. Mixup on HGNet = null (+0.002). |
| exp 030 v5j_control v1 | 0.890 | HGNet + spec interp + dual_val. v1 reproducibility check baseline. |
| exp 029 stage 2 | 0.885 (REJECTED) | 2025-pretrain → 2026-finetune. |

## What happened recently

### 2026-04-26 — Sydorskyi audit + D1 diagnosis (PR #42 MERGED) + exp 030 scaffolded + v1 LBs + v2 retraining (PR #43 OPEN)

**Morning**: Sydorskyi 2025 2nd place reference doc + v5j comparison.
- Cloned the GitHub repo, extracted the paper, mapped the full pipeline
  (architecture, mel pipeline, loss math, full augmentation cascade,
  819K-recording in-domain pretrain corpus, 3-iteration soft-pseudo-
  distillation, TopN N=1 PP) into
  `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md`.
- Side-by-side comparison vs v5j surfaced three potential levers:
  in-domain pretrain at scale, soft pseudo-distillation, full
  augmentation/loss/sampler stack. (Note from session: pushed back on
  jumping to "copy Sydorskyi" — Salman's modeling argument is alive too.)

**MixUp implementation audit (the load-bearing finding)**:
- v5j and prior wave-mixup attempts (v4j/v6b/v10b) all use Zhang-style
  INTERPOLATED multi-label targets: `target = lam·y1 + (1-lam)·y2`.
- Sydorskyi (LB 0.917 single fold, top-3 finishers all do this) uses
  `target = max(y1, y2)`. For binary multi-hot, this means BOTH species
  stay at probability 1.0 in mixed samples, matching the multi-label
  test distribution.
- The audit refutes that Xu 2018 is the source of Sydorskyi's recipe
  (Xu actually uses interpolation on log-mel; Sydorskyi cites him for
  the *concept* of audio mixup, not the formula).
- Reframes our prior "wave-mixup is dead" (3 failed tests) → "interpolated
  wave-mixup is dead; the Sydorskyi-faithful recipe (`(w1+w2)/2` +
  peak-norm + max-targets, p=0.5, deterministic 0.5 lambda) was never
  tested." Documented at
  `analysis/notes/2026-04-26-mixup-implementation-audit.md`.

**MixUp target-distribution simulation** (CPU-only, no model):
- Replicated v5j's mixup math on real label vectors over 100K simulated
  events. **80% of v5j's active training target values are sub-1.**
  **59% of training samples have NO class at full presence (=1.0)** after
  the theta=0.8 cap.
- The model is being trained on a target distribution that DOESN'T match
  the test distribution most of the time. Documented at
  `analysis/notes/2026-04-26-mixup-target-simulation-results.md`.

**D1 train-side attention diagnosis** (the central H1 vs H2 discriminator):
- Diagnostic philosophy: discriminate between H1 (information / data is
  the lever) and H2 (modeling / head is the lever) before committing to
  a recipe change.
- Built `train_side_analysis.py` + extended `precompute_losses.py` for
  `--data-split=train` + extended `per_frame_analysis.py` with
  `--head-type=atthead` mode. Ran on v5j (LSEHead) and v6k (AttHead) over
  the full 33K-sample training set.
- **Headline**: 84% top-1% by-loss overlap between v5j and v6k. The
  hardest training samples are the *same* samples for both heads.
  Failure pattern is data-driven, not head-driven.
- v5j's val pattern (entropy 1.20→1.36, logit_std 5.21→2.85) reproduces
  on train.
- v6k's attention is essentially uniform (entropy 2.5/2.77 max). But
  this is because v6k uses framewise-max-only inference
  (`blend_clip=0.0`); attention has no gradient pressure to localize.
  **H2 not testable from v6k specifically.** Verdict at
  `analysis/notes/2026-04-26-d1-train-side-attention-verdict.md` and
  follow-up discussion at `2026-04-26-d1-followup-discussion.md`.

**PR #42 merged** (commit `3a1e18b`): D1 diagnosis + audit + tooling +
verdict.

**Exp 030 scaffolded (PR #43)**: Sydorskyi-faithful MixUp on v5j as a
2x2 ablation (HGNetV2-B0 × EfficientNetV2-B0 × control × sydorskyi).
- Phase A (HGNet single-variable mixup test) + Phase B (EffNetV2-B0
  backbone, Salman's choice — `tf_efficientnetv2_b0.in1k`, 7.1M params,
  but encoder uses 192-ch stage-5 features not the 1280-d post-pool
  projection).
- New `mixup_mode="average_max"` in `sed_finetune.py:_apply_mixup`
  implementing `(w1+w2)/2` + peak-normalize + `max(y1, y2)`.
- All 4 v1 cells trained, all 4 submitted to LB.

**v1 LB results (with `dual_validation: true`)**:

|  | control (spec interp) | sydorskyi (wave max) | mixup Δ |
|---|---:|---:|---:|
| HGNetV2-B0 | 0.890 | 0.892 | +0.002 (null) |
| EfficientNetV2-B0 | 0.893 | **0.901** | **+0.008** |
| backbone Δ | +0.003 | +0.009 | |

**Best v1 cell**: `effnetv2b0_sydorskyi` at 0.901 — strong interaction
where Sydorskyi mixup helps on v2-B0 but is null on HGNet.

**dual_validation confound diagnosed**:
- All 4 v1 cells were below v5j 0.906. The cause: `dual_validation: true`
  added between exp 019 and exp 030.
- It changes `val/auroc` from "macro-AUROC over the COMBINED focal+SS val"
  to "focal-only" via the new `val_dataloader()` returning
  `[focal_loader, sc_loader]` instead of one ConcatDataset loader.
- Model checkpoint callback monitors `val/auroc`, so checkpoint selection
  happens on a different metric than original v5j (which used the
  combined-val). Best epoch shifts.
- exp 026 already documented this: v5j-dualval = 0.897 LB vs v5j = 0.906
  (−0.009 LB cost).
- **Important: this is NOT an inference-pipeline bug.** The Kaggle
  submission kernels show identical inference flags being applied. The
  difference is which checkpoint we're submitting.

**v2 retraining (in flight at session end)**:
- All 4 configs updated: `dual_validation: true` removed, run_name with
  `-v2` suffix so v1 checkpoints stay intact.
- `experiments/010-submission-pipeline/sbatch/export.sbatch` updated to
  point at the v2 checkpoint dirs.
- 4 SLURM jobs running on rtx6000 (jobs 7455688-7455691):
  v5j_control_v2, v5j_sydorskyi_v2, effnetv2b0_control_v2,
  effnetv2b0_sydorskyi_v2. ~40 min each.

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

### Priority 1: finish exp 030 v2 cycle (resume here next session)

v2 SLURM jobs were running at session end (~40 min each). Pick up by:

**Step 1 — verify v2 training finished cleanly**
```bash
squeue -u $USER  # all 4 should be DONE
ls user/murilo/experiments/030-sydorskyi-mixup/logs/slurm_*_745568[89]*.log \
   user/murilo/experiments/030-sydorskyi-mixup/logs/slurm_*_745569[01]*.log
```
For each: confirm "Job completed successfully!" and inspect val/auroc /
best epoch from `artifacts/training/{model}_v2/metrics.json` (need to
update save_metrics.py output paths if they don't exist; check
`outputs/wandb/run-*/files/wandb-summary.json` as fallback).

**Step 2 — re-export to ONNX** (4 SLURM jobs, ~10 min each)
```bash
cd user/murilo/experiments/010-submission-pipeline
bash sbatch/run.sh export --model hgnet-b0-sed-v5j-control
bash sbatch/run.sh export --model hgnet-b0-sed-v5j-sydorskyi
bash sbatch/run.sh export --model effnetv2-b0-sed-control
bash sbatch/run.sh export --model effnetv2-b0-sed-sydorskyi
```
The 4 export functions in `sbatch/export.sbatch` already point at the
`-v2` checkpoint dirs (committed to PR #43). Output dirs
(`artifacts/weights/{model}/`) overwrite v1 ONNX.

**Step 3 — re-upload weights + push kernels to Kaggle** (Murilo runs)
```bash
python scripts/upload_kaggle.py --model hgnet-b0-sed-v5j-control --push-kernel
python scripts/upload_kaggle.py --model hgnet-b0-sed-v5j-sydorskyi --push-kernel
python scripts/upload_kaggle.py --model effnetv2-b0-sed-control --push-kernel
python scripts/upload_kaggle.py --model effnetv2-b0-sed-sydorskyi --push-kernel
```
Same Kaggle slugs as v1 — replaces v1 weights at the same dataset
identifier; same kernel-metadata files; same MODEL_SLUGS in
`submission.py`.

**Step 4 — submit 4 kernels from Kaggle UI**, collect LBs.

**Step 5 — apply decision matrix from `proposal.md` + write verdict
commit on PR #43**. Update `results.md`, `MODEL-RUNS.md`,
`tasks.md`, and CONTEXT.md.

### What the v2 LBs will tell us

| `v5j_control_v2` LB | Interpretation |
|---|---|
| 0.901-0.911 | dual_val confound CONFIRMED. Pipeline reproduces v5j cleanly. Read remaining v2 numbers normally. |
| 0.895-0.900 | Partial recovery — dual_val explains some but not all of the v1 gap. Maybe additional codebase drift since exp 019. |
| ≤ 0.895 | dual_val isn't the only confound. Halt and investigate (PyTorch/timm version, mixup default behavior). |

**The headline question**: does `effnetv2b0_sydorskyi_v2` land ≥ 0.908?
If yes, **first single-model improvement over v5j 0.906** in this
competition for us. If yes and v5j_control_v2 reproduces cleanly, this
combo is the basis for a 4-fold + ensemble stack going forward.

### Priority 2 (after v2 verdict): pick next experiment

Decision space depends on v2 outcome:

- **If `effnetv2b0_sydorskyi_v2` ≥ 0.908 AND v5j_control_v2 reproduces v5j**:
  endorse the recipe, queue 4-fold version of `effnetv2b0_sydorskyi`
  (matches v5j-4fold-pp2 stack). Then add Sydorskyi's other components
  (label smoothing, BG noise) one at a time.
- **If `effnetv2b0_sydorskyi_v2` ~ 0.901 (flat to v1) AND v5j_control_v2
  reproduces v5j**: dual_validation explains the v1 gap, but the recipe
  itself is null. Pivot to pseudo-distillation on 2026 unlabeled SS
  using our 0.929 ensemble as teacher (Sydorskyi's recipe was +0.029
  Public on top of his pretrain).
- **If v5j_control_v2 doesn't reproduce v5j 0.906**: blocker — halt
  and investigate before another experiment.

### Deferred / superseded

- **Working notes Tier 1+ reading** (Nikita 1st place 2025, Witting
  2024 paper-211 domain shift): captured as future work but lower
  priority than finishing exp 030 v2 cycle. Sydorskyi 2025 is done.
- **exp 030 silent-class candidate**: still queued, hasn't been
  designed; addresses a different problem than mixup target
  distribution.
- **Path A retry (split LR fix in `one_cycle` branch)**: low priority.
- **Triage on exp 029 stage-2 ckpt**: low priority.

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

## Key patterns (updated 2026-04-26)

- **HEADLINE (04-26)**: **First sub-0.005 single-model challenger to v5j**
  is `effnetv2b0_sydorskyi` v1 at 0.901 LB *while carrying the dual_val
  handicap* (~−0.009 LB). v2 retraining (no dual_val) is the test for
  whether this clears 0.906 cleanly. If yes, it's our first confirmed
  single-model improvement.
- **MixUp recipe finding (04-26)**: v5j and prior wave-mixup attempts
  (v4j/v6b/v10b) all use Zhang-style INTERPOLATED multi-label targets:
  `target = lam·y1 + (1-lam)·y2`. Sydorskyi/top-3 winners use
  `target = max(y1, y2)`. CPU simulation showed 80% of v5j's active
  training target values are sub-1; 59% of training samples have NO
  class at full presence after the theta=0.8 cap. **Interpreted as: the
  model rarely sees the multi-label binary test distribution during
  training. The "wave-mixup is dead" claim was about interpolated
  wave-mixup, not the Sydorskyi-faithful recipe.**
- **D1 train-side finding (04-26)**: 84% top-1% by-loss overlap between
  v5j (LSEHead) and v6k (AttHead) on the 33K-sample training set. Failure
  pattern is data-driven, not head-architecture-driven. Reproduced v5j's
  honesty pattern on the train side (entropy 1.20→1.36, logit_std
  5.21→2.85). v6k can't test H2 because its inference recipe
  (`blend_clip=0.0`, framewise-max only) leaves attention with no
  gradient pressure to localize → entropy ≈ uniform.
- **dual_validation confound (04-26)**: `dual_validation: true` (added
  between exp 019 and exp 030) costs ~0.009 LB by changing `val/auroc`
  to focal-only via the new `[focal, sc]` val_dataloader, which shifts
  checkpoint selection. exp 026 already documented this (v5j-dualval
  0.897 vs v5j 0.906). All 4 v1 cells of exp 030 carried this handicap.
  **Lesson: any flag that affects which checkpoint we serialize needs
  an LB receipt, not just an offline-metric receipt.** v2 retraining
  drops the flag.
- **v5j is structurally honest (carried from 04-25, reproduced 04-26)**:
  attention diffuses + logit std collapses on hard crops because the
  call is genuinely weak/absent in those crops. BCE+random-crop training
  penalizes that honesty. Mechanistically explains why 21+ training-
  recipe changes have failed to move LB — most were teaching false
  positives.
- **Direct implication**: stop testing recipe knobs that try to make
  the model more confident on hard crops. Levers are *more information*
  (pretraining, external data, pseudo-distillation), *better inference*
  (PP/TTA), or *backbone change motivated by representation quality*.
  The exp 030 v2 cycle directly tests two of these: backbone change
  (EffNetV2-B0) and a multi-label-faithful target distribution
  (Sydorskyi mixup).
- **The gap is on labeled data**: Boredom (0.938) and Salman (0.937)
  beat us by 0.031 using the same focal + labeled soundscape data.
  Ali Ozan (0.941) used 2025 pretrain at filtered XC scale (much
  larger than exp 029).
- **Offline metrics remain anti-correlated with LB — DO NOT trust them
  for cross-architecture comparisons.** v5j (val/auroc 0.874) → 0.906
  LB. v6l (val/auroc 0.874) → 0.899 LB. **Special case (04-26)**: under
  `dual_validation: true`, val/auroc is *focal-only* and val/cmap +
  val/f1_macro give a directionally cleaner signal — they correctly
  flagged effnetv2b0_sydorskyi as the best v1 cell.
- **CE loss is dead** (7 tests). **Interpolated MixUp is dead** (3
  tests). **Pseudo-labels are dead** (2 approaches). **BG noise mixing
  is demoted** before being run, per the per-frame argument. **2025
  pretrain at our scale is rejected** (exp 029 stage 2 → 0.885 LB).
- **`head_lr` quirk (carried from 04-25)**: silently ignored under
  `scheduler_type: one_cycle` (only used in the `cosine` branch of
  `configure_optimizers`). Worth fixing if we ever want true split LR
  under one_cycle.

## Key files for current work

| File | Purpose |
|------|---------|
| `user/murilo/experiments/030-sydorskyi-mixup/` | **Active experiment — v2 retraining in flight** |
| `user/murilo/experiments/030-sydorskyi-mixup/configs/` | 4 v2 configs: `{v5j,effnetv2b0}_{control,sydorskyi}.yaml` (no dual_validation, run_name `-v2`) |
| `user/murilo/experiments/030-sydorskyi-mixup/results.md` | v1 LBs, dual_val confound diagnosis, v2 plan, decision matrix |
| `user/murilo/experiments/030-sydorskyi-mixup/tasks.md` | Phase A/B done, V2.0 done, V2.1-V2.3 pending |
| `user/murilo/experiments/030-sydorskyi-mixup/proposal.md` | Full hypothesis + method + decision matrix |
| `user/murilo/experiments/010-submission-pipeline/sbatch/export.sbatch` | Updated 04-26 to point at `-v2` checkpoint dirs for all 4 exp 030 cells |
| `user/murilo/experiments/010-submission-pipeline/scripts/upload_kaggle.py` | 4 new MODELS entries for exp 030 |
| `user/murilo/experiments/010-submission-pipeline/kaggle_kernel/submission.py` | MODEL_SLUGS includes 4 exp 030 weight slugs |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | NEW (04-26) `mixup_mode="average_max"` branch in `_apply_mixup` (Sydorskyi recipe). `_find_attention_module` now checks `model.sed_head` first (not `model.head`). |
| `user/murilo/analysis/` | **Active analysis tooling and notes** |
| `user/murilo/analysis/train_side_analysis.py` | **NEW (04-26)** — D1 cross-model bucket comparison (v5j vs v6k by-loss overlap, attention entropy, logit std on training set) |
| `user/murilo/analysis/mixup_target_simulation.py` | **NEW (04-26)** — CPU-only simulation of v5j's mixup target distribution over 100K events |
| `user/murilo/analysis/per_frame_analysis.py` | 04-25 tool, extended 04-26 with `--head-type=atthead` mode for v6k |
| `user/murilo/analysis/precompute_losses.py` | Extended 04-26 with `--data-split=train` + AttHead attention capture via forward hook on `attention_conv` |
| `user/murilo/analysis/triage.py` / `certainty.py` / `species_composition.py` / `export_figures.py` | 04-24 tooling, all still active |
| `user/murilo/analysis/notes/2026-04-26-*.md` | 5 notes from 04-26: mixup audit, target simulation, D1 verdict, D1 followup, Sydorskyi-vs-v5j comparison |
| `user/murilo/analysis/notes/2026-04-25-*.md` | 04-25 per-frame finding + audio validation notes |
| `user/murilo/analysis/data/losses_v5j_train_*.npy` / `losses_v6k_train_*.npy` | NEW (04-26) — per-sample loss + attention dumps for the 33K-sample training set |
| `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` | Competitor insights |
| `user/murilo/FINDINGS.md` | Cross-experiment history |
| `user/murilo/MODEL-RUNS.md` | All model scores. Updated 04-26 with exp 030 v1 LBs (2x2 matrix) + v2 plan |
| `docs/references/kaggle/birdclef-2025/birdclef2025-2nd-place-domain-shift.md` | **NEW (04-26)** — Sydorskyi 2025 2nd place reference doc, source for the exp 030 mixup recipe |
| `docs/references/kaggle/winning-solutions-analysis.md` | Cross-cutting comparison matrix of past BirdCLEF winners |
| `docs/references/kaggle/birdclef-2023/birdclef2023-1st-place-solution.md` | Sydorskyi 2023 1st place |
| `docs/references/kaggle/birdclef-2024/birdclef2024-1st-place-solution.md` | Team Kefir 2024 1st place |
| `docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md` + `birdclef-2025-nikita-1st-place-sed.py` | Nikita Babych 2025 1st place |
| `user/murilo/experiments/029-2025-pretrain/` | REJECTED historical artifact |
| `user/murilo/experiments/028-background-noise-mixing/` | PAUSED / DEMOTED historical artifact |
