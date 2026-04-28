# Session Context

Session handoff document for Murilo's BirdCLEF 2026 experiments.
Read this first, then read the referenced docs to get full context.

Last updated: 2026-04-28 (v5j-resubmit confirmed 0.906 → inference clean, gap is training-side; 3 remaining v2 LBs came back IDENTICAL to v1 → root cause traced to deterministic training making v2 weights bit-identical to v1; seed=43 retrain in flight to confirm the architectural EffNet × Sydorskyi interaction is real)

## Read these files first

Read at session start to get full context. Priority matters — read the
top three fully, the rest can be skimmed.

| File | Purpose | How to read |
|------|---------|-------------|
| `user/murilo/experiments/030-sydorskyi-mixup/results.md` | **ACTIVE EXPERIMENT — seed=43 retrain in flight.** All 4 v2 LBs in: v5j_control 0.896, v5j_sydorskyi 0.892, effnetv2b0_control 0.893, effnetv2b0_sydorskyi 0.901. v5j-resubmit (original v5j ckpt re-exported) → 0.906 → **inference pipeline clean**, all 0.010 gap is training-side. **3/4 v2 LBs literally identical to v1** because deterministic training (`seed_everything(42)` + `Trainer(deterministic=True)`) makes v2 weights bit-identical to v1 at every epoch — `dual_validation: true` only changes val_dataloader, not training trajectory. Only `_val_sc_pos_counts` buffer differs (val-only, stripped from ONNX). Per-key audit: 420/420 effnet weights and 320/320 v5j_sydorskyi weights are bit-identical. Only v5j_control v2 differs (max\|v1−v2\|=511) because best-by-val epoch shifted (14→15) under different val metric. **2x2 reading**: HGNet mixup Δ −0.004 (null), EffNet mixup Δ +0.008 (borderline architectural interaction). Seed=43 retrain (jobs 7488877/7488880/7488881/7488884) is the noise estimate to confirm/reject the EffNet asymmetry. | Read fully (start here) |
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
**Best single SED: 0.906** (v5j original, exp 019) — confirmed reproducible today via `v5j-resubmit` kernel (re-export of exp 019 ckpt on today's submission.py → 0.906 LB).
**Closest single-model challenger: 0.901** (effnetv2b0_sydorskyi, exp 030 v2 — same weights as v1) — within 0.005 of v5j original. Architectural interaction: EffNet+Sydorskyi mixup is +0.008 vs EffNet+control; HGNet+Sydorskyi mixup is null (−0.004). Seed=43 retrain pending to rule out seed noise.
**Reproducibility resolved (2026-04-28)**: v5j-resubmit landed at 0.906 → inference pipeline today is clean, all 0.010 gap (v5j_control_v2 0.896 vs v5j-original 0.906) is training-side. Likely best-by-val epoch shift (v5j-original best=ep 19, v5j_control_v2 best=ep 15) — to be confirmed by seed=43 + `save_top_k=-1` follow-up.
**3/4 v2 LBs identical to v1 (2026-04-28)**: deterministic training (`pl.seed_everything(42)` + `Trainer(deterministic=True)`) reproduces v1 weights bit-identically at every epoch. `dual_validation` only changes val_dataloader; the v2 retrain was a no-op for inference for the 3 cells where best-by-val landed at the same epoch. Per-key audit confirms (e.g., effnet sydorskyi: 420/420 learnable tensors bit-identical, only `_val_sc_pos_counts` non-trainable buffer differs).
**Target: 0.948** (competition #1). Gap: 0.019.

**Active branch**: `mg/exp030-sydorskyi-mixup` — PR #43 (OPEN). exp 030 v2 LBs in; seed=43 retrain in flight (4 SLURM jobs).

**Recent merges to main:**
- PR #42 (2026-04-26) → D1 diagnosis + Sydorskyi 2025 reference doc + MixUp audit + train-side attention tooling. **Merged at `3a1e18b`.** *(Note: not yet present on this branch — fork point predates the merge; will arrive on next rebase/merge from main.)*
- PR #43 (open) → exp 030 scaffolding + Phase A/B v1 results + dual_validation confound diagnosis + v2 retrain configs + v2 training results + v5j-resubmit diagnostic kernel.

| Model | LB | Notes |
|-------|-----|-------|
| **Perch+SED MLP-only (exp 023)** | **0.929** | Current best — ONNX Perch + SED 4-fold |
| Perch ProtoSSM v18 TF (exp 020) | 0.925 | Best individual model |
| SED v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| **SED v5j single (exp 019)** | **0.906** | Best single SED — reproducibility confirmed via v5j-resubmit |
| **exp 030 v5j-resubmit (NEW 04-28)** | **0.906** | Original v5j ckpt re-exported on today's submission.py. Confirms inference pipeline is clean today. |
| **exp 030 effnetv2b0_sydorskyi (v1≡v2)** | **0.901** | EffNetV2-B0 + Sydorskyi mixup. Best non-v5j single. v1 and v2 weights bit-identical (deterministic training); same LB. |
| **exp 030 v5j_control v2** | **0.896** | HGNet + spec interp, NO dual_val, ep 15 best-by-val. 0.010 below v5j-original (training-side gap, likely best-vs-last epoch). |
| **exp 030 effnetv2b0_control (v1≡v2)** | **0.893** | EffNetV2-B0 + spec interp. v1≡v2 weights. EffNet mixup main effect = +0.008 LB. |
| **exp 030 v5j_sydorskyi (v1≡v2)** | **0.892** | HGNet + Sydorskyi mixup. v1≡v2 weights. HGNet mixup main effect = −0.004 (null). |
| exp 030 v5j_control v1 | 0.890 | HGNet + spec interp + dual_val. Best-by-val ep 14 (focal-only metric shifted ckpt selection). |
| exp 029 stage 2 | 0.885 (REJECTED) | 2025-pretrain → 2026-finetune. |

## What happened recently

### 2026-04-28 — v5j-resubmit confirms inference clean (0.906) + 3/4 v2 LBs identical to v1 (root cause: deterministic training) + seed=43 retrain launched

**v5j-resubmit submitted to LB**: came back at exactly **0.906**. This proves
today's inference pipeline (submission.py + Kaggle kernel) reproduces the
original v5j 0.906 LB from exp 019. **Inference-side drift since exp 019 is
zero.** All 0.010 gap between v5j_control_v2 (0.896) and v5j-original (0.906)
is training-side.

**Submitted the other 3 v2 cells from canonical kernels** (no `-exp-030`
suffix, post-409-fix). LBs came back:
- `v5j_sydorskyi_v2` → **0.892** (literally identical to v1 LB 0.892)
- `effnetv2b0_control_v2` → **0.893** (literally identical to v1 LB 0.893)
- `effnetv2b0_sydorskyi_v2` → **0.901** (literally identical to v1 LB 0.901)

This pattern (3/4 v2 LBs equal to v1 to 3 decimals) was suspicious enough
to halt and investigate before drawing conclusions.

**Investigation chain** (took the bulk of the session):

1. Verified Kaggle dataset bytes — local v2 ONNX MD5 matches Kaggle current
   ONNX MD5. The v2 weights are definitely on Kaggle.
2. Compared canonical kernel `submission.csv` MD5 to old `-exp-030` kernel
   `submission.csv` MD5 — **byte-identical**. So either the same ONNX is
   mounted in both, or v1↔v2 ONNX produce identical predictions.
3. Verified the ONNX export logs — each export pulled the correct `-v2`
   checkpoint (ep 14/15/19/19).
4. **Per-key state_dict audit** with shape/dtype-aware comparison:

   | Cell | v1 keys / v2 keys | Δ keys | max\|v1−v2\| (common learnable) |
   |---|---|---|---:|
   | v5j_control | 321 / 320 | `_val_sc_pos_counts` only | **511.0** (real diff — best epoch shifted 14→15) |
   | v5j_sydorskyi | 321 / 320 | `_val_sc_pos_counts` only | **0.0 — bit-identical** |
   | effnetv2b0_control | 421 / 420 | `_val_sc_pos_counts` only | **0.0 — bit-identical** |
   | effnetv2b0_sydorskyi | 421 / 420 | `_val_sc_pos_counts` only | **0.0 — bit-identical** |

5. Located `_val_sc_pos_counts` at `sed_finetune.py:228` — a
   non-trainable Long buffer registered only when `dual_validation=True`,
   used purely for dual-val metric tracking (lines 689, 693, 735, 758).
   It is **not used in `forward()`** and is stripped during ONNX export.

**Root cause confirmed**: `pl.seed_everything(seed=42, workers=True)` +
`Trainer(deterministic=True)` (set in `train.py`) make training fully
reproducible. `dual_validation: true` only changes `val_dataloader` — it
returns `[focal_loader, sc_loader]` instead of one `ConcatDataset`. **It
does not change the training step RNG or trajectory at all.** So v1 and v2
trainings produce bit-identical weights at every epoch.

The `val/auroc` value AT each epoch differs (focal-only vs combined val),
which affects which epoch the `ModelCheckpoint` callback labels as "best":
- 3 cells: same best epoch fired in both regimes → bit-identical weights
  saved → identical LB.
- v5j_control: best shifted from ep 14 (focal-only val/auroc) to ep 15
  (combined val/auroc) → genuinely different weights → LB shifted +0.006.

**EffNet ONNX sanity check** (rule out architecture-specific export bug):
loaded EffNet sydorskyi v2 ONNX and HGNet v5j-resubmit ONNX, ran on the
same fixed mel input. EffNet output: shape (4, 234), std 0.006 (non-
degenerate), top-3 species stable across samples. HGNet: shape (4, 234),
std 0.00001 (very tight to 0.5, expected on random mel). Cross-model
Pearson 0.37. **No EffNet export bug.** EffNet's 0.901 < HGNet's 0.906 is
the documented offline-LB anti-correlation, not a pipeline issue.

**2x2 reading on LB** (after sorting out which scores are real):

|  | control (spec interp) | sydorskyi (wave max) | mixup Δ |
|---|---:|---:|---:|
| HGNetV2-B0 (v5j) | 0.896 (v2, ep 15) | 0.892 (v2≡v1, ep 14) | **−0.004** (null, within seed noise) |
| EfficientNetV2-B0 | 0.893 (v2≡v1, ep 19) | **0.901** (v2≡v1, ep 19) | **+0.008** |

Both Δs sit near the ±0.005 seed-noise band (Mahog's number). The
*direction* is consistent with offline 2x2 (mixup helps both backbones on
val/cmap +0.008/+0.011, val/f1 +0.039/+0.043), but on LB the magnitude
asymmetry is bigger than offline. Possible mechanisms: (a) **v5j is at a
generalization cliff** (per 04-25 per-frame finding) — perturbations don't
help; EffNet hasn't earned the cliff yet so room to improve; (b) capacity
differential (EffNet ~6M params vs HGNet ~3M); (c) feature-dim asymmetry
(EffNet 192-ch vs HGNet 1024-ch at stage 5).

**If +0.008 is real**, `effnetv2b0_sydorskyi` is **the first single-model
training-recipe LB lift in the project's history** (after 22+ HGNet
failures). Worth confirming.

**Seed=43 retrain — diagnostic for the asymmetry**:
- 4 new configs created: `{v5j,effnetv2b0}_{control,sydorskyi}_seed43.yaml`
  with `seed: 43` and `-seed43` run_name suffix. **Default `seed: 42` is
  preserved in the existing 4 configs** for future reference.
- 4 SLURM jobs launched (jobs 7488877, 7488880, 7488881, 7488884) on
  rtx6000 + v100 partitions. ~40 min each.
- **What the seed=43 LBs will tell us**:
  - If EffNet sydorskyi vs control still ≥ +0.005: architectural interaction
    confirmed → endorse `effnetv2b0_sydorskyi`, queue 4-fold + Sydorskyi
    component layering.
  - If EffNet Δ flips sign or shrinks below ±0.003: the +0.008 was seed
    noise. Pivot to pseudo-distillation.
  - If HGNet sydorskyi − control stays within ±0.003: HGNet null is robust;
    "v5j is at a cliff" hypothesis confirmed.
- v5j_control_v2 seed=43 doubles as a partial answer to the 0.010
  training-side gap: if its LB lands ≥ 0.901 (closer to 0.906), the gap
  was largely seed/checkpoint-selection sensitivity, not codebase drift.

### 2026-04-27 — exp 030 v2 training complete + v5j_control_v2 LB 0.896 surfaces reproducibility issue + v5j-resubmit diagnostic kernel built

**Morning**: confirmed all 4 v2 SLURM jobs (7455688–7455691) finished cleanly
overnight. Recorded metrics in `results.md`:

| Cell | val/auroc | best epoch | Δ vs v5j-original 0.874 |
|---|---:|---:|---:|
| v5j_control_v2 | 0.8728 | 15 | −0.001 (within ±0.005 seed band) |
| v5j_sydorskyi_v2 | 0.8700 | 14 | −0.004 |
| effnetv2b0_control_v2 | 0.8745 | 19 | +0.001 |
| effnetv2b0_sydorskyi_v2 | **0.8760** | 19 | **+0.002** (best v2 cell) |

Offline reproducibility check passed: v5j_control_v2 lands within seed
noise of v5j-original 0.874. The +0.024 val/auroc shift across all 4
cells from v1 → v2 matches the dual_validation hypothesis exactly
(focal-only → combined focal+SS macro-AUROC).

**Midday**: started LB submission cycle. ONNX exports + Kaggle uploads
done for all 4 v2 cells (existing-model re-export path; same slugs as v1,
overwrites in place). Submitted v5j_control_v2 first as the
reproducibility check.

**v5j_control_v2 LB came back at 0.896.** This sits in the **0.895–0.900
"reproducibility issue, halt and investigate"** cell of the proposal.md
decision matrix. Only +0.006 above v5j_control v1 (0.890), not the
predicted 0.899–0.906. The dual_val confound (per exp 026, ~−0.009 LB
cost) explains some of the v1 → v2 lift but not the remaining ~0.010 to
v5j-original 0.906.

**Decomposition**: the 0.010 gap is either training-side drift
(codebase/seed/timm/PyTorch changes between exp 019 and exp 030) or
inference-side drift (submission.py + Kaggle kernel changes between exp
019 and today). The two are confounded in the v5j_control_v2 → v5j 0.906
comparison.

**Diagnostic — v5j-resubmit kernel**: re-export the *original* v5j
checkpoint (exp 019, ep 19, training-side state of 2026-03-31) on
*today's* submission.py. This decouples training drift from inference
drift cleanly:
- If resubmit lands at ~0.906 → all gap is training-side. Suspect:
  `pl.seed_everything` semantics, timm version, mixup defaults. Action:
  retrain v5j_control_v2 with no seed call or seed sweep.
- If ~0.901 → roughly even split, both layers drifted ~0.005 each.
- If ~0.896 → all gap is inference-side. Action: git-bisect submission.py
  since exp 019.
- If <0.890 → inference broke. Halt.

**Built kernel + pushed**:
- New kernel slug `hgnet-b0-sed-v5j-resubmit`, new dataset
  `birdclef-2026-hgnet-b0-sed-v5j-resubmit-weights`.
- ONNX re-exported via new `bash sbatch/run.sh export --model
  hgnet-b0-sed-v5j-resubmit` target.
- Two-step Kaggle upload (new dataset, not re-upload): `python
  scripts/upload_kaggle.py --model hgnet-b0-sed-v5j-resubmit` then
  `--push-kernel`.
- Kernel ran cleanly on Kaggle: 29.8s, loaded the right weights
  (`birdclef-2026-hgnet-b0-sed-v5j-resubmit-weights`), produced
  submission.csv. No errors.

**Submission to LB pending**: Kaggle daily 5/day cap hit; waiting ~6h
for reset. v5j-resubmit takes the next available submission slot.

**The other 3 v2 cells (v5j_sydorskyi, effnetv2b0_control,
effnetv2b0_sydorskyi) are GATED behind v5j-resubmit verdict** — we don't
spend daily-cap submissions on them until we know whether to trust 0.906
or 0.896 as the today-reproducible baseline. The 4 ONNX exports +
Kaggle uploads are already done; only LB submissions are paused.

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

### Priority 1 (in flight): seed=43 retrain LB + apply asymmetry decision matrix

4 SLURM jobs launched 2026-04-28 evening — all 4 cells with `seed: 43`,
otherwise identical recipes. Wait for completion (~40 min/job, parallel),
then export + upload + push + submit each to LB.

**Decision matrix on LB outcome**:

| Pattern observed under seed=43 | Conclusion | Action |
|---|---|---|
| EffNet sydorskyi − control ≥ +0.005 AND HGNet sydorskyi − control near 0 | **Architectural interaction is real**: first training-recipe LB lift | Endorse `effnetv2b0_sydorskyi`, queue 4-fold + Sydorskyi component layering |
| Both 2x2 Δs flip sign or change > ±0.005 | All seed=42 readings were noise; mixup is null on both backbones | Pivot to pseudo-distillation (0.929 ensemble as teacher on 2026 unlabeled SS) |
| EffNet Δ shrinks but stays positive (e.g., +0.003) | Borderline. Need 4-fold or more seeds | Provisionally endorse with caveats; 4-fold next |
| Anything weird (one cell crashes, etc.) | Investigate before concluding | Halt LB submissions |

**Side-bet on the 0.010 v5j_control gap**: if v5j_control_seed43 LB lands
≥ 0.901, the gap is mostly seed/checkpoint-selection sensitivity. If it
lands at 0.896 again, the gap is something more structural about the
exp 030 codebase since exp 019.

### Priority 2 (after seed=43 verdict): structural diagnostic on best-vs-last checkpoint selection

`save_top_k: 1` + `save_last: True` here makes `last.ckpt` mirror the most
recent `best.ckpt` (verified empirically: HGNet v2 last.ckpt = ep 14/15,
not ep 19). v5j-original was best=ep 19=last; new HGNet trainings plateau
earlier and never serialize ep 19. To definitively answer whether the
0.010 v5j_control_v2 gap is "best plateau too early" vs "real training
drift", patch `save_top_k: -1` in v5j_control config and submit ep 19 of
the new run. Cheap follow-up after seed=43 results.

### Priority 3 (depends on outcomes above): pick next experiment

Decision space at this point depends on the seed=43 readings:

- **Architectural interaction confirmed (EffNet+Sydorskyi real lift)**:
  4-fold version of `effnetv2b0_sydorskyi` (matches v5j-4fold-pp2 stack).
  Then Sydorskyi component staircase: label smoothing → BG noise →
  SpecAugment → balanced sampler — one at a time, single-variable.
- **Mixup recipe is null on both backbones**: pivot to **pseudo-
  distillation on 2026 unlabeled SS** using our 0.929 ensemble (Perch+SED
  MLP-only) as teacher. Sydorskyi's pseudo-distillation was +0.029 Public
  on top of in-domain pretrain — biggest published 2025 single-experiment
  lift. Engages directly with the 04-25 per-frame finding (information
  lever, not training-recipe lever).
- **Borderline**: 4-fold on EffNet+Sydorskyi as the disambiguator.

### Deferred / superseded

- **Working notes Tier 1+ reading** (Nikita 1st place 2025, Witting
  2024 paper-211 domain shift): captured as future work but lower
  priority than resolving the reproducibility issue. Sydorskyi 2025 is done.
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

## Key patterns (updated 2026-04-28)

- **HEADLINE (04-28) — Deterministic training reproduces v1 weights bit-identically**:
  `pl.seed_everything(seed=42, workers=True)` + `Trainer(deterministic=True)`
  in `train.py` makes training fully reproducible. `dual_validation: true`
  only changes `val_dataloader`; it does not affect the training step RNG.
  So v1 and v2 trainings produce **bit-identical weights at every epoch**
  for cells where best-by-val landed at the same epoch. Per-key audit
  confirmed: 420/420 effnet learnable tensors and 320/320 v5j_sydorskyi
  tensors are bit-identical between v1 and v2; only difference is
  `_val_sc_pos_counts` non-trainable buffer (val-only, stripped from
  ONNX). **Lesson**: any experiment where a config flag only affects val
  signal may be a no-op at the model level — even with new run_name and
  fresh checkpoint dir. Need to verify weight-divergence via state_dict
  hash, not just rely on offline metrics.
- **HEADLINE (04-28) — v5j 0.906 reproduces via v5j-resubmit**: original
  exp 019 ckpt re-exported on today's submission.py landed at 0.906 LB.
  Inference pipeline drift since exp 019 is zero. The 0.010 v5j_control_v2
  → v5j-original gap is fully training-side. Most likely cause:
  best-by-val ep 15 (v5j_control_v2) vs best-by-val ep 19 (v5j-original)
  — same recipe, different epoch's weights. seed=43 + `save_top_k=-1`
  follow-up will confirm.
- **HEADLINE (04-28) — Architectural interaction observed (borderline)**:
  EffNet+Sydorskyi mixup Δ = +0.008 LB; HGNet+Sydorskyi mixup Δ = −0.004
  LB. Both within ±0.005 seed-noise band so not definitive, but the
  *direction* asymmetry is consistent with offline 2x2 (cmap, f1) and
  with the 04-25 "v5j at generalization cliff" hypothesis. If the +0.008
  holds under seed=43, it's the first training-recipe LB lift in the
  project's history. EffNet ONNX sanity-checked: not buggy, std 0.006 on
  random mel, top-K species stable across samples.
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
| `user/murilo/experiments/030-sydorskyi-mixup/` | **Active experiment — v2 training done, LB submissions paused for reproducibility diagnostic** |
| `user/murilo/experiments/030-sydorskyi-mixup/configs/` | 4 v2 configs: `{v5j,effnetv2b0}_{control,sydorskyi}.yaml` (no dual_validation, run_name `-v2`) |
| `user/murilo/experiments/030-sydorskyi-mixup/results.md` | v1 LBs, v2 training results, dual_val confound diagnosis, v5j_control_v2 0.896 reproducibility flag, decision matrix |
| `user/murilo/experiments/030-sydorskyi-mixup/tasks.md` | Phase A/B done, V2.0–V2.1 done, V2.2 (LB submissions) gated on v5j-resubmit, V2.3 pending |
| `user/murilo/experiments/030-sydorskyi-mixup/proposal.md` | Full hypothesis + method + decision matrix |
| `user/murilo/experiments/010-submission-pipeline/sbatch/export.sbatch` | Updated 04-26 to point at `-v2` checkpoint dirs for 4 exp 030 cells; **NEW 04-27**: also `hgnet-b0-sed-v5j-resubmit` target (re-export of original v5j ckpt) |
| `user/murilo/experiments/010-submission-pipeline/scripts/upload_kaggle.py` | 4 exp 030 MODELS entries + **NEW 04-27** `hgnet-b0-sed-v5j-resubmit` MODELS entry |
| `user/murilo/experiments/010-submission-pipeline/kaggle_kernel/submission.py` | MODEL_SLUGS includes 4 exp 030 weight slugs + **NEW 04-27** `hgnet-b0-sed-v5j-resubmit` slug |
| `user/murilo/experiments/010-submission-pipeline/kaggle_kernel/kernel-metadata-hgnet-b0-sed-v5j-resubmit.json` | **NEW 04-27** — diagnostic kernel metadata for re-export of original v5j ckpt |
| `docs/adding-new-models.md` | **Updated 04-27** — documents the 409 Conflict (title-slug-id mismatch) seen during exp 030 kernel push |
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
