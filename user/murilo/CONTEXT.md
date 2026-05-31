# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-30 (night). **Exp 037 (multi-iterative Noisy-Student soft PL) — CLOSED,
REJECT.** The A/B/C-raw-augF trio (clean KD-routing comparison, `last.ckpt` fold0) submitted to LB:
**C 0.832 / B 0.790 / A 0.754 — all far below the 0.906 no-PL base** (best −0.074). PL **poisons** the
distilled-sed-kd student (raw ensemble blend = diffuse soft targets, mean 0.24, ~233 cls at 0.1–0.3/win;
50/50 + loss_weight=1.0 → calibration flattens → ranking collapses). **5th PL failure** (014/015/025/
033/037). One genuine result: the **KD-routing axis transferred offline→LB (C>B>A)** — "let PL reach the
backbone, don't detach" is real — but PL is net-negative regardless. 2 SoftAUC picks NOT submitted
(trio decisive; LB conserved). NOT the val metric (shipped `last.ckpt`) nor epochs (curves plateaued).

**Competition deadline**: 2026-06-03 (~4 days) — now the priority.

**Branch/state**: on `mg/037-noisy-student-pl` (pushed), clean. exp 037 fully closed (REJECT) across
`results.md`/`tasks.md`/`MODEL-RUNS.md`. PL track exhausted (0/5); pretraining capped ~0.88 (exp
034/036); training-recipe base-rate≈0 (exp 027). Deep-research notes + a SoftAUC-PL / Perch-only-PL /
XC-pretrain track remain for the **post-deadline paper** (compute free, doesn't count for LB).

**▶ NEXT SESSION STARTS HERE:** exp 037 is CLOSED (REJECT). All levers (PL, pretrain, recipe) are
exhausted for the deadline. **Pivot to the END-GAME (secure the 0.929 ensemble + PP/TTA final picks
before 06-03)** — see "Pick up here" below. No in-flight jobs; ~14 LB pulls left.

---

## Where we are

**Exp 036 — unlock the XC backbone via a faithful fine-tune recipe.** exp 034 proved the
Sydorskyi EffNetV2-S backbone is bit-identical to source, yet full-FT scored LB 0.879 (<0.906).
The bet: the backbone is fine, the *fine-tune* under-trained it. Phase A varies only the LR
schedule (model + data = exp 034's known-0.879 config), fold 0, 50 ep.

### Phase A results (fold 0) — best-by-val/auroc (NOT last-epoch; cosine decays the tail)

| Cell | best val/auroc | shape over 50 ep |
|---|---:|---|
| a0-fullft (lr 1e-4) | 0.906 @ep3 | peaks ep3 → **decays to 0.875** (overfits) |
| **a1-lower-lr (5e-5)** | **0.921 @ep34** | climbs → **flat 0.91–0.92 plateau (ep19–50)** |
| a2-discriminative | 0.888 | flat-low (dead) |
| a3-gradual-unfreeze | 0.914 @ep9 | peaks ep9 → settles ~0.90 (viable 2nd) |

- **Gate 6 PASS**: a0 ≡ exp034-fullft epoch-by-epoch (~1e-4). Anchor: offline-best 0.906 →
  LB 0.865 raw / **0.879 +PP**. (The "0.883" in old handoffs was a late epoch, not the peak.)
- **a1 wins**: best offline (+0.015 over the anchor), cleanest mechanism (lower LR preserves
  the XC features → no overfit collapse), ideal SWA shape (flat tail).
- **Caveat**: offline anti-correlates with LB ([[feedback_offline_metrics_unreliable]],
  [[project_cv_measurement_failure]]). exp 034 KD-joint was offline 0.9217 → **LB 0.878 (no
  lift)**; a1 lands at the same 0.92 offline. The reason to still bet a1: KD changed the
  *objective*; a1 is LR-only on the same objective, fixing a *visible* pathology. **LB decides.**

Checkpoints (all 4 cells, save_top_k=-1, 50 ep + last):
`/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/036-xc-backbone-finetune/checkpoints/<cell>/fold0/`.
SWA built: `a1-lower-lr/fold0/swa-last5.ckpt` (avg of ep45–49; 786 model.* tensors).

---

## Pick up here (next session) — END-GAME: secure the 0.929 final submission (deadline 2026-06-03)

**Exp 037 is CLOSED (REJECT)** — the PL trio tanked (C 0.832 / B 0.790 / A 0.754 ≪ 0.906; full read in
`037-noisy-student-pl/results.md § LB verdict`). PL is 0/5, pretraining caps ~0.88 (exp 034/036),
recipe changes base-rate≈0 (exp 027). **Every single-model lever is exhausted for the deadline.** The
highest-EV use of the remaining ~4 days + ~14 LB pulls = **lock in the best final submission.**
[[feedback_user_runs_commands]]: give Murilo the commands; don't run SLURM/Kaggle.

Concretely (this is the END-GAME deferred since exp 036):

---

## END-GAME steps — secure the 0.929 final submission (deadline 2026-06-03)

The XC fine-tune + PL tracks are exhausted (PL 0/5; XC ~0.88). **Lock in the best final submission**,
not more single-model training. (Post-deadline paper track: SoftAUC-PL / Perch-only-PL /
PL-target-trim / broad XC-pretrain.) Concretely:

1. **Re-confirm the 0.929 ensemble (exp 023) is reproducible + submittable** as one of the two
   final picks. Verify its kernel still builds/scores; canary `Species mapping: 234 mapped`.
2. **PP/TTA on the best ensemble** — smoothing is +0.014-proven; TTA is untried and cheap. A/B.
3. **(cheap, optional) XC model as a backbone-diverse ensemble member** — prob-space blend the
   a1 XC model (0.882, genuinely different backbone) into the 0.929 ensemble; keep only if the
   blend beats 0.929 on LB. This is the one place the XC backbone might still pay off.
4. **Pick the 2 final submissions** deliberately (one safe = 0.929; one upside = best blend).

Don't restart Phase B (class-balanced data) — decided against (base rate ≈ 0, can't screen
offline, deadline). Wiring/assets stay in the repo if ever revisited.

---

## Best LB (project state)

| Model | LB | Notes |
|---|---:|---|
| Perch + SED MLP-only ensemble (exp 023) | **0.929** | **Current overall best → secure as a final pick** |
| v5j-4fold-pp2 (exp 021) | 0.917 | Best SED ensemble |
| distilled-sed-kd-5fold (exp 032) | 0.915 | Best Tucker-SED ensemble |
| distilled-sed-kd (exp 032, single-fold) | 0.906 | Our best single model (ImageNet + KD) |
| exp 034 KD-joint+PP / full-FT+PP | 0.878 / 0.879 | XC backbone single-fold ceiling |
| exp 036 a1-last / a1-SWA (+PP) | 0.882 / 0.881 | Phase A winner; offline 0.921 → flat on LB (REJECT) |

---

## How to resume after disconnect

1. Read this file top-to-bottom, then `036-xc-backbone-finetune/results.md § LB verdict`.
2. `git log --oneline -6` on `main` → top is `e64a09f` (exp 036 REJECT, merged). No open branch;
   exp 036 is CLOSED and on main. Start a fresh branch for the next experiment.
3. Proceed to the END-GAME steps above (secure the 0.929 final pick; PP/TTA; optional XC-as-
   ensemble-member blend). This is final-submission work, not new single-model training.
4. **[[feedback_user_runs_commands]]**: don't run SLURM/Kaggle yourself — give Murilo the
   commands. Local pytest + light SWA averaging are fine (interactive venv at
   `/storage/home/hcoda1/9/mgustineli3/clef/birdclef-2026/.venv`; [[feedback_use_uv_venv]]).
5. Venv policy: NEVER create a venv under home/project; SLURM jobs build on `$TMPDIR` (CLAUDE.md).
   PACE compute is NOT the bottleneck ([[feedback_pace_compute_not_bottleneck]]); Kaggle LB IS
   rate-limited (~5/day) — be strategic there. Use `--squash` for PR merges ([[feedback_squash_merge]]).

**Prior context**: exp 034 (xcpretrain) REJECT — backbone necessary-not-sufficient, single-fold
ceiling 0.879; KD does not transfer. exp 036 REJECT — fine-tune schedule doesn't unlock it either
(~0.88 ceiling, schedule-independent). exp 032 merged to main. Detail in those experiments'
`results.md`. **Credentials**: `user/murilo/.env` (gitignored), loaded by `train.py`.
