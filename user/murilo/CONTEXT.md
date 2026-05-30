# Session Context

Session handoff for Murilo's BirdCLEF 2026 experiments. Read this first
to resume; follow the linked files for full detail.

**Last updated**: 2026-05-30 (late). **Pivoted to Exp 037 — multi-iterative Noisy-Student SOFT
pseudo-labeling — SCAFFOLDED + verified, ready to run.** After two Claude deep-research reports
(captured in `user/murilo/notes/2026-05-30-deep-research-{noisy-student-pseudo-labeling,
xeno-canto-pretraining}.md`) + a discriminator review, we chose PL (not XC-pretraining) as the
final pre-deadline experiment: pretraining is a *known ~0.88 quantity for us* (exp 034/036 = the
documented "pretrain-only 0.881"), while PL is the +0.036 lever we've **never run correctly** —
our 4 prior PL failures (014/015/025/033) all **sharpened** (T=0.6/P^1.5) + used a weak own-teacher.
Exp 037 inverts that: strong **0.929 ENSEMBLE teacher** (cached), **soft un-sharpened** labels,
**no filtering**, ~50/50 GT:PL, multi-round. Student = `distilled-sed-kd` (0.906/0.915).
XC-pretraining deferred to a parallel/post-deadline paper track (deep-research #2).

**Competition deadline**: 2026-06-03 (4 days out) — the priority.

**Branch/state**: on `mg/037-noisy-student-pl` (pushed to origin), clean. Commits: `8343676`
(exp 037 scaffold) on top of `7c92a07` (deep-research notes, on main). No PR yet (opens after
LB results). **Local verification ALL GREEN**: 20 new unit tests + 310 regression pass; PL
generator runs on all 10,658 files (raw+t2 parquets written); eval harness reproduces reference
bars; species alignment exact. **Teacher is fully CACHED — no new inference needed.**

**▶ NEXT SESSION STARTS HERE:** exp 037 scaffold is done + verified. **Murilo's next step = run it
on PACE** (see "Pick up here" below). Highest-EV remaining task. The 0.929-ensemble END-GAME
(secure final picks + PP/TTA) is the fallback/parallel track if exp 037 round-1 rejects on LB.

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

## Pick up here (next session) — RUN EXP 037 (multi-iterative soft PL)

Scaffold is complete + locally verified on branch `mg/037-noisy-student-pl`. **Murilo runs these
on PACE** (Claude doesn't run SLURM/Kaggle — [[feedback_user_runs_commands]]). PL parquets (raw,
t2) are ALREADY generated. Sequence:

```bash
cd user/murilo/experiments/037-noisy-student-pl
# 0) (optional) regenerate PL parquets — already done; ~15s each on interactive venv:
#    python scripts/generate_ensemble_pl.py --variant raw   (and --variant t2)
# 1) GRADIENT-FLOW GATE (do first): fast-dev-run on a GPU node
bash sbatch/run.sh configs/round1_raw_5050.yaml --fast-dev-run
# 2) round-1 single-fold grid (4 cells, parallel — PACE is not the bottleneck):
bash sbatch/run.sh configs/round1_raw_5050.yaml    --fold 0
bash sbatch/run.sh configs/round1_raw_natural.yaml --fold 0
bash sbatch/run.sh configs/round1_t2_5050.yaml     --fold 0
bash sbatch/run.sh configs/round1_t2_natural.yaml  --fold 0
# 3) offline screen (rough — labeled-SS anti-correlates w/ LB): training logs val/cmap+val/auroc.
#    Gate-A bar = baseline distilled OOF 0.7632:  python scripts/eval_labeled_ss.py --pred <student_oof.parquet>
# 4) 5-fold the best cell (bash sbatch/run.sh <cfg> --all-folds), export ONNX (mirror exp 032),
#    LB: single-fold vs 0.906, 5-fold vs 0.915, blend-into-0.929 vs 0.929.
```
**Falsifiable (single-fold vs 0.906):** CONFIRM ≥0.916 / PARTIAL 0.909–0.915 / WEAK 0.907–0.908 /
REJECT ≤0.906. If round 1 LB-green → round 2 (promote student into teacher, regenerate, retrain).
Full method/gates in `037-noisy-student-pl/{proposal,tasks,README}.md`. **LB is the arbiter.**

---

## Fallback / parallel track — END-GAME: secure the 0.929 final submission (deadline 2026-06-03)

If exp 037 round-1 rejects on LB (base rate for our PL attempts is poor, though all prior ones
used the WRONG recipe), fall back here. The XC fine-tune track is exhausted (exp 034/036, three
recipes ~0.88). **Lock in the best final submission**, not more single-model training. Concretely:

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
