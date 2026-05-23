# 2026-05-22 — Strategic direction: Tracks A (strong single model) → B (own-teacher PL)

**Status**: decision recorded. Discussion captured for reference. Implementation not yet started.

**Time budget**: 12 days to competition deadline (2026-06-03).

**Current best LB**: 0.906 single (v5j HGNet+LSEHead, exp 019). Public top is 0.95+ (Boredom with PL); strongest non-Perch single without unlabeled = 0.946 (Arunodhayan, no recipe disclosed).

---

## The decision

**Primary path (safer, known-recipe, full code reference):** faithfully replicate Tucker Arrants's distilled-SED recipe — EffNet-B0 NS-JFT backbone + logit-space AttHead + dual-supervision + waveform mixup with hard-max labels + SpecAugment + 25 ep cosine + Perch KD + 5-fold + site/hour priors + Gaussian smoothing. Known ceiling: **0.917 single / 0.93+ with 5-fold + PP**.

**Stretch path (higher ceiling, less reference code):** Sydorskyi-yoink + own head + 2026 fine-tune (Ali Ozan's path). Known ceiling: **0.941 single**.

**Final layer on either base (if time):** own-teacher PL on the unlabeled soundscapes. Known lift: **+0.007 to +0.013 LB** (Antoine, Boredom, chenwenqiang_001 data points).

**Naming convention (per Murilo's 2026-05-22 swap):**
- **Track A** = build the strong single model (Tucker primary, Sydorskyi-yoink stretch)
- **Track B** = own-teacher pseudo-labels on top of Track A

---

## Why this ordering

Murilo's framing: "execute a known-recipe path to ~0.93 with high confidence; if time permits, attempt the higher-ceiling path with code we don't have."

This trades theoretical ceiling for execution probability. Reasoning:

1. **We've stalled at 0.906 after 30+ experiments.** Most failures were "right concept, wrong details" (exp 014/015/025 PL with wrong teacher; exp 031 KD with wrong head and stack). Paths we have full code for have lower implementation risk.
2. **Tucker's notebook is fully vendored.** `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`. Code-audited at `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`. Exp 031 already has 70% of the implementation (DistillHead, PerchTeacher, KD branch in `sed_finetune.py`).
3. **Sydorskyi-yoink is methodology-only** in the public discussion. Ali Ozan, Jack, Antoine disclosed *that* they use it, not *how*. The melspec match (Jack: −0.012 LB swing) and attention dim 1028 are warnings that small details are load-bearing.
4. **Own-teacher PL has a simple protocol** (Antoine: no confidence filter, treat predictions as labels). Implementation risk is small.

The compound is: a known-recipe foundation reduces the chance we lose the next 12 days chasing implementation bugs in an unfamiliar pipeline, while still leaving room to try the higher-ceiling stretch if Tucker replication clears the runway early.

---

## Code-reference quality (honest audit, 2026-05-22)

| Path | Code reference we have | Adaptation risk |
|---|---|---|
| **Tucker's distilled-SED full recipe (PRIMARY)** | **FULL CODE** at `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`. Already audited. Exp 031 has KD + Perch ONNX + DistillHead + sed_finetune.py wired (70%). | **LOW** — direct port. Remaining work: swap LSEHead → logit-space AttHead; add dual-supervision (clip+frame_max BCE) blend at inference (0.5+0.5); switch to Tucker's waveform mixup with `MIXUP_HARD=True` (max(y1,y2)); add SpecAugment; switch OneCycle → 25 ep cosine; 5-fold split. |
| **Sydorskyi-yoink + own head + 2026 fine-tune (STRETCH)** | Sydorskyi's 2025 writeup + Kaggle dataset for checkpoints (`vladimirsydor/bird-clef-2025-all-pretrained-models`). **No fine-tuning code.** Ali Ozan / Jack threaded answers only. | **MEDIUM-HIGH** — we invent the head architecture matching attn dim 1028, melspec preprocessing wrapper, fine-tune training loop. Every detail Jack mentioned (melspec mismatch = −0.012) tells us this is brittle to small choices. |
| **Own-teacher PL (FINAL LAYER)** | Antoine's threaded description: pseudo-label unlabeled SS with current best model, treat as additional labeled rows, retrain. No filtering, no confidence threshold. | **LOW** — protocol is simple enough that the description is the implementation. |
| Own XC pretrain (Antoine path) — **OFF THE TABLE for 12-day budget** | None. Antoine kept ASL details secret. | HIGH. Would take 7-10 days alone, leaves no time for the rest. |
| ProtoSSM stack (Karnakbayev 0.949) | FULL CODE at `vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/` + reproduce notebook + public weights at `hideyukizushi/sgkfk-202604041716`. | LOW — but doesn't match Murilo's "train our own from scratch" preference. **Keep as ensemble component possibility, not lead.** |
| Salman's attention investigation | Methodology only. | HIGH (open-ended debugging). **Optional parallel track if time permits, not standalone path.** |

---

## The Perch question, reframed

Murilo: "I'm skeptical about Perch not working for us — most competitors use it."

Honest re-read of our Perch failures:

| Experiment | What we tried | Why it failed (clearer now) |
|---|---|---|
| Exp 014/015/025 | PL on unlabeled SS with Perch's predictions as the labels (category #1) | Cross-architecture distillation on unlabeled data. We converged toward Perch's biases instead of becoming a stronger SED model. Different lever from own-teacher PL. |
| Exp 031 | Tucker's category #3 KD (Perch-embedding MSE on a GAP projection, with `feat.detach()` on cls branch) on **HGNet + LSEHead + 20 ep + α=1.0** | Wrong consumer head (LSEHead pools by magnitude; Perch features encode spatial localization that AttHead would consume). Wrong stack (no AttHead, no dual-supervision, no hard-max mixup, no SpecAugment, wrong schedule). Single-variable KD on our v5j stack is **not Tucker's recipe**. |

**Perch is not dead for us.** Tucker's *full* recipe is what gives him +0.02 LB from KD. Our exp 031 was a single-variable test on a recipe stack incompatible with the KD lever's mechanism. The Tucker-faithful replication in Track A is what actually engages this lever.

This reframing changes the strategic calculus: the high-confidence path (Tucker replication) is also the path that finally tests Perch KD properly. We get to use Perch and have full reference code, simultaneously.

---

## Open questions on the table (from this discussion)

1. **EffNetV2-S vs ECA-NFNet-L0** for Sydorskyi-yoink stretch — discussion settled tentatively on EffNetV2-S (more public recipe coverage, lower implementation risk; ECA-NFNet-L0 is the ensembling-diversity pick for later).
2. **Labeled soundscapes in training**: keep, upsample, or hold out? Not blocking Track A; revisit once new base is up.
3. **Sydorskyi-yoink architecture for our head**: melspec params + 1028 attn dim must match exactly. Implementation cost is the main reason this is stretch, not primary.
4. **PL confidence policy**: use all teacher predictions as Antoine does, or threshold? Antoine's simpler protocol is the default unless we observe failure.

Murilo's leaning (recorded): both ambition and safety. Strive for highest LB possible while having a relatively safe approach with what we know.

---

## What was discussed but NOT committed to

- Building a fresh XC pretrain ourselves (Antoine's actual path) — off table for 12-day budget; possible post-competition learning project.
- Pivoting fully to the ProtoSSM/Perch-frozen stack (EoS.6's 0.949 path) — doesn't match the "train our own" goal; keep weights as potential ensemble component instead.
- Salman attention investigation as standalone path — kept as optional parallel track if pipeline is running smoothly.

---

## Reference materials (all vendored, all readable now)

### Tucker's distilled-SED replication path
- `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb` — Tucker's full notebook
- `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` — code audit
- `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` — Tucker's discussion thread
- `user/murilo/experiments/031-perch-kd-distillation/` — 70% of the implementation already
- Memory: `feedback_atthead_aggregation_domain.md` — the AttHead bug Tucker doesn't have

### Sydorskyi-yoink stretch path
- `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` — Ali Ozan/Jack/Antoine thread (see Thread 10 for Sydorskyi backbone)
- Kaggle dataset: `vladimirsydor/bird-clef-2025-all-pretrained-models`

### Own-teacher PL final layer
- `user/murilo/notes/kaggle-discussion-non-perch-limit.md` — Antoine's full disclosure on PL protocol
- `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` Thread 10 — Boredom's confirmation on 1-iter PL lift

### Background / context
- `user/murilo/notes/2026-05-21-nina2025-eos6-protossm-stack-analysis.md` — the Perch-heavy 0.949 ensemble we're NOT chasing as lead path
- `user/murilo/experiments/031-perch-kd-distillation/results.md` — exp 031 REJECT and post-mortem
- `user/murilo/CONTEXT.md` — session handoff

---

## Concrete next-session starting point

When Murilo resumes:

1. **Re-read this doc + `CONTEXT.md`** to reload context.
2. **Decide whether to start scaffolding exp 032 as full Tucker replication** (vs the single-variable KD+AttHead originally scoped for it).
3. **Implementation order for exp 032 (Tucker replication, Track A primary)**:
   - Port `AttHeadLogitSpace` to `sed_head.py` (per CONTEXT.md step 2)
   - Switch backbone in config from HGNetV2-B0 to `tf_efficientnet_b0.ns_jft_in1k` (Tucker's)
   - Add dual-supervision inference blend (0.5 sigmoid(clip) + 0.5 sigmoid(frame_max)) in `submission.py`
   - Switch mixup to Tucker's: waveform mixup, Beta(0.4,0.4), MIXUP_HARD=True (max(y1,y2))
   - Add SpecAugment (1 freq mask × 10 + 2 time masks × 10)
   - Switch OneCycle → 25 ep cosine with 2 ep linear warmup
   - 90/10 focal/SS batch composition (Tucker uses MixSamp with shares=focal:0.9, sc:0.1)
   - Add focal × labeled-SS mixup (Tucker's second augmentation)
   - 5-fold KFold on filenames + MultiLabelStrat on focal labels
   - Site/hour priors at inference + 5-tap Gaussian smoothing
4. **Submit to LB** as paired control + KD variants (preserve exp 031 paired-control methodology to avoid hardware-drift contamination).
5. **If LB ≥ 0.91 single → proceed to Track B (own-teacher PL).** Else diagnose before iterating.
6. **Stretch (parallel or sequential): Sydorskyi-yoink.** Start at day 4-5 if Tucker pipeline is stable.

The exp 032 scaffolding plan in `CONTEXT.md` is partially correct — the single-variable AttHead+KD-on-HGNet test it describes is now subsumed by the full Tucker replication. The scaffold commands are the same; the configs and head choices are different.
