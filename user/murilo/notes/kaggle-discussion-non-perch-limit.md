# Kaggle Discussion: What's the limit without any Perch involved?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/ (BirdCLEF+ 2026 discussion forum, post by an unidentified user asking about non-Perch ceilings)
- **Posted**: ~2026-05-17 (estimated from "Posted 4 days ago" at capture time)
- **Captured**: 2026-05-21
- **Status at capture**: 14 comments
- **Companion thread**: [`kaggle-discussion-best-single-model-lb.md`](kaggle-discussion-best-single-model-lb.md) — many of the same people show up in both
- **Companion vendored notebook**: `vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/` (the Perch-heavy 0.949 stack — the thing this thread is asking how to compete with *without* Perch)

---

## TL;DR

Three independent recipes hit 0.93+ on the LB **without any Perch** as of 2026-05-21:

| Person | Single-model LB | Recipe summary |
|---|---:|---|
| **Arunodhayan** (7th) | **0.946** | No details disclosed |
| **Antoine Masq** (49th) | **0.943** | EffNetV2-B0 + XC pretrain + custom ASL loss + own-teacher PL + sliding-window TTA + smoothing. 10 min inference. |
| **Boredom** (5th) | **0.938-0.939** (no PL); 0.950+ (1-iter PL) | timm + SED, ~11 min, non-Perch |
| **Salman Ahmed** (39th) | **0.937** | EffNetV2-B0 SED, "modeling on attention" investigation, PP/TTA. Documented in the other thread. |

The two big public takeaways from this thread:

1. **XC pretraining is necessary, not optional.** Antoine pretrained EffNetV2-B0 on Xeno-Canto BEFORE 2026 fine-tuning. Without it, his EffNetV2-B0 stalls at 0.88 (his original baseline). The Sydorskyi-yoink (Ali Ozan 0.941) is the same lever via a different XC checkpoint.
2. **Own-teacher pseudo-labels** add +0.007 LB on top of an already-strong (0.936) teacher (Antoine 0.936 → 0.943 from improved-teacher PL). Boredom reports +0.012 LB from 1-iter PL on his 0.938 base.

This thread also documents the **failure mode for Jack** (rank 21, training from scratch): "I've tried every method that has been discussed this year alongside the top two methods from last year. I haven't managed to see that increase people are getting." → confirms that copying-the-recipe-list-only-half-works without the unspoken details.

---

## Original post

> Hello everyone, I notice nearly all top-ranked notebooks on the leaderboard (including my own solutions) rely heavily on Perch in one way or another. Most either extract audio embeddings via Perch or adopt Perch-based knowledge distillation. I think this may have caused severe lack of model diversity among our approaches. In my exps, merging another extra Perch-based model into ensembles barely brings any effective gain — the final LB improvement is only around 0.001 at most, which is basically within random score fluctuation. For this reason, I've been trying to train pure CNN branches only using mel-spectrogram input with EfficientNet-B0, without introducing any Perch weights, embeddings or distillation strategies at all. Unfortunately, all my attempts can hardly break 0.88 on public LB. Here are my two main questions:
>
> 1. What is the approximate upper bound public LB score we can reach if we stick purely to CNN pipelines with zero Perch involvement?
> 2. Under current mainstream ensemble frameworks, can we still obtain valid leaderboard gains by adding well-trained non-Perch models, even if their single-model scores are below 0.9?

---

## Comments (chronological)

### Arunodhayan (7th) — Posted 3 days ago ⭐ CEILING DATAPOINT

> my single model LB is 0.946

No further details disclosed. This is the strongest non-Perch single-model data point published as of 2026-05-21. Rank suggests serious work behind it.

### yh-nnn (164th) — Posted 3 days ago

> How did you do it? Unbelievable.

(No recorded reply from Arunodhayan.)

### Zejun_ (11th) — Posted 3 days ago

> impressive score

### tennogh (33rd) — Posted 4 days ago ⭐ DIVERSITY CONFIRMS

> If you look at the best LB scores threads you will see people made strong models without any Perch involvement. Even using Perch, different methods can still be complementary, e.g. the public notebooks using Perch embeddings, the one using SED with distillation aren't very correlated.

**Reading**: pure-CNN models DO have non-Perch ensemble value. Tucker's distilled-SED CNN and the ProtoSSM-Perch-pipeline are uncorrelated enough to ensemble usefully (this is what the EoS.6 stack exploits).

### Antoine Masq (49th) — Posted 4 days ago ⭐ THE FULL RECIPE

> I'm only using custom CNN pipelines (EfficientNetV2-B0 only) without using Perch, and now my best single model gets a 0.936 LB score and I can reach 0.948 by ensembling different runs

### To be happy (1781st) — Posted 4 days ago

> Did you initialize your EfficientNetV2-B0 with pretrained weights?

### Antoine Masq — Posted 4 days ago ⭐ XC PRETRAIN CONFIRMED

> Yes and I also pretrained it on Xeno-Canto data

### Antoine Masq — Posted a day ago ⭐ MAJOR UPDATE TO 0.943

> Update: now my best single model is leading to a 0.943 LB score with efficientnetv2_b0 (10min for inference)

### Jack (21st) — Posted a day ago

> Is this with pseudolabels or no? What about tta or postprocessing?

### Antoine Masq — Posted a day ago ⭐ PL CONFIRMED

> Yes with pseudo-labels and the classic TTA with the sliding window and smoothing patterns

### Jack — Posted a day ago ⭐ FAILURE MODE

> Am I the only moron who hasn't managed to benefit from his pseudolabels in this competition? 🫠

### Antoine Masq — Posted a day ago

> Given your rank, you seem to do great anyway haha. Are you also training from scratch or do you use Perch? If it's the former you can have a look at last year's write-ups, they often discuss it

### Jack — Posted a day ago ⭐ "TRIED EVERYTHING, STILL STUCK" PATTERN

> From scratch. I've tried every method that has been discussed this year alongside the top two methods from last year. I haven't managed to see that increase people are getting. I've only managed to distill the ensemble into a solo model that achieves a slightly weaker score than the ensemble. I must be screwing something up but I haven't figured out what…

**Reading**: even rank-21 competitors training from scratch struggle to capture all the published levers. The published recipes are necessary but not sufficient — there's tacit knowledge in the implementation. This is comforting context for our 0.906 → 0.93 gap.

### Antoine Masq — Posted a day ago

> Strange, personally I just incorporated it by using the unlabeled data just as the labeled one but the labels are predicted by the teacher models

**Reading**: Antoine's PL recipe is the simplest possible — pseudo-label the unlabeled SS with a strong teacher, then treat them as additional training data alongside labeled data. No filtering, no confidence thresholding mentioned. Self-distillation in the classical sense.

### Jack — Posted 21 hours ago

> If you don't mind me asking, what loss are you using?

### Antoine Masq — Posted 21 hours ago ⭐ ASL CONFIRMED, DETAILS WITHHELD

> It's a custom ASL, but the details are a secret haha

**Reading**: Antoine's loss is Asymmetric Loss (ASL) with custom modifications. ASL is a multi-label classification loss (Ridnik et al. 2021) designed for long-tail problems. We tested ASL in exp 016 and rejected it on baseline HGNet. **Worth reconsidering now that we have a different starting recipe (XC pretrain + PL)** — the same loss may behave very differently when the backbone has already been XC-aligned and the targets are richer (own-teacher PL).

### Jack — Posted 20 hours ago

> Keep your secrets 😂 Was your 0.936 with pseudolabels and TTA as well or no? Thanks for the answers so far

### Antoine Masq — Posted 19 hours ago ⭐ 0.936 ALSO HAD PL

> The 0.936 was indeed with pseudo labels (in fact it was essentially the same but I improved the teacher model)

**Reading**: Antoine's trajectory was NOT "XC pretrain alone → 0.936 → +PL → 0.943". It was "XC pretrain + PL → 0.936 → improved teacher + PL → 0.943". So PL is load-bearing through the whole 0.88 → 0.943 climb. XC pretrain is what unlocked the recipe; PL is what compounds it.

The 0.936 → 0.943 delta (+0.007) is from **improving the teacher used to generate pseudo-labels**, not from adding more PL iterations.

### Dhruv Pai Dukle (1021st) — Posted a day ago

> Impressive without Perch Distillation was this with or without XC pre-training?

### Antoine Masq — Posted a day ago

> With Pre-training

### Donghui Zhang (1547th) — Posted 10 hours ago

> I'm curious how many iterations your pseudo-tags have gone through?

(No recorded reply.)

### To be happy (1781st) — Posted 9 hours ago ⭐ THE BOOTSTRAP QUESTION

> Curious to know, at what single-model LB score do you typically start ensembling, pseudo-labeling, or distillation?

### Dhruv Pai Dukle — Posted 3 hours ago

> You don't know. That's the beauty, only thing you can reliably do is check the model co-relation

**Reading**: there's no published threshold for "when is the teacher strong enough to PL." Empirically: Tucker quantified +0.013 LB from a 0.886 own-teacher; chenwenqiang_001 confirmed +0.013 from a 0.886-grade self-teacher. Boredom got +0.012 from his 0.938 base. Antoine got +0.007 from his 0.936 base.

The pattern: **PL lift is roughly +0.007-0.013, mostly independent of teacher strength** within the 0.88-0.94 band. So even a 0.886-0.906 teacher is "good enough" to start.

### MengYe (558th) — Posted 4 days ago ⭐ PUBLIC HGNET PATH

> you can use asymmetric loss with the public hgnet notebook and some tuning to reach 0.90+ lb, some public post-processing methods can make it reach 0.91+ lb

**Reading**: a concrete cheap recipe → **ASL + public HGNet notebook + tuning → 0.90+ → +PP → 0.91+**. Two independent levers stacked.

### BNBU Anpeng Yuan (491st) — Posted 3 days ago

> Hi Meng! In the public HGNet notebook, after changing the output chunk to 10 seconds, what corresponding adjustments should be made to the inference code? Since this is not an SED model and cannot produce frame‑level outputs, if the input becomes 10 seconds, the output also seems to become 10 seconds.

### To be happy — Posted 3 days ago ⭐ ASL REPRO FAILURE

> Are you referring to this asymmetric loss? Why did my LB score drop by 0.03 after implementing the modification based on the public hgnet notebook?

(Image attached: ASL code snippet, content not preserved in this capture.)

### Dhruv Pai Dukle — Posted 2 days ago

> Mine To honestly

**Reading**: ASL drops LB by 0.03 for at least one re-implementer. **ASL is recipe-sensitive** — wrong γ_+ / γ_- / threshold values destroy it. This is the same trap Jack flagged ("tried every method").

---

## Key competitor scores (this thread + cross-ref)

| Person | Rank | Single LB | Method | Notes |
|---|---:|---:|---|---|
| **Arunodhayan** | **7th** | **0.946** | No Perch | No details |
| **Boredom** | 5th | 0.938-0.939 (no PL); 0.950+ (PL) | timm+SED, non-Perch | Other thread |
| **Ali Ozan** | 6th | 0.941 | Sydorskyi 2025 XC backbone + own head | Cross-ref [`kaggle-discussion-best-single-model-lb.md`](kaggle-discussion-best-single-model-lb.md) |
| **Salman Ahmed** | 39th | 0.937 | EffNetV2-B0, "modeling on attention" | Other thread |
| **Antoine Masq** | **49th** | **0.943** | EffNetV2-B0 + XC pretrain + own-teacher PL + ASL + sliding-window TTA + smoothing | 10 min inference, full recipe disclosed minus ASL details |
| **Jack** | 21st | (not disclosed in this thread; "didn't manage to see the increase") | From scratch, tried "every method" | Public-recipe-only failure mode |
| **tennogh** | 33rd | (cross-thread) | KD on Perch | Confirms ensemble value across non-Perch and Perch models |
| **MengYe** | 558th | not stated, claims path to 0.91+ | Public HGNet + ASL + PP | The cheap-recipe path |

---

## Antoine Masq's full recipe (synthesized from this thread + best-single-model thread)

This is the cleanest disclosed non-Perch recipe at the 0.94+ tier:

| Stage | Detail |
|---|---|
| **Backbone** | EfficientNetV2-B0 (only — single architecture, no ensembling across architectures within single-model score) |
| **Pretrain** | Xeno-Canto (Antoine's own pretrain, not Sydorskyi-yoinked) |
| **Fine-tune data** | 2026 train_audio + labeled_soundscapes + **pseudo-labeled unlabeled soundscapes** (PL targets from a teacher model) |
| **Loss** | Custom asymmetric loss (ASL variant; specific parameters undisclosed) |
| **Mixup** | (not disclosed in this thread; probably waveform mixup per the broader pattern) |
| **PL recipe** | Use unlabeled SS just like labeled SS, with teacher's predictions as labels. Iterate teacher: each generation of student becomes next teacher. |
| **TTA** | "Classic sliding window" — most likely 5s window slid by some hop across the 60s clip, predictions averaged |
| **Post-processing** | Smoothing patterns (specifics undisclosed) |
| **Inference** | 10 min total |
| **Single-model LB trajectory** | 0.889 (baseline, March) → 0.936 (XC pretrain + initial PL) → 0.943 (improved teacher PL) |
| **Ensemble LB** | 0.948 (ensembling "different runs" of the same recipe) |

**Critical implementation gap** between this and our codebase:
- We don't have XC-pretrained EffNetV2-B0 (we have HGNet+LSEHead from-scratch at 0.906, and Tucker's NS-JFT init at the timm-default level).
- We've never run own-teacher PL (exp 014/015/025 used Perch as teacher, not a 0.906 SED model).
- ASL is implemented but was rejected on baseline HGNet (exp 016).

---

## Implications for our path

### What this thread proves

1. **Non-Perch single-model 0.93+ is reproducible** by at least 4 different people with disclosed (or partly-disclosed) recipes. It's not a one-off lucky run.
2. **The two essential levers are**: (a) in-domain pretrain (XC), (b) own-teacher PL on unlabeled SS. Stack them, and you get 0.94+.
3. **The ASL / loss / head choice is recipe-sensitive but small**: it's the 3rd lever, ~+0.005-0.010 swing depending on tuning. Get the first two right or it doesn't matter.
4. **PL recipe is simple**: pseudo-label unlabeled SS with current best model, treat them like additional labeled rows, retrain. No filtering/confidence-threshold protocol disclosed by anyone. The "iteration" is teacher-quality iteration, not threshold tuning.

### What this thread does NOT settle

1. **How much XC pretrain is "enough"**: Sydorskyi pretrained on 819K-row XC; Antoine doesn't disclose his scale; exp 029 tried 28K and rejected it. The right scale is somewhere between 28K and 819K — we don't know.
2. **ASL parameters**: γ_+/γ_-/clip threshold are the make-or-break knobs. To be happy & Dhruv both report −0.03 LB from wrong-ASL.
3. **Antoine's TTA specifics**: "classic sliding window" — window size, hop, smoothing kernel all undisclosed.
4. **Whether multi-architecture diversity matters within single-model PL chains**: Antoine uses only EffNetV2-B0. Some Perch-stacks blend many architectures. The non-Perch tier seems okay with single-arch.

### Direct implications for our Track 1 (B → A) plan

Track B (Sydorskyi backbone yoink) skips the XC-pretrain blocker entirely:
- We download Sydorskyi's 819K-pretrained EffNetV2-S / ECA-NFNet-L0.
- We attach a new head (matching attention dim 1028 + melspec exactly per Ali Ozan/Jack).
- We fine-tune on 2026 focal + labeled SS.
- LB target: **0.92-0.94** (Ali Ozan 0.941 is the proof point; we'd probably land a bit lower without his unstated tweaks).

Track A (Antoine recipe) then layers PL on top:
- Use the Track B model as the initial teacher.
- Pseudo-label all unlabeled SS.
- Retrain Track B with focal + labeled SS + pseudo-labeled SS.
- LB target: **+0.007 to +0.013** on top of Track B (so 0.93-0.95 if Track B lands 0.92-0.94).
- Iterate teacher: feed student-2 back as teacher for round-3.

**Combined ceiling estimate**: 0.93-0.95 single-model, 0.94-0.96 ensemble. Antoine + Boredom are the proof points.

---

## Open questions to resolve before committing

1. **Storage cost of Sydorskyi backbones**: how large are the checkpoint files? Local download + Kaggle upload needed for inference.
2. **Does Sydorskyi's 2025 backbone work on 2026 species?** The 2025 classifier head saw 2025 species. The backbone is generic acoustic features. Ali Ozan trains a new head on top — confirmed working at 0.941.
3. **Confidence threshold for PL**: should we filter pseudo-labels below threshold X, or trust the teacher as Antoine does (no filtering)?
4. **Inference time budget**: 10 min (Antoine) vs 30 min (Ali Ozan 5-fold ONNX). Kaggle hard limit 90 min. Our v5j-4fold takes ~40 min including ensemble. Plenty of headroom.

---

## Cross-references

- `user/murilo/notes/kaggle-discussion-best-single-model-lb.md` — companion thread; many same people; canonical recipe-comparison home (also being updated 2026-05-21)
- `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` — the Perch-distilled side that this thread is rebelling against
- `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` — Tucker's code audit
- `user/murilo/notes/2026-05-21-nina2025-eos6-protossm-stack-analysis.md` — the Perch-heavy 0.949 ensemble; this thread is the alternative path
- Our exp 016 (ASL rejected on HGNet baseline) — worth re-running with Sydorskyi backbone + PL
- Our exp 029 (2025-only pretrain rejected at 28K-row scale) — was the right idea, wrong scale; Sydorskyi-yoink supersedes
- Our exp 014/015/025 (PL rejected with Perch-as-teacher) — was the right concept, wrong teacher; own-teacher PL is what Antoine/Boredom use
