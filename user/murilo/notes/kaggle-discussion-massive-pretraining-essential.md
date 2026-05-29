# Kaggle Discussion: Is massive pretraining essential now?

- **URL**: https://www.kaggle.com/competitions/birdclef-2026/discussion/ (BirdCLEF+ 2026 discussion forum; post by Olly Powell)
- **Posted**: ~2026-05-28 (estimated from "Posted 15 hours ago" at capture time)
- **Topic author**: Olly Powell (1268th at capture)
- **Comment count at capture**: 1
- **Captured**: 2026-05-29
- **Companion threads**:
  - [`kaggle-discussion-best-single-model-lb.md`](kaggle-discussion-best-single-model-lb.md) — "What is your best single model LB score?" (the big recipe thread)
  - [`kaggle-discussion-non-perch-limit.md`](kaggle-discussion-non-perch-limit.md) — "What's the limit without any Perch involved?"

Notation: `↳` indicates a reply (one level of nesting). ⭐ marks comments referenced by the synthesis at the bottom.

---

## Original post

**Olly Powell (1268th) — Posted 15 hours ago** ⭐

> **Is massive pretraining essential now?**
>
> Has anybody at all been able to achieve a high score without massive pretraining? Or is this the main reason some of us are stuck on single model scores in the 80s whilst others seem to easily cruise to scores over 92?
>
> Last year I pretrained on 506 classes from the 21, 22 & 23 competitions and it seemed quite helpful. This year I notice my pretraining recipe greatly influences the behaviour of subsequent steps. It feels like the pretraining is really doing the heavy lifting this year.
>
> Also it appears that potentially all the high scoring solutions are doing at least one of the following:
>
> - Using, or distilling BirdNet or Perch backbones (trained on thousands of classes)
> - Using pre-trained weights, and feature pipelines from earlier years competitions, all trained on at least 850 classes.
> - Training their own backbones based on the whole XC database.

---

## Comments

**↳ tennogh (37th) — Posted 3 hours ago** ⭐

> Pre-training does help quite a bit, I expect a lot of solutions to use it. Last year some used it, some others re-trained from scratch at each pseudo-labeling stage. You can definitely beat 0.92 without specific pre-training though, I have done it with 1 checkpoint of timm Efficientnet_b0.

---

## Synthesis / relevance to us (exp 034 + final experiment)

**The post crystallizes the three "heavy-lifting" paths the top of the LB uses — and we've now touched two of them:**

| Path (Olly's taxonomy) | Our status |
|---|---|
| **Distill BirdNet/Perch backbones** | exp 032 (+0.050 KD on ImageNet b0 → 0.906/0.915); exp 023 Perch-MLP 0.929. **KD does NOT stack on a strong in-domain backbone** (exp 034 Phase K: joint-KD 0.878 ≈ full-FT+PP 0.879 — [[project_kd_multiplier_not_additive]]). |
| **Reuse earlier-year pretrained weights/feature pipelines** | exp 034 — Sydorskyi's 2025 in-domain EffV2S backbone (819K XC). Validated bit-identical (Gate 1), full-FT+PP = 0.879. Necessary-not-sufficient. |
| **Train own backbone on whole XC database** | **NOT done.** This is what Antoine Masq / Ali Ozan / Sydorskyi did. The proposed final experiment (XC collection + own-teacher PL) is the closest we'd get without months of pretraining. |

**tennogh's data point is the load-bearing one for morale + strategy:** **0.92+ is reachable with a single timm Efficientnet_b0 and NO specific pretraining.** That means our 0.906 plateau is NOT primarily a pretraining-gap — others clear 0.92 from a plain ImageNet b0. The differentiator at our level is therefore the *training-side* stack (PL, PP, TTA, context, modeling), not the backbone. This reframes "are we stuck because we lack a giant pretrain?" → "no — the gap is the PL + PP + context stack we haven't fully landed."

**Cross-check against the best-single-model thread:** many competitors report **0.93+ single-model WITHOUT pseudo-labels** (Jack 0.933 5-fold no-pseudo/no-distill; Salman 0.937–0.939; Boredom 0.937–0.939; Antoine 0.936→0.943). So even PL is not the *only* unlock — strong supervised modeling + PP gets to ~0.93; PL adds the last +0.005–0.01.

### Our draft reply (NOT yet posted — 2026-05-29)

> thanks for sharing! It seems that many people were able to achieve above 0.93 without any pseudo-labels in this post [What is your best single model LB score?](https://www.kaggle.com/competitions/birdclef-2026/discussion/683791). I've been stuck at 0.906 for a while now. I've implemented ~30 different experiments with varying model configurations and approaches, and nothing has worked.
>
> I've even implemented the [Distilled SED Baseline](https://www.kaggle.com/competitions/birdclef-2026/discussion/694479) that @tuckerarrants shared (which was very helpful, thank you for sharing Tucker!) which also got the same 0.906 LB score, and 0.915 with a 5-fold approach.

**Decision pending (per Murilo):** whether to post as a reply here (smaller audience) or as a new comment in the best-single-model thread (larger audience), and whether to expand it with a high-level write-up of our best approach + an explicit ask for advice. See discussion in-session 2026-05-29.
