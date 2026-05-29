# Kaggle help-request post — DRAFT (for Murilo to post)

- **Target thread**: "What is your best single model LB score?"
  (https://www.kaggle.com/competitions/birdclef-2026/discussion/683791)
- **Status**: draft — Murilo posts manually.
- **Tone**: concise; states our stack + plateau, asks 2 sharp questions. Does NOT dump the full recipe.
- Optional: @-mention folks who've replied to Murilo before (@tuckerarrants, @salmanahmedtamu, @hengck23) — but the thread is already where they hang out, so a clean question should get seen.

---

My best single model is **0.906 LB** — HGNetV2-B0 with an SED attention head (ImageNet init), trained on train_audio + labeled soundscapes with BCE. Adding Perch v2 embedding knowledge-distillation keeps the single model at 0.906 but a 5-fold ensemble reaches **0.915**, and a Perch-embedding-probe ⊕ SED ensemble gets me to **0.929**.

I've been stuck around here for a while — ~30 experiments across backbones (HGNet / EffNet / EffNetV2), heads (LSE vs attention), losses (BCE / CE / focal / ASL), waveform vs spectrogram mixup, longer context windows (10–20s), and own-teacher pseudo-labels — and almost nothing moves the LB. I also reproduced @tuckerarrants' excellent Distilled SED baseline (thank you Tucker!) and landed at the same 0.906 / 0.915.

My biggest blocker: **my offline CV anti-correlates with the LB.** Validation macro-AUC improvements consistently fail to translate — or invert — on the leaderboard, so I can't tell a good change from a bad one without spending a submission. Two questions for those of you above 0.93:

1. **What validation scheme actually tracks your LB?** OOF on labeled soundscapes vs focal, GroupKFold-by-site vs random, scoring all 234 classes vs present-only — what gave you a CV you could trust?
2. **What was the single change that first took you past ~0.91?** Architecture/pretraining, pseudo-labels, or post-processing?

Any pointers hugely appreciated with the deadline approaching 🙏
