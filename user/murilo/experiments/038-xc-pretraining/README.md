# Exp 038 — Broad Neotropical XC Pretraining (EffNetV2-S) → fine-tune on BirdCLEF 2026

**Goal (post-deadline paper track):** build our own broad bird-classification backbone — EffNetV2-S,
**ImageNet-21k initialized**, supervised-pretrained on a curated **Neotropical subset of BirdSet-XCL**
(9,734-sp / 480 GB XC snapshot) — then fine-tune on the 2026 234 classes. A genuinely backbone-diverse
ensemble member + the working-note's pretraining contribution.

**Honest framing:** realistic ceiling **~0.88–0.90** single-fold (exp 034/036 evidence: a broad-XC
backbone fine-tunes to ~0.88, below our 0.906 KD). Pretraining strengthens *focal-bird* representation
(which we already have); it does **not** fix the focal→soundscape domain shift that caps our LB.
**Birds-only:** helps the 162 Aves targets; the 72 non-birds (incl. 25 focal-less insects) are **out of
scope** here.

## Pipeline
- **Stage 0 (no GPU):** `scripts/curate_birdset_neotropical.py` → curated `manifest.parquet` +
  `species_list.json` + size report (Neotropical-occurring birds ∪ 162 targets; cap 500/sp, q A/B).
- **Stage 1 (A100 overnight):** supervised multi-label classification pretrain on the subset.
- **Stage 2:** fine-tune on 2026 (all 234, re-init head).
- **Stage 3:** XC-val (pretrain quality) + LB (downstream) + blend into the 0.929 ensemble.

## Run (Murilo, on PACE)
```bash
cd user/murilo/experiments/038-xc-pretraining
# Stage 0 — curate (verify HF schema with --smoke first; needs internet):
python scripts/curate_birdset_neotropical.py --smoke 2000
python scripts/curate_birdset_neotropical.py            # full → manifest
# (selective audio download + Stage 1 pretrain code: built after the manifest lands)
```

Decisions locked (2026-05-30): broad Neotropical (~1.5–3k bird sp), EffNetV2-S ImageNet-21k init,
our-own XC pretrain (not warm-start), Tucker mel (n_mels=256), post-deadline, insects out of scope.
Full plan: `proposal.md`. Status: `tasks.md`.
