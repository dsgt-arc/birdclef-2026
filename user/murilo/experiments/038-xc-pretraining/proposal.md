---
experiment: "038-xc-pretraining"
status: proposed
date: 2026-05-30
author: Murilo Gustineli
branch: mg/038-xc-pretraining
depends_on: ["034-pretrained-backbone", "032-tucker-distilled-sed", "023-perch-sed-ensemble"]
---

# Proposal — Broad Neotropical XC Pretraining (EffNetV2-S) → fine-tune on BirdCLEF 2026

## 1. Overview

Build **our own** broad bird-classification backbone by supervised pretraining EffNetV2-S on a
**curated Neotropical subset of BirdSet-XCL** (a 9,734-species / 480 GB Xeno-Canto snapshot), then
fine-tune on the BirdCLEF 2026 234 classes. This is the recipe the 2025 winners used (broad XC
pretrain → fine-tune → [PL/ensemble]); here it's the **post-deadline working-note contribution** +
a genuinely **backbone-diverse ensemble member**, not a 06-03 LB push.

**Initialization (load-bearing — clarified 2026-05-30):** "from scratch" means **we run our own XC
supervised-classification pretraining**, NOT random init and NOT warm-starting from someone else's
bird checkpoint (BirdSet-ConvNeXT / Perch — that path = exp 034, capped 0.88). The EffNetV2-S backbone
is **ImageNet-21k–initialized** (timm `tf_efficientnetv2_s.in21k`, `pretrained=True`) — standard
practice (matches all our SED models + the winners), strictly better than random init.

## 2. Honest scope + ceiling (decided with discriminator review)

- **Realistic ceiling ~0.88–0.90 single-fold.** Direct evidence: exp 034/036 fine-tuned a broad-XC
  backbone → 0.879–0.882 (< our 0.906 KD). Pretraining strengthens **focal-domain representation**,
  which we already have (ImageNet+KD = 0.906); it does **not** fix the focal→soundscape **domain
  shift** that caps our LB. Value = paper + ensemble diversity, not plateau-break.
- **Birds-only coverage.** BirdSet is birds. The 2026 targets are **162 Aves + 72 non-birds** (35
  Amphibia, 28 Insecta — **25 with zero focal**, 8 Mammalia, 1 Reptilia). This pretraining helps the
  **162 bird targets**; the 72 non-birds (our weakest, focal-less classes) get **nothing** from it
  and are learned only in fine-tuning. **The insect/amphibian gap is OUT OF SCOPE here** (a separate
  targeted-XC pull, deferred).

## 3. Decisions (locked)

| Decision | Value |
|---|---|
| Corpus | **BirdSet-XCL**, curated to a **broad Neotropical bird subset** incl. all 162 bird targets (NOT narrow-to-234 — breadth is the value) |
| Subset breadth | ~1.5–3k species (Neotropical-occurring birds) |
| Backbone | **EffNetV2-S**, **ImageNet-21k init** (`tf_efficientnetv2_s.in21k`) — proven winner + distinct from our HGNet/EffNet-B0 → ensemble diversity |
| Pretrain mode | our own XC supervised-classification pretrain (NOT warm-start from a bird ckpt) |
| Mel | match our SED stack (Tucker mel, n_mels=256, sr=32000) so the model fine-tunes + ensembles cleanly |
| Horizon | **post-deadline paper** (overnight A100 run → model by Sun 5/31; do it right, don't race 06-03) |
| Non-birds | **out of scope** |

## 4. Method — 4 stages

- **Stage 0 — Curate (no GPU; `scripts/curate_birdset_neotropical.py`).** Load BirdSet-XCL **metadata
  only** (HF; `ebird_code`, `lat`, `long`, `length`, `quality`, `filepath`). Define the pretrain
  species pool = {species occurring in the Neotropics (≥N recordings inside a S/Central-America
  lat/long bbox)} ∪ {the 162 BirdCLEF-2026 bird targets, mapped via ebird_code→scientific_name}.
  Per-species cap ~500 (highest-quality/most-recent), min ~10, **quality A/B**. Emit a **file
  manifest + species_list + size report**; selective-download only the manifest's audio (skip the
  full 480 GB + the slow XC API crawl). **Smoke first** (first N shards → verify schema + counts).
- **Stage 1 — Pretrain (A100, overnight after a ~30-min smoke).** EffNetV2-S (ImageNet-21k init),
  supervised **multi-label** classification over the pretrain species (secondary labels as extra
  positives), Tucker mel (n_mels=256), BCE+Focal + label-smoothing 0.05, AdamW cosine→1e-6, ~15–20 ep,
  **no class balancing** (winners' rule), `save_top_k=-1`. **Held-out XC val** = the clean pretraining
  signal. **Discard the pretrain head; keep the backbone.**
- **Stage 2 — Fine-tune on 2026 (all 234).** Re-init a 234-class head; reuse our SED fine-tune recipe
  + full domain-shift aug (waveform MixUp, background mixing, SpecAugment). Non-birds learned here.
- **Stage 3 — Eval / use.** Pretrain quality → XC-val. Downstream → **LB** (the only arbiter;
  labeled-SS anti-correlates — [[project_cv_measurement_failure]]). Blend the fine-tuned model into the
  0.929 ensemble for backbone diversity.

## 5. Falsifiable / decision thresholds

- **Pretrain works** if held-out XC macro-AUC is strong (≳0.95, in-distribution) — sanity that the
  loop learned. (This is NOT a 2026-LB predictor.)
- **Downstream**: fine-tuned single-fold LB vs the 0.906 KD baseline + vs exp 034's 0.879 XC-backbone.
  CONFIRM if it beats 0.906 (would refute the "pretraining caps 0.88" read); else it's a diverse
  ensemble member — keep iff the prob-space blend into 0.929 beats 0.929 on LB.
- **Abandon-from-scratch tripwire** (deep-research): if pretrain doesn't beat ImageNet-init fine-tune
  on XC-val by the smoke checkpoint, stop and reconsider.

## 6. Reuse / build

Reuse: `EfficientNetEncoder` (timm wrapper, in_chans=1), Tucker mel, the SED fine-tune module
(exp 032/034), exp 010 export wiring. Build: Stage-0 curation script + a supervised-classification
pretraining LightningModule (multi-label over N pretrain species) + configs/sbatch. **`feedback_use_uv_venv` / `feedback_user_runs_commands`**: Claude builds + local-tests; Murilo runs the HF
download + A100 pretrain.

## 7. References

- `user/murilo/notes/2026-05-30-deep-research-xeno-canto-pretraining.md` (the recipe)
- BirdSet (Rauch et al. 2024, arXiv:2403.10380; HF `DBD-research-group/BirdSet`, config XCL);
  Perch 2.0 "bittern lesson" (arXiv:2508.04665); Hong et al. TMLR 2024 (fine-grained pretraining).
- Prior in-house: exp 034/036 (XC-backbone → 0.88, the ceiling evidence).
