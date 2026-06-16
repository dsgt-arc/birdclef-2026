---
experiment: "038-xc-pretraining"
status: running   # Stage 0 done; Stage 1 built + 3-cell grid running on rtx6000 (2026-05-31)
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
| Corpus | **BirdSet-XCL**, curated to a **broad Neotropical bird subset** incl. all bird targets (NOT narrow-to-234 — breadth is the value) |
| Subset breadth | **2,683 Neotropical bird species / 160,542 recs / 2,202 h** (curation result); **158/162** bird targets present (`palhor3`/`strher2` absent). 34.6% of recs carry secondary labels. |
| Backbone | **EffNetV2-S**, **ImageNet-21k init** (`tf_efficientnetv2_s.in21k`, embed_dim 1280) — proven winner + distinct from our HGNet/EffNet-B0 → ensemble diversity |
| Pretrain mode | our own XC supervised-classification pretrain (NOT warm-start from a bird ckpt) |
| **Head (load-bearing)** | **SED-mirror**: reuse `DistilledSEDFinetuneModule` UNCHANGED (`AttHeadLogitSpace`, num_classes=2683, KD off). Trains the backbone under the EXACT downstream pooling/mel regime → best transfer + lowest implementation risk (battle-tested module, no new training code). Chosen over a vanilla GAP+linear classifier. |
| Loss | **dual-supervision** `0.5·BCE(clip)+0.5·BCE(frame_max)` + **label-smoothing 0.05** (applied in-loss to a local target copy so `labels.int()` metrics stay hard). Multi-label: primary ∪ in-vocab secondaries. |
| Mel | match our SED stack (Tucker mel, n_mels=256, sr=32000) so the model fine-tunes + ensembles cleanly |
| Schedule | AdamW lr 5e-4, wd 1e-4, 2-ep warmup → cosine→1e-6, 16-mixed, grad-clip 1.0, **25 ep, `save_top_k=-1`** (pick the XC-val peak), eff batch 128, num_workers 8, stratified per-species 2% val. |
| **Run strategy** | **proven primary + parallel insurance arms** (compute is free; LB is the only reliable ranker, so XC-val macro-AUC is a *sanity gate* only). 3 cells: **A** dual-BCE+LS (PRIMARY), **B** focal-BCE (`FocalLossBCE` α0.25/γ2), **C** no-mixup. Fine-tune Cell A to LB first; B/C are ready backups. |
| Horizon | **post-deadline paper** + ensemble-diversity member (do it right, don't race 06-03) |
| Non-birds | **out of scope** |

## 4. Method — 4 stages

- **Stage 0 — Curate (no GPU; `scripts/curate_birdset_neotropical.py`).** Load BirdSet-XCL **metadata
  only** (HF; `ebird_code`, `lat`, `long`, `length`, `quality`, `filepath`). Define the pretrain
  species pool = {species occurring in the Neotropics (≥N recordings inside a S/Central-America
  lat/long bbox)} ∪ {the 162 BirdCLEF-2026 bird targets, mapped via ebird_code→scientific_name}.
  Per-species cap ~500 (highest-quality/most-recent), min ~10, **quality A/B**. Emit a **file
  manifest + species_list + size report**; selective-download only the manifest's audio (skip the
  full 480 GB + the slow XC API crawl). **Smoke first** (first N shards → verify schema + counts).
- **Stage 1 — Pretrain (BUILT + running; rtx6000 grid).** EffNetV2-S (ImageNet-21k init) under the
  **SED-mirror** recipe = reuse `DistilledSEDFinetuneModule` UNCHANGED (`AttHeadLogitSpace`,
  num_classes=2683, **KD off**), Tucker mel (n_mels=256), 5 s random crop, `apply_aug` (gain/noise) +
  focal×focal mixup, SpecAugment. **Dual-supervision** `0.5·BCE(clip)+0.5·BCE(frame_max)` + LS 0.05;
  multi-label = primary ∪ in-vocab secondaries. AdamW lr 5e-4 → cosine→1e-6, 25 ep, **no class
  balancing**, `save_top_k=-1`, stratified per-species 2% **XC val** (= the clean sanity signal, NOT a
  2026-LB predictor). New data layer = `XCPretrainDataset`/`XCPretrainDataModule`; everything else is
  reuse. **3-cell run strategy** (compute free, LB-only ranker): A dual-BCE+LS (primary) / B focal-BCE
  / C no-mixup — fine-tune A first, B/C are backups. **Discard the pretrain head; keep the backbone.**
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

Reuse (UNCHANGED): `EfficientNetEncoder` (timm wrapper, in_chans=1), `TuckerMelSpectrogram`,
`AttHeadLogitSpace`, `DistilledSEDFinetuneModule` (the whole SED training loop + metrics), `SpecAugment`,
`apply_aug`/audio-loader from `FocalDS`, exp 010 export wiring. **Built (minimal):** Stage-0 curation +
download scripts; `src/birdclef/data/xc_pretrain_dataset.py` (the only new data layer); a *gated additive*
`loss_type="focalbce"` + `label_smoothing` in the SED module (bce path bit-identical, regression-tested);
`train.py` + `configs/_common.yaml`+3 cells + `sbatch/{run.sh,pretrain.sh}` + `save_metrics.py`. **`feedback_use_uv_venv` / `feedback_user_runs_commands`**: Claude builds + local-tests; Murilo runs the HF
download + GPU pretrain.

## 7. References

- `user/murilo/notes/2026-05-30-deep-research-xeno-canto-pretraining.md` (the recipe)
- BirdSet (Rauch et al. 2024, arXiv:2403.10380; HF `DBD-research-group/BirdSet`, config XCL);
  Perch 2.0 "bittern lesson" (arXiv:2508.04665); Hong et al. TMLR 2024 (fine-grained pretraining).
- Prior in-house: exp 034/036 (XC-backbone → 0.88, the ceiling evidence).
