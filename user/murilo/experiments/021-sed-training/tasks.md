# Experiment 021: SED Training — Tasks

## Goal

Push single SED model from 0.906 → 0.920+ LB by applying competitor insights
(priors, CE+sigmoid, waveform augs) and exploring larger backbones.

**Current Best SED**: 0.906 LB (v5j single), 0.913 LB (v5j-4fold)
**Target**: 0.920+ single, 0.925+ 4-fold

### Status: **Phase B: v6a/b/c/d training on PACE. v6a+v6b running, v6c+v6d pending.**

- **Branch**: `mg/021-sed-training`

---

## Config Comparison

| Config | Backbone | Params | Embed | Head | Loss | MixUp | Epochs | Change from v5j |
|--------|----------|--------|-------|------|------|-------|--------|-----------------|
| v5j (baseline) | HGNetV2-B0 | 6M | 1024 | LSEHead | BCE | spec α=1.0 | 20 | — |
| **v6a** | HGNetV2-B0 | 6M | 1024 | LSEHead | **CE** | spec α=1.0 | 20 | CE(primary_label) + sigmoid infer |
| **v6b** | HGNetV2-B0 | 6M | 1024 | LSEHead | BCE | **wave** α=1.0 | 20 | Waveform MixUp |
| **v6c** | HGNetV2-B0 | 6M | 1024 | LSEHead | BCE | **wave+noise+stretch** | 20 | Full waveform aug stack |
| **v6d** | HGNetV2-B0 | 6M | 1024 | LSEHead | BCE | spec α=1.0 | **25** | Longer training |
| **v6f** | **HGNetV2-B2** | **12M** | **1536** | LSEHead | BCE | spec α=1.0 | 20 | Larger HGNet |
| **v6g** | **ConvNeXt-V2 Nano** | **15M** | **640** | LSEHead | BCE | spec α=1.0 | 20 | Modern ConvNet |
| **v6h** | **EffNet-B3 NS** | **12M** | **384→1024** | LSEHead | BCE | spec α=1.0 | 20 | MBConv comparison |

---

## Phase A: Site/Hour Priors on SED (1 day, 1-2 LB subs)

Uses exp 010's existing `_apply_metadata_priors` (multiplicative in probability
space, taxa-aware lambda). Generated priors via `build_inference_priors.py`.
Created separate `-pp2` weights directories (base v5j unchanged).

- [x] A.1 Generate metadata_priors.json + taxa_mapping.json for v5j and v5j-4fold.
      Created `hgnet-b0-sed-v5j-pp2` and `hgnet-b0-sed-v5j-4fold-pp2` weights dirs.
      Registered in exp 010: upload_kaggle.py, submission.py MODEL_SLUGS, kernel-metadata.
- [x] A.2 Upload v5j-pp2 to Kaggle, push kernel, submit to LB.
      Priors: 9 sites, 13 hours, 1478 segments (Laplace α=0.01).
      **LB: 0.909** (+0.003 over v5j 0.906, below 0.006 target)
- [x] A.3 Submit v5j-4fold-pp2 → **LB: 0.917** (+0.004 over v5j-4fold 0.913)

---

## Phase B: Training Recipe Ablations (3-5 days, 4-6 LB subs)

Single-variable changes from v5j baseline.

- [x] B.1 Implement CE(primary_label) training mode — already in train.py
- [x] B.2 Implement waveform augmentations (pink noise, time stretch) in sed_finetune.py
- [x] B.3 Add reproducibility: pl.seed_everything(42), Trainer(deterministic=True), git hash logging
- [x] B.4 Train v6a (CE loss) on PACE — running on V100 (job 6265569)
- [x] B.5 Train v6b (waveform MixUp) on PACE — running on RTX 6000 (job 6265571)
- [x] B.6 Train v6c (full waveform augs) on PACE — pending (job 6265572)
- [x] B.7 Train v6d (25 epochs) on PACE — pending (job 6265576)
- [ ] B.8 Submit winners to LB
- [ ] B.9 Train combined recipe (v6e) from best Phase B findings

---

## Phase C: Backbone Scaling (2-3 days, 2-3 LB subs)

Larger non-MBConv backbones with ONNX inference validation.

- [ ] C.1 Verify HGNetV2-B2, ConvNeXt-V2 Nano, EffNet-B3 NS work with train.py
- [ ] C.2 Train v6f (HGNetV2-B2) on PACE
- [ ] C.3 Train v6g (ConvNeXt-V2 Nano) on PACE
- [ ] C.4 Train v6h (EffNet-B3 NS) on PACE
- [ ] C.5 ONNX export + Kaggle inference timing validation
- [ ] C.6 Submit best to LB — target ≥0.920
- [ ] C.7 4-fold ensemble of best backbone — target ≥0.925+

---

## Results Tracking

| Submission | LB | val/auroc | Notes |
|------------|-----|-----------|-------|
| v5j (exp 019 baseline) | **0.906** | 0.874 | HGNetV2-B0, NNAudio AmplitudeToDB |
| v5j-pp2 | **0.909** | — | v5j + site/hour priors (+0.003, 11 min runtime) |
| v5j-4fold (exp 019) | **0.913** | — | 4-fold ensemble |
| **v5j-4fold-pp2** | **0.917** | — | v5j-4fold + site/hour priors (+0.004) |
| Perch V18 (exp 020) | **0.925** | — | Two-kernel TF, reference |
