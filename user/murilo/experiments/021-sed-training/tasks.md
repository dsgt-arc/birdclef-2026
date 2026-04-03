# Experiment 021: SED Training — Tasks

## Goal

Push single SED model from 0.906 → 0.920+ LB by applying competitor insights
(priors, CE+sigmoid, waveform augs) and exploring larger backbones.

**Current Best SED**: 0.906 LB (v5j single), 0.913 LB (v5j-4fold)
**Target**: 0.920+ single, 0.925+ 4-fold

### Status: **Phase A in progress — implementing site/hour priors on SED output.**

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

Port prior fusion from exp 020 Perch pipeline to SED submission kernel.
Pure post-processing — no retraining.

- [ ] A.1 Port `fuse_scores_with_tables` + `fit_prior_tables` to SED submission kernel
- [ ] A.2 Submit v5j + priors → target ≥0.912
- [ ] A.3 Submit v5j-4fold + priors → target ≥0.919

---

## Phase B: Training Recipe Ablations (3-5 days, 4-6 LB subs)

Single-variable changes from v5j baseline.

- [ ] B.1 Implement CE(primary_label) training mode (verify in train.py)
- [ ] B.2 Implement waveform augmentations (noise, time stretch) in train.py
- [ ] B.3 Train v6a (CE loss) on PACE
- [ ] B.4 Train v6b (waveform MixUp) on PACE
- [ ] B.5 Train v6c (full waveform augs) on PACE
- [ ] B.6 Train v6d (25 epochs) on PACE
- [ ] B.7 Submit winners to LB
- [ ] B.8 Train combined recipe (v6e) from best Phase B findings

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
| v5j-4fold (exp 019) | **0.913** | — | 4-fold ensemble |
| Perch V18 (exp 020) | **0.925** | — | Two-kernel TF, reference |
