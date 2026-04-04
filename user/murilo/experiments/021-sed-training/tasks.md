# Experiment 021: SED Training — Tasks

## Goal

Push single SED model from 0.906 → 0.920+ LB by applying winning solution
techniques (AttHead + dual supervision, CE loss, backbone scaling).

**Current Best SED**: 0.909 LB (v5j-pp2 single), 0.917 LB (v5j-4fold-pp2)
**Target**: 0.920+ single, 0.925+ 4-fold

### Status: **Phase C: v6e-v6j training on PACE. AttHead + dual supervision on multiple backbones.**

- **Branch**: `mg/021-sed-training`

---

## Config Comparison

### Phase B — Recipe ablations (COMPLETED, no gains)

| Config | Backbone | Head | Loss | MixUp | val/auroc | LB | Change from v5j |
|--------|----------|------|------|-------|-----------|-----|-----------------|
| v5j (baseline) | HGNetV2-B0 | LSEHead | BCE | spec | 0.874 | **0.906** | — |
| v6a | HGNetV2-B0 | LSEHead | CE | spec | 0.854 | — | −0.020 (regressed) |
| v6b | HGNetV2-B0 | LSEHead | BCE | wave | 0.871 | — | −0.003 |
| v6c | HGNetV2-B0 | LSEHead | BCE | wave+noise | 0.861 | — | −0.013 (regressed) |
| v6d | HGNetV2-B0 | LSEHead | BCE | spec | 0.873 | — | −0.001 (25ep, flat) |

**Lesson**: Recipe changes on LSEHead don't help. The gap is architectural (head type).

### Phase C — AttHead + backbone scaling (IN PROGRESS)

All use **AttHead with dual supervision** (clip 0.5 + frame 0.5, blend 75% clip + 25% frame max).
Motivated by BirdCLEF 2023/2024/2025 winning solutions + Salman's explicit recommendation.

| Config | Backbone | Params | Embed | Head | Loss | Priority | Rationale |
|--------|----------|--------|-------|------|------|----------|-----------|
| **v6e** | HGNetV2-B0 | 6M | 1024 | **AttHead** | BCE | **Highest** | Isolate head: LSEHead → AttHead |
| **v6f** | HGNetV2-B0 | 6M | 1024 | **AttHead** | **CE** | **High** | Winning combo (2024+2025 1st) |
| **v6g** | EffNet-B3 NS | 12M | 384→1024 | **AttHead** | **CE** | **High** | Winning recipe on larger EffNet |
| **v6h** | HGNetV2-B2 | 12M | 1536 | **AttHead** | BCE | Medium | Larger backbone + best head |
| **v6i** | ConvNeXt-V2 Nano | 15M | 640 | **AttHead** | BCE | Medium | Architecture diversity |
| **v6j** | EffNet-B0 NS | 5.3M | 320→1024 | **AttHead** | BCE | Medium | Boredom says EffNet reaches 925+ |

---

## Phase A: Site/Hour Priors on SED — COMPLETE

- [x] A.1 Generate priors, create -pp2 weights dirs, register in exp 010
- [x] A.2 v5j-pp2 → **LB: 0.909** (+0.003)
- [x] A.3 v5j-4fold-pp2 → **LB: 0.917** (+0.004)

---

## Phase B: Training Recipe Ablations — COMPLETE (no gains)

- [x] B.1-B.3 Implementation (CE loss, waveform augs, reproducibility)
- [x] B.4 v6a (CE+LSEHead): val/auroc 0.854 (−0.020, regressed)
- [x] B.5 v6b (waveform MixUp): val/auroc 0.871 (−0.003)
- [x] B.6 v6c (waveform augs): val/auroc 0.861 (−0.013, regressed)
- [x] B.7 v6d (25 epochs): val/auroc 0.873 (−0.001, flat)

**Key finding**: v6a (CE+LSEHead) regressed because CE needs AttHead with
framewise max to work. 2024+2025 winners use CE on AttHead, not LSEHead.

---

## Phase C: AttHead + Backbone Scaling (IN PROGRESS)

Motivated by winning solution analysis:
- 2023 1st: AttHead (GeM freq + attention), 75/25 clip/frame blend
- 2024 1st: CE loss "significantly better" than BCE, EfficientNet-B0
- 2025 1st: AttHead + CE, EfficientNet B0/B3/B4, 7-model ensemble
- Salman (0.921): "Attn head's framewise max is better than LSE head"

- [x] C.1 Create AttHead configs (v6e-v6j) — dual supervision, 75/25 blend
- [x] C.2 Train v6e (HGNetV2-B0 + AttHead + BCE) — submitted to PACE
- [x] C.3 Train v6f (HGNetV2-B0 + AttHead + CE) — submitted to PACE
- [x] C.4 Train v6g (EffNet-B3 NS + AttHead + CE) — submitted to PACE
- [x] C.5 Train v6h (HGNetV2-B2 + AttHead + BCE) — submitted to PACE
- [x] C.6 Train v6i (ConvNeXt-V2 Nano + AttHead + BCE) — submitted to PACE
- [x] C.7 Train v6j (EffNet-B0 NS + AttHead + BCE) — submitted to PACE
- [ ] C.8 Check training results, compare val/auroc
- [ ] C.9 ONNX export + Kaggle inference timing validation
- [ ] C.10 Submit winners to LB (no priors) — target ≥0.920 single
- [ ] C.11 Add priors to best model (-pp2 variant)
- [ ] C.12 4-fold ensemble of best backbone — target ≥0.925+

---

## Results Tracking

| Submission | LB | val/auroc | Notes |
|------------|-----|-----------|-------|
| v5j (exp 019 baseline) | **0.906** | 0.874 | HGNetV2-B0, LSEHead, BCE |
| v5j-pp2 | **0.909** | — | v5j + site/hour priors (+0.003) |
| v5j-4fold (exp 019) | **0.913** | — | 4-fold ensemble |
| **v5j-4fold-pp2** | **0.917** | — | v5j-4fold + priors (+0.004), current best SED |
| v6a (CE+LSEHead) | — | 0.854 | −0.020 offline, regressed |
| v6b (waveform MixUp) | — | 0.871 | −0.003 offline |
| v6c (waveform augs) | — | 0.861 | −0.013 offline, regressed |
| v6d (25 epochs) | — | 0.873 | −0.001 offline, flat |
| Perch V18 (exp 020) | **0.925** | — | Two-kernel TF, reference |
