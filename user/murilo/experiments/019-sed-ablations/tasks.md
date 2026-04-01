      # Experiment 019: Perch-Guided SED Training — Tasks

## Goal

Push SED from 0.860 → 0.900-0.915 LB by combining HGNet-inspired architecture
improvements with Perch pseudo-labels on 10K+ unlabeled soundscapes.

**Current SED Best**: 0.860 LB (EfficientNet-B0 SED v4b, exp 017)
**Teacher**: Perch ProtoSSM v4.1 ~0.920 LB (exp 018)
**Target**: 0.900-0.915 LB (SED alone, enabling future Perch+SED ensemble)

### Reference Chain

| Score | Model | Key technique |
|-------|-------|---------------|
| 0.860 | Our SED v4b | EfficientNet-B0, AmplitudeToDB, AttHead, soft CE |
| 0.884 | ttahara HGNetV2-B0 | LSEHead, BCE, MixUp α=1.0, 4-fold ensemble |
| 0.908 | Salman Ahmed SED | Single EfficientNet-B0 (unknown techniques) |
| ~0.920 | Our Perch ProtoSSM | Teacher for pseudo-labels |
| 0.939 | Competition #1 | Likely ensemble + pseudo-labels |

### Status: **v5j-4fold 0.913 LB — best trained model!** v5j single 0.906, v5g-4fold 0.901. v5m (embed_dim test) training.

- **Branch**: `mg/019-sed-4fold-ensemble`
- **Architecture analysis**: `docs/vendor/efficientnet-vs-hgnetv2.md`

---

## Phase A: Improved SED Baseline (target: 0.870-0.885 LB)

Establish a stronger SED model before pseudo-labeling. Apply quick wins
from HGNet gap analysis, then 4-fold ensemble.

Reference: `user/murilo/notes/gap-analysis-hgnet-vs-effnet.md`

### Step 1: Quick wins ablation (parallel runs on PACE)

All ablations start from v4b (0.860 LB) with a single change.
90/10 split, 50 epochs, same data pipeline.

Code changes implemented (all backward compatible with exp 017):
- `src/birdclef/encoders/efficientnet.py`: generic timm support via `in_chans=1`
- `src/birdclef/heads/sed_head.py`: added `LSEPooling` + `LSEHead`
- `src/birdclef/lightning/sed_finetune.py`: added `mixup_warmup_epochs`,
  `mixup_theta`, weighted label mixing, `lam >= 0.5` enforcement
- `experiments/017-mel-overhaul/train.py`: registered LSEHead + new mixup params

Fast-dev-run: all 5 configs passed. GPU check: >95% utilization on all.

- [x] 1.1 **v5a — BCEWithLogitsLoss**: Replace soft CE with standard BCE.
      Config: `loss_type: "bce"`. Submitted to PACE (2026-03-30).
- [x] 1.2 **v5b — Stronger MixUp**: α=1.0 (uniform), θ=0.8 label clipping,
      5-epoch warmup. Config: `mixup_alpha=1.0, mixup_theta=0.8,
      mixup_warmup_epochs=5`. Submitted to PACE (2026-03-30).
- [x] 1.3 **v5c — Lower weight_decay**: 1e-4 (from 0.01). Config:
      `weight_decay: 1.0e-4`. Submitted to PACE (2026-03-30).
- [x] 1.4 **v5d — LSEHead**: Log-Sum-Exp pooling head (GeM freq + LSE time).
      Config: `sed_head.type: "lse_head"`. Submitted to PACE (2026-03-30).
- [x] 1.5 **v5e — HGNetV2-B0 backbone**: `hgnetv2_b0.ssld_stage2_ft_in1k`
      (embed_dim=1024). Submitted to PACE (2026-03-30).
- [x] 1.6 Collect results: val/auroc, val/cmap for all 5 runs
      - v5a: 0.875 / 0.620 (LB 0.851, -0.009)
      - v5b: 0.869 / 0.638 (LB 0.840, -0.020)
      - v5c: 0.867 / 0.624 (LB pending)
      - v5d: 0.861 / — (LB pending)
      - v5e: 0.861 / 0.612 (LB pending)
      - **Conclusion**: All ablations regressed on LB despite some offline gains.
        val/auroc is a poor LB predictor for single-variable changes on v4b.

### Step 2: ttahara Full Recipe Reproduction

Pivoted from "combine winners" to faithful reproduction of ttahara's complete
training pipeline (0.884 LB). The Step 1 results showed that individual changes
don't help — the full recipe must be adopted together.

Code changes: LSEHead(freq_pool=mean), TtaharaMelTransform, per-sample MixUp,
OneCycleLR. See `src/birdclef/heads/sed_head.py`, `src/birdclef/preprocessing/ttahara_mel.py`,
`src/birdclef/lightning/sed_finetune.py`. 22 pytest tests passing.

- [x] 2.1 Implement LSEHead with `freq_pool="mean"` option (matching ttahara)
- [x] 2.2 Implement TtaharaMelTransform (torchaudio, resize 256x256, min-max)
- [x] 2.3 Implement per-sample MixUp lambda in SEDFinetuneModule
- [x] 2.4 Implement OneCycleLR scheduler option (single param group, per-step)
- [x] 2.5 Update train.py: mel type routing, freq_pool passthrough, new params
- [x] 2.6 Create v5f config (EfficientNet-B0 + ttahara recipe)
- [x] 2.7 Create v5g config (HGNetV2-B0 + ttahara recipe)
- [x] 2.8 Audit: cross-referenced all 12 components against ttahara ground truth
- [x] 2.9 Submit v5f and v5g to PACE (2026-03-30)
- [x] 2.10 Collect v5g results: **0.872 val/auroc**, epoch 18, 21 min training
      - +0.011 over v5e (same backbone, old recipe: 0.861)
      - +0.007 over v4b (EfficientNet, old recipe: 0.865)
      - Training 8-10× faster (21 min vs ~3-4h)
- [x] 2.10b Collect v5f results: **0.866 val/auroc**, epoch 14
      - +0.001 over v4b (same backbone, old recipe: 0.865) — recipe barely helps EfficientNet
      - -0.006 vs v5g (HGNet with same recipe: 0.872) — backbone is the main factor
- [x] 2.11 Compare results:
      - Recipe impact on EfficientNet: v5f (0.866) vs v4b (0.865) = +0.001 (negligible)
      - Backbone impact under ttahara recipe: v5g (0.872) vs v5f (0.866) = +0.006
      - Recipe unlocking HGNet: v5e (0.861) → v5g (0.872) = +0.011
- [x] 2.12 Decision gate: v5g 0.872 ≥ 0.870 target → proceed to Step 3 (4-fold ensemble)
- [x] 2.13 Add v5f/v5g to exp 010 submission pipeline:
      - export_efficientnet_sed.py: pass freq_pool to LSEHead, handle ttahara mel_config
      - export.sbatch: add export functions + dispatch for v5f/v5g
      - upload_kaggle.py: add MODELS entries
      - submission.py: add compute_mel_ttahara + MODEL_SLUGS (per-segment, not full-signal)
      - kernel-metadata JSON files for both models
- [x] 2.14 Export v5f/v5g to ONNX + upload weights + push v5f kernel to Kaggle
- [x] 2.15 Push v5g kernel to Kaggle (2026-03-31)
- [x] 2.16 Check v5f LB result: **0.855** (−0.005 vs v4b 0.860)
      - ttahara recipe on EfficientNet regressed on LB despite matching offline (0.866 vs 0.865)
      - Consistent with prior pattern: offline gains don't translate for EfficientNet
- [x] 2.17 Check v5g LB result: **0.892** (+0.032 vs v4b) — val under-predicted LB!

### Step 3: 4-fold ensemble

- [x] 3.1 Implement MultiLabel Stratified GroupKFold (from ttahara, group by audio_id)
- [x] 3.2 Train 4 folds of HGNet v5g on PACE (~21 min each, all 4 parallel)
      - Fold 0: 0.939 val/auroc (epoch 14) — 55/66 soundscape files in val
      - Fold 1: 0.842 val/auroc (epoch 19)
      - Fold 2: 0.826 val/auroc (epoch 18)
      - Fold 3: 0.802 val/auroc (epoch 16)
      - Fold imbalance (0.137 spread) didn't prevent ensemble from working on LB
- [ ] 3.3 Optionally train 4 folds of the **other backbone** too — mixed-backbone
      ensemble (EfficientNet + HGNet) may outperform single-backbone ensemble
- [x] 3.4 Export all 4 fold models to ONNX
- [x] 3.5 Implement fold-averaged inference kernel (ensemble detection + compute mel once)
- [x] 3.6 Validate on LB: **0.901** (target was ≥0.885) — **exceeded target by +0.016!**
      - v5g single: 0.892 (+0.032 over v4b)
      - v5g 4-fold: **0.901** (+0.009 from ensemble, +0.041 over v4b)
      - Surpassed ttahara's 0.884 reference by +0.017

### Step 4: v5g Recipe Ablations (baseline: v5g 0.892 LB)

Iterate on the validated v5g recipe. Each ablation changes ONE thing from v5g.
Train single-fold first (~21 min), submit best to LB, then 4-fold the winner.

Evidence-based (from our prior experiments):
- [x] 4.1 **v5h — n_fft=4096**: Submitted to PACE (2026-03-31).
      Keep ttahara recipe, change mel n_fft from 2048→4096. Resize absorbs frame count.
- [x] 4.2 **v5i — 10s context windows**: Submitted to PACE (2026-03-31).
      Change window_duration 5s→10s. Training ~2x slower (~42 min).
- [x] 4.3 **v5j — AmplitudeToDB norm**: Submitted to PACE (2026-03-31).
      NNAudio mel (no resize, 256×511 output). Confounded: also changes win_length + resize.
- [x] 4.4 Collect results, compare vs v5g (0.892 LB)
      - v5h (n_fft=4096): 0.869 val → **0.893 LB** (+0.001 vs v5g)
      - v5i (10s windows): **0.878 val** → **0.880 LB** (−0.012 vs v5g)
      - **v5j (AmplitudeToDB): 0.874 val → 0.906 LB (+0.014 vs v5g, best single-fold!)**
      - v5k (3-channel mel): 0.868 val → LB pending
      - **Conclusion: offline val/auroc cannot rank configs. v5j best on LB despite
        being only +0.002 offline. v5i best offline but worst on LB.**
- [x] 4.5 Decision: v5j is the LB winner → proceed to 4-fold ensemble
- [x] 4.6 **v5j 4-fold ensemble**: Trained on PACE (2026-03-31). ~38 min/fold on V100.
      - Fold 0: 0.951 val/auroc (epoch 15)
      - Fold 1: 0.843 val/auroc (epoch 19)
      - Fold 2: 0.823 val/auroc (epoch 19)
      - Fold 3: 0.808 val/auroc (epoch 18)
      - Same fold imbalance as v5g-4fold. Marginally better on folds 0/1/3.
- [x] 4.7 Export v5j-4fold to ONNX, upload weights, push kernel — done (2026-03-31)
- [x] 4.8 **Submit v5j-4fold to LB**: **0.913 LB** — best trained model! +0.007 from 4-fold
- [x] 4.9 **Submit v5k to LB**: **0.886 LB** — 3-channel mel, within noise of v5g
- [x] 4.10 **v5l — v5j + n_fft=4096**: Trained on PACE (2026-04-01).
      0.874 val/auroc (same as v5j), epoch 16. LB pending.
- [ ] 4.11 Submit v5l to LB — submitted to Kaggle (2026-04-01), awaiting result

Embed_dim hypothesis test:
- [x] 4.15 **v5m — EfficientNet-B0 + projection 320→1024**: Trained on PACE (2026-04-01).
      **0.873 val/auroc** (epoch 16), 0.645 val/cmap. SLURM job 6149183.
      **0.898 LB** — projection closes ~83% of the v5f→v5j gap (+0.043 of 0.051).
      Remaining 0.008 gap to v5j (0.906) likely from HGNet architectural advantages.
- [x] 4.16 Submit v5m to LB: **0.898** (+0.043 vs v5f, −0.008 vs v5j)
- [x] 4.17 **v5n — EfficientNet-B0 + v5j recipe, NO projection** (control): Trained on PACE (2026-04-01).
      **0.871 val/auroc** (epoch 19), 0.629 val/cmap. **0.875 LB**.
      2×2 grid: mel (v5f→v5n) = +0.020 LB, **projection (v5n→v5m) = +0.023 LB**.
      Confirms embed_dim is the primary bottleneck (45% of v5f→v5j gap).
- [x] 4.18 Submit v5n to LB: **0.875** (+0.020 vs v5f from mel, −0.023 vs v5m from projection)
- [x] 4.19 Update analysis in `docs/vendor/efficientnet-vs-hgnetv2.md` with LB results

Medium confidence (from competitors):
- [ ] 4.20 **5-fold CV** (vs current 4-fold) — pantanal-distill uses 5-fold
- [ ] 4.21 **Post-processing**: temperature scaling, confidence-sharpened smoothing,
      per-class thresholds on v5j predictions (zero training cost)
- [ ] 4.22 **NS-JFT pretrained weights** on EfficientNet or HGNet equivalent
- [x] 4.13 **v5k — 3-channel mel input**: Trained on PACE (2026-03-31).
      Repeat mono mel to 3 channels, encoder in_chans=3 (preserves full pretrained weights).

Lower priority:
- [ ] 4.23 **Perch + SED ensemble**: Combine best SED (v5j-4fold) with Perch ProtoSSM
      (0.918) for a combined submission — different architectures = diverse errors
- [ ] 4.24 **EfficientNetV2 backbone**: If v5m confirms embed_dim matters, try
      `efficientnetv2_rw_t.ra2_in1k` (13.7M, 82.4% top-1, 208-dim) + projection.
      Stronger backbone features + projection may outperform HGNet.

---

## Phase B/C: Perch Pseudo-Label Training — DEFERRED

> **Descoped from exp 019.** Perch pseudo-label extraction and multi-round
> pseudo-label training will be a separate experiment. The idea: use Perch
> ProtoSSM to extract soft pseudo-labels from all ~10,600 unlabeled soundscapes,
> then fine-tune the best SED config (v5j/HGNet) on labeled + pseudo-labeled
> data with conservative weighting and iterative refinement.
>
> See proposal.md Phase B/C for the original plan.

---

## Results Tracking

| Submission | val/auroc | LB | Notes |
|------------|----------|-----|-------|
| SED v4b (baseline) | 0.865 | 0.860 | EfficientNet-B0, AmplitudeToDB, AttHead, soft CE |
| v5a (BCE) | **0.875** | **0.851** | `loss_type: "bce"`, epoch 29, **+0.010 val but −0.009 LB vs baseline** |
| v5b (MixUp) | **0.869** | **0.840** | `mixup_alpha=1.0, theta=0.8, warmup=5`, epoch 31, **−0.020 LB vs baseline** |
| v5c (weight_decay) | **0.867** | pending | `weight_decay=1e-4`, epoch 35 |
| v5d (LSEHead) | **0.861** | pending | `sed_head.type: "lse_head"`, epoch 21 |
| v5e (HGNetV2) | 0.861 | pending | `hgnetv2_b0.ssld_stage2_ft_in1k`, epoch 23 |
| v5f (EffNet+ttahara) | **0.866** | **0.855** | Full ttahara recipe on EfficientNet-B0, epoch 14 |
| v5g (HGNet+ttahara) | **0.872** | **0.892** | Full ttahara recipe on HGNetV2-B0, epoch 18 |
| v5g-4fold (HGNet×4) | 0.939/0.842/0.826/0.802 | **0.901** | **Broke 0.900!** 4-fold ensemble |
| v5h (n_fft=4096) | 0.869 | **0.893** | n_fft 2048→4096, epoch 16, +0.001 LB vs v5g |
| v5i (10s windows) | **0.878** | **0.880** | window 5→10s, epoch 17, −0.012 LB vs v5g |
| **v5j (AmplitudeToDB)** | 0.874 | **0.906** | **Best single-fold! NNAudio+AmpToDB, +0.014 LB vs v5g** |
| v5k (3-channel mel) | 0.868 | **0.886** | in_chans=3, epoch 16, within noise of v5g |
| v5l (v5j+n_fft=4096) | 0.874 | pending | v5j + n_fft=4096, epoch 16, same val as v5j |
| **v5j-4fold (HGNet×4)** | 0.951/0.843/0.823/0.808 | **0.913** | **Best trained model! +0.007 from 4-fold, +0.012 over v5g-4fold** |
| v5m (EffNet+proj1024) | **0.873** | **0.898** | EfficientNet-B0 + 1x1 Conv2d 320→1024, v5j recipe. −0.008 vs v5j, +0.043 vs v5f |
| v5n (EffNet, no proj) | 0.871 | **0.875** | EfficientNet-B0, v5j recipe, no projection. Projection = +0.023 LB. |

---

## Runtime Budget (Kaggle kernel, 90 min)

| Component | v5j single | v5j 4-fold (actual) | v5g 4-fold (actual) |
|-----------|-----------|-------------------|-------------------|
| Model loading | ~1s | ~4s | ~4s |
| Audio → mel (NNAudio) | ~8 min | ~8 min | ~15 min (ttahara+resize) |
| SED inference (ONNX) | ~10 min | ~40 min | ~15 min |
| Post-processing | ~1s | ~1s | ~1s |
| **Total** | **~20 min** | **~50 min** | **~30 min** |

v5j-4fold actual: ~50 min on Kaggle CPU kernel. Well within 90 min budget,
leaving ~40 min for potential Perch+SED ensemble.
