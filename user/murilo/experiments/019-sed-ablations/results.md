# Experiment 019: Perch-Guided SED Training — Results

## Phase A: Improved SED Baseline

### Step 1: Quick Wins Ablation

All ablations start from v4b (0.865 val/auroc, 0.860 LB) with a single change.
90/10 split, 50 epochs, same data pipeline.

| Slug | Change | Best val/auroc | val/cmap | Best Epoch | Stopped | Delta vs v4b | LB | Checkpoint |
|------|--------|---------------|----------|------------|---------|-------------|-----|------------|
| `effnet-b0-sed-v5a` | BCEWithLogitsLoss | 0.875 | 0.620 | 29 | 35 | +0.010 | **0.851** | `2026-effnet-b0-sed-v5a/best-epochepoch=29.ckpt` |
| `effnet-b0-sed-v5b` | Stronger MixUp (α=1.0, θ=0.8, warmup=5) | 0.869 | 0.638 | 31 | 42 | +0.004 | **0.840** | `2026-effnet-b0-sed-v5b/best-epochepoch=31.ckpt` |
| `effnet-b0-sed-v5c` | weight_decay=1e-4 | 0.867 | 0.624 | 35 | 42 | +0.002 | pending | `2026-effnet-b0-sed-v5c/best-epochepoch=35.ckpt` |
| `effnet-b0-sed-v5d` | LSEHead (GeM+LSE pooling) | 0.861 | — | 21 | — | -0.004 | pending | `2026-effnet-b0-sed-v5d/best-epochepoch=21.ckpt` |
| `effnet-b0-sed-v5e` | HGNetV2-B0 backbone (embed=1024) | 0.861 | 0.612 | 23 | 34 | -0.004 | pending | `2026-hgnet-b0-sed-v5e/best-epochepoch=23.ckpt` |

Checkpoint base: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/`

### Step 2: ttahara Full Recipe Reproduction

Faithful reproduction of ttahara's 0.884 LB training pipeline (LSEHead with mean freq,
BCEWithLogitsLoss, per-sample MixUp α=1.0/θ=0.8, OneCycleLR, 5s segments, 20 epochs).
Two backbone variants to isolate recipe vs architecture impact.

Reference: `user/murilo/notes/ttahara-hgnetv2-b0-annotated.md`

| Slug | Backbone | Best val/auroc | val/cmap | Best Epoch | Stopped | LB | Checkpoint |
|------|----------|---------------|----------|------------|---------|-----|------------|
| `effnet-b0-sed-v5f` | EfficientNet-B0 (embed=320) | **0.866** | 0.599 | 14 | 20 | **0.855** | `2026-effnet-b0-sed-v5f/best-epochepoch=14.ckpt` |
| `hgnet-b0-sed-v5g` | HGNetV2-B0 (embed=1024) | **0.872** | 0.618 | 18 | 20 | **0.892** | `2026-hgnet-b0-sed-v5g/best-epochepoch=18.ckpt` |

Key differences from Step 1 ablations: LSEHead(freq_pool=mean), BCE loss (clipwise only),
per-sample MixUp(α=1.0, θ=0.8, warmup=5, 100%), OneCycleLR(lr=5e-4 uniform, wd=1e-4),
torchaudio mel(n_fft=2048, resize 256x256, min-max norm), 5s segments, batch=64, 20 epochs.

Expected: v5g single-fold ~0.870 (ttahara 4-fold = 0.884).

### v5g Detailed Metrics (Best Checkpoint, Epoch 18)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8720 |
| val/cmap | 0.6180 |
| val/f1_macro | 0.5191 |
| val/f1_micro | 0.6614 |
| val/top1_acc | 0.7393 |
| val/top5_acc | 0.8515 |
| val/loss | 0.0125 |

Training config: HGNetV2-B0 (`hgnetv2_b0.ssld_stage2_ft_in1k`), LSEHead(mean freq,
hidden=1024), BCEWithLogitsLoss, per-sample MixUp(α=1.0, θ=0.8, warmup=5),
OneCycleLR(lr=5e-4, wd=1e-4), 5s segments, 20 epochs, batch=64.
GPU: SLURM job 5953608. **Training time: ~21 min** (vs ~3-4h for v4b/v5a-e).

### v5f Detailed Metrics (Best Checkpoint, Epoch 14)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8657 |
| val/cmap | 0.5985 |
| val/f1_macro | 0.4936 |
| val/f1_micro | 0.6718 |
| val/top1_acc | 0.7379 |
| val/top5_acc | 0.8582 |
| val/loss | 0.0125 |

Training config: EfficientNet-B0, LSEHead(mean freq, hidden=320), BCEWithLogitsLoss,
per-sample MixUp(α=1.0, θ=0.8, warmup=5), OneCycleLR(lr=5e-4, wd=1e-4), 5s segments,
20 epochs, batch=64. GPU: SLURM job 5953606.

### Analysis

**v5g (0.872) matches our target.** The full ttahara recipe on HGNetV2-B0 achieves
0.872 val/auroc, within range of the ~0.870 single-fold expectation. Key comparisons:

| Model | Recipe | val/auroc | Delta vs v4b |
|-------|--------|----------|-------------|
| v4b (EfficientNet) | v4b recipe | 0.865 | baseline |
| v5e (HGNet) | v4b recipe | 0.861 | -0.004 (backbone alone hurt) |
| **v5f (EfficientNet)** | **ttahara recipe** | **0.866** | **+0.001** (recipe only) |
| **v5g (HGNet)** | **ttahara recipe** | **0.872** | **+0.007** (recipe + backbone) |

**Recipe impact** (same backbone): v5f (0.866) vs v4b (0.865) = +0.001 on EfficientNet.
The recipe alone barely moves the needle on EfficientNet — the gains are mostly from the backbone.

**Backbone impact** (same recipe): v5g (0.872) vs v5f (0.866) = +0.006 from HGNet.
HGNet is clearly the better backbone under the ttahara recipe.

**Recipe + backbone combined**: v5e (0.861, HGNet+old) → v5g (0.872, HGNet+ttahara) = +0.011.
The ttahara recipe unlocks HGNet's potential that the v4b recipe was suppressing.

**Training speed**: 21 min for 20 epochs (5s, batch=64, no grad accum) vs ~3-4h for v4b
recipe (50 epochs, 10s, 4× grad accum). The ttahara recipe is 8-10× faster to train.

**LB validation needed**: offline val/auroc has been a poor LB predictor in prior experiments.
v5g's 0.872 offline is similar to v5a's 0.875 (which scored only 0.851 LB). The question is
whether the ttahara recipe produces better-calibrated predictions that transfer to LB.

**Next**: If v5g LB ≥ 0.870 → proceed to 4-fold ensemble (target ~0.884).

### Step 3: 4-Fold Ensemble (v5g-4fold)

MultiLabelStratifiedGroupKFold (n_splits=4, seed=1086), grouped by audio_id (filename stem).
Per-segment soundscape entries for stratification. Trained with `--fold N` flag.

| Fold | val/auroc | val/cmap | Best Epoch | Checkpoint |
|------|----------|----------|------------|------------|
| 0 | **0.939** | 0.545 | 14 | `fold0/best-epochepoch=14.ckpt` |
| 1 | 0.842 | 0.566 | 19 | `fold1/best-epochepoch=19.ckpt` |
| 2 | 0.826 | 0.557 | 18 | `fold2/best-epochepoch=18.ckpt` |
| 3 | 0.802 | 0.556 | 16 | `fold3/best-epochepoch=16.ckpt` |

**Fold imbalance issue**: Fold 0 val/auroc (0.939) is 0.137 higher than fold 3 (0.802).
The root cause is soundscape file distribution: fold 0 has 55 of 66 soundscape files in
its validation set (and only 11 for training), while folds 2-3 have only 2 each in val.
With only 66 soundscape file-level groups, the GroupKFold constraint prevents even
distribution. The per-segment stratification fix (from per-file) barely changed this
(57→55 files in fold 0).

**Impact**: Fold 0 model trains on very few soundscape files, potentially weaker on
soundscape-like test data. The ensemble averaging may still help (each fold's weakness
is different), but the extreme spread suggests the folds are not comparable.

### LB Results

| Submission | LB | Notes |
|-----------|-----|-------|
| v5g single-fold | **0.892** | +0.032 over v4b (0.860), +0.037 over v5f (0.855) |
| **v5g-4fold ensemble** | **0.901** | **Broke 0.900 barrier!** +0.009 from 4-fold, +0.041 over v4b |

**v5g (0.892 LB)** is a massive improvement. The ttahara recipe + HGNet backbone
translates to LB far better than any prior SED model. Notably, offline val/auroc
(0.872) UNDER-predicted LB (0.892) — the opposite of all prior experiments where
val over-predicted LB. The ttahara recipe produces better-calibrated predictions.

**v5g-4fold (0.901 LB)** confirms the ensemble adds +0.009 despite the fold imbalance
(fold 0 has 55/66 soundscape files). This is within the expected +0.010-0.015 range
from ttahara's 4-fold approach.

Comparison to competition:
- ttahara HGNet 4-fold: 0.884 LB (our reproduction: **0.901**, +0.017 above reference)
- Our Perch ProtoSSM v4: 0.918 LB (still higher, but SED closing the gap)
- Salman single EfficientNet-B0 SED: 0.908 (we're at 0.892 single-fold, gap = 0.016)

### Step 4: v5g Recipe Ablations (v5h/v5i/v5j/v5k)

Single-variable ablations from v5g (0.872 val/auroc, 0.892 LB). Each changes ONE thing.
All use HGNetV2-B0 + ttahara recipe, 20 epochs, same data pipeline.

| Slug | Change from v5g | Best val/auroc | val/cmap† | Best Epoch | Delta vs v5g | Checkpoint |
|------|----------------|---------------|----------|------------|-------------|------------|
| `hgnet-b0-sed-v5g` | (baseline) | **0.872** | 0.618 | 18 | — | `2026-hgnet-b0-sed-v5g/best-epochepoch=18.ckpt` |
| `hgnet-b0-sed-v5h` | n_fft 2048→4096 | 0.869 | 0.615 | 16 | **−0.003** | `2026-hgnet-b0-sed-v5h/best-epochepoch=16.ckpt` |
| `hgnet-b0-sed-v5i` | window 5s→10s | **0.878** | **0.652** | 17 | **+0.006** | `2026-hgnet-b0-sed-v5i/best-epochepoch=17.ckpt` |
| `hgnet-b0-sed-v5j` | AmplitudeToDB norm‡ | 0.874 | 0.643 | 19 | +0.002 | `2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt` |
| `hgnet-b0-sed-v5k` | 3-channel mel (in_chans=3) | 0.868 | 0.611 | 16 | **−0.004** | `2026-hgnet-b0-sed-v5k/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v5m` | EffNet-B0 + proj 320→1024§ | **0.873** | **0.645** | 16 | +0.001 | `2026-effnet-b0-sed-v5m/best-epochepoch=16.ckpt` |
| `effnet-b0-sed-v5n` | EffNet-B0, no proj (v5j recipe)¶ | 0.871 | 0.629 | 19 | −0.001 | `2026-effnet-b0-sed-v5n/best-epochepoch=19.ckpt` |

† val/cmap values from end-of-training metrics (may differ slightly from best-epoch values).
‡ v5j is confounded: also changes win_length (626→2048), removes resize (256×256→256×511), and uses NNAudio instead of torchaudio.
§ v5m uses EfficientNet-B0 backbone (320 native) + 1x1 Conv2d projection to 1024, with v5j recipe. Tests embed_dim hypothesis.
¶ v5n uses EfficientNet-B0 with v5j recipe (NNAudio+AmplitudeToDB) but NO projection. Control for v5m to isolate projection vs mel pipeline.

Checkpoint base: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/`

### v5h Detailed Metrics (Best Checkpoint, Epoch 16)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8692 |
| val/cmap | 0.6145 |
| val/f1_macro | 0.5111 |
| val/f1_micro | 0.6636 |
| val/top1_acc | 0.7427 |
| val/top5_acc | 0.8618 |
| val/loss | 0.0124 |

Training config: HGNetV2-B0, ttahara recipe, n_fft=4096 (vs 2048), win_length=626,
resize 256×256. GPU: SLURM job 6077424.

### v5i Detailed Metrics (Best Checkpoint, Epoch 17)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8782 |
| val/cmap | 0.6523 |
| val/f1_macro | 0.5479 |
| val/f1_micro | 0.7124 |
| val/top1_acc | 0.8002 |
| val/top5_acc | 0.9025 |
| val/loss | 0.0109 |

Training config: HGNetV2-B0, ttahara recipe, window_duration=10s (vs 5s), resize
256×256 (more temporal compression). GPU: SLURM job 6077429.

### v5j Detailed Metrics (Best Checkpoint, Epoch 19)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8737 |
| val/cmap | 0.6429 |
| val/f1_macro | 0.5515 |
| val/f1_micro | 0.6882 |
| val/top1_acc | 0.7663 |
| val/top5_acc | 0.8764 |
| val/loss | 0.0118 |

Training config: HGNetV2-B0, ttahara recipe, NNAudio mel with AmplitudeToDB norm
(no resize, 256×511 output, win_length=2048). GPU: SLURM job 6077436.

### v5k Detailed Metrics (Best Checkpoint, Epoch 16)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8677 |
| val/cmap | 0.6108 |
| val/f1_macro | 0.5183 |
| val/f1_micro | 0.6659 |
| val/top1_acc | 0.7486 |
| val/top5_acc | 0.8596 |
| val/loss | 0.0124 |

Training config: HGNetV2-B0, ttahara recipe, in_chans=3 (mono mel repeated to 3
channels for full pretrained weight preservation). GPU: SLURM job 6077443.

### Step 4 Analysis

All four ablations land within ±0.006 val/auroc of v5g (0.872). Given that v5g
showed a +0.020 gap between val (0.872) and LB (0.892), these offline deltas are
smaller than the val→LB prediction error. We cannot confidently rank v5h, v5j,
or v5k relative to v5g on offline metrics alone — they are effectively tied.

**v5i (10s windows) stands out** (+0.006 val/auroc, +0.034 cmap). Unlike the others,
the gain is consistent across *every* metric: f1_micro 0.661→0.712, top1_acc
0.739→0.800, top5_acc 0.862→0.903. This breadth of improvement is harder to
dismiss as noise. The longer context likely helps capture temporal patterns for
continuous-stridulation species (insects, amphibians).

**v5j (AmplitudeToDB)** shows +0.002 val/auroc and +0.025 cmap, but is confounded
by three simultaneous changes (norm pipeline, win_length 626→2048, resize removal).
The cmap improvement suggests better rare-species calibration but the gain cannot
be attributed to any single factor.

**v5h (n_fft=4096)** and **v5k (3-channel mel)** score −0.003 and −0.004 val/auroc
respectively vs v5g. These are within the noise floor of our val set — they could
plausibly perform anywhere from slightly worse to slightly better than v5g on LB.
The only way to know is LB validation.

### Step 4 LB Results

| Variant | val/auroc | LB | val→LB gap | LB delta vs v5g |
|---------|----------|-----|-----------|----------------|
| v5g (baseline) | 0.872 | **0.892** | +0.020 | — |
| v5i (10s windows) | **0.878** | **0.880** | +0.002 | **−0.012** |

**v5i (0.880 LB) regressed −0.012 vs v5g (0.892) despite being the best offline
variant (+0.006 val).** The val→LB gap collapsed from +0.020 (v5g) to +0.002 (v5i),
suggesting the 10s window context that helps offline validation does not transfer
to LB — likely because the test soundscapes have different temporal structure than
training soundscapes, or the 10s→256×256 temporal compression loses detail that
matters on the test distribution.

**The val→LB disconnect makes offline metrics unreliable for ranking configs.**
The full val→LB mapping across all HGNet ablations so far:

| Variant | val/auroc | LB | val→LB gap | Inference time |
|---------|----------|-----|-----------|---------------|
| v5g | 0.872 | 0.892 | +0.020 | ~45 min (ttahara, 5s) |
| v5i | 0.878 | 0.880 | +0.002 | ~37s†† (10s) |
| **v5h** | 0.869 | **0.893** | +0.024 | ~33 min (ttahara, n_fft=4096) |
| **v5j** | 0.874 | **0.906** | **+0.032** | **~20 min** (NNAudio, no resize) |
| v5k | 0.868 | **0.886** | +0.018 | |
| v5l | 0.874 | pending | ? | (NNAudio, n_fft=4096, no resize) |
| v5m | 0.873 | **0.898** | +0.025 | **~15 min** (EffNet-B0 + proj, NNAudio) |
| v5n | 0.871 | **0.875** | +0.004 | (EffNet-B0, no proj, v5j recipe) |
| **v5j-4fold** | 0.951/0.843/0.823/0.808 | **0.913** | — | **~50 min** |

†† v5i ran on 2 validation files only (36.9s total), not comparable.

**v5l (v5j + n_fft=4096)** scores 0.874 val/auroc — identical to v5j (0.874).
The higher frequency resolution adds nothing offline when combined with
AmplitudeToDB norm and no resize. LB validation needed to confirm.

**The offline ranking is completely inverted on LB.** v5j was tied-second-worst
offline (0.874) but is the best on LB by a wide margin (0.906). v5i was the
best offline (0.878) but worst on LB (0.880). The val→LB gap varies from +0.002
to +0.032 — offline metrics cannot rank these configs.

**v5k (0.886 LB)** scored higher than v5g (0.892 LB) would suggest from offline
(0.868 val, −0.004 vs v5g), reinforcing that small offline differences are noise.
The 3-channel input didn't help but didn't catastrophically hurt either.

**v5j (0.906 LB) is the best single-fold SED model**, beating v5g-4fold
ensemble (0.901) with a single model. Key differences from v5g:
- NNAudio mel with AmplitudeToDB norm (vs ttahara min-max + resize)
- No resize to 256×256 (raw 256×511 output)
- win_length=n_fft=2048 (vs ttahara's win_length=626)

**v5j inference is ~2x faster** (~20 min vs ~45 min for v5g). The speed gain
comes from skipping the resize interpolation step and using the simpler NNAudio
mel pipeline. This is significant for kernel time budget — leaves ~70 min for
potential Perch+SED ensemble or 4-fold inference.

**v5h (0.893 LB)** confirms that n_fft=4096 provides a small +0.001 gain over
v5g (0.892) under the ttahara recipe, consistent with the direction from exp 017
(where n_fft=4096 was a bigger win). The 33 min inference is slower than v5g due
to the larger FFT.

### v5m Detailed Metrics (Best Checkpoint, Epoch 16)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8734 |
| val/cmap | 0.6453 |
| val/f1_macro | 0.5495 |
| val/f1_micro | 0.7033 |
| val/top1_acc | 0.7744 |
| val/top5_acc | 0.8784 |
| val/loss | 0.0117 |

Training config: EfficientNet-B0 + 1x1 Conv2d projection (320→1024), v5j recipe
(NNAudio mel, AmplitudeToDB, LSEHead(mean), BCE, MixUp α=1.0/θ=0.8, OneCycleLR).
GPU: SLURM job 6149183.

**v5m (0.873 val, 0.898 LB)** matches v5j offline but falls short on LB (−0.008).
The projection closes most of the gap: v5m is +0.043 over v5f (0.855 LB) and
only −0.008 behind v5j (0.906 LB). The val→LB gap (+0.025) is between v5g's
(+0.020) and v5j's (+0.032), suggesting the projection captures most but not
all of HGNet's LB advantage.

**Comparison of v5m to key baselines:**

| Model | Backbone | embed_dim | val/auroc | LB | val→LB gap |
|-------|----------|----------|----------|-----|-----------|
| v5f | EffNet-B0 | 320 (native) | 0.866 | 0.855 | −0.011 |
| **v5m** | **EffNet-B0 + proj** | **320→1024** | **0.873** | **0.898** | **+0.025** |
| v5j | HGNet-B0 | 1024 (native) | 0.874 | 0.906 | +0.032 |

The projection accounts for ~83% of the v5f→v5j LB gap (0.043 of 0.051).
The remaining 0.008 gap (v5m→v5j) likely comes from HGNet's architectural
advantages: regular convolutions, dense connections within HG blocks, and
SSLD-pretrained features that are qualitatively different from projected
EfficientNet features.

### v5n Detailed Metrics (Best Checkpoint, Epoch 19)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8708 |
| val/cmap | 0.6293 |
| val/f1_macro | 0.5369 |
| val/f1_micro | 0.6940 |
| val/top1_acc | 0.7621 |
| val/top5_acc | 0.8700 |
| val/loss | 0.0121 |

Training config: EfficientNet-B0 (no projection, native 320 embed_dim), v5j recipe
(NNAudio mel, AmplitudeToDB, LSEHead(mean, hidden=320), BCE, MixUp, OneCycleLR).
GPU: SLURM job for v5n. Control for v5m to isolate projection vs mel pipeline.

### 2×2 Ablation Grid: Mel Pipeline × Projection

|  | ttahara mel (resize, min-max) | v5j mel (NNAudio, AmpToDB, no resize) | Delta (mel) |
|--|-------------------------------|---------------------------------------|-------------|
| **EffNet 320 (no proj)** | v5f: 0.866 val / 0.855 LB | v5n: 0.871 val / 0.875 LB | +0.005 val / **+0.020 LB** |
| **EffNet 320→1024 (proj)** | — | v5m: 0.873 val / 0.898 LB | — |
| **Delta (proj)** | — | +0.002 val / **+0.023 LB** | — |
| **HGNet 1024 (native)** | v5g: 0.872 val / 0.892 LB | v5j: 0.874 val / 0.906 LB | +0.002 val / +0.014 LB |

**The projection is the single biggest LB factor despite being negligible offline.**
Decomposing the v5f→v5j LB gap (0.855→0.906 = +0.051):
- Mel pipeline (v5f→v5n): +0.020 LB (39% of gap)
- **Projection (v5n→v5m): +0.023 LB (45% of gap)**
- HGNet native architecture (v5m→v5j): +0.008 LB (16% of gap)

The projection's +0.023 LB from a single 1x1 Conv2d layer (320→1024) is
remarkable — it's a ~328K parameter addition that outweighs the mel pipeline
change on LB. Offline, this same change was only +0.002 val/auroc, confirming
that offline metrics cannot detect the capacity bottleneck that matters on the
harder test distribution.

**Conclusion: EfficientNet was doubly disadvantaged.** Two distinct problems
explain the gap:
1. **SED head capacity bottleneck** — 320 dims too few for 234 classes. The
   projection fixes this (+0.023 LB), but it's a linear rearrangement of the
   same 320 features, not new information.
2. **Weaker backbone features** — even at matched 1024 embed_dim, HGNet still
   leads by +0.008 LB. Its regular convolutions, dense connections, and SSLD
   pretraining produce genuinely richer representations for spectrograms.

HGNet wins because it solves both problems natively: high embed_dim (1024)
AND better feature quality. The projection recovers 84% of the gap by fixing
problem #1, but can't close the remaining 16% feature quality gap.

Full architecture comparison and recommendations: `docs/vendor/efficientnet-vs-hgnetv2.md`.

### Step 5: v5j 4-Fold Ensemble

MultiLabelStratifiedGroupKFold (n_splits=4, seed=1086), same fold assignments as
v5g-4fold. Trained with v5j config (NNAudio mel, AmplitudeToDB norm, no resize).

| Fold | val/auroc | val/cmap† | Best Epoch | Checkpoint |
|------|----------|----------|------------|------------|
| 0 | **0.951** | 0.570 | 15 | `fold0/best-epochepoch=15.ckpt` |
| 1 | 0.843 | 0.581 | 19 | `fold1/best-epochepoch=19.ckpt` |
| 2 | 0.823 | 0.564 | 19 | `fold2/best-epochepoch=19.ckpt` |
| 3 | 0.808 | 0.570 | 18 | `fold3/best-epochepoch=18.ckpt` |

† val/cmap from end-of-training (epoch 20), not best-epoch.

Checkpoint base: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/`

**Comparison to v5g 4-fold:**

| Fold | v5g val/auroc | v5j val/auroc | Delta |
|------|-------------|-------------|-------|
| 0 | 0.939 | **0.951** | +0.012 |
| 1 | 0.842 | 0.843 | +0.001 |
| 2 | 0.826 | 0.823 | −0.003 |
| 3 | 0.802 | **0.808** | +0.006 |

Same fold imbalance pattern (fold 0 ≫ folds 1-3). v5j is marginally better on
folds 0/1/3, marginally worse on fold 2.

Training: ~38 min/fold on V100 (SLURM jobs 6107175/6107185/6107187/6107188).

### Step 5 LB Results

| Submission | LB | Notes |
|-----------|-----|-------|
| v5k single-fold | **0.886** | 3-channel mel, +0.018 val→LB gap |
| **v5j-4fold ensemble** | **0.913** | **Best trained model!** +0.007 from 4-fold, +0.012 over v5g-4fold |

**v5j-4fold (0.913 LB) is our best trained model**, only 0.005 behind Perch
ProtoSSM v4 (0.918 LB) which relies on a frozen foundation model. The 4-fold
ensemble adds +0.007 over v5j single-fold (0.906), consistent with the +0.009
gain from v5g→v5g-4fold.

Comparison to competition:
- **v5j-4fold: 0.913 LB** (our best trained SED)
- Perch ProtoSSM v4: 0.918 LB (frozen foundation model, gap = 0.005)
- ttahara HGNet 4-fold: 0.884 LB (our reproduction beats by +0.029)
- Competition #1: 0.939 LB (gap = 0.026)

### v5b Detailed Metrics (Best Checkpoint, Epoch 31)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8685 |
| val/cmap | 0.6376 |
| val/f1_macro | 0.4196 |
| val/f1_micro | 0.4303 |
| val/top1_acc | 0.8041 |
| val/top5_acc | 0.9126 |
| val/loss | 1.3404 |

Training config: EfficientNet-B0, mixup_alpha=1.0, mixup_theta=0.8,
mixup_warmup_epochs=5, batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789084.

### v5d Detailed Metrics (Best Checkpoint, Epoch 21)

LSEHead (Log-Sum-Exp temporal pooling) slightly regressed vs attention head.
val/auroc 0.861 vs 0.865 baseline (-0.004).

### v5a Detailed Metrics (Best Checkpoint, Epoch 29)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8750 |
| val/cmap | 0.6196 |
| val/f1_macro | 0.4218 |
| val/f1_micro | 0.6196 |
| val/top1_acc | 0.7797 |
| val/top5_acc | 0.8955 |
| val/loss | 0.0133 |

Training config: EfficientNet-B0, loss_type=bce (BCEWithLogitsLoss),
batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789083.

### v5e Detailed Metrics (Best Checkpoint, Epoch 23)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8613 |
| val/cmap | 0.6124 |
| val/f1_macro | 0.5242 |
| val/f1_micro | 0.6307 |
| val/top1_acc | 0.7822 |
| val/top5_acc | 0.9086 |
| val/loss | 1.4121 |

Training config: HGNetV2-B0 (`hgnetv2_b0.ssld_stage2_ft_in1k`), embed_dim=1024,
batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789090.

### Analysis

**v5a (BCE)** is the strongest ablation (+0.010 val/auroc over v4b baseline).
BCEWithLogitsLoss is a clear improvement over soft CE for multi-label SED training.
The large f1_micro jump (0.430 → 0.620) suggests BCE produces better-calibrated
predictions. Needs LB validation — offline val/auroc has been a poor LB predictor
in prior experiments (v4e: 0.868 val → 0.840 LB, v4j: 0.868 val → 0.853 LB).

**v5b (MixUp)** is the second-best ablation (+0.004 val/auroc). Stronger MixUp
regularization (uniform α=1.0, label clipping θ=0.8, 5-epoch warmup) helps the model
generalize better.

**v5e (HGNetV2-B0)** regressed slightly (-0.004 val/auroc). Despite higher embed_dim
(1024 vs 320), HGNet doesn't outperform EfficientNet-B0 with default hyperparameters.
Interestingly, v5e has the best f1_macro (0.524 vs 0.420) — may benefit from different
tuning.

**v5d (LSEHead)** regressed slightly (-0.004 val/auroc). LSE pooling doesn't help on
EfficientNet-B0.

**v5c (weight_decay=1e-4)** had marginal improvement (+0.002 val/auroc over v4b). Lower
weight decay doesn't meaningfully change performance — the model is not significantly
over-regularized at the default 0.01.

### v5c Detailed Metrics (Best Checkpoint, Epoch 35)

| Metric | Value |
|--------|-------|
| val/auroc | 0.8669 |
| val/cmap | 0.6240 |
| val/f1_macro | 0.3987 |
| val/f1_micro | 0.3865 |
| val/top1_acc | 0.8027 |
| val/top5_acc | 0.9114 |
| val/loss | 1.3720 |

Training config: EfficientNet-B0, weight_decay=1e-4 (default 0.01),
batch_size=64, precision=16-mixed.
GPU: Quadro RTX 6000, SLURM job 5789087.
