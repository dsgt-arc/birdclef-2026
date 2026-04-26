# 2026-04-26 — MixUp implementation audit (Step 0 of diagnosis)

Code-only audit, no experiments. Verifies whether our v5j and prior wave-mixup attempts (v4j, v6b, v10b) implement multi-label mixup the same way as Sydorskyi 2025 (LB 0.917 single, 0.925 ensemble) and the cited Xu et al. 2018 paper.

**TL;DR**: confirmed three concrete divergences. The largest is **target-vector combination** — v5j and v6b interpolate (`lam·y1 + (1-lam)·y2`), Sydorskyi takes max (`max(y1, y2)`). For multi-label binary targets these are not equivalent, and Sydorskyi's choice is structurally aligned with the soundscape test distribution (multiple birds at full probability per chunk) while ours is not. v10b had the right target semantics but the wrong waveform scaling. **None of our four mixup variants reproduces Sydorskyi's exact recipe.**

This does not prove that fixing it would move LB. It only refutes the conclusion "wave-mixup is dead" — that conclusion was based on three regressions, all of which used the wrong target combination relative to the recipe published by every top-3 finisher in 2025.

---

## What Xu 2018 actually does (the cited source)

Found in `/tmp/xu_2018.txt` (extracted from arXiv:1805.07319), Section IV equations (1)–(2):

```
x = α · x_i + (1 − α) · x_j        (1)
y = α · y_i + (1 − α) · y_j        (2)
```

α ∈ [0, 1]. Best α reported: **0.5** (Table II, both VGGNet and Xception).

**Crucially**: Xu applies this to the **log-mel spectrogram**, not the raw waveform. From the paper's conclusion (lines 270-273): *"Presently, the mixup processing is relied on the log mel spectrum of the audio signal. We did not observe significant improvement by mixup of the raw audio signal directly."*

Targets are interpolated (Zhang 2017 style). Xu does not use `max`.

So Sydorskyi's paper text "Two audio waveforms were added in the audio domain, and the resulting one-hot target vector was computed as the element-wise maximum across the original vectors" is **not faithful to its [35] cite**. The cite is for the *concept* of "mixup applied to audio classification," not the formula. Sydorskyi's actual recipe diverges from Xu in two dimensions: (a) waveform domain not mel, (b) max targets not interpolation.

---

## What Sydorskyi actually does (the reference implementation)

Found in `/tmp/sydorskyi_2025/code_base/datasets/wave_dataset.py:WaveDataset.__getitem__` lines 762-820. The selected configs `train_configs/selected_ebs.py` and `selected_eca.py` use `mixup_params={"prob": 0.5, "alpha": None}`.

With `alpha=None`, the code path (lines 788-803) executes:

```python
wave = (mixup_wave + wave) / 2                                     # equal-weight average
target = mixup_target + target                                     # default target_aggregation="sum"
target = torch.clamp(target, min=0, max=1.0)                        # clip to [0, 1]
if self.late_normalize:
    wave = librosa.util.normalize(wave)                            # peak-normalize to [-1, 1]
```

For binary multi-hot targets, `clip(y1 + y2, 0, 1)` ≡ `max(y1, y2)`, matching the paper text.

Summary:
- **Domain**: waveform (CPU, in dataloader, before mel computation)
- **Lambda**: deterministic 0.5
- **Wave**: `(w1 + w2) / 2`, then peak-renormalize
- **Targets**: `min(1, y1 + y2)` ≡ `max(y1, y2)`
- **No theta cap, no warmup**
- **Probability**: 0.5 per sample

---

## What our v5j actually does

Lightning module: `user/murilo/src/birdclef/lightning/sed_finetune.py`. Config: `user/murilo/experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml`.

v5j config has `mixup_prob: 1.0`, `mixup_alpha: 1.0`, `mixup_per_sample: true`, `mixup_warmup_epochs: 5`, `mixup_theta: 0.8`. **No** `raw_waveform_mixup` key (defaults to `false`) and **no** `mixup_mode` key (defaults to `"interpolate"`).

Call site `sed_finetune.py:589-594` confirms domain — when `waveform_mixup=False`, mel is computed first, then `_apply_mixup` runs on the mel:

```python
elif self.mel_transform is not None:
    mel = self.mel_transform(x.float())               # GPU mel computation
    if do_mixup:
        mel, labels, loss_weight = self._apply_mixup(mel, labels, loss_weight)
```

The `_apply_mixup` per-sample interpolate path (`sed_finetune.py:327-341`):

```python
beta_dist = torch.distributions.Beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
lam = beta_dist.sample(sample_shape=(x.size(0),)).to(x.device)
lam = torch.maximum(lam, 1 - lam).float()                     # enforce lam >= 0.5

x = x_lam * x + (1 - x_lam) * x[indices]                      # spec interpolation
labels = label_lam * labels + (1 - label_lam) * labels[indices]   # target interpolation
```

Then the theta cap (`sed_finetune.py:350-352`):

```python
theta = self.hparams.mixup_theta                              # 0.8 in v5j
if theta < 1.0:
    labels = torch.where(labels >= theta, torch.ones_like(labels), labels)
```

So v5j's recipe:
- **Domain**: mel-spectrogram (NOT waveform)
- **Lambda**: Beta(1, 1) per-sample, max-flipped to [0.5, 1] (uniform on [0.5, 1])
- **Wave/spec**: `lam·m1 + (1-lam)·m2` — Zhang interpolation
- **Targets**: `lam·y1 + (1-lam)·y2` — Zhang interpolation
- **Theta cap**: labels ≥ 0.8 snapped to 1.0 (partial mitigation; covers ~40% of mixed samples where lam ≥ 0.8)
- **Warmup**: skipped during first 5 epochs
- **Probability**: 1.0 (whenever active)

---

## What our prior wave-mixup attempts did

### v4j (exp 017, LB 0.853, regressed −0.006 vs v4b 0.859)

Config `efficientnet_b0_sed_v4j.yaml:63-65`:
```yaml
mixup_prob: 0.5
mixup_alpha: 0.5
raw_waveform_mixup: true
```

Translation in `experiments/017-mel-overhaul/train.py:172`:
```python
waveform_mixup = training_cfg.get("raw_waveform_mixup", False)
```

Confirmed: `raw_waveform_mixup` is read and passed through as `waveform_mixup` to the module — **not a silent no-op**.

So v4j actually ran:
- **Domain**: waveform (mel computed AFTER mixup) — matches Sydorskyi ✓
- **Lambda**: Beta(0.5, 0.5), per-batch (not per-sample because `mixup_per_sample` not set), max-flipped — bimodal toward 0/1
- **Wave**: `lam·w1 + (1-lam)·w2` — Zhang interpolation, NOT Sydorskyi's `(w1+w2)/2`
- **Targets**: `lam·y1 + (1-lam)·y2` — interpolation, NOT max ✗
- **No theta cap** (theta defaults to 1.0 since not set)
- **Probability**: 0.5 — matches Sydorskyi ✓

### v6b (exp 021, LB 0.887, regressed −0.019 vs v5j 0.906)

Config `hgnet_b0_sed_v6b.yaml:62-68`. Same as v5j *plus* `raw_waveform_mixup: true`. So:
- **Domain**: waveform ✓
- **Lambda**: Beta(1, 1), per-sample, max-flipped to [0.5, 1]
- **Wave**: Zhang interpolation
- **Targets**: Zhang interpolation ✗
- **Theta cap**: 0.8 (partial mitigation)
- **Probability**: 1.0 (NOT 0.5 like Sydorskyi)

### v10b (exp 027, LB 0.875, regressed −0.031 vs v5j 0.906)

Config `v10b_addmix_bce.yaml:72-78`:
```yaml
raw_waveform_mixup: true
mixup_mode: "additive"
```

Triggers the additive branch in `sed_finetune.py:322-326`:
```python
x = x + x[indices]                                              # no /2, no normalize
labels = torch.maximum(labels, labels[indices])                 # max — matches Sydorskyi ✓
```

So v10b:
- **Domain**: waveform ✓
- **Wave combination**: `x1 + x2` — pure additive, **NO `/2`, NO peak-normalization** ✗
- **Targets**: `max(y1, y2)` — matches Sydorskyi ✓
- **Probability**: 1.0 (NOT 0.5)

v10b is the closest of our four to Sydorskyi on target semantics, but the waveform path is wrong: doubled peak amplitude, no renormalization. Inputs to mel after additive mixing are out-of-distribution (peaks beyond [-1, 1]).

---

## Side-by-side

| | Xu 2018 (cited) | **Sydorskyi 2025** | v5j (0.906) | v4j (0.853) | v6b (0.887) | v10b (0.875) |
|---|---|---|---|---|---|---|
| Domain | mel | **waveform** | mel | waveform ✓ | waveform ✓ | waveform ✓ |
| Wave/spec combination | `α·x1 + (1-α)·x2` | **`(w1+w2)/2` + peak-renormalize** | `lam·m1 + (1-lam)·m2` | `lam·w1 + (1-lam)·w2` | `lam·w1 + (1-lam)·w2` | `w1+w2` (no /2, no norm) ✗ |
| Target combination | `α·y1 + (1-α)·y2` (interpolation) | **`min(1, y1+y2)` ≡ `max(y1, y2)`** | interpolation ✗ | interpolation ✗ | interpolation ✗ | **`max` ✓** |
| Lambda distribution | α ∈ [0, 1], best 0.5 | **deterministic 0.5** | Beta(1,1) on [0.5, 1] | Beta(0.5,0.5) on [0.5, 1] | Beta(1,1) on [0.5, 1] | n/a (full additive) |
| Per-sample or per-batch | per-batch | per-sample (one mix per draw) | per-sample | per-batch | per-sample | per-sample (just additive) |
| Theta cap | none | none | **0.8** (partial mitigation) | none | 0.8 | 0.8 |
| Warmup | none | none | 5 epochs | none | 5 | 5 |
| Probability | implicit 1.0 (paper trains "with mixup") | **0.5** | 1.0 | 0.5 ✓ | 1.0 | 1.0 |

---

## What this confirms / refutes

**Claim 1 — Xu 2018 differs from Sydorskyi.** ✅ Confirmed. Xu uses interpolated targets on log-mel; Sydorskyi uses max targets on waveforms. The Sydorskyi cite to [35] is for the concept of "audio mixup," not the formula.

**Claim 2 — v5j uses interpolated multi-label targets.** ✅ Confirmed. `sed_finetune.py:339`: `labels = label_lam * labels + (1 - label_lam) * labels[indices]`. Theta cap (line 350-352) partially compensates but only for samples with lam ≥ 0.8.

**Claim 3 — Our three prior wave-mixup attempts (v4j, v6b, v10b) did NOT replicate Sydorskyi's recipe.**
- v4j and v6b: wave-domain ✓ but interpolated-target ✗
- v10b: wave-domain ✓ and max-target ✓, but `x1+x2` without `/2` or peak-normalize → input distribution shift
- None used Sydorskyi's complete `(w1+w2)/2` + peak-normalize + max-target combination

The conclusion "wave-mixup is dead" (CONTEXT.md note: "Additive MixUp is dead (3 tests)") is over-reaching. **Three different wave-mixup recipes failed; the Sydorskyi-faithful recipe was never tested.**

---

## What this does and does not prove

**Proves**:
- Concrete implementation gap exists between v5j (and our prior wave-mixup attempts) and Sydorskyi.
- The "wave-mixup is dead" prior was based on three different recipes, none of which match what the top-3 BirdCLEF 2025 finishers all use.
- The dimension that differs across all four of our recipes vs Sydorskyi is **target combination** (Zhang interpolation vs `max`) — and that dimension has a clear test-distribution-alignment argument.

**Does not prove**:
- That a Sydorskyi-faithful wave-mixup would actually move our LB. We have not tested it. The information lever (H1) and modeling lever (H2) hypotheses still need their own diagnostics.
- That fixing this is the highest-EV next experiment. It's one of several candidates, now better-characterized.
- That the failures of v4j / v6b / v10b are *purely* explained by the target-combination issue. Other variables differed too (probability, warmup, scaling).

## What this changes for the diagnosis plan

1. **Update to D1** (train-side per-frame analysis on v5j + v6k): worth flagging in the analysis that v5j's target interpolation produces sub-1 multi-hot values during training. If v5j attention is "honest" (diffuse on hard crops), one component of "honest" might be that its training targets are themselves uncertain — model isn't being told "this bird is fully present" most of the time. Worth characterizing.

2. **Sydorskyi-faithful wave-mixup is now a candidate experiment** — but it should sit *behind* D1 + D2 in priority, because we still need to know whether the H1 vs H2 framing is correct before picking specific interventions.

3. **The "22 training-recipe failures" prior is slightly weaker now**. At least 3 of them (v4j, v6b, v10b) were tests of recipes that don't exist anywhere in the published top-3 solutions. They tell us those specific recipes don't work. They don't say "all wave-mixup variants are dead."

## Files inspected (with line numbers)

| File | Lines | What it confirms |
|---|---|---|
| `/tmp/xu_2018.txt` | 168-178 | Xu 2018 equations (1)-(2): interpolated wave + interpolated target |
| `/tmp/xu_2018.txt` | 270-273 | Xu found mel-domain best, "did not observe significant improvement by mixup of raw audio" |
| `/tmp/sydorskyi_2025/code_base/datasets/wave_dataset.py` | 788-803 | alpha=None branch: `(w1+w2)/2`, target sum-clip, peak-renormalize |
| `/tmp/sydorskyi_2025/train_configs/selected_ebs.py` | 41-42 | `mixup_params={"prob": 0.5, "alpha": None}` |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | 297-353 | `_apply_mixup` — both interpolate and additive branches |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | 322-326 | additive branch: `x1+x2` (no /2), `max(y1, y2)` |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | 327-341 | per-sample interpolate branch: full Zhang interpolation |
| `user/murilo/src/birdclef/lightning/sed_finetune.py` | 583-594 | call site — domain controlled by `waveform_mixup` flag |
| `user/murilo/experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml` | 72-76 | v5j: spec-domain, alpha=1.0, theta=0.8, prob=1.0, no waveform_mixup |
| `user/murilo/experiments/017-mel-overhaul/config/efficientnet_b0_sed_v4j.yaml` | 63-65 | v4j: wave-domain, alpha=0.5, prob=0.5, per-batch |
| `user/murilo/experiments/021-sed-training/config/hgnet_b0_sed_v6b.yaml` | 62-68 | v6b: wave-domain, alpha=1.0 per-sample, theta=0.8, prob=1.0 |
| `user/murilo/experiments/027-model-training-fixes/configs/v10b_addmix_bce.yaml` | 72-78 | v10b: wave-domain, additive (no /2 or normalize), max targets |
| `user/murilo/experiments/017-mel-overhaul/train.py` | 172, 218 | Confirms `raw_waveform_mixup` → `waveform_mixup` remap is wired correctly (no silent no-op) |
