---
experiment: "034-pretrained-backbone-yoink"
phase: A (source reading + verified config)
status: COMPLETE (Phase A.1 + A.2; ready for Phase B implementation)
date_started: 2026-05-28
date_completed: 2026-05-28
author: Murilo Gustineli
---

# Sydorskyi 2025 — Verified Config Reference

Every parameter that goes into our Phase B implementation, line-cited against the Sydorskyi 2025 source code @ `vendor/sydorskyi-2025/` commit `053594f`. The paper leaves several parameters ambiguous; the source code is canonical.

**Source repo**: `github.com/VSydorskyy/BirdCLEF_2025_2nd_place` @ `053594f` ("Update paper reference in README.md"). Cloned as git submodule at `vendor/sydorskyi-2025/`.

**Pretrained ckpts**: `vladimirsydor/bird-clef-2025-all-pretrained-models` at `/storage/project/ps-clef2026_birdclef-0/shared/pretrained-models/sydorskyi-2025/`.

**Confidence levels**:
- ✅ **VERIFIED** — confirmed against source, with line-number citation
- 🔬 **CKPT-VERIFIED** — confirmed by direct ckpt inspection (Phase A.1.4)
- ⚠️ **CORRECTED** — previous belief (proposal / deep research / CONTEXT.md) was wrong

---

## Architecture stack (top-down, traced from selected_ebs.py / selected_eca.py)

### Backbone

| Component | EffNetV2-S branch | NFNet-L0 branch |
|---|---|---|
| timm name | `tf_efficientnetv2_s_in21k` | `eca_nfnet_l0` |
| Instantiation | `timm.create_model(..., features_only=True, pretrained=False, exportable=True, in_chans=1)` | same |
| Loadable output channels | **256** ⚠️ (NOT 1280 — `conv_head`+`bn2` stripped by features_only) | **2304** ✅ (NOT 1024) |
| Pretrained ckpt loaded | `models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth` ⚠️ (NOT the 2025 SWA) | `models_2025/eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt` |
| Source | `wave_clasifier.py:115-123` | same |

✅ **VERIFIED — features_only mode** strips `conv_head`+`bn2` from the timm model automatically. Combined with the pre-stripped ckpts (`create_pretrain_backbone.py` outputs flat OrderedDict), `model.load_state_dict(ckpt, strict=True)` will work cleanly.

⚠️ **CORRECTED — EffNetV2-S uses the 2024 ckpt**, not the 2025 SWA. Both `selected_ebs.py:217` and `best_solo_ebs.py:219` reference `models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth`. The 2025 SWA ckpt (`models_2025/tf_efficientnetv2_s_in21k_pretrain_from_bigXCV2Ext_swa.ckpt`) is NOT referenced by any final config — it's an intermediate experiment artifact.

### Mel pipeline (model-level)

| Component | Value | Source |
|---|---|---|
| Wrapper | `TraceableMelspec` (when `exportable=True`; our case) | `wave_clasifier.py:187-188`, `blocks.py:448-524` |
| STFT backend | **nnAudio.Spectrogram.STFT** (`window="hann"`, `freq_scale="no"`, `output_format="Complex"`) | `blocks.py:487-502` |
| MelScale backend | **torchaudio.transforms.MelScale** | `blocks.py:510` |
| `sample_rate` | 32000 | config |
| `n_mels` | 128 | config |
| `n_fft` | 2048 | config |
| `hop_length` | 512 | config |
| `f_min` | 20 | config |
| `f_max` | **None** (defaults to sr/2 = 16000 in MelScale) | config (omitted) |
| `power` | 2.0 (default in TraceableMelspec; not overridden) | `blocks.py:453` |
| `normalized` | **True** ✅ | config + `blocks.py:514-515` (divides by `||hann_window||₂`) |
| `center` | True (default) | `blocks.py:455` |
| `pad_mode` | "reflect" (default) | `blocks.py:456` |

⚠️ **Important nuance**: `normalized=True` here means after STFT, divide by `||hann_window||_2 = sqrt(sum(hann²))`. This is NOT the same as torchaudio.MelSpectrogram's `normalized=True` (which normalizes by window energy *before* power); the math differs slightly. For Phase B parity, use TraceableMelspec verbatim (or torchaudio with caveat).

⚠️ **Hybrid library**: nnAudio is needed (for STFT) but NOT directly for the mel filterbank. The deep research's "use nnAudio.Spectrogram.MelSpectrogram" framing was imprecise — Sydorskyi uses **nnAudio STFT + torchaudio MelScale**.

### Mel post-norm (model-level, applied after backbone instantiates the feature extractor)

Chain (from `wave_clasifier.py:_create_feature_extractor` lines 218-224):
```
TraceableMelspec → ChannelAgnosticAmplitudeToDB(top_db=80, amin=1e-10) → NormalizeMelSpec(exportable=True)
```

`NormalizeMelSpec` (`blocks.py:23-54`):
```python
def forward(self, X):
    mean = X.mean((1, 2), keepdim=True)         # per-sample, over (n_mels, time)
    std  = X.std((1, 2), keepdim=True)          # per-sample, over (n_mels, time)
    Xstd = (X - mean) / (std + 1e-6)            # standardize
    # exportable=True branch:
    norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
    norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
    return (Xstd - norm_min) / (norm_max - norm_min + 1e-6)   # min-max to [0, 1]
```

✅ **VERIFIED — standardize axes = (1, 2)** = (n_mels, time), **per-sample, over the full spectrogram** (open question from proposal resolved). Min-max[0,1] uses the same axes.

### SpecAugment (model-level, inside `forward` when training)

Applied at `wave_clasifier.py:261-262`, only in `training=True` mode. Implementation in `code_base/augmentations/spec_augment.py`.

| Component | Value | Source |
|---|---|---|
| FrequencyMasking | wraps `torchaudio.transforms.FrequencyMasking(freq_mask_param=10)` | `spec_augment.py:CustomFreqMasking` |
| TimeMasking | wraps `torchaudio.transforms.TimeMasking(time_mask_param=20)` | `spec_augment.py:CustomTimeMasking` |
| Per-direction `p` | 0.3 | config |
| `mask_max_masks` | 3 (per direction) | config |
| `inplace` | True | config |

Per-sample logic: with prob 0.3, sample N = randint(1, mask_max_masks+1) and apply N independent torchaudio masks.

### Head — AttHead (`blocks.py:202-365`)

Constructor signature (`blocks.py:202-215`):
```python
AttHead(
    in_chans,                       # backbone output channels: 256 (EffNetV2-S) or 2304 (NFNet)
    hidden_chans=512,               # ✅ HIDDEN DIM CONFIRMED 512
    p=0.5,                          # ✅ dropout master (head_config sets this)
    num_class=397,                  # we'll override to 234 for BirdCLEF 2026
    train_period=15.0,              # config overrides to 5.0
    infer_period=5.0,
    infer_framewise_max_coef=0.5,
    output_type=None,
    exportable=False,
    rnn_config=None,                # No RNN by default
)
```

Forward chain (`blocks.py:268-320` — training-mode path with `train_period == infer_period == 5.0`):
1. **GeMFreq pool** (`p=3.0` learnable, `eps=1e-6`) — over freq axis. `blocks.py:228, 64-69, 71-79`.
   - `gem_freq(x, p=3) = x.clamp(min=eps).pow(p).mean(dim=2).pow(1/p)` — averages over freq axis.
2. **Squeeze + permute** to (B, W_time, C) — `blocks.py:275, 280`.
3. **Dense block** (`blocks.py:245-250`):
   ```python
   nn.Sequential(
       nn.Dropout(p / 2),              # 0.5 / 2 = 0.25 ✅
       nn.Linear(in_chans, 512),       # 256 → 512 (EffNetV2-S) or 2304 → 512 (NFNet)
       nn.ReLU(),                      # ✅ ReLU
       nn.Dropout(p),                  # 0.5 ✅
   )
   ```
4. **Two parallel Conv1d heads** (kernel=1, stride=1, bias=True):
   - `attention` (512 → num_class) → `tanh(.)` → time attention weights
   - `fix_scale` (512 → num_class) → value branch (no activation)
5. **Soft-attention-weighted aggregation** (`blocks.py:301-304`):
   ```python
   clipwise_logits_long = sum_t( feat_v[t] * softmax_t(time_att) )
   ```

✅ **VERIFIED — aggregation in LOGIT space** (matches `feedback_atthead_aggregation_domain` memory's "correct" direction).

✅ **VERIFIED — dropout positions**: 0.25 before linear, 0.5 after ReLU. From master `p=0.5` arg.

Training-mode output dict:
- `clipwise_logits_long` (B, num_class) — **input to loss** (loss_type="baseline")
- `clipwise_pred_long` (B, num_class) — probability-space aggregation (for metric tracking)
- `framewise_logits_long` (B, time, num_class) — pre-aggregation logits

### Loss — `FocalLossBCE` (`focal_loss.py:28-54`)

Selected via `loss_type="baseline"` + `use_bce_focal_loss=True` in the forward init (`cls_forward.py:100-106, 221-223`).

```python
loss = bce_weight * BCEWithLogitsLoss(input, target)
     + focal_weight * torchvision.ops.focal_loss.sigmoid_focal_loss(input, target, α, γ, reduction)
```

| Param | Value | Source |
|---|---|---|
| `α` | **0.25** ✅ | `focal_loss.py:31` default; no override |
| `γ` | **2.0** ✅ | `focal_loss.py:32` default; no override |
| `bce_weight` | **1.0** ✅ | `focal_loss.py:34` default |
| `focal_weight` | **1.0** ✅ | `focal_loss.py:35` default |
| `reduction` | **"mean"** (since `use_weights=False` in config) | `cls_forward.py:102-106` |
| Input | `clipwise_logits_long` (raw logits) | `cls_forward.py:223` |

Implementation calls `torchvision.ops.focal_loss.sigmoid_focal_loss` directly. NO additional framewise-max term (`loss_type="baseline"` only uses clipwise; the `_and_max_frame` variants are separate loss_types not used in selected configs).

### Forward wrapper — `MultilabelClsForwardLongShort` (`cls_forward.py:11-273`)

Pre-loss waveform pipeline (training only, `cls_forward.py:174-218`):
1. Unpack `(wave, labels) = batch`
2. **batch_aug = `RandomFiltering(min_db=-20, is_wave=True, normalize_wave=True)`** applied to waveform (`cls_forward.py:199`)
3. MixUp here is **NO-OP** in our configs (forward-level `mixup_alpha=None`; mixup happens in the dataset)
4. `output = runner.model(wave)` → AttHead dict
5. `loss = FocalLossBCE(output["clipwise_logits_long"], labels)`

### RandomFiltering (`code_base/models/augmentations.py:RandomFiltering`)

4-band random-EQ filter applied to **waveform** (via STFT → freq-domain gain → iSTFT).

| Param | Value | Source |
|---|---|---|
| `n_bands` | 4 (default) | `augmentations.py:7` |
| `min_db` | **−20** | config (`selected_ebs.py:182`) |
| `is_wave` | True | config |
| `normalize_wave` | True (peak-normalize after) | config (`LATE_NORMALIZE=True`) |
| `n_fft` | 1024 (default; ⚠️ different from mel n_fft=2048) | `augmentations.py:12` |
| `hop_length` | 512 (default) | `augmentations.py:13` |

Algorithm:
1. STFT (n_fft=1024, hop=512, hann window)
2. Sample 4 random dB values from `Uniform[-20, 0]` (per-sample, per-band)
3. Linear interpolate to n_fft/2+1 frequency bins
4. Convert dB → amp: `gain = 10^(dB / 20)`
5. Multiply STFT bins by gains
6. iSTFT
7. Peak-normalize (if `normalize_wave=True`)

---

## Dataset pipeline (`code_base/datasets/wave_dataset.py:WaveDataset`)

### Augmentation order inside `__getitem__`

Per `wave_dataset.py:756-905`:
1. `_prepare_sample_target_from_idx(index)` → original sample (possibly **PL-swapped**, see below)
2. (if `do_mixup` AND `p=0.5` Bernoulli fires) — **MixUp** with random partner from `_get_mixup_idx`
3. (if `do_montage` — NOT in our configs) — Montage with N extra samples
4. **`late_aug(wave)`** = `OneOf([nocall-SS BG noise, ESC-50 subset BG noise])`
5. **`librosa.util.normalize`** (peak normalize) — at multiple points in chain
6. (if `unlabeled_files` set — NOT in our final configs) — Unlabeled wave handling
7. Return `(wave, target)`

Then forward applies `RandomFiltering` to waveform (waveform-level EQ) before mel extraction.

### MixUp (`wave_dataset.py:756-819`) — the "sum-clip" trick

With `alpha=None` (our config), this is the **"PL × focal fusion" mechanism** flagged in the exp 033 audit as Sydorskyi's "cleverest piece":

```python
if self.do_mixup and np.random.binomial(1, mixup_params["prob"]):  # prob=0.5
    mixup_idx = self._get_mixup_idx()       # random partner
    mixup_wave, mixup_target = self._prepare_sample_target_from_idx(mixup_idx)
    # alpha=None path (target_aggregation="sum" default):
    wave   = (mixup_wave + wave) / 2        # EQUAL-WEIGHT 50/50 wave blend
    target = mixup_target + target           # SUM targets
    target = torch.clamp(target, min=0, max=1.0)  # CLIP to [0, 1]
    if late_normalize: wave = librosa.util.normalize(wave)
```

✅ **VERIFIED — sum-clip target rule with equal-weight wave blend.** Matches audit description.

**Coupling with PL**: each call to `_prepare_sample_target_from_idx` independently rolls a 40% PL-swap probability. Per mixup'd batch slot:
- P(focal × focal) = 0.6 × 0.6 = 36%
- P(focal × PL or PL × focal) = 2 × 0.6 × 0.4 = 48%
- P(PL × PL) = 0.4 × 0.4 = 16%
→ **64% of mixup'd samples have ≥1 PL component** (much higher PL exposure than the audit estimated).

### PL swap mechanism (`wave_dataset.py:603-609`)

```python
if soundscape_df is not None and np.random.binomial(1, sampling_prob):  # sampling_prob = 0.4
    initial_class = self.df[target_col].iloc[idx]
    if initial_class in pseudo_birds:
        idx = np.random.choice(np.where(soundscape_df[target_col] == initial_class)[0])
        is_pseudo_sample = True
```

✅ **VERIFIED — 40% Bernoulli swap, primary-class-matched** (audit description correct).

### PL filter (applied once at dataset init)

`wave_dataset.py:368-394`:
```python
ss_df = ss_df[ss_df["primary_label_prob"] > 0.5]   # chunk discard
...
pseudo_probs[pseudo_probs < 0.1] = 0                # per-class trim
```

✅ Confirms `primary_label_min_prob = 0.5` (chunk discard) + `trim_min_prob = 0.1` (cell-level zero).
- Note: `primary_label_prob` is the probability of the chunk's argmax class (effectively max(probs)).
- No `use_oof` flag in selected configs — PLs are NOT fold-disjoint, just sample-deduped across PL CSV sources.

### Label smoothing (`wave_dataset.py:585-602`, only on GT samples)

```python
if label_smoothing is not None:    # α = 0.05
    all_tgt = all_tgt * (1 - α) + α * all_tgt.sum() / all_tgt.shape[-1]
```

Mass-preserving: for K positives in N classes:
- Positive class: `0.95 + 0.05 * K / N`
- Negative class: `0.05 * K / N`

✅ **VERIFIED — label smoothing α=0.05 mass-preserving** (matches deep research description).

**Only applied to GT samples** (called inside `_prepare_target`); PL samples get raw soft targets from parquet (line 702: `target = row[indices_for_soundscapes].values`).

### Background noise (`code_base/augmentations/transforms.py:OneOf` + `BackgroundNoise`)

Configured as:
```python
OneOf([
    BackgroundNoise(p=0.5, esc50_root=nocall_ss, ...),       # 50% pick this branch
    BackgroundNoise(p=0.5, esc50_root=esc50_subset, ...),    # 50% pick this branch
])
```

- `OneOf` picks one transform uniformly at random; ALWAYS calls it.
- Each `BackgroundNoise` has its own internal `p=0.5` apply-probability.
- Net per sample: 25% nocall-SS noise + 25% ESC-50 noise + 50% no BG noise.

ESC-50 subset cats: dog, rain, insects, hen, engine, hand_saw, pig, rooster, sea_waves, cat, crackling_fire, thunderstorm, chainsaw, train, sheep, wind, footsteps, frog, cow, crickets (20 categories).

---

## Training hyperparameters

| Hyperparam | EffNetV2-S branch | NFNet-L0 branch |
|---|---|---|
| Batch size | 64 | 64 |
| `train_period` / `infer_period` | 5.0 / 5.0 | 5.0 / 5.0 |
| Epochs | 50 | 50 |
| Precision | `"32-true"` (NO AMP) ✅ | same |
| Seed | 1243 | 1243 |
| Optimizer | `AdamW(lr=1e-4, eps=1e-8, betas=(0.9, 0.999))` | `RAdam(lr=1e-3)` |
| LR schedule | `CosineAnnealingWarmRestarts(T_0=50*len_train, T_mult=1, eta_min=1e-6)` (step-level) | same |
| Warmup | None | None |
| Folds | 5 | 5 |
| Strategy | `ddp_find_unused_parameters_true` | same |
| Class-balance | `class_weights_path="balanced"` → `w_c = 1/n_c` (equal balancing, γ=−1) | `class_weights_path="sqrt"` → `w_c = (n_c/N)^-0.5` (sqrt balancing, γ=−0.5) ⚠️ |
| Sampler | `WeightedRandomSampler(sample_weights, replacement=False)` | same |
| Checkpoint selection | `last` (`save_last=True` + `n_checkpoints_to_save=3`) | same |

⚠️ **CORRECTED — class-balance γ mapping**: NFNet uses **sqrt (γ=-0.5)**, EffNetV2-S uses **balanced (γ=-1)**. The proposal table had this backwards. Verified at `train_lightning.py:197 (power_value=-0.5 default)`, `train_lightning.py:240-247` (formula).

### MixUp params

```python
mixup_params = {"prob": 0.5, "alpha": None}    # alpha=None → equal-weight 50/50 + sum-clip target
```

### PL config

```python
soundscape_pseudo_config = {
    "primary_label_min_prob": 0.5,    # chunk discard
    "trim_min_prob": 0.1,             # per-class trim
    "sampling_prob": 0.4,             # 40% Bernoulli swap per slot
}
label_smoothing = 0.05                # mass-preserving, GT-only
```

---

## Pretrained ckpt loading

### Ckpt format (`scripts/create_pretrain_backbone.py`)

The script that produced the distributed ckpts:
```python
chkp = torch.load(input_path)["state_dict"]   # input is a Lightning ckpt
distilled_chkp = OrderedDict()
for key in chkp:
    if key.startswith("model.backbone."):
        distilled_chkp[key.replace("model.backbone.", "")] = chkp[key]
torch.save(distilled_chkp, save_path)
```

✅ Distributed ckpts are **flat `OrderedDict`**, prefix-stripped. Confirmed by direct inspection (Phase A.1.4):
- NFNet-L0 ckpt: 183 keys, no wrapping
- EffNetV2-S ckpts: 774 keys, no wrapping

### Loading pattern for Phase B

```python
import timm
import torch

# Backbone
backbone = timm.create_model(
    "tf_efficientnetv2_s_in21k",     # or "eca_nfnet_l0"
    features_only=True,
    pretrained=False,                # don't load timm's default weights
    exportable=True,                  # matches Sydorskyi
    in_chans=1,                       # single mel channel
)

# Load Sydorskyi ckpt
ckpt = torch.load(
    "/storage/project/ps-clef2026_birdclef-0/shared/pretrained-models/sydorskyi-2025/"
    "models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth",
    map_location="cpu",
    weights_only=False,
)
backbone.load_state_dict(ckpt, strict=True)   # MUST be strict=True; assert no missing/unexpected

# Sanity: backbone output channels
assert backbone.feature_info.channels()[-1] == 256   # EffNetV2-S features_only
# For NFNet: assert == 2304
```

---

## Open questions resolved by Phase A.2

| Question | Resolution |
|---|---|
| Head hidden dim 512 or 1028? | **512** ✅ (`blocks.py:206`) |
| GeMFreq p, eps? | `p=3.0` (learnable param), `eps=1e-6` (`blocks.py:64, 71-79`) |
| Standardize axes? | **(1, 2) = (n_mels, time), per-sample** ✅ (`blocks.py:30-31`) |
| min-max[0,1] axes? | Same: (1, 2), per-sample (`blocks.py:34-35`) |
| Focal α, γ? | **α=0.25, γ=2.0** ✅ (`focal_loss.py:31-32`) |
| Class-balance γ NFNet vs EffNetV2-S? | NFNet sqrt (γ=−0.5); EffNetV2-S balanced (γ=−1) ⚠️ CORRECTED |
| Ckpt key prefix? | None — flat `OrderedDict`, no wrapping (`create_pretrain_backbone.py:13-14`) |
| EffNetV2-S backbone output channels? | **256** ⚠️ CORRECTED (features_only mode strips conv_head+bn2) |
| NFNet-L0 backbone output channels? | **2304** ✅ |
| Mel library? | **nnAudio STFT + torchaudio MelScale** (hybrid) ⚠️ partial correction |
| `f_max` for mel? | None (defaults to sr/2 = 16000) |
| `mel_scale="htk"` or "slaney"? | `"htk"` (default in `MelScale`) |
| MixUp `alpha=None` semantics? | Equal-weight 50/50 wave + sum-clip target ✅ |
| MixUp target rule (sum vs max)? | **Sum + clip to [0, 1]** ✅ |
| PL swap mechanism? | 40% Bernoulli, primary-class-matched ✅ |
| Label smoothing implementation? | Mass-preserving α=0.05, GT-only ✅ |
| OOF PL filtering? | NOT used in selected configs (sample-dedup only) |
| MixUp partner via class-balanced sampler? | No — uniform random across full dataset (`_get_mixup_idx`) |
| Which ckpt for EffNetV2-S? | **2024 ckpt** (`tf_efficientnetv2_s_in21k_Pretrainversion1.pth`) ⚠️ — NOT 2025 SWA |
| Which ckpt for NFNet-L0? | 2025 ckpt (`eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt`) |
| BG noise mix probability? | OneOf picks branch (50/50); each branch has internal p=0.5 → 25% nocall + 25% ESC50 + 50% none |
| 5s crop strategy? | `_prepare_sample_piece` — random crop in training, center in inference (file size dependent — see line 497-583 for details) |
| Mixed precision? | **NO AMP** (`precision_mode="32-true"`) ✅ |

---

## Phase B implementation TODO derived from Phase A findings

| Component | Implementation note |
|---|---|
| `nnAudio` dep | Required for STFT only. Add to `pyproject.toml`. Use `nnAudio.Spectrogram.STFT` with `freq_scale="no"`. |
| `TraceableMelspec` | Port `vendor/sydorskyi-2025/code_base/models/blocks.py:448-524` verbatim. |
| `NormalizeMelSpec` | Port `vendor/sydorskyi-2025/code_base/models/blocks.py:23-54` (exportable=True branch). |
| `ChannelAgnosticAmplitudeToDB` | Port `vendor/sydorskyi-2025/code_base/models/blocks.py:822-880` OR use torchaudio.functional.amplitude_to_DB with `top_db=80, amin=1e-10`. |
| `GeMFreq` | Port `vendor/sydorskyi-2025/code_base/models/blocks.py:71-79` (p=3 learnable, eps=1e-6). |
| `AttHead` | Port `vendor/sydorskyi-2025/code_base/models/blocks.py:202-365`. Override `num_class=234` for BirdCLEF 2026. |
| `WaveCNNAttenClasifier` | Port `wave_clasifier.py:37-298`. Default `head_type="AttHead"`. |
| `RandomFiltering` | Port `code_base/models/augmentations.py` verbatim. |
| `CustomFreqMasking` / `CustomTimeMasking` | Port `code_base/augmentations/spec_augment.py`. |
| `FocalLossBCE` | Port `code_base/losses/focal_loss.py:28-54` OR write fresh using `torch.nn.BCEWithLogitsLoss` + `torchvision.ops.focal_loss.sigmoid_focal_loss`. |
| MixUp in dataset | Port `_prepare_sample_target_from_idx` + `__getitem__` MixUp branch (sum-clip target). |
| PL swap | Port the 40% Bernoulli + primary-class-matched logic. |
| Label smoothing | Port `_prepare_target` (mass-preserving formula). |
| WeightedRandomSampler | `power_value=-0.5` for NFNet, or `1/n_c` for EffNetV2-S. |
| Ckpt loader | `timm.create_model(features_only=True, in_chans=1)` + `strict=True` load. |

Reuse Sydorskyi's exact source where possible (`vendor/sydorskyi-2025/code_base/`) to maximize parity. We can import modules directly from `vendor/sydorskyi-2025/code_base/` as long as their imports resolve (they should — Phase B test 1).

---

## Cross-references

- Proposal: `user/murilo/experiments/034-pretrained-backbone-yoink/proposal.md`
- Tasks: `user/murilo/experiments/034-pretrained-backbone-yoink/tasks.md`
- Deep research review: `user/murilo/notes/2026-05-28-research-report.md`
- Source (submodule): `vendor/sydorskyi-2025/` @ `053594f`
- Pretrained ckpts: `/storage/project/ps-clef2026_birdclef-0/shared/pretrained-models/sydorskyi-2025/`
- Paper: Sydorskyi & Gonçalves 2025, CEUR Vol-4038 paper_256
- Reference inference notebook (Phase C Gate 1 anchor): [`vladimirsydor/bird-clef-2025-minimul-inference`](https://www.kaggle.com/code/vladimirsydor/bird-clef-2025-minimul-inference) (scriptVersionId 245080754)
