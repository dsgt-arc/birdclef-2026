# 2026-05-09 — Tucker Arrants's distilled SED notebook: code audit

Source: `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`
(pulled via `kaggle kernels pull tuckerarrants/bc2026-distilled-sed`).

Purpose: verify the structural claims in
`user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md` against
the actual code, and compare Tucker's attention head + distillation forward
against our `AttHead` and `LSEHead`.

---

## TL;DR

1. **The KD setup is exactly what the discussion-summary note described.**
   Embedding-space MSE on a GAP-pooled backbone projection vs Perch v2's
   per-clip 1536-d embedding, with `feat.detach()` on the classification
   branch. Confirmed.
2. **Tucker's attention head is NOT our `AttHead`.** Two important
   divergences vs ours:
   - Tucker aggregates clip predictions in **logit space**: `clip_logits =
     sum(softmax(tanh(att)) * framewise_logits, dim=time)`. We aggregate in
     **probability space** then `logit(clamp(probs))` back to logits. The
     sigmoid+inverse-sigmoid round-trip is the suspicious part of our recipe.
   - Tucker uses **dual-supervision inference**: `0.5 * sigmoid(clip_logits)
     + 0.5 * sigmoid(framewise.max(dim=time))`. v6k used `blend_clip=0.0,
     blend_frame=1.0` (framewise-only), which is what made v6k's attention
     uniform per the 04-26 D1 verdict.
3. **Tucker's mixup default is Sydorskyi-style** (`MIXUP_HARD = True`,
   `lb = np.maximum(lb1, lb2)`) — exactly the exp 030 recipe. So his 0.93+
   recipe stacks Sydorskyi-faithful mixup + Perch KD + dual-supervision
   AttHead + 5-fold + simple PP. The mixup was never the load-bearing piece
   for him, it's just the natural choice.
4. **Backbone: `tf_efficientnet_b0.ns_jft_in1k`** (NS-JFT pretrained
   EfficientNet-B0). NOT EfficientNetV2-B0 — that was Salman. Different
   architecture from exp 030's `tf_efficientnetv2_b0.in1k`.

---

## Tucker's attention head (verbatim, cleaned)

```python
class GeMFreqPool(nn.Module):
    """Generalized Mean pooling over frequency. Learnable p starts at 3.0."""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)              # CLAMP — we don't have this
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)

class BirdSEDModel(nn.Module):
    def __init__(self, ..., hidden_dim=512):
        ...
        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.att.weight)         # explicit init
        nn.init.xavier_uniform_(self.cla.weight)
        self.att.bias.data.fill_(0.)
        self.cla.bias.data.fill_(0.)
        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)                              # (B, C, F, T)
        distill_emb = None
        if return_distill and hasattr(self, 'distill_head'):
            distill_emb = self.distill_head(h)            # GAP+Linear → (B, 1536)

        # ⭐ Stop-gradient ONLY when distillation is on
        h_cls = h.detach() if USE_PERCH_DISTILL else h

        h_cls = self.gem_freq(h_cls)                      # (B, C, T)
        h_cls = h_cls.permute(0, 2, 1)                    # (B, T, C)
        h_cls = self.dense(h_cls)                         # (B, T, 512)
        h_cls = h_cls.permute(0, 2, 1)                    # (B, 512, T)

        # ⭐ Aggregation in LOGIT space, not probability space
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        ...
```

## Our `AttHead` (current, sed_head.py:120-149)

```python
def forward(self, features):
    x = self.gem(features)                                # (B, C, T)
    x = x.permute(0, 2, 1)                                # (B, T, C)
    x = self.project(x)                                   # (B, T, hidden)
    x = x.permute(0, 2, 1)                                # (B, hidden, T)

    frame_logits = self.frame_logits_conv(x)              # (B, num_classes, T)
    att_raw = torch.tanh(self.attention_conv(x))
    att_weights = torch.softmax(att_raw, dim=-1)

    # ⚠️ Aggregate in PROBABILITY space, then logit-back
    frame_probs = torch.sigmoid(frame_logits)
    clipwise_probs = (att_weights * frame_probs).sum(dim=-1)
    clipwise_logits = torch.logit(clipwise_probs.clamp(1e-7, 1.0 - 1e-7))

    return clipwise_logits, frame_logits
```

## Side-by-side

| Component | Tucker's BirdSEDModel | Our `AttHead` |
|---|---|---|
| Freq pool | GeMFreqPool, p=3 init, **`p.clamp(min=1.0)`** | GeM (same module), p=3 init, **no clamp** |
| Dense block | Dropout(0.25) → Linear(C, 512) → ReLU(inplace) → Dropout(0.5) | Dropout(0.25) → Linear(C, 512) → ReLU → Dropout(0.5) — **identical** |
| Attention conv | `Conv1d(512, num_cls, k=1, bias=True)`, **xavier_uniform_ init**, bias=0 | Same shape, **default torch init** |
| Class conv | `Conv1d(512, num_cls, k=1, bias=True)`, **xavier_uniform_ init**, bias=0 | Same shape, **default torch init** |
| Attention transform | `softmax(tanh(att(x)), dim=-1)` | `softmax(tanh(att(x)), dim=-1)` — **identical** |
| **Clipwise aggregation** | **`sum(att_weights * raw_logits, dim=time)`** in logit space | `logit(clamp(sum(att_weights * sigmoid(logits), dim=time), 1e-7, 1-1e-7))` in probability space |
| Returns | `(clip_logits, frame_logits)` from training; sometimes `frame_logits.permute(0, 2, 1)` | `(clipwise_logits, frame_logits)` |

### Why the aggregation-domain difference matters

**Logit-space aggregation (Tucker)**:
- `clip = Σ_t att_t · logit_t` is a linear weighted sum.
- `d clip / d logit_t = att_t`. Gradient flows cleanly to per-frame logits.
- Big-magnitude logits dominate (so the head can express "this frame is
  decisive" by producing a large logit there, regardless of attention).
- BCE on `clip` → standard sigmoid+log-loss math, no numerical traps.

**Probability-space aggregation (ours)**:
- `clip_prob = Σ_t att_t · σ(logit_t)`, then `clip_logit = log(clip_prob /
  (1 - clip_prob))` after clamping.
- The `sigmoid → average → inverse-sigmoid` round-trip saturates: if any
  per-frame `σ(logit_t)` is near 1, `clip_prob` saturates to ~1, and the
  inverse-sigmoid blows up. The `clamp(1e-7, 1-1e-7)` is a band-aid for that.
- Gradient through `logit(clamp(prob))` is `1/(prob·(1-prob))` — *infinite*
  in the saturation regions, killed by the clamp.
- High-confidence frames near 1.0 contribute *less* useful gradient than they
  should.

This is plausibly part of why our AttHead variants regressed on LB:
the head is learning to push individual frames near 1.0, which then saturates
the aggregation and kills training signal. Tucker's logit-space aggregation
doesn't have this failure mode.

**This is also exactly what hengck23 / Nikita 1st place 2025 do** — Tucker's
docstring says "1st-place-inspired design." Our `AttHead` docstring claims
the same provenance but we drifted on aggregation domain. Worth a careful
re-read of Nikita's published code to confirm which one matches.

---

## Tucker's loss (training cell 14, paraphrased)

```python
clip_logits, framewise, distill_emb = m(mel, return_framewise=True, return_distill=True)
frame_max_logits = framewise.max(dim=1).values   # (B, num_classes)

# ⭐ Dual-supervision: 50/50 clip and framewise-max BCE
bce_clip  = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
bce = 0.5 * bce_clip + 0.5 * bce_frame
ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
cls_loss = (ps * sw).mean()

if USE_PERCH_DISTILL:
    with torch.no_grad():
        wav_5s = wav.squeeze(1)                  # crop/pad to exactly 160000 samples
        perch_emb = perch_teacher.embed(wav_5s).to(device)   # (B, 1536)
    distill_loss = F.mse_loss(distill_emb, perch_emb)
    loss = cls_loss + ALPHA_DISTILL * distill_loss   # ALPHA_DISTILL = 1.0
else:
    loss = cls_loss
```

And inference:
```python
clip_logits, framewise = m(mel, return_framewise=True)
frame_max = framewise.max(dim=1).values
p_clip  = torch.sigmoid(clip_logits)
p_fmax  = torch.sigmoid(frame_max)
p_blend = 0.5 * p_clip + 0.5 * p_fmax       # ⭐ blend at inference too
```

### Key observations

1. **Dual-supervision is enabled at BOTH training AND inference**.
   - Training: `0.5 * BCE(clip_logits, lb) + 0.5 * BCE(frame_max_logits, lb)`.
   - Inference: `0.5 * sigmoid(clip_logits) + 0.5 * sigmoid(frame_max)`.
   - This gives the attention head real gradient pressure: if attention is
     uniform, `clip_logits = mean(framewise_logits)`, which behaves
     differently from `frame_max`. The two terms only agree when attention
     correctly localizes onto the high-magnitude frames.
   - **This is the structural fix to v6k.** Our v6k had `blend_clip=0.0,
     blend_frame=1.0`, which removed inference-time gradient pressure on
     attention → attention learned to be uniform (D1 verdict, 04-26).
     Tucker's `blend_clip=0.5, blend_frame=0.5` is what gets attention to
     localize.

2. **`ALPHA_DISTILL = 1.0`**. Distillation MSE has the same nominal weight as
   classification BCE, but their *magnitudes* differ wildly:
   - End-of-training: `cls_loss ≈ 0.032`, `dist_loss ≈ 0.003`. So MSE term
     contributes ~10× less to total loss by epoch 25.
   - Crucially: **only `dist_loss` updates the backbone** (because of
     `feat.detach()` on the cls branch). So `cls_loss`'s 10× larger gradient
     is irrelevant for backbone learning — it only shapes the head.
   - The backbone is being driven ENTIRELY by Perch matching. The SED head
     learns to read off whatever representation that produces.

3. **Mixup happens BEFORE the Perch teacher forward.** `wav` is the
   already-mixed waveform from `FocalDS.__getitem__`. The teacher embeds the
   mixed audio. So the supervision target is "what does Perch say about this
   mixed soundscape?" — Perch labels the mixture itself, no consistency
   issue. This is the right move (alternative would be: embed the un-mixed
   pre-images and weight-combine, which is more code and probably worse).

---

## Tucker's mixup default (cell 4 + 12)

```python
USE_FOCAL_MIXUP    = True
MIXUP_PROB         = 0.5
MIXUP_ALPHA        = 0.4
MIXUP_HARD         = True              # ⭐ union labels, max(y1, y2)

USE_FOCAL_SC_MIXUP     = True          # focal × labeled-soundscape mixup too
FOCAL_SC_MIXUP_PROB    = 0.5
FOCAL_SC_MIXUP_ALPHA   = 0.4
```

```python
# In FocalDS.__getitem__ (lines paraphrased):
lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)    # bimodal toward 0/1
ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
lb = np.maximum(lb1, lb2) if MIXUP_HARD else (lam * lb1 + (1 - lam) * lb2)
```

So Tucker uses:
- Waveform mixup (not spec mixup)
- Beta(0.4, 0.4) lam (bimodal toward 0/1, but with significant mass in the
  middle — *different* from Sydorskyi's `lam=0.5` deterministic)
- **`max(lb1, lb2)` hard labels — Sydorskyi-style!**
- p=0.5 application probability (same as Sydorskyi)

This is *almost* the exp 030 sydorskyi recipe — different lam distribution
(Beta(0.4, 0.4) vs deterministic 0.5), no peak-renormalize after mix, but
same target combination. **The hard-max target is the exp 030 lever, applied
as a default in Tucker's pipeline.** Worth noting that Tucker's 0.93+ stacks
this on top of Perch KD, dual-supervision AttHead, and 5-fold + PP — so
the mixup target choice isn't load-bearing for him, it's just the natural
default.

He also adds **focal × labeled-soundscape mixup** (`USE_FOCAL_SC_MIXUP=True`,
p=0.5), where 50% of focal samples get blended with a random labeled
soundscape window with `max` of the two labels. This is its own thing —
data-augmentation that fakes more multi-label "soundscape-like" training
samples. We don't do this.

---

## Tucker's distillation head

```python
class DistillHead(nn.Module):
    """Projects backbone features to Perch's 1536-d space via GAP + Linear."""
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, feature_map):
        gap = feature_map.mean(dim=[2, 3])   # (B, C, F, T) -> (B, C)
        return self.proj(gap)                 # (B, embed_dim)
```

**One vector per sample, not per-frame.** `mean(dim=[2, 3])` averages over
both frequency and time → single (B, C) → projected to (B, 1536).

Perch v2's ONNX export returns `(B, 1536)` per 5s clip — single embedding,
not framewise. So MSE is between two (B, 1536) vectors. Cleaner than my
earlier `(B, 1536, T)` mental model — there's no temporal axis to align,
and Perch v2 doesn't expose per-frame embeddings via this export anyway.

The Perch teacher itself is **ONNX**, not PyTorch:

```python
import onnxruntime as ort

class PerchTeacher:
    def __init__(self, onnx_path, device_str="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device_str == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        ...

    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy()
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()
```

The Perch ONNX path Tucker uses:
`/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx`

We have Perch v2 in our pipeline already (it's the "Perch" half of the 0.929
ensemble). We'd need to either: (a) reuse the same dataset Tucker
references, (b) re-export our Perch ONNX as a no-DFT variant, or (c) use
Perch via PyTorch instead of ONNX. ONNX has a tax: `wav.cpu().numpy() →
ort.run() → torch.from_numpy().to(device)` per batch. PyTorch teacher would
keep everything on GPU. Tucker's framing implicitly suggests "no-DFT" means
the mel computation is in the student, not the teacher.

---

## Tucker's mel pipeline (cell 4 + 10)

```python
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 256          # ⭐ same as v5j (256 mels)
FMIN       = 20
FMAX       = 16000
SR         = 32000

class MelSpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
        return self.db_transform(self.mel_spec(waveform))
```

Plus per-sample mean/std normalization in the training step:
```python
for i in range(B):
    mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
```

Then SpecAugment (1 freq mask × 10, 2 time masks × 10).

**Compared to our v5j**: same n_fft / hop / n_mels / fmin (we use fmax higher
at 16000 too). We use NNAudio mel + AmplitudeToDB; Tucker uses torchaudio
MelSpectrogram + AmplitudeToDB. We use NNAudio's normalization; Tucker does
explicit per-sample (mel - mean) / std. Functionally equivalent.

The big difference: **Tucker applies SpecAugment** (1 freq × 10 + 2 time ×
10). v5j has none. Tucker also flagged in the discussion that SpecAugment
on student but not teacher contributes to the dist_loss floor — so the
SpecAugment is enabled by default here.

---

## Tucker's optimizer + schedule (cell 14)

```python
EPOCHS = 25
BATCH  = 64        (16 in DEBUG mode, but 64 is the production setting)
LR     = 5e-4
WD     = 1e-4
WARMUP_EPOCHS = 2

opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
warmup_sched = LinearLR(opt, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
cosine_sched = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
sch = SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
```

Standard recipe: AdamW + 2-epoch linear warmup → 23-epoch cosine decay to
`eta_min=1e-6`. We use OneCycleLR; this is a different schedule but
qualitatively similar. Not the lever.

Mixed precision via `GradScaler` + `autocast()`. We use 16-mixed too. Same.

`torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)` — gradient clipping
at 1.0. We don't currently clip. Likely not load-bearing.

---

## Tucker's data composition (`MixSamp` batch sampler)

```python
ACTIVE_SOURCES = ["focal", "sc"]
SHARES = {"focal": 0.9, "sc": 0.1}     # 90% focal, 10% labeled-soundscape per batch
SOURCE_WEIGHTS = {
    "focal":         1.0,
    "focal_missing": 0.0,
    "sc":            1.0,
}
```

Each batch is constructed to have ~58 focal samples + ~6 labeled-soundscape
samples per batch of 64. He's deliberately under-weighting labeled
soundscapes in the *batch composition* but giving them full per-sample
weight at loss time. Seems like a "calibration" choice — labeled SS is
small (~1500 windows) so heavy upsampling would mean repeats; instead, he
keeps each batch mostly focal but ensures labeled-SS is always represented.

We currently feed both as one ConcatDataset; we don't have explicit
batch-share control. Worth noting if we replicate.

---

## Cross-references to our 04-26 / 04-28 findings

- **Mixup audit (2026-04-26)**: Tucker uses `MIXUP_HARD=True → max(lb1,
  lb2)` by default. Same target distribution we tested in exp 030 (which we
  found null at single-fold resolution on HGNet). Tucker stacks it with
  Perch KD + dual-supervision AttHead + 5-fold + PP, so the mixup variant
  isn't his lever — it's just the natural default.

- **Target-distribution simulation (2026-04-26)**: Tucker's `MIXUP_HARD`
  produces `target = max(y1, y2)` for binary multi-hot, matching the test
  distribution (both species at 1.0). His default supervision distribution
  is much closer to test than v5j's interpolated targets.

- **D1 verdict (2026-04-26)**: v6k's attention was uniform because
  `blend_clip=0.0` removed gradient pressure on the attention branch.
  Tucker's `0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)` blend at
  inference forces attention to localize. **This is the structural fix to
  v6k.** A `v6k_v2` with `blend_clip=0.5, blend_frame=0.5` would actually
  test whether AttHead architecture works on our codebase.

- **2026-04-28 PL vs KD distinction**: Tucker is doing **#3 (KD on labeled,
  embedding space, with stop-gradient)**. Confirmed by the code. The
  detach is conditional on `USE_PERCH_DISTILL=True`; with KD off, gradient
  flows normally through the SED head. So the stop-gradient is part of the
  KD recipe specifically.

- **2026-04-28 next-experiment direction**: my mental model of Tucker's
  recipe was structurally correct (embedding KD + stop-grad on cls branch),
  but I missed:
  1. The **dual-supervision** AttHead (clip + framewise-max, 50/50)
  2. The fact that distillation is on **GAP-pooled clip embedding**, not
     framewise (because Perch v2 only exposes clip embeddings via ONNX)
  3. The **`MIXUP_HARD` default** (Sydorskyi-style targets included by
     default)
  4. The **focal × labeled-SS mixup** as a second augmentation
  5. The **`tf_efficientnet_b0.ns_jft_in1k`** backbone (NS-JFT, NOT
     EffNetV2-B0)

---

## What the LSEHead ablation would look like (single-variable test)

Two ways to wire Tucker's KD into our codebase, ranked by tightness of
single-variable comparison:

### Option α — minimum-change KD on top of v5j

Keep v5j-as-is (HGNet + LSEHead + BCE + spec mixup), add a Perch KD branch:

```python
# In sed_finetune.py forward:
feat = self.backbone(mel)                         # (B, C, F, T)

# NEW distillation branch
distill_emb = self.distill_head(feat)             # GAP+Linear → (B, 1536)
with torch.no_grad():
    perch_emb = self.perch_teacher.embed(wav_5s)  # (B, 1536)
distill_loss = F.mse_loss(distill_emb, perch_emb)

# Existing LSEHead branch — but feat.detach() now
feat_for_head = feat.detach() if self.use_perch_distill else feat
clipwise_logits, framewise_logits = self.lse_head(feat_for_head)
cls_loss = bce(clipwise_logits, labels)

loss = cls_loss + alpha_distill * distill_loss
```

Single variable change vs v5j-original: add Perch KD MSE + stop-grad on the
LSEHead. Predicts +0.02 LB if Tucker's quantification holds. Most directly
tests whether KD is the lever, with everything else (head, loss, mixup
recipe) held at v5j's setting.

**Cost**: implement DistillHead + PerchTeacher (ONNX path) + the loss-side
branch. Reuse our exp 020 Perch ONNX if possible.

### Option β — full Tucker replication (multi-variable)

Port the full BirdSEDModel: AttHead with logit-space aggregation +
dual-supervision (clip + frame_max BCE) + Perch KD + SpecAugment + hard
mixup + 90/10 focal/SS batch composition. 5-fold + PP at inference.

Closest replication, but multi-variable. If LB lands at 0.91-0.93+, we
can't attribute the lift to any single component without follow-up
ablations. Better to start with Option α.

### Option γ — fix v6k first (free check)

Take the existing v6k checkpoint and resubmit with `blend_clip=0.5,
blend_frame=0.5` at inference. Zero training cost. Tests whether attention
becomes useful when the inference recipe gives it weight. Doesn't replicate
Tucker's pipeline but does engage the dual-supervision finding directly.
**Probably won't move LB much** — v6k's attention is already uniform per
the D1 verdict; flipping the inference blend wouldn't fix what's already
not localizing — but it's a free sanity check.

---

## Files

- Source notebook: `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`
- Discussion summary: `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md`
- This audit: `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`
- Cross-references:
  - `user/murilo/src/birdclef/heads/sed_head.py` (our `AttHead`, `LSEHead`)
  - `user/murilo/analysis/notes/2026-04-26-d1-train-side-attention-verdict.md`
    (v6k uniform-attention finding, fixed by Tucker's blend)
  - `user/murilo/analysis/notes/2026-04-26-mixup-implementation-audit.md`
    (Tucker's `MIXUP_HARD=True` is the same `max(y1, y2)` target we audited)
  - `user/murilo/analysis/notes/2026-04-28-next-experiment-direction.md`
    ADDENDUM (the PL vs KD #3 category this notebook embodies)
