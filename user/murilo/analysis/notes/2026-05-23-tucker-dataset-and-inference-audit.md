# 2026-05-23 — Tucker Arrants's distilled SED notebook: dataset, sampler, fold, weights, and inference audit (gap-fill)

**Source**: `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`
(byte-identical to the 2026-05-09 vendored copy; re-pulled 2026-05-23 to confirm no upstream updates since v2). Convenience `.py` extraction at `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.py` (gitignored).

**Companion**: `analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md` covered the forward pass, the AttHead aggregation domain, the loss formula, mixup defaults, and the optimizer. This note fills the gaps the prior audit explicitly flagged: dataset class internals, split construction, per-sample weights, soundscape label assembly, inference pipeline, post-processing, and a few **public-notebook bugs** we should not inherit.

---

## TL;DR

1. **Split construction is mixed**: focal uses `StratifiedKFold(n_splits=5)` on `primary_label`; soundscape uses `GroupKFold(n_splits=5)` on `filename`. All 66 SS files participate (no S22 holdout) — S22 is masked out only in the validation metric (`non_s22_macro`).
2. **Per-sample loss weights (`wt`, `mk`, `sw`) are mostly inert.** The dataset returns `ones` for `wt` and `mk` in every code path. `sw` is `SOURCE_WEIGHTS["focal"|"sc"|"focal_missing"] ∈ {1.0, 1.0, 0.0}`. **Effective loss is plain per-sample mean BCE over 234 classes, with missing-focal samples zeroed out**.
3. **Rare-species upsampling** copies focal rows so any species with `< MIN_SAMPLE=20` exemplars is replicated up to 20. Upsampling happens *after* fold assignment, so duplicates stay in the same fold as the original (no leak).
4. **Soundscape window labels** are built by matching `(filename, start_sec)` tuples against `train_soundscapes_labels.csv`. **Only 66 of 10,658 SS files have any labels** — the other 10,592 files (× 12 windows = 127,104 windows) never enter Tucker's pipeline at all; his pre-built cache only covers labeled files. Within the 66 labeled files × 12 windows = 792 cached SS windows, 739 are labeled and **53 are all-zero**. The all-zero rows sit *between* annotated windows in otherwise-labeled files — closer to "true negative" than "missed positive."
5. **MixUp Beta(0.4, 0.4) is symmetric — no `lam = max(lam, 1-lam)` trick.** Different from the Sydorskyi/Yuriy convention we've sometimes assumed. `MIXUP_HARD=True` then takes `max(lb1, lb2)` regardless of `lam`.
6. **The notebook has TWO real bugs**:
   - **Mel-size mismatch between training and inference**: training uses `N_MELS=256`, the SEDExportWrapper hardcodes `INF_N_MELS=128` for ONNX tracing, but cell 20's inference path also uses `INF_N_MELS=256`. The publicly-uploaded fold ONNXes likely embed 128-mel shapes; the published training code produces 256-mel models. Tucker's actual 0.917 LB must come from a different end-to-end path.
   - **Checkpoint-naming inversion**: file written as `fold{k}_best_ns22.pt` but contains `best_macro_state` weights — the bug Tucker self-disclosed.
7. **Site/hour priors are NOT in the public notebook.** Only 5-tap Gaussian smoothing in logit space across the 12 windows-per-file is present. Site/hour priors must live in a private branch or different notebook.
8. **Determinism is off**: `torch.backends.cudnn.deterministic = False`, `benchmark = True`. Run-to-run training is not bit-reproducible.

---

## 1. Fold construction (cell 07)

### 1.1 Focal — StratifiedKFold on `primary_label`

```python
audio_for_split = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
audio_for_split["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(audio_for_split, audio_for_split["primary_label"])):
    audio_for_split.loc[val_idx, "fold"] = fold
audio_cache_meta = audio_cache_meta.merge(
    audio_for_split[["original_idx", "fold"]], on="original_idx", how="left"
)
```

- 5-fold stratified on `primary_label`, deduplicating by `original_idx` so any cache-side row duplication doesn't bias the split.
- `random_state=SEED=42`.
- Distribution gets printed per-fold for sanity.

### 1.2 Soundscape — GroupKFold on `filename`

```python
sc_files = sc_cache_meta[["filename", "site"]].drop_duplicates().reset_index(drop=True)
gkf = GroupKFold(n_splits=N_FOLDS)
sc_files["fold"] = -1
for fold, (_, val_idx) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
    sc_files.loc[sc_files.index[val_idx], "fold"] = fold
file_to_fold = dict(zip(sc_files["filename"], sc_files["fold"]))
sc_cache_meta["fold"] = sc_cache_meta["filename"].map(file_to_fold).fillna(-1).astype(int)
```

- File-level fold assignment (all 12 windows of a given SS file share a fold).
- All 66 SS files participate, including the dominant `S22` site. Tucker's inline comment: *"All files including S22 participate in CV for maximum species coverage (46 multi-fold species vs 32 with S22 holdout)."*
- **S22 is masked OUT in the primary validation metric** (`non_s22_macro`) via `non_s22_mask_sc = sc_sites != "S22"`. So S22 windows train but don't drive checkpoint selection on the dominant non-S22 site metric.

### 1.3 Upsample rare species (after fold assignment)

```python
counts = audio_cache_meta["primary_label"].value_counts()
rare_species = counts[counts < MIN_SAMPLE].index   # MIN_SAMPLE = 20
extra_rows = []
for sp in rare_species:
    sp_rows = audio_cache_meta[audio_cache_meta["primary_label"] == sp]
    n_copies = int(np.ceil(MIN_SAMPLE / len(sp_rows))) - 1
    for _ in range(n_copies):
        extra_rows.append(sp_rows)
audio_cache_meta = pd.concat([audio_cache_meta] + extra_rows, ignore_index=True)
```

- Pure row-replication (not augmentation). Each rare species gets `ceil(20 / count) - 1` extra copies of its rows.
- Critical: **fold column is copied from the original rows**, so upsampled duplicates stay in the same fold as their source. No cross-fold leak.

### 1.4 Soundscape labels per-window (the part the prior audit missed)

```python
sc_labels_raw = pd.read_csv(LABELS_PATH).drop_duplicates()
sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)

Y_SC = np.zeros((len(sc_cache_meta), NUM_CLASSES), dtype=np.float32)
for i, row in sc_cache_meta.iterrows():
    matches = sc_labels_raw[
        (sc_labels_raw["filename"] == row["filename"]) &
        (sc_labels_raw["start_sec"] == row["start_sec"])
    ]
    for _, m in matches.iterrows():
        for lbl in str(m["primary_label"]).split(";"):
            lbl = lbl.strip()
            if lbl in LABEL2IDX:
                Y_SC[i, LABEL2IDX[lbl]] = 1.0
```

- `Y_SC[i]` is multi-hot for SS window `i`, populated only if `(filename, start_sec)` matches a row in `train_soundscapes_labels.csv`.
- `labeled_sc_mask = Y_SC.sum(axis=1) > 0` — convenience flag, but **NOT used to filter training data**. ALL `sc_cache_meta` windows (labeled or not) go into the training ScDS via `build_active_datasets`; unlabeled rows contribute as **all-negative** supervision across all 234 species.

**Actual scope of the "all-zero" population** (verified against `train_soundscapes_labels.csv` on 2026-05-23):

| Quantity | Count |
|---|---:|
| Total `train_soundscapes/*.ogg` files on disk | 10,658 |
| Files with ANY labels in `train_soundscapes_labels.csv` | 66 |
| Files with NO labels (excluded from Tucker's cache entirely) | 10,592 |
| Unique labeled `(filename, start_sec)` windows | 739 |
| Cached SS windows in Tucker's pipeline (66 × 12) | 792 |
| **All-zero windows within labeled files (792 − 739)** | **53** |
| Unlabeled-file windows NOT in Tucker's pipeline (10,592 × 12) | 127,104 |

So the "all-zero label" issue is small: **53 windows out of 792 (~6.7%) of the SS training pool**, or ~0.7% of total training samples at MixSamp's 10% SC batch share. And these 53 windows sit *between* annotated windows in otherwise-labeled files — so they're much closer to "annotator reviewed and chose not to mark anything" (true negative) than to "missed positive" in randomly-unlabeled audio.

The 127,104 unlabeled-file windows are the **Track B unlabeled pool** (own-teacher pseudo-labels per Antoine/Boredom/chenwenqiang_001). Not used in Tucker's recipe.

---

## 2. FocalDS — the dataset class (cell 12)

### 2.1 Return signature

```python
def __getitem__(self, i):
    ...
    return (waveform_tensor, label_tensor, weight_tensor, mask_tensor, source_tag)
```

`weight_tensor` and `mask_tensor` are **always `torch.ones(NUM_CLASSES)`** in every code path I can find. `source_tag` is `"focal"`, `"focal_missing"`, or `"sc"`.

### 2.2 Chunk extraction

```python
def _load_chunk(self, r):
    w = load_focal(r["cache_file"])
    if w is None: return None, None
    if self.aug:
        start = np.random.randint(0, max(1, len(w) - TRAIN_SAMPLES + 1)) if len(w) > TRAIN_SAMPLES else 0
    else:
        start = int(r.get("start_sec", 0)) * SR
    ch = extract_chunk_np(w, start, TRAIN_SAMPLES)
    lb = np.zeros(NUM_CLASSES, dtype=np.float32)
    if str(r["primary_label"]) in self.l2i:
        lb[self.l2i[str(r["primary_label"])]] = 1.0
    if self.secondary_lookup is not None and "original_idx" in self.df.columns:
        for s in self.secondary_lookup.get(int(r["original_idx"]), []):
            if s in self.l2i: lb[self.l2i[s]] = 1.0
    return ch, lb
```

- **Training mode**: random start offset (uniform over valid range). This is the de-facto time-shift augmentation — `apply_aug` does NOT shift in time.
- **Eval mode**: fixed offset at `start_sec * SR` (deterministic).
- **Label**: primary one-hot + secondary one-hots from `train.csv["secondary_labels"]` (controlled by `USE_FOCAL_SECONDARY = True`). Secondaries lookup is built once at load time:
  ```python
  for idx, row in train_df.iterrows():
      sec = row.get("secondary_labels", "")
      if pd.isna(sec) or sec in ("", "[]"): continue
      try:
          sec_list = eval(sec) if isinstance(sec, str) else []
      except: continue
      valid = [s for s in sec_list if s in LABEL2IDX]
      if valid: focal_secondary_labels[idx] = valid
  ```
  So a focal recording with `primary_label="rabest1"` and `secondary_labels="['gretwh1', 'comhaw1']"` gets a 3-hot label.

### 2.3 The MixUp / augmentation pipeline (per sample)

Order of operations in `FocalDS.__getitem__`:

1. Load `ch1`, `lb1` (the "primary" sample).
2. **MixUp branch A (Focal-Focal)**: with prob `MIXUP_PROB=0.5`, sample a second focal example; if loaded, mix waveforms `(lam * ch1 + (1-lam) * ch2)` with `lam = Beta(0.4, 0.4)` and label `max(lb1, lb2)` (MIXUP_HARD=True). Apply `apply_aug` to mixed waveform. Return.
3. **MixUp branch B (Focal-SC)**: only reached if branch A didn't fire. With prob `FOCAL_SC_MIXUP_PROB=0.5`, sample a *labeled* SS window from the same training-fold pool, mix `(lam * ch1 + (1-lam) * sc_chunk)` with `lam = Beta(0.4, 0.4)`, label `max(lb1, lb_sc)`. Apply `apply_aug`. Return.
4. **No MixUp**: apply `apply_aug` to `ch1`. Return.

So **at most one mixup applies per sample**, and the effective per-sample probabilities are:
- 50% focal-focal mixup
- 25% focal-SC mixup (= 50% × 50%)
- 25% no mixup

### 2.4 `apply_aug` — what actually augments at the waveform level

```python
def apply_aug(w):
    if np.random.random() < AUG_PROB:  # 0.5
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))   # ±6 dB
    if np.random.random() < AUG_PROB:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(
                sp / (10 ** (np.random.uniform(*AUG_NOISE_SNR_DB_RANGE) / 10)))   # 10–30 dB SNR
    return w
```

**Just two augmentations**: gain jitter (±6 dB) and SNR-controlled Gaussian noise (10–30 dB).

**The markdown claims three augmentations including time-shift, but the code has only two.** Time shift comes from the random `_load_chunk` offset, not from `apply_aug`.

### 2.5 ScDS — labeled soundscape dataset

```python
class ScDS(Dataset):
    def __init__(self, Y, sc_df, aug=False):
        self.Y, self.df, self.aug = Y, sc_df.reset_index(drop=True), aug
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav_full = load_sc_waveform_from(WAVEFORM_CACHE_DIR, row.get("cache_file")) if row.get("cache_file") else None
        if wav_full is None:
            wav_t = torch.zeros(1, TRAIN_SAMPLES)
        else:
            chunk = extract_chunk_np(wav_full, int(row["start_sec"]) * SR, TRAIN_SAMPLES)
            if self.aug: chunk = apply_aug(chunk)
            wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
        return (wav_t, torch.from_numpy(self.Y[i].astype(np.float32)),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "sc")
```

- No mixup branch for SS samples.
- `apply_aug` only (gain + noise).
- Returns `Y_SC[i]` directly — including all-zero rows for unlabeled SS windows.

---

## 3. Batch composition — `MixSamp` (cell 12)

```python
class MixSamp(torch.utils.data.Sampler):
    def __init__(self, sizes, names, shares, bs, nst, seed=0):
        ...
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        # SHARES = {"focal": 0.9, "sc": 0.1}, BATCH = 64
        # → per_src = [int(round(64*0.9))=58, int(round(64*0.1))=6] → [58, 6] (sums to 64) ✓
    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0: continue
                idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch
```

- For each batch: independently sample 58 random indices from focal + 6 from SC (with replacement, since `rng.integers` does not deduplicate). Concatenate and shuffle.
- `nst = max(100, sum(sizes) / BATCH)` — at least 100 steps per epoch regardless of dataset size.
- **Sampling is with replacement, controlled by per-epoch seed `seed=42+ep`**, so the same batch composition shifts each epoch.
- DataLoader uses `num_workers=0` (single-threaded). That's fine for `BATCH=64` but limits throughput.

---

## 4. Loss math (cell 14)

```python
bce_clip  = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")   # (B, 234)
bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")  # (B, 234)
bce = 0.5 * bce_clip + 0.5 * bce_frame
ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)   # per-sample, mean over classes
cls_loss = (ps * sw).mean()
```

With `wt = mk = ones(B, 234)` and `mk.sum(1) = 234` (constant):
- `ps[b] = mean(bce[b])` over the 234 classes
- `sw[b] = 1.0` for focal/sc, `0.0` for focal_missing
- `cls_loss = mean over batch of (ps[b] * sw[b])`

So **effective loss is plain mean BCE, with focal_missing samples zeroed out** (but they still occupy a batch slot and dilute the gradient slightly because the divisor is `B`, not `sum(sw)`).

The `wt` / `mk` plumbing is set up for class-weighting / class-masking but is unused in the public notebook.

### Plus distillation

```python
if USE_PERCH_DISTILL and perch_teacher is not None:
    with torch.no_grad():
        wav_5s = wav.squeeze(1)
        N = wav_5s.shape[1]
        if N > 160000:
            start = (N - 160000) // 2
            wav_5s = wav_5s[:, start:start + 160000]
        elif N < 160000:
            wav_5s = F.pad(wav_5s, (0, 160000 - N))
        perch_emb = perch_teacher.embed(wav_5s).to(device)
    distill_loss = F.mse_loss(distill_emb, perch_emb)
    loss = cls_loss + ALPHA_DISTILL * distill_loss   # ALPHA_DISTILL = 1.0
```

- Teacher embeds the **already-mixed** waveform (mixup happens before this branch). Perch labels the mixture itself — no consistency issue.
- 5s slicing: center-crop if too long, right-pad with zeros if too short. (The dataset returns `TRAIN_SAMPLES = SR * 5 = 160000` so this is a no-op in practice.)
- ONNX teacher → numpy → torch conversion per batch (the ~30 s/epoch overhead Tucker mentioned).

---

## 5. Optimization (cell 14)

```python
opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4)
scaler = GradScaler()
warmup_steps = nst * 2   # 2 epochs of linear warmup
total_steps  = nst * 25
warmup_sched = LinearLR(opt, start_factor=1/25, end_factor=1.0, total_iters=warmup_steps)
cosine_sched = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
sch = SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
```

Per step:
```python
opt.zero_grad()
scaler.scale(loss).backward()
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
scaler.step(opt)
scaler.update()
sch.step()
```

- AdamW (β default), lr=5e-4, weight_decay=1e-4.
- 2 epoch linear warmup (start_factor=1/25, so lr starts at lr/25 = 2e-5) → 23 epoch cosine decay to 1e-6.
- gradient clipping at norm 1.0 — applied AFTER `unscale_` so it operates on real (not scaled) gradients.
- Mixed-precision (fp16 autocast + GradScaler).

---

## 6. Checkpoint selection (cell 14 + cell 16)

```python
if r["non_s22_macro"] > best_ns22:
    best_ns22 = r["non_s22_macro"]
    best_state_ns22 = {k: v.cpu().clone() for k, v in m.state_dict().items()}
if r["macro_auc_all"] > best_macro:
    best_macro = r["macro_auc_all"]
    best_state_macro = {k: v.cpu().clone() for k, v in m.state_dict().items()}
```

Two checkpoints tracked per fold:
- **best by `non_s22_macro`** (the dominant-site-excluded metric)
- **best by `macro_auc_all`** (all sites including S22)

Then in cell 16:
```python
if best_macro_state is not None:
    torch.save(best_macro_state, OUT_DIR / f"fold{fold_k}_best_ns22.pt")
    m.load_state_dict(best_macro_state, strict=False)
    ...
```

**Bug**: file written as `fold{k}_best_ns22.pt` but contains `best_macro_state` weights. Filename inversion, not a correctness bug — Tucker self-disclosed this in v2 ("I use best macro for inference not best ns22"). The ONNX export uses `best_macro_state`.

---

## 7. Inference pipeline (cells 20, 22, 24, 26)

### 7.1 Mel computation (cell 20)

```python
INF_N_MELS   = 256
INF_N_FFT    = 2048
INF_HOP      = 512
INF_FMIN     = 20
INF_FMAX     = 16000
INF_TOP_DB   = 80
INF_SR       = 32000
INF_CHUNK_S  = 5
INF_CHUNK_N  = INF_SR * INF_CHUNK_S    # 160,000
INF_N_FRAMES = INF_CHUNK_N // INF_HOP + 1   # 313

def audio_to_mel(chunks):
    """Raw audio chunks (N, 160000) -> normalized mel dB (N, 1, 256, 313)."""
    mels = []
    for i in range(chunks.shape[0]):
        S = librosa.feature.melspectrogram(
            y=chunks[i], sr=INF_SR, n_fft=INF_N_FFT, hop_length=INF_HOP,
            n_mels=INF_N_MELS, fmin=INF_FMIN, fmax=INF_FMAX, power=2.0,
        )
        S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)
        S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)
        mels.append(S_dB)
    return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)
```

- librosa (not torchaudio) at inference. Different backends could produce small numerical differences vs training.
- Per-instance mean/std normalization, same as training.
- **`INF_N_MELS = 256`** — matches training's `N_MELS = 256`. Good.

### 7.2 Bug: SEDExportWrapper hardcodes 128 mels (cell 16)

```python
class SEDExportWrapper(nn.Module):
    ...
INF_N_MELS = 128
INF_N_FRAMES = VAL_SAMPLES // HOP_LENGTH + 1

dummy_mel = torch.randn(1, 1, INF_N_MELS, INF_N_FRAMES).to(device)
torch.onnx.export(
    export_model, dummy_mel, str(onnx_path),
    input_names=["mel"],
    output_names=["clip_logits", "framewise_logits"],
    dynamic_axes={"mel": {0: "batch"},
                  "clip_logits": {0: "batch"},
                  "framewise_logits": {0: "batch"}},
    opset_version=17,
)
```

**Real bug**: `dynamic_axes` marks only the batch axis as dynamic. The mel and time axes are baked into the ONNX from `dummy_mel`'s shape `(1, 1, 128, 313)`. So the exported ONNX expects **128 mels** at inference, but the inference path produces **256 mels**. Shape mismatch at runtime.

**Implication for us**: Tucker's 0.917 LB must come from a local pipeline that doesn't have this mismatch. Either:
- (a) His training also used `N_MELS=128` somewhere
- (b) The publicly-shared fold ONNXes in `tuckerarrants/bc2026-distilled-sed-public` were exported from a different code path
- (c) The published notebook has a real bug and the LB shown is from a different run

We should **use `N_MELS=256` consistently** at training and inference, and add either `dynamic_axes` for freq/time or set the export dummy to match the real input shape. Easy fix.

### 7.3 Per-fold inference + blend (cell 24)

```python
for sess in fold_sessions:
    outs = sess.run(None, {'mel': mel})
    clip_logits = outs[0]
    frame_max = outs[1].max(axis=1)
    logits_sum += 0.5 * clip_logits + 0.5 * frame_max
logits_mean = logits_sum / len(INF_FOLDS)

# Smooth across windows in logit space, then sigmoid once.
logits_smoothed = gauss_smooth_final(logits_mean)
probs = sigmoid_inf(logits_smoothed)
```

**Everything happens in logit space until the very end**:
1. Per fold: `0.5 * clip_logits + 0.5 * frame_max_logits`
2. Sum across folds, divide by n_folds (mean)
3. 5-tap Gaussian smooth across the 12 windows per file (`weights = [0.1, 0.2, 0.4, 0.2, 0.1]`)
4. Sigmoid once at the end

This is different from "sigmoid → average → smooth" — and likely better-conditioned numerically.

### 7.4 5-tap Gaussian smoothing

```python
from scipy.ndimage import convolve1d
GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
N_WINDOWS = 12

def gauss_smooth_final(scores, weights=GAUSSIAN_KERNEL):
    smoothed = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
    return smoothed.reshape(-1, scores.shape[1])
```

- Reshape per-file: `(n_rows_total, 234) → (n_files, 12, 234)`
- 1D convolution along the 12-window axis with kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` (sums to 1.0, symmetric).
- `mode='nearest'`: replicate boundary values (no zero-padding).
- Re-flatten back to `(n_rows_total, 234)`.

Smoothing is **across time within a file**, not across species. Different from MengYe's `+0.004` lift (which smooths across labels).

### 7.5 What's NOT in the public notebook

Tucker's discussion mentions "site/hour priors + Gaussian smoothing." **Only the Gaussian smoothing is in the public notebook.** No site/hour prior fusion code anywhere. The 0.93+ LB from his discussion ("with PP") presumably includes priors that are in private code.

We have a clean reference for site/hour priors in the EoS.6 (LB 0.949) stack at `vendor/kaggle/notebooks/birdclef-2026/birdclef-2026-eos-6-bz/`, which Karnakbayev uses — same fundamental design (priors derived from labeled SS positives × site/hour, fused additively into logit space). We can port that pattern if we want priors.

---

## 8. The architecture from the figures

Re-confirming from `notes/figures/tucker-distilled-sed-arch.png` (1536-d teacher, default) and `arch2.png` (384-d cheap variant):

- Same waveform feeds two paths:
  - **Mel → backbone → DistillHead (GAP+Linear) → MSE vs Perch v2 frozen embedding** (the "Perch student" branch — drives the backbone)
  - **Mel → backbone → `feat.detach()` → SED head (GeMFreq → Dense → Att/Cla → clip + frame) → 0.5·σ(clip) + 0.5·σ(frame_max)** (the classification branch — never updates the backbone)
- The 384-d variant (`arch2.png`) targets Perch's last MBConv stage instead of the final embedding: same LB, 20% faster.

---

## 9. Determinism (cell 04)

```python
SEED = 42
random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

- `cudnn.deterministic = False`, `benchmark = True` → fastest kernels, **not** reproducible across runs.
- DataLoader has `num_workers=0` and the MixSamp seed shifts each epoch (`seed=42 + ep`).
- Our codebase is more strict (`Trainer(deterministic=True)`). Decision: do we copy Tucker's looser determinism for speed, or keep ours for repeatability? Probably keep ours.

---

## 10. Bugs to NOT inherit when porting

1. **`SEDExportWrapper` `INF_N_MELS = 128`** is wrong if training is `N_MELS = 256`. Use the training mel size at export, or make freq/time dynamic.
2. **Checkpoint filename inversion** — write the best-by-macro checkpoint to a file named `*_best_macro.pt`, not `*_best_ns22.pt`.
3. **`apply_aug` markdown lies** about time-shift — code has only gain + noise. (Not a bug; just inconsistent docs.)
4. **`secondary_labels = eval(sec)`** — using `eval` on a CSV string is a security smell. `ast.literal_eval` is the safer port. (Real impact ≈ 0 since this is offline training, but worth doing right.)

---

## 11. Items still NOT settled (gaps remaining)

| Item | Source we'd need |
|---|---|
| Tucker's actual site/hour prior code | Not in the public notebook; would need to ask or use the EoS.6 implementation as a stand-in |
| Whether the publicly-shared fold ONNXes in `bc2026-distilled-sed-public` were exported at 128 mels or 256 mels | Would need to download that dataset and inspect ONNX input shapes |
| How Tucker handles the all-zero-label SS windows during training (just plain BCE? Or some masking?) | Code is unambiguous: plain BCE. After verifying actual counts (53 zero-label windows out of 792 SS cache, ~0.7% of training samples), this is a small effect on a population that's likely true-negative, not "thousands of missed positives." Default to matching Tucker. |
| Whether his discussion-reported 0.917 LB used the 5-tap Gaussian smoothing in logit space (cell 24) or a different pipeline | Discussion text and code disagree on details; we should treat the public notebook code as our reference and accept that our LB might differ |
| MIN_SAMPLE choice (20 vs 10 vs 50) sensitivity | Not ablated by Tucker |
| Whether ALPHA_DISTILL = 1.0 is the right scale for our codebase's BCE magnitude | Tucker reports cls_loss ≈ 0.032 vs dist_loss ≈ 0.003 at convergence — but our v5j's cls_loss at convergence is ~0.016, smaller. May need to retune. |

---

## 12. Implication for exp 032 scaffold

What this audit changes about our previously-scoped Track A:

### Items I'd add to the implementation order

1. **Fold construction code**: don't just port the model — port the StratifiedKFold(focal) + GroupKFold(SS) + rare-species upsampling pattern. This is methodologically different from our v5j single-fold-with-CSV approach.
2. **Sampler implementation**: MixSamp (90/10 batch share) is non-trivial to bolt onto Lightning. We'd write a custom `BatchSampler` or use a `ConcatDataset` with a `WeightedRandomSampler`-driven equivalent.
3. **All-zero-label SS handling**: explicit decision. Tucker trains on them as all-negative. With the actual count now verified (53 windows ≈ 0.7% of training samples at 10% SC share, sitting between annotated windows in otherwise-labeled files), the "missed positive" concern is weak. **Default to (a) match Tucker** — the data is cheap, the signal is plausibly clean, and the implementation overhead of filtering or branch-separating these few rows isn't worth it.
4. **Inference in logit space, single sigmoid at end**: simpler than what our submission.py does and likely more numerically stable for the smoothing step.
5. **Fix the ONNX export bug**: use `N_MELS=256` consistently and either set dummy correctly or mark mel/time as dynamic axes.
6. **Don't inherit the looser determinism**: keep `Trainer(deterministic=True)` and our paired-control methodology.

### Items the audit didn't change

- AttHead with logit-space aggregation (port verbatim from `BirdSEDModel.forward`)
- DistillHead (GAP + Linear, port verbatim)
- PerchTeacher (ONNX wrapper, already have)
- Dual-supervision loss (already understood from prior audit)
- Mixup (waveform, Beta(0.4, 0.4), MIXUP_HARD=True, p=0.5; same for focal×SC)
- SpecAugment (1 freq×10 + 2 time×10)
- AdamW + 2ep warmup + 23ep cosine (LinearLR + CosineAnnealingLR via SequentialLR)
- backbone `tf_efficientnet_b0.ns_jft_in1k`

---

## Files

- This audit: `user/murilo/analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md`
- Prior forward+loss audit: `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`
- Verbatim notebook: `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`
- Convenience .py extraction (gitignored): `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.py`
- Discussion summary: `user/murilo/notes/kaggle-discussion-tucker-distilled-sed-baseline.md`
- Architecture figures: `user/murilo/notes/figures/tucker-distilled-sed-arch.png`, `arch2.png`
