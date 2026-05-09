# v5j vs Tucker `bc2026-distilled-sed` — full pipeline comparison

Side-by-side audit of every load-bearing pipeline component, to scope what's
actually different between our 0.906 single-fold v5j and Tucker's
publicly-quantified 0.91-without-PP / 0.917-with-PP recipe.

**Captured 2026-05-09.** Sources:
- v5j: `user/murilo/experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml`,
  `user/murilo/src/birdclef/lightning/sed_finetune.py`,
  `user/murilo/src/birdclef/heads/sed_head.py`
- Tucker: `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb`
  (cells 4, 10, 12, 14)
- Code audit: `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`

## Color key for "Risk of blunting Tucker's +0.02 KD lift"

- 🟢 = same as Tucker, no concern
- 🟡 = different but unlikely to interact with KD specifically
- 🔴 = different in a way that could plausibly blunt the KD lift in our codebase
- ⭐ = the KD lever itself (the variable Phase 1 is testing)

## 1. Data and splits

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Focal recordings | ~35,549 (BirdCLEF 2026 train.csv) | Same competition data | 🟢 |
| Labeled soundscapes | ~1,478 segments | Same | 🟢 |
| Train/val split | **90/10 random + 8 held-out SS files** (`val_fraction: 0.1`, `soundscape_val_files: 8`, `seed: 42`) | **5-fold StratifiedKFold by primary_label + SS filename split** | 🟡 — different validation regime, but Phase 1 only reads LB anyway |
| Unlabeled soundscapes | Not used | Not used | 🟢 |
| Per-class min sample upsampling | Default (none) | `MIN_SAMPLE = 20` per class | 🟡 |

**Reading**: split structure differs but Phase 1 doesn't depend on val
dynamics — we read LB. The MIN_SAMPLE=20 upsampling is a small data-balance
detail Tucker has on by default; we don't. Probably not load-bearing for
KD lift.

## 2. Sampler / batch composition

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Sources per batch | Focal + labeled SS combined into one ConcatDataset | Two streams via `MixSamp` batch sampler | 🟡 |
| Per-batch composition | Implicit (proportional to each set's size) | Explicit: `SHARES = {"focal": 0.9, "sc": 0.1}` (~58 focal + 6 SS per batch=64) | 🟡 |
| Per-source sample weight | Equal | `SOURCE_WEIGHTS = {"focal": 1.0, "sc": 1.0}` (also equal in practice) | 🟢 |
| Class-balanced sampling | None | None | 🟢 |
| `num_workers` | 8 | 0 (Kaggle constraint) | 🟢 (just speed) |

**Reading**: Tucker's explicit 90%/10% batch composition keeps SS represented
in every batch. Our concat-then-shuffle gives SS a smaller per-batch
expected count by recording-count ratio (~1478 / (35549 + 1478) = 4%, not
10%). For *KD* specifically this matters because the Perch teacher embeds
the *exact training waveforms it sees* — if SS samples are rare per batch,
the dist_loss on SS-domain inputs is rare too. Could blunt the KD lift on
the soundscape side specifically. **🟡 → 🔴 if the lift fails.**

## 3. Mel spectrogram pipeline 🔴 LARGEST DIVERGENCE

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Library | **NNAudio** (`NNAudioMelSpectrogram`, GPU) | **torchaudio** (`torchaudio.transforms.MelSpectrogram`) | 🔴 |
| Sample rate | 32,000 Hz | 32,000 Hz | 🟢 |
| `n_mels` | 256 | 256 | 🟢 |
| `n_fft` | 2,048 | 2,048 | 🟢 |
| **`hop_length`** | **313** | **512** | 🔴 |
| **Frames per 5s** | **~512** (160,000 / 313) | **~313** (160,000 / 512) | 🔴 |
| `f_min` | (default — 0?) | 20 | 🟡 |
| `f_max` | `null` (Nyquist = 16 kHz) | 16,000 | 🟢 |
| `power` | (NNAudio default) | 2.0 | 🟡 |
| Amplitude-to-dB | `top_db=80.0` | `top_db=80` | 🟢 |
| Post-norm | NNAudio's standardize + min-max baked in | Per-sample explicit `(mel - mel.mean()) / (mel.std() + 1e-6)` after AmpToDB | 🟡 |

**Reading**: This is the largest pipeline divergence and the most likely
place a teacher-student distribution mismatch could blunt the KD lift.
Specifically:

1. **`hop_length` 313 vs 512** changes our temporal resolution by ~1.6×.
   Our backbone sees a feature map with ~512 time frames per 5s; Tucker's
   sees ~313. This affects:
   - The GAP-pooled (B, C) feature distribution — averaging over 512 vs
     313 frames produces a different distribution.
   - Per-frame logit magnitudes downstream in the SED head.
   - Whether the projection layer needs to be sized differently.
2. **NNAudio vs torchaudio MelSpectrogram** can produce numerically
   different mel matrices even with identical parameters (different STFT
   windowing defaults, different filter bank construction). This is the
   "mel mismatch" Tucker flagged as a dist_loss floor cause — and we're
   adding *another* mismatch on top by using a different mel library.
3. **Per-sample mean/std normalization** in Tucker happens after the mel
   computation; ours is baked into NNAudio's transform. Functionally
   similar but the order of operations and exactly-which-stats differ.

**Phase 0 mitigation candidates** (in order of effort):
- (i) Switch student mel to torchaudio with `hop_length=512` to fully match
  Tucker's input distribution. ~1 day of refactor (touches dataset,
  training, ONNX export, inference kernel).
- (ii) Numerically compare NNAudio output to torchaudio output on a fixed
  100-sample batch with our current hyperparameters; if difference is
  small (e.g., per-pixel max abs diff < 0.1 in dB), proceed with NNAudio.
- (iii) Accept the higher dist_loss floor and let the projection head do
  the work to bridge the gap. Risk: lift is blunted but not zero.

## 4. Backbone

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Architecture | **HGNetV2-B0** (`hgnetv2_b0.ssld_stage2_ft_in1k`, ~3.1M params) | **EfficientNet-B0 NS-JFT** (`tf_efficientnet_b0.ns_jft_in1k`, ~5.3M params) | 🟡 |
| Input channels | 1 (mono mel) | 1 | 🟢 |
| Pretrain | ImageNet (SSLD stage 2 ft) | ImageNet → NS-JFT (Noisy Student on JFT-300M) | 🟡 |
| Final feature dim | 1024 (per `embed_dim`) | 1280 (B0 standard) | 🟡 |
| `num_classes=0, global_pool=""` | (we control this in code) | Same | 🟢 |
| `drop_path_rate` | (default 0?) | 0.1 | 🟡 |

**Reading**: Different backbone architecture, different feature dim. This
is intentionally held constant for Phase 1's single-variable discipline.
Phase 3 candidate: swap to NS-JFT EffNet-B0 to test Tucker's exact backbone.

## 5. SED head 🔴 STRUCTURAL DIVERGENCE

| Component | v5j (LSEHead) | Tucker (BirdSEDModel head) | Risk |
|---|---|---|---|
| Architecture family | LSE pool over time | Attention head | 🔴 |
| Frequency pool | `freq_pool: "mean"` (simple average) | `GeMFreqPool(p=3.0, learnable, p.clamp(min=1.0))` | 🔴 |
| Hidden dim | 1024 | 512 | 🟡 |
| Dropout | 0.5 (post) | 0.25 pre, 0.5 post — `Dropout → Linear → ReLU(inplace) → Dropout` | 🟢 (functionally equivalent) |
| Frame logits | `Linear(1024, 234)` applied per-frame | `Conv1d(512, 234, k=1)` | 🟢 |
| Attention branch | None (LSE pools logits) | `Conv1d(512, 234, k=1, bias=True)`, **xavier_uniform_ init**, bias=0 | 🔴 |
| Clipwise aggregation | `lse_pool(framewise_logits, T=1.0)` | `sum(softmax(tanh(att)) * framewise_logits, dim=time)` ← **logit space** | 🔴 |
| Framewise output | Permuted to `(B, num_classes, T)` | Same | 🟢 |

**Reading**: v5j and Tucker use **fundamentally different aggregation
strategies**:

- v5j: `clip = (1/τ) · log(Σ_t exp(τ · z_t))` — soft-max-over-time of
  per-frame logits. Frame-wise differentiable approximation of
  `max_t(z_t)`. No explicit attention.
- Tucker: `clip = Σ_t softmax(tanh(att_t)) · logit_t` — explicit learned
  attention weighting over per-frame logits. Logit-space aggregation
  (per the AttHead-aggregation memory entry, this is the *correct*
  Nikita-faithful version; our `AttHead` got this wrong).

For Phase 1 (KD test on v5j with LSEHead held fixed) this is intentional.
But it's worth flagging that **Tucker's published lift is on his head, not
ours**. Possible Tucker's KD interacts with attention specifically — the
backbone learns Perch-aligned features, the attention learns to focus on
where those features are most discriminative per-frame. LSEHead just
pools-by-magnitude regardless of where the signal is. KD might still help
LSEHead (it's a representation lever, not a localization lever) but the
size of the lift could differ.

**Phase 3+ candidate**: re-run KD with the bug-fixed AttHead (logit-space
aggregation) to test Tucker's full head architecture.

## 6. Loss

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Classification loss | `BCE(clipwise_logits, labels)` only (no framewise term) | `0.5 * BCE(clip_logits, lb) + 0.5 * BCE(frame_max_logits, lb)` | 🔴 |
| `clipwise_weight` | 1.0 | 0.5 | 🔴 |
| `framewise_weight` | **0.0** | 0.5 | 🔴 |
| Loss reduction | `.mean()` after sample/class masks | Same (sample weight + class mask + reduction='none' → mean) | 🟢 |
| Distillation loss ⭐ | **None** | `F.mse_loss(distill_emb, perch_emb) * alpha_distill` (alpha=1.0) | ⭐ KD lever |
| Loss combination ⭐ | n/a | `loss = cls_loss + alpha_distill * distill_loss` | ⭐ |
| Stop-gradient on cls branch ⭐ | n/a | `feat.detach()` when distill is on | ⭐ |

**Reading**: Tucker uses **dual-supervision** (clip + framewise-max BCE,
50/50). v5j is clip-only. This matters for KD because:

- The framewise-max term gives gradient pressure on per-frame logit
  magnitudes everywhere in the temporal axis.
- Combined with attention, this is what gives Tucker's attention real
  gradient pressure to localize (the v6k uniform-attention finding from
  04-26).
- LSEHead doesn't have an attention branch, so the framewise-max term is
  redundant with LSE-pooling anyway. Adding it to v5j may not help.

**Phase 1 holds v5j's clip-only loss.** The KD branch adds MSE on top.

## 7. MixUp

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Domain | **Spectrogram** (post-mel, GPU) | **Waveform** (pre-mel, dataloader) | 🟡 |
| Distribution of `lam` | `Beta(α=1.0, α=1.0)` per-sample (uniform on [0, 1], max-flipped to [0.5, 1]) | `Beta(α=0.4, α=0.4)` per-sample (bimodal toward 0/1) | 🟡 |
| Wave/spec combination | `lam·m1 + (1-lam)·m2` — Zhang interpolation | `lam·w1 + (1-lam)·w2` — Zhang interpolation (NOT Sydorskyi's `(w1+w2)/2`) | 🟡 |
| **Target combination** | **`lam·y1 + (1-lam)·y2`** (interpolated) | **`max(y1, y2)`** (`MIXUP_HARD = True`) | 🔴 |
| Theta cap | 0.8 (post-mix `≥ 0.8 → 1.0`) | None | 🟡 |
| Application probability | 1.0 (post warmup) | 0.5 | 🟡 |
| Warmup | 5 epochs disabled | None | 🟡 |
| Per-sample / per-batch | per-sample | per-sample | 🟢 |
| Focal × labeled-SS mixup | None | `USE_FOCAL_SC_MIXUP=True`, p=0.5 (on top of focal-focal) | 🟡 |

**Reading**: v5j and Tucker differ on **target combination** — Tucker uses
the same `max(y1, y2)` recipe we tested in exp 030 (sydorskyi mixup) and
found null at single-fold resolution on HGNet. Tucker has it on by
default. Per the 04-26 simulation, our v5j's interpolated targets produce
80% sub-1 active values; Tucker's hard-max produces 100% at 1.0. The test
distribution (multi-label binary) matches Tucker's training distribution,
not ours.

**However**: Tucker's KD lift is +0.02 with hard mixup; we don't know the
counterfactual ("Tucker without hard mixup"). Could be that hard mixup
is doing some of the work he attributes to KD. We can't disentangle from
public information.

**Phase 1 holds v5j's mixup recipe.** This is intentional — single-variable
discipline. Phase 3+ candidate: KD on top of `v5j_sydorskyi` (which
already has hard-max targets). We have those weights from exp 030 (LB
0.892).

## 8. Other augmentations

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| SpecAugment | **None** | `FREQ_MASK_PARAM=10, NUM_FREQ_MASKS=1, TIME_MASK_PARAM=10, NUM_TIME_MASKS=2` | 🟡 |
| RandomFiltering | None | None (Sydorskyi has it; Tucker doesn't) | 🟢 |
| Background noise injection | None | None | 🟢 |
| Waveform gain jitter | None | `apply_aug` with prob=0.5: gain ~ U(-6, 6) dB | 🟡 |
| Waveform noise injection | None | `apply_aug`: SNR ~ U(10, 30) dB Gaussian | 🟡 |
| Per-sample peak-normalize | None | None (Sydorskyi has it; Tucker doesn't) | 🟢 |

**Reading**: Tucker has more augmentation than v5j — SpecAugment + waveform
gain/noise jitter. None of these are KD-specific; they help generalization.
But Tucker himself flagged that SpecAugment-on-student-but-not-teacher
contributes to the dist_loss floor — so the augmentation actually
*hurts* the dist_loss curve while presumably helping the cls_loss curve.

For our Phase 1: v5j has no SpecAugment. The dist_loss floor should be
*lower* than Tucker's (since student and teacher see closer-to-identical
mels). This is a free indication that the teacher hookup is working
correctly.

## 9. Optimizer / schedule / training duration

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Optimizer | AdamW | AdamW | 🟢 |
| Initial LR | 5e-4 | 5e-4 | 🟢 |
| Weight decay | 1e-4 | 1e-4 | 🟢 |
| LR schedule | OneCycleLR (`div=25, final_div=4`) | LinearLR warmup (2 ep, start=1/25) → CosineAnnealingLR (T_max=23ep, eta_min=1e-6) | 🟡 |
| Warmup | 5 epochs (built into OneCycle) | 2 epochs (separate Linear) | 🟡 |
| Total epochs | 20 | 25 | 🟡 |
| Batch size | 64 | 64 (16 in Kaggle DEBUG mode) | 🟢 |
| Precision | 16-mixed | 16-mixed | 🟢 |
| Gradient clipping | None (`grad_clip_val: 0`) | `clip_grad_norm_(_, 1.0)` | 🟡 |
| `accumulate_grad_batches` | 1 | 1 | 🟢 |
| Mixup warmup vs schedule warmup | 5 epochs (mixup off during warmup) | 0 (always on) | 🟡 |
| Determinism | `seed_everything(42, workers=True) + Trainer(deterministic=True)` | `seed=42, cudnn.deterministic=False, cudnn.benchmark=True` | 🟡 |

**Reading**: Schedule shape and total epochs differ. Tucker's 25 epochs +
cosine is the more standard choice; v5j's OneCycleLR is unusual but
worked. Probably not the lever. Our determinism is stricter than Tucker's;
this may or may not interact with KD reproducibility.

## 10. Validation

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Primary monitor | `val/auroc` (combined focal + SS macro-AUROC) | `non_s22_macro` AUC (excluding dominant site) AND `macro_auc_all` | 🟡 |
| Save top-k | 1 (best only) | Saves both `best_state_ns22` and `best_state_macro`, picks `best_state_macro` for inference | 🟡 |
| Save last | True (mirrors best per exp 030 finding) | n/a | 🟡 |
| `dual_validation` | False (per exp 030 v2 fix) | n/a (single val set) | 🟢 |
| Best-by-val epoch typical | 14-19 | ~17-25 | 🟡 |

**Reading**: Tucker tracks two checkpoint candidates and explicitly notes
in his thread that `best_macro` ≠ `best_ns22` and there's a real LB
difference. Our `save_top_k=1` gives us only one; we don't have the option
to compare. Worth the `save_top_k=-1` ablation eventually but not in scope
for Phase 1.

## 11. Inference / postprocessing

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Window length | 5s | 5s | 🟢 |
| Aggregation across folds | 4-fold simple average (in `v5j-4fold-pp2`) | 5-fold simple average | 🟡 |
| TTA | ±2.5s shifts, weights 50/25/25 | None mentioned | 🟡 |
| Temporal smoothing | Gaussian (in pp2) | 5-tap Gaussian (in v2 of public notebook) | 🟢 |
| TopN | TopN N=1 | None mentioned (Sydorskyi has it; Tucker doesn't seem to) | 🟡 |
| Site/hour priors | Yes (+0.003 in pp2) | "Site/hour priors + Gaussian smooth" → 0.93+ | 🟡 |
| **Inference blend** | `blend_clip=1.0, blend_frame=0.0` (clip-only) | **`0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)`** | 🔴 |
| Sigmoid timing | Per-segment | Per-segment | 🟢 |

**Reading**: Tucker's blend is the structural fix to v6k's uniform
attention (D1 verdict). For LSEHead it's less critical because LSEHead
already approximates `max_t`. **Phase 1 inference**: v5j-default (clip-only
LSE output, our existing pp2 inference path). Don't add Tucker's blend —
that's Phase 3+ if we ever swap to AttHead.

## 12. Knowledge distillation ⭐

| Component | v5j (no KD) | Tucker | Phase 1 |
|---|---|---|---|
| Teacher | n/a | Frozen Perch v2 (ONNX, no-DFT variant) | Same as Tucker (Phase 0 vendors his ONNX) |
| Teacher input | n/a | 5s waveforms (B, 160000) | Same |
| Teacher output | n/a | (B, 1536) clip embeddings | Same |
| Distill loss | n/a | `F.mse_loss(distill_emb, perch_emb)` | Same |
| Distillation head | n/a | `Linear(backbone_dim, 1536)` after GAP over `dim=[2,3]` | Same |
| `alpha_distill` | n/a | 1.0 | 1.0 |
| Stop-gradient on cls | n/a | `feat.detach()` when KD is on | Same |
| Teacher inference timing | n/a | Per-batch in training loop, on-the-fly | Same |
| Teacher precomputed embeddings | n/a | Mentioned as a speed-up Tucker recommends but doesn't use himself | Defer to Phase 2/3 |

## 13. External data

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Pretraining corpus | ImageNet (SSLD) for backbone | ImageNet → NS-JFT for backbone; Perch v2 = teacher (trained on 14,795 species, multi-corpus) | 🟡 |
| 2025 BirdCLEF data | None | None | 🟢 |
| BG noise pool (other comps) | None | None | 🟢 |

## 14. Output / submission format

| Component | v5j | Tucker | Risk |
|---|---|---|---|
| Submission stride | 5s non-overlap | 5s non-overlap | 🟢 |
| Output | `(N_segments, 234)` probabilities | Same | 🟢 |
| Kernel runtime | ~30 min | ~50-60 min (5-fold) | 🟢 |

---

## Summary — components that could blunt the KD lift in Phase 1

Ranked by likelihood × magnitude of impact:

| # | Difference | Why it matters | Phase 1 mitigation |
|---|---|---|---|
| 1 | **Mel pipeline (NNAudio vs torchaudio + hop_length 313 vs 512)** | Student feeds different mel distribution to backbone than what Perch was trained on. Backbone has to learn a harder projection. Could raise dist_loss floor and reduce backbone's capacity to match Perch's representation. | **Phase 0 task**: numerically compare NNAudio mel vs torchaudio mel on a 100-sample fixed batch. If max abs diff > 0.5 dB, consider switching student mel to torchaudio. Also flag the hop_length difference — likely needs torchaudio with `hop_length=512` to match Tucker. |
| 2 | **Head architecture (LSEHead vs AttHead)** | Tucker's KD lift is published on AttHead with logit-space aggregation + dual supervision. LSEHead may benefit less from a Perch-aligned representation because it pools by magnitude not localization. | Accept for Phase 1 (single-variable discipline). If REJECT, Phase 3 candidate: KD on bug-fixed AttHead. |
| 3 | **Mixup target combination** | Tucker has hard-max default; v5j has interpolated. Possible Tucker's lift compounds with hard targets. | Accept for Phase 1. Phase 3 candidate: KD + hard mixup (use exp 030's v5j_sydorskyi_v2 as the base). |
| 4 | **Dual-supervision loss** | Tucker's `0.5 BCE(clip) + 0.5 BCE(frame_max)`. v5j is clip-only. May give the head more gradient, especially in interaction with KD. | Accept for Phase 1 (LSEHead doesn't need dual). Phase 3+ candidate. |
| 5 | **No SpecAugment** | Tucker has it; v5j doesn't. SpecAugment hurts dist_loss floor (Tucker confirmed) but helps generalization. | Phase 1 *advantage*: lower dist_loss floor → free indicator that teacher hookup is working. |
| 6 | **Schedule / epochs** | v5j is 20-epoch OneCycle; Tucker is 25-epoch warmup+cosine. Different shape. | Probably not load-bearing. Accept. |
| 7 | **Determinism** | We're stricter (`Trainer(deterministic=True)`). | Probably not load-bearing. |

**Priority ranking for Phase 0 diagnostic checks:**
- (a) Mel-pipeline numerical equivalence check — required before Phase 1.
- (b) Hop-length 313 vs 512 — should we switch to 512 for Phase 1 to
  match Tucker exactly? If the temporal resolution change disrupts the
  backbone's existing learned representation, it's a poor first KD test.
  If it's transparent, no change needed.
- (c) Confirm Phase 1 success criteria are read against the right
  baseline — see proposal correction.

## Reproducing v5j-original (0.906) end-to-end — what's known

User concern: can we reproduce v5j-original 0.906 LB end-to-end (training
+ Kaggle inference) today? **Per exp 030 evidence:**

| Step | Status |
|---|---|
| Inference reproduces v5j-original | ✅ Confirmed by `v5j-resubmit` LB 0.906 (exp 030, 04-28). Same checkpoint + today's submission.py = same LB. |
| Re-training v5j with same config | ⚠️ `v5j_control_v2` (deterministic-baseline retrain of v5j config in exp 030) lands at LB **0.896**, not 0.906. **0.010 LB gap is training-side.** |
| Root cause of 0.010 gap | Best-by-val plateau is at ep 14-15 in new runs vs ep 19 in v5j-original. With `save_top_k=1`, ep 19 is never serialized. Reasonable hypothesis: codebase drift between exp 019 (v5j-original) and exp 030, OR seed-sensitivity of when val/auroc peaks. |
| `save_top_k=-1` retrain to test "ep 19 weights = 0.906?" | Deferred. Not blocking Phase 1 if we read against v5j_control_v2 baseline. |

**Implication for Phase 1 baseline**: don't compare to v5j-original 0.906;
compare to **v5j_control_v2 = 0.896** (the structurally-identical no-KD
single-fold seed=42 run). Add KD on top, submit, read the delta.

## Cross-references

- Code audit: `user/murilo/analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md`
- Proposal: `proposal.md`
- Exp 030 reproducibility findings: `030-sydorskyi-mixup/results.md`
  (v5j-resubmit verdict, v5j_control_v2 0.896 LB)
- v5j config: `019-sed-ablations/config/hgnet_b0_sed_v5j.yaml`
- v5j_control_v2 config: `030-sydorskyi-mixup/configs/v5j_control.yaml`
- Tucker notebook: `vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/`
