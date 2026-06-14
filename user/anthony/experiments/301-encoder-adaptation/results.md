# Experiment 301 Results

**Status**: COMPLETE. First run 2026-06-11 (5 s baseline, historical); **native-geometry
fix is the best result** (2026-06-13); 20-epoch ladders + held-out SSL val **finished
2026-06-13 for both encoders** (jobs 9911259 SSAMBA, 9913471 EAT + downstream chains, all
COMPLETED). The "first run" sections below are the historical 5 s baseline — read the
Update + Epoch-ladder sections first.

## Update — native-geometry fix is the best result (2026-06-13)

The first run (below) DAPT'd at the wrapper's **5 s** geometry. But SSAMBA's checkpoint is
natively **1024-frame (10 s)**, and `_waveform_to_fbank` pads a 5 s window into that
buffer — so **~half the masked MPG/MPC targets were silence** (trivially reconstructed),
diluting the objective. Scaling that 5 s objective to 10 epochs was null-to-negative
(SSAMBA frame-max +0.017→+0.003): 10× overfitting on a half-degenerate target.

**Fix: feed native-length 10 s audio** (`data.window_samples: 160000`) to fill the buffer
with real audio. EAT has no silence disease (true-5 s fills its 512 buffer), so its analog
is the *length-match* test: `target_length: 512→1024` + 10 s window = native 10 s geometry.
Eval/extract stay 5 s (the backbone is geometry-shared; the buffer is identical at train
and eval). **One epoch at native geometry beats everything, on both encoders:**

| Encoder | Readout | Frozen | 5 s ep1 | Δ | **10 s ep1** | **Δ** |
|---------|---------|--------|---------|------|--------------|-------|
| ssamba_hf | ridge (linear) | 0.5625 | 0.5504 | −0.012 | **0.5713** | **+0.009** |
| ssamba_hf | proto frame-max | 0.2335 | 0.2505 | +0.017 | **0.2606** | **+0.027** |
| eat | ridge (linear) | 0.5689 | 0.5557 | −0.013 | **0.5737** | **+0.005** |
| eat | proto frame-max | 0.2725 | 0.2828 | +0.010 | **0.2884** | **+0.016** |

**Reading it:**
- **Frame-max is the best in the experiment** for both encoders (SSAMBA +0.027, EAT +0.016),
  and **ridge FLIPS positive** on both — once the objective isn't half-reconstructing
  silence (SSAMBA) / is at native length (EAT), the linear probe stops reading DAPT as
  damage. Same checkpoints; the geometry was the bottleneck, not the SSL.
- **Cross-encoder lesson: match the encoder's native input length for DAPT.** It holds even
  for EAT, which had no silence dilution — so the mechanism is train/pretrain length-match,
  not only silence. Both encoders show the same KS signature: 10 s moved *less* (median KS
  SSAMBA 0.88, EAT 0.90 vs 5 s-ep1 0.92/0.98) but helped *more* — a smaller, more useful shift.
- Corroborating: SSAMBA 10 s `loss_mpg` 0.78 vs ~0.40 at 5 s — the 10 s objective is
  genuinely harder because the 5 s one was partly reconstructing silence.

## Epoch ladder (native 10 s, both encoders) — COMPLETE

The 20-epoch ladders (`dapt-{ssamba,eat}-native-10s-ep20`, milestone snapshots at
{3,10,20} + ep1, all from the same run) finished 2026-06-13, each with a **held-out SSL
val loss** (`data.val_frac: 0.05`, deterministic by-file 5 % holdout scored every epoch).
Full curve (frozen → ep1 → ep3 → ep10 → ep20), pooled-OOF macro AUROC on the 739 LOSO
windows, both readouts (`probe_<enc>_dapt-native-10s-ep*.json` ridge, `proto_<enc>_*.json`
frame-max):

| Encoder | Readout | Frozen | ep1 | ep3 | ep10 | ep20 |
|---------|---------|--------|-----|-----|------|------|
| ssamba_hf | ridge (linear) | 0.5625 | **0.5713** | 0.5460 | 0.5516 | 0.5529 |
| ssamba_hf | proto frame-max | 0.2335 | **0.2606** | 0.2499 | 0.2163 | 0.2024 |
| eat | ridge (linear) | 0.5689 | **0.5737** | 0.5715 | 0.5525 | 0.5509 |
| eat | proto frame-max | 0.2725 | 0.2884 | 0.2998 | **0.3050** | 0.3013 |

**The epoch curve is peak-early, and the readout sets *where* the peak is:**
- **SSAMBA peaks at ep1 under both readouts**, then erodes monotonically on proto
  (0.2606 → 0.2024) and falls back to ~frozen on ridge. One pass is the whole gain.
- **EAT splits by readout: ridge peaks ep1 (0.5737) while proto frame-max peaks ep10
  (0.3050, EAT's best in the experiment).** Same checkpoints — the two instruments
  disagree not just on the *sign* of the Δ (the original finding) but on *when to stop
  training*. This is the sharpest on-our-data form of "the linear yardstick under-reads
  masked-SSL": read off ridge you'd stop EAT at ep1 and miss its actual proto peak.
- Neither encoder shows a still-climbing curve past its peak. **The epoch axis of native
  masked-SSL DAPT is saturated** — gains on *this knob* are spent.

**KS-shift across the ladder (`ks_<enc>_dapt-native-10s-ep*.json`, median per-dim KS) — the
mechanism is direction, not magnitude:**

| Encoder | ep1 | ep3 | ep10 | ep20 |
|---------|-----|-----|------|------|
| ssamba_hf | 0.876 | 0.989 | 0.996 | 0.994 |
| eat | 0.900 | 0.674 | 0.614 | 0.605 |

SSAMBA **drifts ever further** from frozen with epochs (KS ↑) while getting *worse*
downstream — more movement in a less useful direction. EAT **relaxes back toward frozen**
(KS ↓, the data2vec EMA target stabilizing) while getting *better* — a smaller, more useful
shift. Both confirm the rep moves hugely in one pass (KS ≈ 0.88–0.90 at ep1); duration buys
drift, not quality.

**Held-out SSL val — no overfitting at either encoder (the train/val gap detector):**
- **SSAMBA**: train `mpg` ≈ `val_mpg` within ~0.002 the whole way (ep19: 0.5188 / 0.5207);
  the objective plateaus ~0.52 by ep5–6; `val_acc` creeps to 0.41. No gap.
- **EAT**: `val_loss` ≈ train `d2v_loss` within ~0.0001 throughout; `d2v_loss` dips ep1
  (0.00342) → rises ep8 (0.00428) → falls ep19 (0.00381) — the non-stationary EMA-target
  wobble, not memorization.

**Methodological correction — do NOT read downstream usefulness off the SSL objective.**
EAT is the proof of decoupling: its SSL `d2v_loss` was *flat-to-rising* from ep1→ep10 while
the downstream proto readout *climbed to its best at ep10*. The training-side curve pointed
the opposite way from the thing we care about. The held-out SSL val's only load-bearing job
here is the overfitting check (train ≈ val → no memorization, which it passed); it is **not**
a proxy for downstream quality. The conclusions above rest on the proto frame-max curve, the
readout the literature prescribes — never the objective loss. This is the same
training-metric ≠ readout-metric trap the whole experiment exists to flag.

**Verdict.** Native-geometry DAPT gives a small, consistent gain that lands at ep1 (SSAMBA;
EAT ridge) or by ~ep10 (EAT proto), and the epoch axis is then saturated — more epochs buy
drift, not quality. The gains-exhausted claim is scoped to *this knob* (epochs of
unsupervised masked-SSL DAPT); see Deferred / next for the levers that are still open.

## Headline (first run — 5 s geometry, historical)

Native-objective continued pretraining (DAPT) on the unlabeled soundscape pool
**helps under the MAE-appropriate prototypical readout but hurts under the linear
ridge probe** — the *same* full-DAPT checkpoints give opposite-sign Δ depending on the
readout. This is direct, on-our-data evidence for the Bird-MAE result that linear
probing under-reads masked-SSL features: the exp 300 ridge yardstick is the wrong
instrument for measuring domain SSL, and the gain only appears under the prototypical
frame-max readout the literature prescribes.

Method (vs the shelved generic latent-bootstrap `dapt.py`, which was degenerate):
- **SSAMBA**: native masked-spectrogram SSL — MPG (per-patch normalized MSE) + MPC
  (cross-sample NCE), the loss SSAMBA was pretrained with (logic from exp 103),
  continued from `ssamba_small_400.pth` at the wrapper's exact 16 kHz / 128-mel Kaldi /
  5 s / 16×16 geometry.
- **EAT**: native data2vec-2.0 self-distillation — EMA teacher averaging all-12-block
  FFN targets, student encodes visible patches, lightweight decoder reconstructs masked
  targets (frame-level `d2v_loss` MSE). From-scratch PyTorch port (no fairseq).
- Both `peft: {none|lora}` arms. 1 epoch, ~666 (SSAMBA) / 888 (EAT) steps.

## Frozen → adapted Δ macro AUROC (LOSO, pooled-OOF, 739 labeled windows)

Frozen baseline re-probed here (matches exp 300: EAT 0.569, SSAMBA 0.563 ridge).
`full` = `dapt-native-ep1` (state_dict), `lora` = `dapt-native-lora-ep1` (adapter).
Three readouts, all the identical pooled-OOF macro ROC-AUC, differing only in the
classifier: ridge (linear), proto-A (clip-level prototype cosine on the pooled vector),
proto-B (frame-level max prototype cosine on the patch grid — the Bird-MAE gap-closer).

| Encoder | Readout | Frozen | Full | Δ Full | LoRA | Δ LoRA |
|---------|---------|--------|------|--------|------|--------|
| eat | ridge (linear) | 0.5689 | 0.5557 | **−0.0132** | 0.5689 | +0.0000 |
| eat | proto-A (clip) | 0.2846 | 0.2933 | **+0.0086** | 0.2846 | +0.0000 |
| eat | proto-B (frame-max) | 0.2725 | 0.2828 | **+0.0103** | 0.2725 | +0.0000 |
| ssamba_hf | ridge (linear) | 0.5625 | 0.5504 | **−0.0122** | 0.5624 | −0.0001 |
| ssamba_hf | proto-A (clip) | 0.2916 | 0.2885 | −0.0031 | 0.2918 | +0.0003 |
| ssamba_hf | proto-B (frame-max) | 0.2335 | 0.2505 | **+0.0170** | 0.2165 | −0.0170 |

**Reading it:**
- **Ridge Δ is negative for full DAPT on both encoders** (−0.013 EAT, −0.012 SSAMBA):
  full fine-tuning on the native masked objective *disrupts linear separability*.
- **Proto frame-max Δ is positive for full DAPT on both** (+0.010 EAT, +0.017 SSAMBA):
  under the readout that fits masked-SSL features, the same adaptation *helps*. proto-A
  (clip) agrees in sign for EAT (+0.009); for SSAMBA the clip readout is flat (−0.003)
  while the frame-level readout is clearly positive — consistent with Bird-MAE's point
  that the *patch-level* readout is the one that surfaces the gain.
- The sign flip between ridge and proto on identical checkpoints is the result: **the
  ridge yardstick cannot measure native domain-SSL; it would have reported "DAPT hurts."**

## KS-shift (frozen vs adapted pooled embeddings)

`artifacts/ks_<encoder>_<variant>.json`. Higher = adaptation moved the representation
more (per-dim KS, Bonferroni-significant fraction).

| Encoder | Variant | median KS | % dims shifted |
|---------|---------|-----------|----------------|
| eat | full (dapt-native-ep1) | 0.982 | 99.6% |
| eat | **lora** | **0.000** | **0.0%** |
| ssamba_hf | full (dapt-native-ep1) | 0.922 | 99.0% |
| ssamba_hf | lora | 0.816 | 98.4% |

**The LoRA arms are the cautionary finding:**
- **EAT-LoRA moved the representation by exactly zero** (KS 0.0, Δ 0.0000 everywhere).
  In data2vec the trainable params are the LoRA adapters *and* the decoder; the decoder
  (discarded at save) absorbed the entire objective while the LoRA B-matrices never left
  their zero init, so the merged adapter is an exact no-op. **Parameter-efficient
  latent-bootstrap with a discarded decoder doesn't transfer to a frozen readout.**
- **SSAMBA-LoRA moved a lot (KS 0.82) but unhelpfully** (proto-B −0.017): its MPG/MPC
  target is the raw mel patch (no decoder shortcut), so the adapters *had* to move — but
  the movement didn't help the frozen prototypical readout.
- Net: for this objective family, the useful adaptation lives in the backbone (full DAPT),
  not in low-rank adapters paired with a throwaway head.

## EAT-port validation gate (numerical stability + directional learning)

Before any full run, the from-scratch EAT data2vec port was gated on a GPU node
(`sbatch/validate-eat.sbatch`, `tests/test_eat_dapt.py`) — **10/10 passed**:
- *Stability*: `AltBlock` FFN-target hand-recompute matches returned `t` (the teacher
  target's correctness anchor); target shape/finite; anti-collapse variance
  (target_var > 0.1, pred_var > 0.01); **loss > 1e-2 at init** (vs the degenerate
  bootstrap's ~7e-4 — masks genuinely hidden); finite grads to the blocks; state_dict
  round-trip (after the EATModel-prefix fix below); LoRA merge.
- *Directional*: tiny-overfit halves the loss; beats predict-the-mean and predict-zeros;
  representation moves but stays bounded.

SSAMBA gate (`validate-ssamba.sbatch`, 6/6): resolved the pos_embed geometry risk
(checkpoint is natively 1024-frame/512-patch; the wrapper feeds matching fbank, so
`num_patches+1 == pos_embed` — no interpolation needed) and confirmed gradients flow
through the Mamba CUDA kernel.

Bug caught at re-extract (the gate initially tested the wrong load target): EAT
full-DAPT must save the **outer `EATModel`** state_dict (`model.`-prefixed keys),
because exp 300's `_apply_overlay` does `wrapper.model.load_state_dict(...)` strict —
saving the inner `EAT` module fails. Fixed in `eat_dapt.py`; the round-trip test now
mirrors `_apply_overlay` exactly.

## Caveats

- Small eval set (739 LOSO windows): the deltas (0.01–0.017) are small. They are not
  flat, and the *sign* is consistent across both encoders and both proto strategies for
  full DAPT — but absolute magnitudes warrant more epochs / windows-per-file before any
  strong claim (the cheap first knob; this run sees ~8% of the pool, 1 epoch).
- Prototypical absolute AUROC (0.23–0.29) is far below ridge (0.56): the prototype-cosine
  readout is a weaker absolute classifier on these encoders. The Δ *direction* under a
  fixed readout is the apples-to-apples quantity; the readout disagreement on identical
  checkpoints is the finding, not the absolute proto level.
- The generic latent-bootstrap baseline (`dapt-ep1`) remains in the cache as the
  "homemade objective" comparison point (it gave broad KS shift but null/noise Δ).

## Reproduce

```bash
# gates (GPU) — go/no-go
sbatch sbatch/validate-eat.sbatch ; sbatch sbatch/validate-ssamba.sbatch
# DAPT (GPU, ~5 min each)
CMD=dapt-native CONFIG=configs/dapt-ssamba-native.yaml      sbatch sbatch/dapt.sbatch
CMD=dapt-native CONFIG=configs/dapt-ssamba-native-lora.yaml sbatch sbatch/dapt.sbatch
CMD=dapt-native CONFIG=configs/dapt-eat-native.yaml         sbatch sbatch/dapt.sbatch
CMD=dapt-native CONFIG=configs/dapt-eat-native-lora.yaml    sbatch sbatch/dapt.sbatch
# re-extract pooled+patch (GPU), then probe ridge+proto (CPU), then KS (CPU)
ENCODER=<e> VARIANT=<v> STATE_DICT=<…>|ADAPTER=<…> PATCH=1 sbatch sbatch/reextract.sbatch
VARIANT=<v> ENCODERS=eat,ssamba_hf [READOUT=proto] sbatch sbatch/probe.sbatch
ENCODER=<e> VARIANT=<v> sbatch sbatch/ksshift.sbatch
```

## Deferred / next

- ~~More epochs (the cheap knob given small Δ).~~ **DONE and exhausted** — the 20-epoch
  ladders above are peak-early (SSAMBA ep1, EAT proto ep10) with no still-climbing curve;
  KS / SSL-val both show the rep moves fully in one pass. Don't spend more compute here.
- **Bigger / cleaner eval is the real bottleneck.** Δ is 0.01–0.03 on 739 LOSO windows —
  near the measurement floor. "No further gain on the epoch knob" ≠ "no gain possible";
  more labeled windows would be needed to resolve any of the levers below.
- **Objective-direction levers (untested):** EAT `cls_loss` (utterance term) + exact
  `Decoder2d`; block (vs random) masking. Since direction (not duration) is what moves the
  readout, these are the only DAPT knobs that could plausibly beat the geometry-fix peak.
- **Supervised track (untested):** the LoRA arms were *mechanical* no-ops (zero-init B /
  data2vec decoder sink), not evidence the supervised signal is tapped out. LoRA-done-right
  (keep the adapted rep — EMA-distill into the LoRA path, or eval the decoder-free encoder
  under proto directly) or a small supervised head on the strong + weak-focal labels is a
  different gain source from unsupervised DAPT.
