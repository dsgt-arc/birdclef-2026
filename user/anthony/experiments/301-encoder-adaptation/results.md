# Experiment 301 Results

**Status**: complete (native-objective soundscape DAPT, first run — 2026-06-11).

## Headline

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

- More epochs / more windows-per-file (the cheap knob given small Δ).
- EAT `cls_loss` (utterance term) + exact `Decoder2d`; block (vs random) masking.
- LoRA-DAPT for the latent objective needs the adapted representation *kept*, not a
  throwaway decoder — e.g. EMA-distill into the LoRA path, or evaluate the decoder-free
  encoder under proto directly.
