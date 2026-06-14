# Experiment 301: Encoder Adaptation

## What this experiment does

Adapts EAT/SSAMBA-HF to the bird/soundscape domain via **each encoder's native SSL
objective** (continued pretraining / DAPT on the unlabeled soundscape pool) and reports
the frozen→adapted Δ under both exp 300's ridge probe and an MAE-appropriate
**prototypical** readout. Feeds §7 of the working note. It is a thin spoke — it never
reimplements extraction or probing, it shells out to `birdclef-300`.

- **SSAMBA**: native masked-spectrogram MPG (MSE) + MPC (NCE), logic reused from exp 103,
  continued from `ssamba_small_400.pth` at the wrapper's exact eval geometry.
- **EAT**: native data2vec-2.0 self-distillation, a from-scratch PyTorch port (no fairseq)
  over the HF EAT ViT — gated by `tests/test_eat_dapt.py` (numerical stability +
  directional learning) before any full run.
- The shelved generic pooled latent-bootstrap (`dapt.py`) is kept only as the
  "homemade objective" baseline (it was degenerate; see `results.md`).
- **Supervised-focal LoRA** (all three encoders) is the other track, deferred.

## Hub/spoke contract

301 writes one of:
- **LoRA** → a PEFT adapter dir (`<outdir>/adapter`)
- **DAPT** → a full state_dict (`<outdir>/state_dict.pt`)

exp 300 consumes them via overlay flags it already supports:
```
birdclef-300 extract --encoder eat --variant lora-r8  --adapter    <outdir>/adapter
birdclef-300 extract --encoder eat --variant dapt-ep1 --state-dict <outdir>/state_dict.pt
```
All four torch wrappers expose `.model`, so the overlay (merge LoRA / load
state_dict) is encoder-agnostic.

## Running

```bash
cd user/anthony/experiments/301-encoder-adaptation

# 0. Smoke gate (cheap, ~5-10 min; builds the venv once, validates the never-run
#    paths before committing full jobs). Fault-tolerant: all encoders run even if
#    one fails, so a bad target_modules / dep is obvious in one job.
sbatch sbatch/smoke.sbatch        # 3x LoRA --max-steps 2 (venv + target_modules + grad fwd)
sbatch sbatch/smoke-dapt.sbatch   # 2x DAPT --max-steps 2 (SoundscapeDataset + EMA + state_dict)

# 1. Adapt (GPU)
CONFIG=configs/dapt-eat.yaml    sbatch sbatch/dapt.sbatch     # also dapt-ssamba (this push)
CONFIG=configs/lora-eat.yaml    sbatch sbatch/lora.sbatch     # also lora-beats, lora-ssamba (deferred)

# 2. Re-extract adapted embeddings into the shared exp 300 cache (GPU)
ENCODER=eat VARIANT=lora-r8 ADAPTER=artifacts/checkpoints/lora-eat/adapter \
    sbatch sbatch/reextract.sbatch
ENCODER=eat VARIANT=dapt-ep1 STATE_DICT=artifacts/checkpoints/dapt-eat/state_dict.pt \
    sbatch sbatch/reextract.sbatch

# 3. Probe adapted variants (CPU) and KS-shift diagnostic (CPU)
VARIANT=lora-r8 ENCODERS=beats,eat,ssamba_hf sbatch sbatch/probe.sbatch
ENCODER=eat VARIANT=lora-r8 sbatch sbatch/ksshift.sbatch
```

Venv: `~/scratch/birdclef-2026/.venv-301` (CUDA torch + mamba-ssm + transformers/peft).

## Modules

| File | Purpose |
|------|---------|
| `data.py` | FocalDataset (multi-label) + SoundscapeDataset (unlabeled DAPT pool) |
| `losses.py` | multi-label BCE-focal + latent-bootstrap cosine |
| `encoder_forward.py` | grad-enabled pooled forward (wrappers' `extract_features` is no-grad) |
| `lora_adapt.py` | build wrapper + LoRA + head, train, save adapter |
| `ssamba_dapt.py` | **native** masked-spectrogram MPG/MPC DAPT (reuses exp 103 logic) |
| `eat_dapt.py` | **native** data2vec-2.0 port (EMA teacher + decoder), `peft:{none\|lora}` |
| `dapt.py` | generic pooled latent-bootstrap — SHELVED baseline (degenerate) |
| `ks_shift.py` | per-dim KS between two embedding caches (ported from exp 106) |
| `tests/test_eat_dapt.py` | EAT-port validation gate (run via `sbatch/validate-eat.sbatch`) |

Eval-side additions live in exp 300: `probe/proto_probe.py` (prototypical LOSO),
`extract --patch` (per-patch cache), `extract_patch_features` on the EAT/SSAMBA wrappers.

## Smoke tests (no cluster)

```bash
# 2-step LoRA: adapters attach, one focal batch flows through the focal loss
birdclef-301 lora --config configs/lora-ssamba.yaml --output-dir /tmp/lora-smoke --max-steps 2
# KS null check: identical parquet -> KS ~ 0
birdclef-301 ksshift --frozen X.parquet --adapted X.parquet --output /tmp/ks.json
```

## Key decisions

- **Ridge re-probe is the reported Δ** (apples-to-apples with exp 300 frozen).
- **DAPT is a generic latent-bootstrap** objective (genericity tradeoff in proposal).
- **BEATs LoRA-only** (tokenizer-bound SSL out of scope).
- `module load cuda/12.6.1` in the sbatch body, never in `_setup-venv.sh`.

## Status

**COMPLETE — geometry fix is the best result; epoch ladders finished 2026-06-13.** Full
results + history in `results.md`. The arc:

1. **First run (2026-06-11):** native full-DAPT at the wrapper's *5 s* geometry **hurt the
   ridge probe (−0.012/−0.013) but helped the prototypical frame-max readout (+0.010 EAT,
   +0.017 SSAMBA)** — same checkpoints, opposite signs → the ridge yardstick under-reads
   native domain-SSL (on-our-data Bird-MAE confirmation). Both validation gates GREEN
   (EAT port 10/10, SSAMBA 6/6).
2. **ep10 scale-up was null-to-negative:** SSAMBA frame-max +0.017→+0.003. Root cause:
   SSAMBA pads a 5 s window into its native **1024-frame buffer → ~half the masked targets
   are silence** (trivial); 10× longer overfits that degraded objective.
3. **Geometry fix = best result, at 1 epoch (HEADLINE):** feeding **native-length 10 s
   audio** (`data.window_samples: 160000`) fills the buffer with real audio. **SSAMBA
   frame-max +0.027 (from +0.017) and ridge FLIPS −0.012→+0.009**; **EAT +0.016 (from
   +0.010), ridge −0.013→+0.005** (EAT via `target_length: 1024`, no silence disease — its
   gain is train/pretrain length-match). Cross-encoder lesson: **match the encoder's native
   input length for DAPT.** Eval/extract stay 5 s (backbone geometry-shared).
4. **COMPLETE — 20-epoch ladders, both encoders** (`dapt-{ssamba,eat}-native-10s-ep20`,
   {3,10,20} + ep1, jobs 9911259/9913471, finished 2026-06-13). The epoch curve is
   **peak-early and the readout sets where the peak is: SSAMBA peaks ep1 under both readouts;
   EAT ridge peaks ep1 (0.5737) but EAT proto frame-max peaks ep10 (0.3050, its best).** Same
   checkpoints → the two instruments disagree on *when to stop*, the sharpest form of "ridge
   under-reads masked-SSL." No still-climbing curve past the peak → **epoch axis saturated.**
   **Held-out SSL val: no overfitting** at either encoder (train≈val within ~0.002 / ~0.0001).
   **Methodological catch:** don't read downstream off the SSL loss — EAT's `d2v_loss` was
   flat-to-rising ep1→ep10 while proto *climbed*; the SSL val's only job is the overfit check.
   Verdict: gains exhausted on the *epoch knob* of unsupervised DAPT; bigger eval + objective-
   direction / supervised levers still open (see `results.md` Deferred / next).

- **LoRA arms remain a cautionary finding:** EAT-LoRA moved the rep by *exactly zero* (KS
  0.0 — discarded data2vec decoder absorbed the objective); SSAMBA-LoRA moved a lot but
  unhelpfully. Useful adaptation lives in the backbone (full DAPT), not low-rank adapters.
- **Infra added this push:** per-epoch resumable `last.pt` (resubmit same OUTDIR
  auto-resumes) + `save_epochs` milestone snapshots in both runners; `probe.sbatch FORCE=1`
  to dodge the Luigi variant-collision (combined `probe_metrics_<variant>` is keyed by
  variant not encoder, so encoders sharing a variant string must force a rerun).
- **EAT save:** must save the outer `EATModel` state_dict (`model.`-prefixed) for
  `_apply_overlay`'s strict load (milestone snapshots and final save both do this).
- **Venv:** `_setup-venv.sh` mirrors exp 300's closure (pandas, timm, `transformers<5`,
  pinned torch trio, `mamba-ssm --no-build-isolation`); validate sbatch add `pytest`.

See `proposal.md` for hypothesis/rationale and `results.md` for the full Δ + KS tables.
