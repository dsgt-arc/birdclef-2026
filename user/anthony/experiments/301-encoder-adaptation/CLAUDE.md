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

**Complete — native-objective DAPT, first run (2026-06-11).** Full results in
`results.md`. Headline: native full-DAPT on soundscapes **hurts the linear ridge probe
(−0.012/−0.013) but helps under the MAE-appropriate prototypical frame-max readout
(+0.010 EAT, +0.017 SSAMBA)** — the same checkpoints, opposite signs → the ridge
yardstick under-reads native domain SSL (on-our-data confirmation of Bird-MAE).

- **Both validation gates GREEN** before full runs: EAT port 10/10 (FFN-target
  hand-recompute, non-degeneracy, state_dict round-trip, LoRA merge, overfit), SSAMBA 6/6
  (pos_embed geometry, grad-through-Mamba, round-trip, overfit).
- **LoRA arms are a cautionary finding:** EAT-LoRA moved the representation by *exactly
  zero* (KS 0.0) — the discarded data2vec decoder absorbed the objective; SSAMBA-LoRA
  moved a lot but unhelpfully. Useful adaptation lives in the backbone (full DAPT).
- **EAT save bug fixed:** must save the outer `EATModel` state_dict (`model.`-prefixed)
  for `_apply_overlay`'s strict load — see `results.md`.
- **Venv:** `_setup-venv.sh` mirrors exp 300's closure (pandas, timm, `transformers<5`,
  pinned torch trio, `mamba-ssm --no-build-isolation`); validate sbatch add `pytest`.
- **Scale:** 1 epoch, ~666 (SSAMBA) / 888 (EAT) steps, ~5 min/encoder. More epochs /
  windows-per-file is the cheap next knob (deltas are small but sign-consistent).

See `proposal.md` for hypothesis/rationale and `results.md` for the full Δ + KS tables.
