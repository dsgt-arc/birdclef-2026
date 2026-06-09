# Experiment 301: Encoder Adaptation

## What this experiment does

Adapts BEATs/EAT/SSAMBA-HF to the bird/soundscape domain and reports the
frozen→adapted Δ on exp 300's ridge LOSO probe. Two tracks: supervised-focal LoRA
(all three) and soundscape DAPT (EAT + SSAMBA-HF; BEATs LoRA-only). Feeds §7 of the
working note. It is a thin spoke — it never reimplements extraction or probing, it
shells out to `birdclef-300`.

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

# 1. Adapt (GPU)
CONFIG=configs/lora-eat.yaml    sbatch sbatch/lora.sbatch     # also lora-beats, lora-ssamba
CONFIG=configs/dapt-eat.yaml    sbatch sbatch/dapt.sbatch     # also dapt-ssamba

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
| `dapt.py` | data2vec/BYOL latent DAPT (EMA teacher), save state_dict |
| `ks_shift.py` | per-dim KS between two embedding caches (ported from exp 106) |

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

Scaffolded; not yet run. See `proposal.md` and `results.md`.
