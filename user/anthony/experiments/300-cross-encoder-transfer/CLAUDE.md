# Experiment 300: Cross-Encoder Transfer

## What this experiment does

Places 6 audio encoders (BEATs, AST, EAT, SSAMBA-HF + Perch/BirdNET anchors) on
one BirdCLEF 2026 yardstick: extract a frozen embedding per labeled 5 s window,
then score each encoder with a ridge leave-one-site-out (LOSO) probe. Feeds §6 of
the working note. Exp 301 (encoder adaptation) reuses this extractor + probe.

## The unified cache schema (the 300 ↔ 301 contract)

One parquet per `(encoder, variant)` under `output.base_dir`:

```
embeddings/encoder={enc}/variant={var}/source=soundscape-labeled/data.parquet
```

Columns: `filename, window_idx, start_sec, source, split, primary_label,
ground_truth, encoder, variant, embed_dim, embedding (list[float32])`.
`variant` is `frozen` here; exp 301 writes `lora-r8`, `dapt-ep10`, … into the same
tree and re-runs the probe. 301 never imports 300's Python — it shells out to
`birdclef-300 extract --variant <tag> --checkpoint <ckpt>` then `birdclef-300 probe`.

## Encoder backends (registry.py)

| Backend | Encoders | Path | Venv |
|---|---|---|---|
| `torch` | beats, ast, eat, ssamba_hf | shared `birdclef.encoders` wrappers, 16 kHz | `.venv-300` (GPU) |
| `perch_cache` | perch | ingest exp 106 `soundscape_labeled.parquet` (no recompute) | `.venv-300` |
| `birdnet_bmz` | birdnet | bioacoustics-model-zoo TFLite, 3 s→5 s pooled | `.venv-300-birdnet` (TF) |

## Running

```bash
cd user/anthony/experiments/300-cross-encoder-transfer

# 1. Extract frozen embeddings (torch encoders, one GPU job each)
ENCODER=beats     sbatch sbatch/extract-torch.sbatch
ENCODER=ast       sbatch sbatch/extract-torch.sbatch
ENCODER=eat       sbatch sbatch/extract-torch.sbatch
ENCODER=ssamba_hf sbatch sbatch/extract-torch.sbatch

# 2. Anchors (CPU)
sbatch sbatch/ingest-perch.sbatch        # reuses exp 106 cache
sbatch sbatch/extract-birdnet.sbatch     # TF venv

# 3. Ridge LOSO probe over the whole roster
sbatch sbatch/probe.sbatch               # -> artifacts/probe_metrics_frozen.json
```

Venvs: `~/scratch/birdclef-2026/.venv-300` (torch) and `.venv-300-birdnet` (TF).

## Checkpoints expected on scratch

- BEATs: `~/scratch/birdclef-2026/checkpoints/beats/BEATs_iter3_plus_AS2M.pt`
- SSAMBA: `~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth`

(Both downloaded by exp 001's `download-checkpoints.sbatch`.) EAT/AST pull from HF.

## Smoke tests (no cluster)

```bash
# CPU-only end-to-end on 4 windows (SSAMBA-HF is CPU-capable + cosine-exact)
birdclef-300 extract --config configs/exp300.yaml --encoder ssamba_hf \
    --variant frozen --limit 4
# Probe math on the existing exp 106 labeled parquet (ingest perch first):
birdclef-300 extract --config configs/exp300.yaml --encoder perch --variant frozen
birdclef-300 probe --config configs/exp300.yaml --encoders perch --variant frozen
```

## Key decisions

- **Ridge probe only** (the exp 107 Hopfield head is intentionally dropped).
- **Per-fold standardization** (z-score on train sites) so the λ sweep is
  comparable across encoders with very different embedding scales.
- **Species list derived from the labeled parquet**, not exp 106's mapping —
  keeps exp 300 self-contained.
- `module load cuda/12.6.1` lives in the sbatch body, never in `_setup-venv.sh`.

## Status

**Complete (2026-06-11).** Ran end-to-end on PACE; full ranking + analysis in
`results.md`. Primary metric is **pooled-OOF macro ROC-AUC** (LB-aligned):
Perch 0.620 > BEATs 0.612 > BirdNET 0.588 > AST 0.576 > EAT 0.569 > SSAMBA 0.563.
Key finding: pooled-vs-per-fold aggregation reorders the field (BirdNET 6th→3rd).
Token pooling is configurable (`pooling: mean_patch|cls`); mean_patch is the
primary convention. See `proposal.md` for the hypothesis.
