<!-- Handoff prompt for the next Claude Code session. Copy-paste as opening message. -->

We'll be working from user/murilo/ in this session. I'm continuing from the previous session where we verified inference outputs and restructured directories.

## What happened last session

1. **v5j 4-fold inference COMPLETE** — SLURM job 6328686 finished successfully. All 4 folds × 2 sources × 3 parquet files verified:
   - `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_audio/fold{0-3}/` — 265,924 chunks each
   - `shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/fold{0-3}/` — 127,896 chunks each
   - Each folder has: `embeddings.parquet` (1024-dim), `logits.parquet` (234 species), `framewise_logits.parquet` (16 frames × 234)

2. **Directory layout restructured** — changed from `fold{N}/source/` to `source/fold{N}/` to match single-fold pattern. Code in `infer.py` (line 821) and `run.sh` (line 177) updated. Committed as `ba6e791`.

3. **Single-fold outputs also available** at `shared/inference/birdclef-2026/hgnet-b0-sed-v5j/{train_audio,train_soundscapes}/` (same chunk counts, same 3 parquet files).

## Current scoreboard

| Model | LB | Notes |
|-------|-----|-------|
| Perch ProtoSSM v18 (exp 020) | **0.925** | Best overall — two-kernel TF pipeline |
| v5j-4fold + pp2 (exp 021) | **0.917** | Best SED |
| v5j-4fold raw (exp 019) | 0.913 | |
| v5j single + pp2 (exp 021) | 0.909 | |
| v5j single raw (exp 019) | 0.906 | |
| Competition #1 | 0.939 | Target |

## What needs to happen now

Decide on the next experiment (exp 023). The extracted embeddings/logits from v5j open several paths:

### Option A: Perch + SED Ensemble (likely highest impact)
- Combine Perch ProtoSSM (0.925) with SED v5j-4fold (0.917) at the logit level
- Simple weighted average of calibrated probabilities
- Could push past 0.930 if models are complementary (Perch = frozen foundation, SED = trained from scratch)
- Can prototype quickly using existing Kaggle submission kernels

### Option B: Knowledge Distillation
- Use 4-fold averaged logits as soft labels for a new student model
- Train a single model that captures ensemble knowledge
- v5j teacher labels are calibrated across folds (OOF-style for train_audio)
- Risk: exp 015 showed pseudo-labels hurt with clip-level arch (but SED-to-SED might work)

### Option C: OOF Linear Probes on SED Embeddings
- Train linear probes on v5j embeddings (1024-dim) similar to Perch probe approach
- 4-fold structure enables proper OOF evaluation
- Could reveal if SED embeddings carry complementary information to Perch

### Option D: Framewise Analysis
- Use `framewise_logits.parquet` to analyze temporal detection patterns
- Identify species where framewise detection is weak → targeted data augmentation
- Build framewise ensembles (max vs mean vs attention-pooled across frames)

## Key learned patterns (from exp 019-022)

- **Training recipe changes never improve LB** — every modification to v5j's recipe (CE loss, waveform MixUp, AttHead, multi-context, manifold mixup, PCEN, longer epochs) regressed on LB
- **Offline metrics are anti-correlated with LB** — best val/auroc models score worst on LB
- **v5j recipe is a local optimum** — HGNetV2-B0 + LSEHead + BCE + spectrogram MixUp + AmplitudeToDB + NNAudio mel + 20 epochs
- **Ensembling/post-processing are the remaining levers** — pp2 priors give +0.003-0.004, 4-fold gives +0.007

## Key context files

- `user/murilo/inference/infer.py` — inference pipeline (SED + ProtoCLR support)
- `user/murilo/inference/config/hgnet_b0_sed_v5j_4fold.yaml` — 4-fold config
- `user/murilo/MODEL-RUNS.md` — single source of truth for all model scores
- `user/murilo/experiments/020-perch-onnx/` — Perch ProtoSSM v18 submission code
- `user/murilo/experiments/021-sed-training/` — SED training experiments (v6a-v8b)
- `docs/STATUS.md` — project status (needs updating)
